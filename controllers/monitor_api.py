import time
import logging
import threading
import requests
from typing import Dict, List, Optional

from prometheus_client import start_http_server, Gauge, Counter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ------------------------------------------------------------------
# Cấu hình cluster
# ------------------------------------------------------------------

CONTROLLER_CONFIGS = [
    {"id": 1, "host": "127.0.0.1", "rest_port": 8080, "ofp_port": 6633},
    {"id": 2, "host": "127.0.0.1", "rest_port": 8081, "ofp_port": 6634},
    {"id": 3, "host": "127.0.0.1", "rest_port": 8082, "ofp_port": 6635},
]

SCRAPE_INTERVAL = 5       # giây giữa mỗi lần scrape
PROMETHEUS_PORT = 9100    # port Prometheus scrape
REQUEST_TIMEOUT = 2.0     # timeout gọi REST API mỗi controller

# ------------------------------------------------------------------
# Prometheus metrics definitions
# ------------------------------------------------------------------
# Label "controller_id" để phân biệt từng controller trên Grafana

_CPU = Gauge("sdn_controller_cpu_percent", "CPU usage của Ryu controller process (normalize 0-1)", ["controller_id"])
_RAM = Gauge("sdn_controller_ram_percent", "RAM usage của Ryu controller process (normalize 0-1)", ["controller_id"])
_PKT_IN_RATE = Gauge("sdn_controller_packet_in_rate", "Packet-in rate (normalize 0-1, cap tại 1000 pkt/s)", ["controller_id"])
_PKT_IN_TOTAL = Counter("sdn_controller_packet_in_total", "Tổng số packet-in từ khi khởi động", ["controller_id"])
_SWITCH_COUNT = Gauge("sdn_controller_switch_count", "Số switches đang kết nối với controller", ["controller_id"])
_CONTROLLER_UP = Gauge("sdn_controller_up", "Controller có đang hoạt động không (1=up, 0=down)", ["controller_id"])

# RL metrics — được cập nhật bởi rl_agent khi training
_RL_REWARD = Gauge("sdn_rl_reward", "Reward nhận được tại mỗi step của RL agent")
_RL_MIGRATION_COUNT = Gauge("sdn_rl_migration_count", "Số lần migrate trong episode hiện tại")
_RL_VARIANCE_CPU = Gauge("sdn_rl_variance_cpu", "Variance CPU giữa các controllers (mục tiêu minimize)")

# ------------------------------------------------------------------
# Scraper
# ------------------------------------------------------------------

class ControllerScraper:
    """
    Scrape metrics từ từng Ryu controller qua REST API /monitor/load.
    Chạy trong background thread, cập nhật Prometheus Gauge mỗi SCRAPE_INTERVAL giây.
    """

    def __init__(self, configs: List[Dict], interval: int = SCRAPE_INTERVAL):
        self.configs = configs
        self.interval = interval
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Track total packet-in để tính Counter đúng (Counter chỉ tăng)
        self._pkt_in_prev: Dict[int, int] = {c["id"]: 0 for c in configs}

    def _scrape_one(self, cfg: Dict) -> None:
        """Scrape một controller và cập nhật Prometheus metrics."""
        cid = str(cfg["id"])
        url = f"http://{cfg['host']}:{cfg['rest_port']}/monitor/load"

        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()

            _CPU.labels(controller_id=cid).set(data.get("cpu", 0.0))
            _RAM.labels(controller_id=cid).set(data.get("ram", 0.0))
            _PKT_IN_RATE.labels(controller_id=cid).set(data.get("packet_in_rate", 0.0))
            _SWITCH_COUNT.labels(controller_id=cid).set(data.get("switch_count", 0))
            _CONTROLLER_UP.labels(controller_id=cid).set(1)

            # Counter: chỉ tăng, tính delta
            total = int(data.get("packet_in_total", 0))
            prev = self._pkt_in_prev.get(cfg["id"], 0)
            delta = max(0, total - prev)
            if delta > 0:
                _PKT_IN_TOTAL.labels(controller_id=cid).inc(delta)
            self._pkt_in_prev[cfg["id"]] = total

        except requests.exceptions.ConnectionError:
            # Controller chưa khởi động hoặc đang down
            _CONTROLLER_UP.labels(controller_id=cid).set(0)
            logger.warning(f"[Scraper] Controller {cfg['id']} không phản hồi tại {url}")

        except Exception as e:
            _CONTROLLER_UP.labels(controller_id=cid).set(0)
            logger.error(f"[Scraper] Lỗi scrape controller {cfg['id']}: {e}")

    def _scrape_loop(self) -> None:
        """Vòng lặp scrape tất cả controllers."""
        logger.info(f"[Scraper] Bắt đầu scrape {len(self.configs)} controllers "
                    f"mỗi {self.interval}s")
        while self._running:
            for cfg in self.configs:
                self._scrape_one(cfg)
            time.sleep(self.interval)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._scrape_loop, daemon=True)
        self._thread.start()
        logger.info("[Scraper] Background thread đã khởi động")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[Scraper] Đã dừng")

# ------------------------------------------------------------------
# Public API — dùng bởi RL agent để push metrics lên Prometheus
# ------------------------------------------------------------------

def update_rl_metrics(reward: float, migration_count: int, variance_cpu: float) -> None:
    """
    RL agent gọi hàm này sau mỗi step để cập nhật Prometheus.

    Ví dụ trong sdn_env.py step():
        from controllers.monitor_api import update_rl_metrics
        update_rl_metrics(reward, len(self.migration_history), new_variance)
    """
    _RL_REWARD.set(reward)
    _RL_MIGRATION_COUNT.set(migration_count)
    _RL_VARIANCE_CPU.set(variance_cpu)

def get_cluster_state(configs: List[Dict] = None) -> List[Dict]:
    """
    Lấy state của tất cả controllers trong cluster qua REST API.
    Hàm này được dùng bởi api_client.py.

    Returns:
        List[Dict] — mỗi dict chứa metrics của 1 controller.
                     Trả về list rỗng nếu tất cả đều down.
    """
    if configs is None:
        configs = CONTROLLER_CONFIGS

    results = []
    for cfg in configs:
        url = f"http://{cfg['host']}:{cfg['rest_port']}/monitor/load"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            data = resp.json()
            data["ofp_port"] = cfg["ofp_port"]
            data["host"] = cfg["host"]
            results.append(data)
        except Exception as e:
            logger.warning(f"[ClusterState] Controller {cfg['id']} không phản hồi: {e}")
            results.append({
                "controller_id": cfg["id"],
                "cpu": 0.0, "ram": 0.0, "packet_in_rate": 0.0,
                "switch_count": 0, "up": False,
                "ofp_port": cfg["ofp_port"], "host": cfg["host"],
            })
    return results

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SDN Prometheus Exporter")
    parser.add_argument("--port", type=int, default=PROMETHEUS_PORT, help="Port để Prometheus scrape (default: 9100)")
    parser.add_argument("--interval", type=int, default=SCRAPE_INTERVAL, help="Scrape interval tính bằng giây (default: 5)")
    args = parser.parse_args()

    logger.info(f"Khởi động Prometheus exporter tại port {args.port}")
    start_http_server(args.port)
    logger.info(f"Metrics có tại: http://localhost:{args.port}/metrics")

    scraper = ControllerScraper(CONTROLLER_CONFIGS, interval=args.interval)
    scraper.start()

    logger.info("Đang chạy... Nhấn Ctrl+C để dừng")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        scraper.stop()
        logger.info("Đã dừng exporter")