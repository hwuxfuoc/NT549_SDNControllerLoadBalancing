import time
import logging
import threading
import requests
from typing import Dict, List, Optional

from prometheus_client import start_http_server, Gauge, Counter, Info

from monitoring.system_monitor import ControllerProcessMonitor

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Cấu hình — khớp với monitor_api.py CONTROLLER_CONFIGS
# ------------------------------------------------------------------

CONTROLLER_CONFIGS = [
    {"id": 1, "host": "127.0.0.1", "rest_port": 8080, "ofp_port": 6633},
    {"id": 2, "host": "127.0.0.1", "rest_port": 8081, "ofp_port": 6634},
    {"id": 3, "host": "127.0.0.1", "rest_port": 8082, "ofp_port": 6635},
]

SCRAPE_INTERVAL = 5       # giây giữa mỗi lần scrape
PROMETHEUS_PORT = 9100
REQUEST_TIMEOUT = 2.0

# ------------------------------------------------------------------
# Prometheus metric objects
# (dùng tên metric KHÁC với monitor_api.py để tránh duplicate nếu chạy cùng lúc)
# ------------------------------------------------------------------

_CPU = Gauge("sdn_cpu_percent", "CPU usage Ryu process (psutil, normalize 0-1)", ["controller_id"])
_RAM = Gauge("sdn_ram_percent", "RAM usage Ryu process (psutil, normalize 0-1)", ["controller_id"])
_PKT_RATE = Gauge("sdn_packet_in_rate", "Packet-in rate từ Ryu REST API (normalize 0-1)", ["controller_id"])
_SWITCH_COUNT = Gauge("sdn_switch_count", "Số switches kết nối với controller", ["controller_id"])
_UP = Gauge("sdn_controller_up", "Controller hoạt động (1=up, 0=down)", ["controller_id"])
_PKT_TOTAL = Counter("sdn_packet_in_total", "Tổng packet-in từ khi khởi động", ["controller_id"])

# RL metrics — push bởi RL agent qua update_rl_metrics()
_RL_REWARD = Gauge("sdn_rl_reward", "Reward mỗi step của RL agent")
_RL_VARIANCE = Gauge("sdn_rl_variance_cpu", "Variance CPU giữa các controllers")
_RL_MIGRATION = Gauge("sdn_rl_migration_count", "Số migration trong episode")
_RL_LATENCY = Gauge("sdn_rl_latency", "Latency normalize (mỗi step)")

# ------------------------------------------------------------------
# Public API — RL agent gọi sau mỗi step
# ------------------------------------------------------------------

def update_rl_metrics(reward: float, variance_cpu: float, migration_count: int, latency: float = 0.0) -> None:
    """
    RL agent gọi hàm này sau mỗi step để push metrics lên Prometheus.

    Ví dụ trong sdn_env.py step():
        from monitoring.prometheus_exporter import update_rl_metrics
        update_rl_metrics(reward, new_variance, len(self.migration_history), self.current_latency)
    """
    _RL_REWARD.set(float(reward))
    _RL_VARIANCE.set(float(variance_cpu))
    _RL_MIGRATION.set(float(migration_count))
    _RL_LATENCY.set(float(latency))


# ------------------------------------------------------------------
# Exporter core
# ------------------------------------------------------------------

class SDNPrometheusExporter:
    """
    Kết hợp psutil (qua ControllerProcessMonitor) và Ryu REST API
    để expose metrics lên Prometheus.

    Luồng dữ liệu:
        psutil → _process_monitor → CPU/RAM metrics
        Ryu REST API → packet_in_rate, switch_count
        RL agent → update_rl_metrics() → RL metrics
    """

    def __init__(self, controller_configs: List[Dict] = CONTROLLER_CONFIGS, scrape_interval: float = SCRAPE_INTERVAL, controller_pids: Optional[Dict[int, int]] = None):
        self.configs = controller_configs
        self.scrape_interval = scrape_interval

        # psutil monitor — nguồn chính cho CPU/RAM
        self._sys_monitor = ControllerProcessMonitor(controller_pids = controller_pids, num_controllers = len(controller_configs), monitor_interval = scrape_interval)

        # Tracking packet_in_total để tính Counter đúng
        self._pkt_total_prev: Dict[int, int] = {c["id"]: 0 for c in controller_configs}

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def _fetch_ryu_metrics(self, cfg: Dict) -> Optional[Dict]:
        """Gọi Ryu REST API để lấy packet_in_rate và switch_count."""
        url = f"http://{cfg['host']}:{cfg['rest_port']}/monitor/load"
        
        try:
            resp = requests.get(url, timeout = REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        
        except Exception:
            return None

    def _scrape_and_update(self) -> None:
        """Một vòng scrape: lấy metrics từ psutil + Ryu, cập nhật Prometheus."""
        for cfg in self.configs:
            cid = cfg["id"]
            label = str(cid)

            # 1. CPU/RAM từ psutil (chính xác, không phụ thuộc network)
            cpu, ram = self._sys_monitor.get_cpu_ram(cid)
            _CPU.labels(controller_id = label).set(cpu)
            _RAM.labels(controller_id = label).set(ram)

            # 2. Packet-in rate và switch count từ Ryu REST API
            ryu_data = self._fetch_ryu_metrics(cfg)
            if ryu_data is not None:
                _PKT_RATE.labels(controller_id = label).set(float(ryu_data.get("packet_in_rate", 0.0)))
                _SWITCH_COUNT.labels(controller_id = label).set(int(ryu_data.get("switch_count", 0)))
                _UP.labels(controller_id = label).set(1)

                # Counter: chỉ tăng
                total = int(ryu_data.get("packet_in_total", 0))
                delta = max(0, total - self._pkt_total_prev.get(cid, 0))
                if delta > 0:
                    _PKT_TOTAL.labels(controller_id = label).inc(delta)
                self._pkt_total_prev[cid] = total
            else:
                _PKT_RATE.labels(controller_id = label).set(0.0)
                _SWITCH_COUNT.labels(controller_id = label).set(0)
                _UP.labels(controller_id = label).set(0)

    def _scrape_loop(self) -> None:
        logger.info(f"[Exporter] Scrape loop bắt đầu (interval = {self.scrape_interval}s)")
        while self._running:
            self._scrape_and_update()
            time.sleep(self.scrape_interval)

    def start(self) -> None:
        """Khởi động psutil monitor + scrape loop."""
        self._sys_monitor.start()
        self._running = True
        self._thread = threading.Thread(target = self._scrape_loop, daemon = True)
        self._thread.start()
        logger.info("[Exporter] Đã khởi động")

    def stop(self) -> None:
        self._running = False
        self._sys_monitor.stop()
        if self._thread:
            self._thread.join(timeout = 5)
        logger.info("[Exporter] Đã dừng")

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description = "SDN Prometheus Exporter")
    parser.add_argument("--port", type = int, default = PROMETHEUS_PORT)
    parser.add_argument("--interval", type = float, default = SCRAPE_INTERVAL)
    args = parser.parse_args()

    logger.info(f"Khởi động Prometheus exporter tại port {args.port}")
    start_http_server(args.port)
    logger.info(f"Metrics: http://localhost:{args.port}/metrics")

    exporter = SDNPrometheusExporter(scrape_interval = args.interval)
    exporter.start()

    logger.info("Đang chạy... Ctrl+C để dừng")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        exporter.stop()