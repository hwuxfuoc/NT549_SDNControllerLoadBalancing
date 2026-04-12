import os
import time
import logging
import threading
import psutil
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Cấu hình
# ------------------------------------------------------------------

MONITOR_INTERVAL = 2.0      # giây giữa mỗi lần đo
CPU_INTERVAL = 1.0          # giây psutil dùng để tính CPU% (blocking)
PACKET_IN_NORM_CAP = 1000   # packet/s tương ứng với normalize = 1.0

# Mapping controller_id (1-indexed) → cổng OFP (để tìm process)
CONTROLLER_OFP_PORTS = {1: 6633, 2: 6634, 3: 6635}
NUM_CONTROLLERS = 3

class ControllerProcessMonitor:
    """
    Theo dõi tài nguyên của từng Ryu controller process.

    Cách tìm process:
    - Ưu tiên: dùng PID được truyền vào trực tiếp (chính xác nhất)
    - Fallback: tìm process có cmdline chứa tên file ryu_app_cN.py
    """

    def __init__(self, controller_pids: Optional[Dict[int, int]] = None, num_controllers: int = NUM_CONTROLLERS, monitor_interval: float = MONITOR_INTERVAL):
        """
        Args:
            controller_pids: Dict {controller_id: pid} — nếu biết trước PID.
                             Nếu None, tự động tìm process theo tên.
            num_controllers: Số controller trong cluster.
            monitor_interval: Chu kỳ đo (giây).
        """
        self.num_controllers = num_controllers
        self.monitor_interval = monitor_interval

        # {controller_id: psutil.Process}
        self._processes: Dict[int, Optional[psutil.Process]] = {}

        # Cache metrics mới nhất — được cập nhật bởi background thread
        # {controller_id: {"cpu": float, "ram": float, "timestamp": float}}
        self._latest_metrics: Dict[int, Dict] = {
            i: {"cpu": 0.0, "ram": 0.0, "timestamp": 0.0}
            for i in range(1, num_controllers + 1)
        }

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Gán process từ PID nếu có
        if controller_pids:
            for cid, pid in controller_pids.items():
                try:
                    self._processes[cid] = psutil.Process(pid)
                    logger.info(f"[Monitor] C{cid} → PID {pid}")
                except psutil.NoSuchProcess:
                    logger.warning(f"[Monitor] PID {pid} không tồn tại cho C{cid}")
                    self._processes[cid] = None
        else:
            self._auto_discover_processes()

    # ------------------------------------------------------------------
    # Process discovery
    # ------------------------------------------------------------------

    def _auto_discover_processes(self) -> None:
        """
        Tự động tìm Ryu controller processes theo tên file trong cmdline.
        Tìm 'ryu_app_c1.py', 'ryu_app_c2.py', 'ryu_app_c3.py'.
        """
        logger.info("[Monitor] Tự động tìm Ryu processes...")
        found = {}

        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                for cid in range(1, self.num_controllers + 1):
                    if f"ryu_app_c{cid}" in cmdline or f"ryu-manager" in cmdline:
                        if cid not in found:
                            found[cid] = proc
                            logger.info(f"[Monitor] Tìm thấy C{cid}: PID = {proc.pid} | {cmdline[:60]}")

            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        for cid in range(1, self.num_controllers + 1):
            self._processes[cid] = found.get(cid, None)
            if self._processes[cid] is None:
                logger.warning(f"[Monitor] Không tìm thấy process cho C{cid} — sẽ dùng giá trị 0")

    def rediscover(self) -> None:
        """Gọi lại auto-discovery — dùng khi controller restart."""
        self._auto_discover_processes()

    # ------------------------------------------------------------------
    # Đo metrics
    # ------------------------------------------------------------------

    def _measure_one(self, controller_id: int) -> Dict:
        """
        Đo CPU và RAM của một controller process.

        Returns:
            Dict {"cpu": float [0,1], "ram": float [0,1], "pid": int, "alive": bool}
        """
        proc = self._processes.get(controller_id)

        if proc is None:
            return {"cpu": 0.0, "ram": 0.0, "pid": -1, "alive": False}

        try:
            # cpu_percent(interval=None) → non-blocking, dùng delta từ lần gọi trước
            cpu_raw = proc.cpu_percent(interval = None)
            mem_info = proc.memory_percent()

            # Số CPU core → normalize CPU% về [0,1] per-core
            n_cpu = psutil.cpu_count(logical = True) or 1
            cpu_norm = min(cpu_raw / (n_cpu * 100.0), 1.0)
            ram_norm = min(mem_info / 100.0, 1.0)

            return {
                "cpu": round(cpu_norm, 4),
                "ram": round(ram_norm, 4),
                "pid": proc.pid,
                "alive": True,
            }
        except psutil.NoSuchProcess:
            logger.warning(f"[Monitor] C{controller_id} process đã tắt (PID={proc.pid})")
            self._processes[controller_id] = None
            return {"cpu": 0.0, "ram": 0.0, "pid": -1, "alive": False}
        
        except psutil.AccessDenied:
            logger.warning(f"[Monitor] Không có quyền đọc process C{controller_id}")
            return {"cpu": 0.0, "ram": 0.0, "pid": -1, "alive": False}

    def measure_all(self) -> Dict[int, Dict]:
        """
        Đo tất cả controllers ngay lập tức (blocking).

        Returns:
            Dict {controller_id: {"cpu": float, "ram": float, "pid": int, "alive": bool}}
        """
        results = {}
        for cid in range(1, self.num_controllers + 1):
            results[cid] = self._measure_one(cid)
        return results

    # ------------------------------------------------------------------
    # Background monitoring loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """Background thread: đo định kỳ và cache kết quả."""
        # Lần đầu gọi cpu_percent để khởi tạo baseline (kết quả = 0.0, bỏ qua)
        for cid in range(1, self.num_controllers + 1):
            proc = self._processes.get(cid)
            if proc:
                try:
                    proc.cpu_percent(interval = None)
                except Exception:
                    pass

        while self._running:
            time.sleep(self.monitor_interval)
            metrics = self.measure_all()
            with self._lock:
                for cid, m in metrics.items():
                    self._latest_metrics[cid] = {
                        "cpu": m["cpu"],
                        "ram": m["ram"],
                        "timestamp": time.time(),
                    }

    def start(self) -> None:
        """Khởi động background monitoring thread."""
        self._running = True
        self._thread = threading.Thread(target = self._monitor_loop, daemon = True)
        self._thread.start()
        logger.info(f"[Monitor] Background thread khởi động (interval = {self.monitor_interval}s)")

    def stop(self) -> None:
        """Dừng background monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout = 5)
        logger.info("[Monitor] Đã dừng")

    # ------------------------------------------------------------------
    # Public API cho RL agent
    # ------------------------------------------------------------------

    def get_cpu_ram(self, controller_id: int) -> Tuple[float, float]:
        """
        Trả về (cpu, ram) normalize [0,1] của controller_id từ cache.
        Không blocking — lấy từ lần đo gần nhất.
        """
        with self._lock:
            m = self._latest_metrics.get(controller_id, {"cpu": 0.0, "ram": 0.0})
        return m["cpu"], m["ram"]

    def get_state_vector(self, packet_in_rates: Optional[Dict[int, float]] = None) -> List[float]:
        """
        Trả về state vector cho RL agent:
            [cpu_c1, ram_c1, pkt_in_c1, cpu_c2, ram_c2, pkt_in_c2, ...]

        Args:
            packet_in_rates: Dict {controller_id: rate_normalized} từ Ryu API.
                             Nếu None → dùng 0.0 cho tất cả.

        Returns:
            List[float] length = num_controllers * 3, mỗi giá trị trong [0,1].
        """
        state = []
        pkt_rates = packet_in_rates or {}
        for cid in range(1, self.num_controllers + 1):
            cpu, ram = self.get_cpu_ram(cid)
            pkt = float(pkt_rates.get(cid, 0.0))
            state.extend([cpu, ram, pkt])
        return state

    def get_summary(self) -> Dict:
        """Trả về tóm tắt trạng thái cluster."""
        with self._lock:
            metrics = dict(self._latest_metrics)
        cpus = [m["cpu"] for m in metrics.values()]
        rams = [m["ram"] for m in metrics.values()]
        import numpy as np

        return {
            "controllers": metrics,
            "cpu_variance": float(np.var(cpus)),
            "cpu_mean": float(np.mean(cpus)),
            "ram_mean": float(np.mean(rams)),
        }

    def print_summary(self) -> None:
        """In trạng thái cluster ra console."""
        summary = self.get_summary()
        print("\n=== System Monitor Summary ===")
        for cid, m in summary["controllers"].items():
            print(f"  Controller {cid}: CPU={m['cpu']:.3f}  RAM={m['ram']:.3f}")
        print(f"  CPU Variance: {summary['cpu_variance']:.4f}")
        print(f"  CPU Mean:     {summary['cpu_mean']:.3f}")
        print("=" * 30)

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="SDN System Monitor")
    parser.add_argument("--interval", type = float, default = 2.0, help = "Chu kỳ đo (giây, default: 2)")
    parser.add_argument("--duration", type = int, default = 60, help = "Thời gian chạy (giây, default: 60)")
    args = parser.parse_args()

    monitor = ControllerProcessMonitor(monitor_interval = args.interval)
    monitor.start()

    print(f"Đang theo dõi {NUM_CONTROLLERS} controllers trong {args.duration}s...\n")
    try:
        for _ in range(args.duration // max(1, int(args.interval))):
            time.sleep(args.interval)
            monitor.print_summary()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()