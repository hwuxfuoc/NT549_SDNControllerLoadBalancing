import requests
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cấu hình mặc định cluster — khớp với monitor_api.py CONTROLLER_CONFIGS
DEFAULT_CONTROLLER_CONFIGS = [
    {"id": 1, "host": "127.0.0.1", "rest_port": 8080, "ofp_port": 6633},
    {"id": 2, "host": "127.0.0.1", "rest_port": 8081, "ofp_port": 6634},
    {"id": 3, "host": "127.0.0.1", "rest_port": 8082, "ofp_port": 6635},
]

DEFAULT_TIMEOUT = 2.0  # giây

class RyuAPIClient:
    """
    Client gọi REST API của cluster Ryu controllers.

    Cung cấp:
    - get_controller_metrics(controller_id): metrics của 1 controller
    - get_cluster_state(): metrics tất cả controllers → state vector cho RL
    - get_state_vector(): trả về np.ndarray đúng format sdn_env.py
    - get_switch_list(controller_id): danh sách switches của controller
    - health_check(): kiểm tra controller nào đang up
    """

    def __init__(self, controller_configs: Optional[List[Dict]] = None, timeout: float = DEFAULT_TIMEOUT):
        self.configs = controller_configs or DEFAULT_CONTROLLER_CONFIGS
        self.timeout = timeout

        # Cache URL để không tính lại mỗi lần
        self._urls: Dict[int, str] = { cfg["id"]: f"http://{cfg['host']}:{cfg['rest_port']}" for cfg in self.configs }

        logger.info(f"[APIClient] Khởi tạo với {len(self.configs)} controllers: " + ", ".join(f"C{c['id']}:{c['rest_port']}" for c in self.configs))

    # ------------------------------------------------------------------
    # Core fetch methods
    # ------------------------------------------------------------------

    def get_controller_metrics(self, controller_id: int) -> Optional[Dict]:
        """
        Lấy metrics của 1 controller: cpu, ram, packet_in_rate, switch_count.

        Returns:
            Dict hoặc None nếu controller không phản hồi.
        """
        url = self._urls.get(controller_id)
        if url is None:
            logger.error(f"[APIClient] Không tìm thấy config cho controller_id = {controller_id}")
            return None

        try:
            resp = requests.get(f"{url}/monitor/load", timeout = self.timeout)
            resp.raise_for_status()
            return resp.json()
        
        except requests.exceptions.ConnectionError:
            logger.warning(f"[APIClient] Controller {controller_id} không phản hồi ({url})")
            return None
        
        except Exception as e:
            logger.error(f"[APIClient] Lỗi lấy metrics controller {controller_id}: {e}")
            return None

    def get_cluster_state(self) -> List[Optional[Dict]]:
        """
        Lấy metrics của tất cả controllers trong cluster.

        Returns:
            List[Dict | None] — theo thứ tự controller_id tăng dần.
            None tại vị trí controller đang down.
        """
        results = []

        for cfg in sorted(self.configs, key = lambda c: c["id"]):
            metrics = self.get_controller_metrics(cfg["id"])
            results.append(metrics)

        return results

    def get_state_vector(self) -> np.ndarray:
        """
        Trả về state vector đúng format của sdn_env.py:
            [cpu_c1, ram_c1, pkt_in_c1, cpu_c2, ram_c2, pkt_in_c2, ...]
        Normalize về [0, 1].

        Nếu controller down → dùng giá trị fallback 0.5 (unknown).

        Returns:
            np.ndarray shape (num_controllers * 3,)
        """
        cluster = self.get_cluster_state()
        state = []

        for metrics in cluster:
            if metrics is None:
                state.extend([0.5, 0.5, 0.5])  # fallback khi controller down
            else:
                state.extend([
                    float(metrics.get("cpu", 0.5)),
                    float(metrics.get("ram", 0.5)),
                    float(metrics.get("packet_in_rate", 0.5)),
                ])

        return np.clip(np.array(state, dtype=np.float32), 0.0, 1.0)

    def get_switch_list(self, controller_id: int) -> List[str]:
        """
        Lấy danh sách dpid của switches đang kết nối với controller.

        Returns:
            List[str] — list hex dpid, ví dụ ["0x1", "0x2"]
        """
        url = self._urls.get(controller_id)

        if url is None:
            return []
        
        try:
            resp = requests.get(f"{url}/monitor/switches", timeout = self.timeout)
            resp.raise_for_status()
            data = resp.json()
            return data.get("switches", [])
        
        except Exception as e:
            logger.warning(f"[APIClient] Không lấy được switch list từ C{controller_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[int, bool]:
        """
        Kiểm tra từng controller có đang up không.

        Returns:
            Dict {controller_id: True/False}
        """
        status = {}

        for cfg in self.configs:
            cid = cfg["id"]
            metrics = self.get_controller_metrics(cid)
            status[cid] = metrics is not None
            state_str = "UP" if status[cid] else "DOWN"
            logger.info(f"[HealthCheck] Controller {cid}: {state_str}")

        return status

    def get_most_loaded_controller(self) -> Tuple[int, float]:
        """
        Tìm controller đang bị tải nặng nhất theo CPU.

        Returns:
            (controller_id, cpu_value) của controller overloaded nhất.
        """
        cluster = self.get_cluster_state()
        max_cpu = -1.0
        max_ctrl = self.configs[0]["id"]

        for i, metrics in enumerate(cluster):
            if metrics is None:
                continue
            cpu = float(metrics.get("cpu", 0.0))
            if cpu > max_cpu:
                max_cpu = cpu
                max_ctrl = self.configs[i]["id"]

        return max_ctrl, max_cpu

    def get_least_loaded_controller(self, exclude_id: Optional[int] = None) -> Tuple[int, float]:
        """
        Tìm controller đang ít tải nhất theo CPU.

        Args:
            exclude_id: controller_id cần loại trừ (thường là controller đang migrate ra).

        Returns:
            (controller_id, cpu_value) của controller ít tải nhất.
        """
        cluster = self.get_cluster_state()
        min_cpu = 2.0
        min_ctrl = self.configs[0]["id"]

        for i, metrics in enumerate(cluster):
            cid = self.configs[i]["id"]
            if cid == exclude_id:
                continue
            if metrics is None:
                continue
            cpu = float(metrics.get("cpu", 1.0))
            if cpu < min_cpu:
                min_cpu = cpu
                min_ctrl = cid

        return min_ctrl, min_cpu

    def print_cluster_summary(self) -> None:
        """In tóm tắt trạng thái cluster ra console — dùng để debug."""
        cluster = self.get_cluster_state()
        print("\n=== Cluster Controller Summary ===")

        for i, metrics in enumerate(cluster):
            cid = self.configs[i]["id"]
            if metrics is None:
                print(f"  Controller {cid}: [DOWN]")
            else:
                print(
                    f"  Controller {cid}: "
                    f"CPU={metrics.get('cpu', 0):.3f}  "
                    f"RAM={metrics.get('ram', 0):.3f}  "
                    f"PktIn={metrics.get('packet_in_rate', 0):.3f}  "
                    f"Switches={metrics.get('switch_count', 0)}"
                )
        print("=" * 34)

if __name__ == "__main__":
    logging.basicConfig(level = logging.INFO)

    client = RyuAPIClient()

    print("\n--- Health Check ---")
    status = client.health_check()
    print(status)

    print("\n--- State Vector (RL input) ---")
    vec = client.get_state_vector()
    print(f"Shape: {vec.shape} | Values: {vec}")

    client.print_cluster_summary()