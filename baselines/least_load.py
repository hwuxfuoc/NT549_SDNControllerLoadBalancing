"""
least_load.py
-------------
Baseline: Least-Load Balancing cho SDN Cluster Controllers.

Logic: Tại mỗi bước, tìm controller đang bị overload nhất (CPU cao nhất)
và migrate switch có ảnh hưởng lớn nhất sang controller ít tải nhất (CPU thấp nhất).

Khác với Round-Robin: Least-Load NHÌN VÀO load thực tế để quyết định,
nhưng vẫn là thuật toán greedy (tham lam tại thời điểm hiện tại),
không học từ kinh nghiệm như RL agent.

Dùng để so sánh với RL agent trong evaluate.py.
"""

import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class LeastLoadBalancer:
    """
    Least-Load balancer cho cluster SDN controllers.

    Attributes:
        num_controllers:   Số controller trong cluster.
        num_switches:      Tổng số switch trong mạng.
        switch_assignment: Mảng ánh xạ switch_id → controller_id.
        overload_threshold: Ngưỡng CPU để kích hoạt migration (mặc định 0.7).
        imbalance_threshold: Chênh lệch CPU tối thiểu để migrate (mặc định 0.15).
    """

    def __init__(
        self,
        num_controllers: int = 3,
        num_switches: int = 12,
        overload_threshold: float = 0.7,
        imbalance_threshold: float = 0.15,
    ):
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.overload_threshold = overload_threshold
        self.imbalance_threshold = imbalance_threshold

        self.switch_assignment: np.ndarray = np.full(num_switches, -1, dtype=int)
        self._initial_assignment()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_assignment(self) -> None:
        """Phân phối switch ban đầu (mất cân bằng để test)."""
        self.switch_assignment = np.array([i % self.num_controllers for i in range(self.num_switches)])
        # Giả lập controller 0 bị overload (giống sdn_env.py)
        n_overload = int(self.num_switches * 0.6)
        self.switch_assignment[:n_overload] = 0

        counts = np.bincount(self.switch_assignment, minlength=self.num_controllers).tolist()
        logger.info(f"[LeastLoad] Phân phối ban đầu: {counts} switches/controller")

    def _mock_load(self) -> np.ndarray:
        """Mock load tỷ lệ với số switch mỗi controller giữ."""
        counts = np.bincount(self.switch_assignment, minlength=self.num_controllers).astype(float)
        load = np.zeros((self.num_controllers, 3), dtype=np.float32)
        for i in range(self.num_controllers):
            base = counts[i] / self.num_switches
            load[i, 0] = np.clip(base + np.random.normal(0, 0.04), 0.0, 1.0)  # cpu
            load[i, 1] = np.clip(base * 0.8 + np.random.normal(0, 0.03), 0.0, 1.0)  # ram
            load[i, 2] = np.clip(base + np.random.normal(0, 0.05), 0.0, 1.0)  # packet_in
        return load

    def _find_best_switch_to_migrate(self, from_controller: int) -> Optional[int]:
        """
        Chọn switch nào của `from_controller` nên migrate.

        Chiến lược: Chọn switch đầu tiên tìm thấy (greedy đơn giản).
        Có thể mở rộng: chọn switch có packet_in_rate cao nhất.

        Returns:
            switch_id hoặc None nếu controller không giữ switch nào.
        """
        switches = np.where(self.switch_assignment == from_controller)[0]
        if len(switches) == 0:
            return None
        return int(switches[0])

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def decide_migration(self, current_loads: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Quyết định migrate dựa trên load thực tế (greedy).

        Logic:
        1. Tìm controller có CPU cao nhất (most overloaded).
        2. Tìm controller có CPU thấp nhất (least loaded).
        3. Nếu chênh lệch CPU > imbalance_threshold VÀ
           controller overloaded vượt ngưỡng → migrate.

        Args:
            current_loads: np.ndarray shape (num_controllers, 3)
                           cột 0 = cpu, cột 1 = ram, cột 2 = packet_in_rate.

        Returns:
            (switch_id, target_controller) hoặc None nếu không cần migrate.
        """
        if current_loads.shape != (self.num_controllers, 3):
            logger.error(
                f"[LeastLoad] current_loads shape sai: "
                f"{current_loads.shape} ≠ ({self.num_controllers}, 3)"
            )
            return None

        cpu_loads = current_loads[:, 0]
        most_loaded_ctrl = int(np.argmax(cpu_loads))
        least_loaded_ctrl = int(np.argmin(cpu_loads))

        max_cpu = float(cpu_loads[most_loaded_ctrl])
        min_cpu = float(cpu_loads[least_loaded_ctrl])

        # Không migrate nếu:
        # - Không có sự chênh lệch đáng kể
        # - Controller overload chưa vượt ngưỡng
        # - most == least (chỉ có 1 controller)
        if most_loaded_ctrl == least_loaded_ctrl:
            return None
        if max_cpu < self.overload_threshold:
            return None
        if max_cpu - min_cpu < self.imbalance_threshold:
            return None

        switch_id = self._find_best_switch_to_migrate(most_loaded_ctrl)
        if switch_id is None:
            logger.warning(
                f"[LeastLoad] Controller {most_loaded_ctrl} overload "
                f"nhưng không có switch để migrate"
            )
            return None

        logger.debug(
            f"[LeastLoad] Overload ctrl={most_loaded_ctrl} (CPU={max_cpu:.2f}) → "
            f"migrate s{switch_id} sang ctrl={least_loaded_ctrl} (CPU={min_cpu:.2f})"
        )
        return switch_id, least_loaded_ctrl

    def execute_migration(self, switch_id: int, target_controller: int) -> bool:
        """
        Thực thi migration: cập nhật switch_assignment.

        Returns:
            True nếu thành công.
        """
        if not (0 <= switch_id < self.num_switches):
            logger.error(f"[LeastLoad] switch_id {switch_id} không hợp lệ")
            return False
        if not (0 <= target_controller < self.num_controllers):
            logger.error(f"[LeastLoad] target_controller {target_controller} không hợp lệ")
            return False

        old_ctrl = int(self.switch_assignment[switch_id])
        if old_ctrl == target_controller:
            return False  # no-op

        self.switch_assignment[switch_id] = target_controller
        logger.debug(f"[LeastLoad] s{switch_id}: ctrl {old_ctrl} → {target_controller}")
        return True

    def get_load_distribution(self) -> Dict[int, int]:
        """Trả về số switch mỗi controller đang giữ."""
        counts = np.bincount(self.switch_assignment, minlength=self.num_controllers)
        return {i: int(counts[i]) for i in range(self.num_controllers)}

    def run_episode(self, num_steps: int = 200, load_fn=None) -> Dict[str, float]:
        """
        Chạy một episode để lấy metrics so sánh với RL agent.

        Args:
            num_steps: Số bước mỗi episode.
            load_fn:   Hàm nhận switch_assignment → np.ndarray shape (num_controllers, 3).
                       Nếu None dùng mock.

        Returns:
            Dict metrics: mean_variance_cpu, final_variance_cpu,
                          migration_count, mean_max_cpu.
        """
        self.reset()
        migration_count = 0
        variances = []
        max_cpus = []

        for step in range(num_steps):
            loads = load_fn(self.switch_assignment) if load_fn else self._mock_load()

            cpu = loads[:, 0]
            variances.append(float(np.var(cpu)))
            max_cpus.append(float(np.max(cpu)))

            result = self.decide_migration(loads)
            if result is not None:
                sw_id, tgt = result
                if self.execute_migration(sw_id, tgt):
                    migration_count += 1

        metrics = {
            "mean_variance_cpu": float(np.mean(variances)),
            "final_variance_cpu": float(variances[-1]) if variances else 0.0,
            "migration_count": migration_count,
            "mean_max_cpu": float(np.mean(max_cpus)),
        }
        logger.info(
            f"[LeastLoad] Episode done — "
            f"Var: {metrics['mean_variance_cpu']:.4f} | "
            f"MaxCPU: {metrics['mean_max_cpu']:.3f} | "
            f"Migrations: {migration_count}"
        )
        return metrics

    def reset(self) -> None:
        """Reset về trạng thái mất cân bằng ban đầu để test."""
        self._initial_assignment()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    balancer = LeastLoadBalancer(
        num_controllers=3,
        num_switches=12,
        overload_threshold=0.7,
        imbalance_threshold=0.15,
    )

    print("\n=== Phân phối ban đầu (mất cân bằng) ===")
    print(balancer.get_load_distribution())

    print("\n=== Chạy episode 200 bước ===")
    metrics = balancer.run_episode(num_steps=200)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")