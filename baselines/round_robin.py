import numpy as np
import logging
from typing import Dict, Optional, Tuple

from baselines.policy_utils import encode_action

logger = logging.getLogger(__name__)

class RoundRobinBalancer:
    """
    Round-Robin balancer cho cluster SDN controllers.

    Attributes:
        num_controllers:   Số controller trong cluster.
        num_switches:      Tổng số switch trong mạng.
        switch_assignment: Mảng ánh xạ switch_id → controller_id.
    """

    def __init__(self, num_controllers: int = 3, num_switches: int = 12):
        self.num_controllers = num_controllers
        self.num_switches = num_switches

        # Con trỏ vòng tròn — controller nào sẽ nhận switch tiếp theo
        self._rr_pointer: int = 0

        # switch_assignment[i] = controller đang giữ switch i
        self.switch_assignment: np.ndarray = np.full(num_switches, -1, dtype = int)

        # Phân phối ban đầu theo Round-Robin
        self._initial_assignment()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _initial_assignment(self) -> None:
        """Phân phối switch đều ban đầu theo Round-Robin."""
        for sw in range(self.num_switches):
            self.switch_assignment[sw] = sw % self.num_controllers
        self._rr_pointer = self.num_switches % self.num_controllers
        counts = np.bincount(self.switch_assignment, minlength = self.num_controllers).tolist()
        logger.info(f"[RoundRobin] Phân phối ban đầu: {counts} switches/controller")

    def _mock_load(self) -> np.ndarray:
        """Mock load tỷ lệ với số switch mỗi controller giữ."""
        counts = np.bincount(self.switch_assignment, minlength = self.num_controllers).astype(float)
        load = np.zeros((self.num_controllers, 3), dtype = np.float32)
        for i in range(self.num_controllers):
            base = counts[i] / self.num_switches
            load[i, 0] = np.clip(base + np.random.normal(0, 0.04), 0.0, 1.0)  # cpu
            load[i, 1] = np.clip(base * 0.8 + np.random.normal(0, 0.03), 0.0, 1.0)  # ram
            load[i, 2] = np.clip(base + np.random.normal(0, 0.05), 0.0, 1.0)  # packet_in
        return load

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def decide_migration(self, current_loads: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Quyết định có nên migrate không.

        Round-Robin không dựa vào load thực tế — chỉ điều chỉnh khi
        số switch giữa các controller chênh lệch > 1.

        Args:
            current_loads: shape (num_controllers, 3) — nhận để interface
                           đồng nhất với LeastLoadBalancer, không dùng ở đây.

        Returns:
            (switch_id, target_controller) hoặc None nếu đã cân bằng.
        """
        counts = np.bincount(self.switch_assignment, minlength = self.num_controllers)
        max_ctrl = int(np.argmax(counts))
        min_ctrl = int(np.argmin(counts))

        if counts[max_ctrl] - counts[min_ctrl] <= 1:
            return None  # Đã cân bằng, không cần migrate

        switches_of_max = np.where(self.switch_assignment == max_ctrl)[0]
        if len(switches_of_max) == 0:
            return None

        # Luân phiên: chọn switch đầu tiên của controller nhiều tải nhất
        switch_to_migrate = int(switches_of_max[0])
        return switch_to_migrate, min_ctrl

    def execute_migration(self, switch_id: int, target_controller: int) -> bool:
        """
        Thực thi migration: cập nhật switch_assignment.

        Returns:
            True nếu thành công.
        """
        if not (0 <= switch_id < self.num_switches):
            logger.error(f"[RoundRobin] switch_id {switch_id} không hợp lệ")
            return False
        if not (0 <= target_controller < self.num_controllers):
            logger.error(f"[RoundRobin] target_controller {target_controller} không hợp lệ")
            return False

        old_ctrl = int(self.switch_assignment[switch_id])
        if old_ctrl == target_controller:
            return False  # no-op

        self.switch_assignment[switch_id] = target_controller
        logger.debug(f"[RoundRobin] s{switch_id}: ctrl {old_ctrl} → {target_controller}")
        return True

    def get_load_distribution(self) -> Dict[int, int]:
        """Trả về số switch mỗi controller đang giữ."""
        counts = np.bincount(self.switch_assignment, minlength = self.num_controllers)
        return {i: int(counts[i]) for i in range(self.num_controllers)}

    def run_episode(self, num_steps: int = 200, load_fn=None) -> Dict[str, float]:
        """
        Chạy một episode để lấy metrics so sánh với RL agent.

        Args:
            num_steps: Số bước mỗi episode.
            load_fn:   Hàm nhận switch_assignment → np.ndarray shape (num_controllers, 3).
                       Nếu None dùng mock.

        Returns:
            Dict metrics: mean_variance_cpu, final_variance_cpu, migration_count.
        """
        self.reset()
        migration_count = 0
        variances = []

        for _ in range(num_steps):
            loads = load_fn(self.switch_assignment) if load_fn else self._mock_load()
            variances.append(float(np.var(loads[:, 0])))

            result = self.decide_migration(loads)
            if result is not None:
                sw_id, tgt = result
                if self.execute_migration(sw_id, tgt):
                    migration_count += 1

        metrics = {
            "mean_variance_cpu": float(np.mean(variances)),
            "final_variance_cpu": float(variances[-1]) if variances else 0.0,
            "migration_count": migration_count,
        }

        logger.info(
            f"[RoundRobin] Episode done — "
            f"Var: {metrics['mean_variance_cpu']:.4f} | "
            f"Migrations: {migration_count}"
        )
        return metrics

    def reset(self) -> None:
        """Reset về trạng thái ban đầu (mất cân bằng) để test."""
        # Giả lập mất cân bằng như trong sdn_env.py
        self.switch_assignment = np.array([i % self.num_controllers for i in range(self.num_switches)])
        n_overload = int(self.num_switches * 0.6)
        self.switch_assignment[:n_overload] = 0
        self._rr_pointer = 0

    def select_action(self, obs: np.ndarray, env) -> int:
        """Sinh action cho SDNLoadBalancingEnv theo vòng tròn trên switch + target."""
        _ = obs
        self.switch_assignment = env.switch_assignment.copy()

        switch_id = self._rr_pointer % self.num_switches
        self._rr_pointer = (self._rr_pointer + 1) % self.num_switches

        current_ctrl = int(self.switch_assignment[switch_id])
        target_ctrl = (current_ctrl + 1) % self.num_controllers
        return encode_action(self.switch_assignment, switch_id, target_ctrl, self.num_controllers)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    balancer = RoundRobinBalancer(num_controllers = 3, num_switches = 12)

    print("\n=== Phân phối ban đầu ===")
    print(balancer.get_load_distribution())

    print("\n=== Chạy episode 200 bước ===")
    metrics = balancer.run_episode(num_steps=200)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")