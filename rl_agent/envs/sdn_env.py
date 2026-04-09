import gymnasium as gym
import numpy as np
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class SDNLoadBalancingEnv(gym.Env):
    """
    Gymnasium environment cho bài toán SDN Controller Load Balancing.

    State:  [cpu_c0, ram_c0, pkt_in_c0,  cpu_c1, ram_c1, pkt_in_c1, ...]  shape=(n_controllers*3,)
    Action: Discrete — index tương ứng cặp (switch_id, target_controller_id)
    Reward: -alpha*Var(CPU) - beta*latency + gamma*bonus_cân_bằng - migration_cost
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_controllers: int = 3,
        num_switches: int = 12,
        use_mock: bool = True,
        ryu_api_urls: Optional[list] = None,   # FIX: list URL cho từng controller trong cluster
        alpha: float = 2.0,   # trọng số penalty variance CPU
        beta: float = 0.8,    # trọng số penalty latency
        gamma: float = 1.5,   # trọng số bonus cân bằng
        migration_cost: float = 0.05,
    ):
        super().__init__()
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.use_mock = use_mock
        self.ryu_api_urls = ryu_api_urls or [
            f"http://127.0.0.1:{8080 + i}" for i in range(num_controllers)
        ]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.migration_cost = migration_cost

        # State: mỗi controller có 3 chỉ số [cpu, ram, packet_in_rate] → normalize về [0,1]
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(num_controllers * 3,),
            dtype=np.float32
        )

        # Action: chọn cặp (switch_id, target_controller_id)
        # switch_id: 0..num_switches-1 , target_controller: 0..num_controllers-1
        # nhưng không migrate về chính controller đang giữ → num_controllers-1 lựa chọn/switch
        self.action_space = gym.spaces.Discrete(num_switches * (num_controllers - 1))

        # Trạng thái nội bộ
        self.current_load: Optional[np.ndarray] = None
        self.switch_assignment: Optional[np.ndarray] = None  # switch_assignment[i] = controller id
        self.current_latency: float = 0.0
        self.migration_history: list = []
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_switch_assignment(self) -> np.ndarray:
        """Phân phối switch đều ban đầu."""
        assignment = np.array([i % self.num_controllers for i in range(self.num_switches)])
        return assignment

    def _get_state_mock(self) -> np.ndarray:
        """
        Mock state: giả lập controller 0 bị overload để agent có động lực migrate.

        BUG CŨ: Hàm _get_state() luôn trả về random hoàn toàn (kể cả nhánh use_mock=False
        cũng random) → state không phản ánh hành động migrate, agent không học được.
        FIX: Mock state tính từ switch_assignment → load tỷ lệ với số switch mỗi controller giữ.
        """
        load = np.zeros((self.num_controllers, 3), dtype=np.float32)
        switch_counts = np.bincount(self.switch_assignment, minlength=self.num_controllers)

        for i in range(self.num_controllers):
            # CPU tỷ lệ với số switch + noise
            base_cpu = switch_counts[i] / self.num_switches
            load[i, 0] = np.clip(base_cpu + np.random.normal(0, 0.05), 0.0, 1.0)  # cpu
            load[i, 1] = np.clip(base_cpu * 0.8 + np.random.normal(0, 0.04), 0.0, 1.0)  # ram
            load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.06), 0.0, 1.0)  # packet_in_rate

        return load.flatten()

    def _get_state_real(self) -> np.ndarray:
        """
        Lấy state thực tế từ Ryu REST API + psutil.
        Sẽ được implement khi tích hợp với Mininet thật.
        """
        try:
            import requests
            state = []
            for url in self.ryu_api_urls:
                resp = requests.get(f"{url}/monitor/load", timeout=1.0)
                data = resp.json()
                state.extend([
                    float(data.get("cpu", 0.5)),
                    float(data.get("ram", 0.5)),
                    float(data.get("packet_in_rate", 0.5)),
                ])
            return np.clip(np.array(state, dtype=np.float32), 0.0, 1.0)
        except Exception as e:
            logger.warning(f"Không lấy được state thực: {e} — fallback về mock")
            return self._get_state_mock()

    def _get_state(self) -> np.ndarray:
        if self.use_mock:
            return self._get_state_mock()
        return self._get_state_real()

    def _decode_action(self, action: int) -> Tuple[int, int]:
        """
        Giải mã action index → (switch_id, target_controller_id).

        BUG CŨ: target_controller = target_offset if target_offset < switch_id % num_controllers
        → logic sai, có thể trả về target_controller == current_controller (migrate vào chính mình).
        FIX: Bỏ qua controller đang giữ switch, map offset → controller id đúng.
        """
        switch_id = action // (self.num_controllers - 1)
        offset = action % (self.num_controllers - 1)

        current_ctrl = self.switch_assignment[switch_id]
        # Tạo list controllers loại trừ current
        candidates = [c for c in range(self.num_controllers) if c != current_ctrl]
        target_controller = candidates[offset]

        return switch_id, target_controller

    def _execute_migration(self, switch_id: int, target_controller: int) -> bool:
        """
        Thực thi switch migration.
        - Mock mode: cập nhật switch_assignment.
        - Real mode: gửi OpenFlow Role Request (implement sau).
        """
        current_ctrl = self.switch_assignment[switch_id]
        if current_ctrl == target_controller:
            return False  # No-op

        self.switch_assignment[switch_id] = target_controller
        self.migration_history.append((self.step_count, switch_id, current_ctrl, target_controller))
        logger.debug(f"Migrate switch s{switch_id}: controller {current_ctrl} → {target_controller}")
        return True

    def _calculate_reward(self, old_variance: float, new_variance: float) -> float:
        """
        Reward function:
            R = -alpha * Var(CPU) - beta * latency - migration_cost + gamma * bonus

        BUG CŨ: old_variance không được dùng trong reward → agent không được
        khuyến khích nếu variance giảm so với bước trước.
        FIX: Thêm delta variance làm tín hiệu cải thiện.
        """
        reward = -self.alpha * new_variance
        reward -= self.beta * self.current_latency
        reward -= self.migration_cost

        # Bonus nếu variance cải thiện so với bước trước
        delta = old_variance - new_variance
        if delta > 0:
            reward += 0.5 * delta  # khuyến khích giảm mất cân bằng

        # Bonus lớn khi đạt trạng thái cân bằng tốt
        if new_variance < 0.02:
            reward += self.gamma

        return float(reward)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.step_count = 0
        self.migration_history.clear()
        self.current_latency = 0.3  # bắt đầu với latency tương đối cao

        # Khởi tạo phân phối switch ban đầu (controller 0 nhận nhiều hơn để test)
        self.switch_assignment = self._init_switch_assignment()
        # Giả lập mất cân bằng ban đầu: controller 0 giữ 60% switch
        n_overload = int(self.num_switches * 0.6)
        self.switch_assignment[:n_overload] = 0

        self.current_load = self._get_state()
        return self.current_load, {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1

        old_variance = float(np.var(self.current_load[::3]))  # CPU của các controllers

        switch_id, target_controller = self._decode_action(action)
        migration_success = self._execute_migration(switch_id, target_controller)

        # Cập nhật state sau migration
        self.current_load = self._get_state()

        # Cập nhật latency: giảm dần nếu variance giảm
        new_variance = float(np.var(self.current_load[::3]))
        self.current_latency = max(0.05, self.current_latency - 0.02 * (old_variance - new_variance + 0.1))

        reward = self._calculate_reward(old_variance, new_variance)

        terminated = self.step_count >= 200
        truncated = False

        info = {
            "variance_cpu": new_variance,
            "latency": self.current_latency,
            "migration_count": len(self.migration_history),
            "migration_success": migration_success,
            "switch_migrated": switch_id,
            "target_controller": target_controller,
            "switch_assignment": self.switch_assignment.tolist(),
        }

        return self.current_load, reward, terminated, truncated, info

    def render(self):
        if self.current_load is None:
            return
        print("\n=== SDN Load Balancing State ===")
        for i in range(self.num_controllers):
            cpu = self.current_load[i * 3]
            ram = self.current_load[i * 3 + 1]
            pkt = self.current_load[i * 3 + 2]
            n_sw = int(np.sum(self.switch_assignment == i))
            print(f"  Controller {i}: CPU={cpu:.2f} RAM={ram:.2f} PktIn={pkt:.2f} | Switches={n_sw}")
        print(f"  Variance(CPU): {np.var(self.current_load[::3]):.4f} | Latency: {self.current_latency:.3f}")

    def close(self):
        pass