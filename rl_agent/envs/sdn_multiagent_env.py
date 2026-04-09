import gymnasium as gym
from pettingzoo import ParallelEnv
import numpy as np
from typing import Dict, Optional, List

class SDNMultiAgentEnv(ParallelEnv):
    """
    Multi-Agent PettingZoo environment cho SDN Controller Load Balancing.

    Mỗi agent tương ứng với 1 Ryu controller.
    - Observation: global state [cpu_c0, ram_c0, pkt_in_c0, ..., cpu_cN, ...]
    - Action:     
        - Discrete — chọn switch nào (trong danh sách mình đang giữ) để migrate đi.
        - Action = num_switches → no-op (không migrate).
    - Reward: Local reward (giảm load bản thân) + Global reward (giảm variance toàn cluster).
    """

    metadata = {"render_modes": ["human"], "name": "sdn_multiagent_v0"}

    def __init__(
        self,
        num_controllers: int = 3,
        num_switches: int = 12,
        use_mock: bool = True,
        alpha: float = 1.5,    # trọng số penalty variance toàn cluster (global)
        beta: float = 1.0,     # trọng số penalty load bản thân (local)
        migration_cost: float = 0.05,
    ):
        super().__init__()
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.use_mock = use_mock
        self.alpha = alpha
        self.beta = beta
        self.migration_cost = migration_cost

        self.possible_agents = [f"controller_{i}" for i in range(num_controllers)]
        self.agents: List[str] = []

        # Observation: toàn bộ state cluster (mỗi controller biết load của tất cả)
        self._obs_shape = (num_controllers * 3,)

        self.observation_spaces = {
            agent: gym.spaces.Box(low=0.0, high=1.0, shape=self._obs_shape, dtype=np.float32)
            for agent in self.possible_agents
        }

        # Action: chọn switch (0..num_switches-1) để migrate sang controller ít tải nhất,
        # hoặc num_switches = no-op.
        self.action_spaces = {
            agent: gym.spaces.Discrete(num_switches + 1)
            for agent in self.possible_agents
        }

        # Internal state
        self.switch_assignment: Optional[np.ndarray] = None
        self.load: Optional[np.ndarray] = None   # shape (num_controllers, 3)
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _agent_idx(self, agent: str) -> int:
        return int(agent.split("_")[1])

    def _compute_load(self) -> np.ndarray:
        """
        Tính load từ switch_assignment (mock).
        load[i] = [cpu_i, ram_i, packet_in_i] normalize về [0,1].
        """
        counts = np.bincount(self.switch_assignment, minlength=self.num_controllers).astype(np.float32)
        load = np.zeros((self.num_controllers, 3), dtype=np.float32)
        for i in range(self.num_controllers):
            base = counts[i] / self.num_switches
            load[i, 0] = np.clip(base + np.random.normal(0, 0.04), 0.0, 1.0)  # cpu
            load[i, 1] = np.clip(base * 0.8 + np.random.normal(0, 0.03), 0.0, 1.0)  # ram
            load[i, 2] = np.clip(base + np.random.normal(0, 0.05), 0.0, 1.0)  # packet_in
        return load

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Tất cả agents nhận cùng global observation."""
        flat = self.load.flatten().astype(np.float32)
        return {agent: flat.copy() for agent in self.agents}

    def _find_least_loaded_controller(self, exclude: int) -> int:
        """Tìm controller ít tải nhất (trừ controller hiện tại)."""
        cpu_loads = self.load[:, 0].copy()
        cpu_loads[exclude] = np.inf
        return int(np.argmin(cpu_loads))

    def _calculate_rewards(self, old_variance: float) -> Dict[str, float]:
        """
        Reward = global_reward (giảm variance cluster) + local_reward (giảm load bản thân).
            - global_reward: alpha * (old_variance - new_variance)
        """
        new_variance = float(np.var(self.load[:, 0]))
        delta_variance = old_variance - new_variance  # dương = cải thiện

        rewards = {}
        for agent in self.agents:
            i = self._agent_idx(agent)
            cpu_i = float(self.load[i, 0])
            mean_cpu = float(np.mean(self.load[:, 0]))

            # Local: phần thưởng nếu load bản thân giảm về gần mean
            local_r = -self.beta * abs(cpu_i - mean_cpu)

            # Global: phần thưởng nếu variance cluster giảm
            global_r = self.alpha * delta_variance

            rewards[agent] = float(local_r + global_r - self.migration_cost)

        return rewards

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.step_count = 0

        # Phân phối mất cân bằng ban đầu: controller 0 giữ 60% switches
        self.switch_assignment = np.array([i % self.num_controllers for i in range(self.num_switches)])
        n_overload = int(self.num_switches * 0.6)
        self.switch_assignment[:n_overload] = 0

        self.load = self._compute_load()
        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: Dict[str, int]):
        """
        Xử lý actions của tất cả agents đồng thời (Parallel).
        """
        self.step_count += 1
        old_variance = float(np.var(self.load[:, 0]))

        for agent, action in actions.items():
            i = self._agent_idx(agent)

            # action = num_switches → no-op
            if action >= self.num_switches:
                continue

            switch_id = int(action)
            # Chỉ migrate nếu switch này đang thuộc controller i
            if self.switch_assignment[switch_id] != i:
                continue

            # Migrate sang controller ít tải nhất (trừ bản thân)
            target = self._find_least_loaded_controller(exclude=i)
            self.switch_assignment[switch_id] = target

        # Cập nhật load sau khi tất cả agents đã hành động
        self.load = self._compute_load()

        observations = self._get_obs()
        rewards = self._calculate_rewards(old_variance)

        # Episode kết thúc sau 200 steps hoặc khi đạt cân bằng tốt
        new_variance = float(np.var(self.load[:, 0]))
        done = self.step_count >= 200 or new_variance < 0.01

        terminated = {agent: done for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        infos = {
            agent: {
                "variance": new_variance,
                "switch_assignment": self.switch_assignment.tolist(),
                "load_cpu": self.load[:, 0].tolist(),
            }
            for agent in self.agents
        }

        if done:
            self.agents = []

        return observations, rewards, terminated, truncated, infos

    def observation_space(self, agent: str) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Space:
        return self.action_spaces[agent]

    def render(self):
        if self.load is None:
            return
        print(f"\n=== Multi-Agent SDN State (step {self.step_count}) ===")
        for i, agent in enumerate(self.possible_agents):
            n_sw = int(np.sum(self.switch_assignment == i))
            print(f"  {agent}: CPU={self.load[i,0]:.2f} RAM={self.load[i,1]:.2f} "
                  f"PktIn={self.load[i,2]:.2f} | Switches={n_sw}")
        print(f"  Variance(CPU): {np.var(self.load[:,0]):.4f}")

    def close(self):
        pass