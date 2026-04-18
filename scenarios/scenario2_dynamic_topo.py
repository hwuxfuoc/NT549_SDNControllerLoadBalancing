import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from baselines.round_robin import RoundRobinBalancer
from baselines.least_load import LeastLoadBalancer
from utils.visualizer import plot_scenario_summary, plot_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario2")

# ------------------------------------------------------------------
# Cấu hình kịch bản
# ------------------------------------------------------------------

NUM_CONTROLLERS = 3
INITIAL_SWITCHES = 9        # switch ban đầu (3 switch/controller, cân bằng)
ADDED_SWITCHES = 5          # switch thêm vào controller 0
REMOVED_SWITCHES = 3        # switch bị xóa ở phase 3
MAX_SWITCHES = INITIAL_SWITCHES + ADDED_SWITCHES  # 14

PHASE1_STEPS = 60           # topology cân bằng, traffic thấp
PHASE2_STEPS = 100          # topology mất cân bằng, traffic tăng dần
PHASE3_STEPS = 40           # một số switch rời mạng
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS  # 200

N_EPISODES = 20
TRAFFIC_RAMP_RATE = 0.003   # tốc độ tăng traffic mỗi bước ở phase 2


class DynamicTopoEnv(SDNLoadBalancingEnv):
    """
    Override env để mô phỏng topology thay đổi động:
    - Phase 1: 9 switch, cân bằng, traffic thấp
    - Phase 2: 14 switch (thêm 5 vào C0), traffic tăng dần
    - Phase 3: 11 switch (xóa 3), traffic ổn định
    """

    def __init__(self, **kwargs):
        # Khởi tạo với MAX_SWITCHES để action space không thay đổi giữa phases
        kwargs["num_switches"] = MAX_SWITCHES
        super().__init__(**kwargs)
        self._active_switches = INITIAL_SWITCHES  # số switch thực sự active
        self._traffic_level = 0.2                 # mức traffic hiện tại [0,1]
        self._phase = 1

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed = seed, options = options)
        self._active_switches = INITIAL_SWITCHES
        self._traffic_level = 0.2
        self._phase = 1

        # Phase 1: phân đều 9 switch ban đầu
        self.switch_assignment = np.array([i % NUM_CONTROLLERS for i in range(MAX_SWITCHES)])
        # Switch từ index INITIAL_SWITCHES trở đi là "inactive" (gán vào C0 nhưng load = 0)
        self.switch_assignment[INITIAL_SWITCHES:] = 0

        return self._get_state(), info

    def _get_state_mock(self) -> np.ndarray:
        """Load tính trên active switches + traffic_level."""
        load = np.zeros((self.num_controllers, 3), dtype = np.float32)

        # Đếm switch active của mỗi controller
        active_counts = np.zeros(NUM_CONTROLLERS, dtype = int)
        for sw_idx in range(self._active_switches):
            ctrl = int(self.switch_assignment[sw_idx])
            active_counts[ctrl] += 1

        for i in range(NUM_CONTROLLERS):
            base_cpu = (active_counts[i] / max(self._active_switches, 1)) * self._traffic_level
            load[i, 0] = np.clip(base_cpu + np.random.normal(0, 0.03), 0.0, 1.0)
            load[i, 1] = np.clip(base_cpu * 0.8 + np.random.normal(0, 0.02), 0.0, 1.0)
            load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.04), 0.0, 1.0)

        return load.flatten()

    def step(self, action):
        # Cập nhật phase và topology trước khi step
        self._update_phase()
        return super().step(action)

    def _update_phase(self):
        step = self.step_count

        if step < PHASE1_STEPS:
            self._phase = 1
            self._traffic_level = 0.2  # traffic thấp, ổn định

        elif step == PHASE1_STEPS:
            # Transition → Phase 2: thêm 5 switch mới vào controller 0
            self._phase = 2
            self._active_switches = MAX_SWITCHES  # kích hoạt tất cả 14 switch
            # 5 switch mới (index 9-13) được gán vào C0 → overload
            self.switch_assignment[INITIAL_SWITCHES:] = 0
            logger.info(f"[Topo] Phase 2: Thêm {ADDED_SWITCHES} switch → "
                        f"C0 giờ giữ {np.sum(self.switch_assignment == 0)} switches")

        elif PHASE1_STEPS <= step < PHASE1_STEPS + PHASE2_STEPS:
            self._phase = 2
            # Traffic tăng dần tuyến tính
            self._traffic_level = min(0.9, 0.2 + TRAFFIC_RAMP_RATE * (step - PHASE1_STEPS))

        elif step == PHASE1_STEPS + PHASE2_STEPS:
            # Transition → Phase 3: xóa 3 switch (giảm active count)
            self._phase = 3
            self._active_switches = MAX_SWITCHES - REMOVED_SWITCHES
            logger.info(f"[Topo] Phase 3: Xóa {REMOVED_SWITCHES} switch → "
                        f"Còn {self._active_switches} switches")

        else:
            self._phase = 3
            self._traffic_level = 0.7  # traffic ổn định cao

# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_rl_agent(model_path: str, n_episodes: int = N_EPISODES) -> Dict:
    model = DQN.load(model_path)
    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []
    phase_variances = {1: [], 2: [], 3: []}

    for ep in range(n_episodes):
        env = DynamicTopoEnv(num_controllers=NUM_CONTROLLERS, use_mock=True)
        obs, _ = env.reset()
        ep_rewards, ep_variances, ep_latencies = [], [], []
        done = truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_rewards.append(float(reward))
            v = float(info.get("variance_cpu", 0.0))
            ep_variances.append(v)
            ep_latencies.append(float(info.get("latency", 0.0)))
            phase_variances[env._phase].append(v)

        all_rewards.append(sum(ep_rewards))
        all_variances.extend(ep_variances)
        all_latencies.extend(ep_latencies)
        all_migrations.append(info.get("migration_count", 0))
        logger.info(f"[RL] Episode {ep+1:2d}: reward={sum(ep_rewards):.3f} | "
                    f"migrations={info.get('migration_count', 0)}")

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu":  float(np.mean(all_variances)),
        "variance_phase1": float(np.mean(phase_variances[1])) if phase_variances[1] else 0.0,
        "variance_phase2": float(np.mean(phase_variances[2])) if phase_variances[2] else 0.0,
        "variance_phase3": float(np.mean(phase_variances[3])) if phase_variances[3] else 0.0,
        "mean_latency":   float(np.mean(all_latencies)),
        "mean_migrations": float(np.mean(all_migrations)),
        "rewards_per_ep": all_rewards,
        "variances":  all_variances[:TOTAL_STEPS],
        "latencies":  all_latencies[:TOTAL_STEPS],
        "migrations": all_migrations,
    }

def run_baseline(baseline_name: str, n_episodes: int = N_EPISODES) -> Dict:
    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []

    for ep in range(n_episodes):
        env = DynamicTopoEnv(num_controllers=NUM_CONTROLLERS, use_mock=True)
        obs, _ = env.reset()

        if baseline_name == "round_robin":
            balancer = RoundRobinBalancer(NUM_CONTROLLERS, MAX_SWITCHES)

        else:
            balancer = LeastLoadBalancer(NUM_CONTROLLERS, MAX_SWITCHES)
        balancer.switch_assignment = env.switch_assignment.copy()

        ep_rewards, ep_variances, ep_latencies = [], [], []
        migration_count = 0
        done = truncated = False

        while not done and not truncated:
            load_matrix = obs.reshape(NUM_CONTROLLERS, 3)
            result = balancer.decide_migration(load_matrix)

            if result is not None:
                sw_id, tgt_ctrl = result
                current_ctrl = int(env.switch_assignment[sw_id])
                candidates = [c for c in range(NUM_CONTROLLERS) if c != current_ctrl]
                if tgt_ctrl in candidates:
                    offset = candidates.index(tgt_ctrl)
                    action = sw_id * (NUM_CONTROLLERS - 1) + offset
                    balancer.execute_migration(sw_id, tgt_ctrl)
                    migration_count += 1
                else:
                    action = 0

            else:
                action = 0

            obs, reward, done, truncated, info = env.step(action)
            balancer.switch_assignment = env.switch_assignment.copy()
            ep_rewards.append(float(reward))
            ep_variances.append(float(np.var(obs[::3])))
            ep_latencies.append(float(info.get("latency", 0.0)))

        all_rewards.append(sum(ep_rewards))
        all_variances.extend(ep_variances)
        all_latencies.extend(ep_latencies)
        all_migrations.append(migration_count)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu": float(np.mean(all_variances)),
        "mean_latency":  float(np.mean(all_latencies)),
        "mean_migrations": float(np.mean(all_migrations)),
    }


def main(model_path: str, output_dir: str = "data/scenario2"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 2: TOPOLOGY ĐỘNG")
    logger.info(f"  Phase 1: {PHASE1_STEPS} bước (9 switch, cân bằng)")
    logger.info(f"  Phase 2: {PHASE2_STEPS} bước (+5 switch vào C0, traffic tăng)")
    logger.info(f"  Phase 3: {PHASE3_STEPS} bước (-3 switch)")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy DQN Agent...")
    rl_result = run_rl_agent(model_path, n_episodes=N_EPISODES)

    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes=N_EPISODES)

    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes=N_EPISODES)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 2")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Mean Reward':>12} {'Var CPU':>10} {'Migrations':>12}")
    logger.info("-" * 58)
    for name, res in [("DQN Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(f"{name:<20} {res['mean_reward']:>12.4f} "
                    f"{res['mean_variance_cpu']:>10.4f} "
                    f"{res['mean_migrations']:>12.1f}")

    logger.info(f"\n[RL] Variance theo phase:")
    logger.info(f"  Phase 1 (stable):  {rl_result['variance_phase1']:.4f}")
    logger.info(f"  Phase 2 (+switch): {rl_result['variance_phase2']:.4f}")
    logger.info(f"  Phase 3 (-switch): {rl_result['variance_phase3']:.4f}")

    throughput_loss_rr = rr_result["mean_variance_cpu"] / max(rl_result["mean_variance_cpu"], 1e-6)
    logger.info(f"\n✓ Throughput imbalance ratio RL/RR: {1/throughput_loss_rr:.2f}x tốt hơn")

    plot_scenario_summary(
        rewards=rl_result["rewards_per_ep"],
        variances=rl_result["variances"],
        latencies=rl_result["latencies"],
        migration_counts=rl_result["migrations"],
        output_dir=output_dir,
        scenario_name="Dynamic Topology",
    )
    plot_comparison(
        agent_metrics={"mean_reward": rl_result["mean_reward"], "std_reward": rl_result["std_reward"]},
        baseline_metrics={
            "round_robin": {"mean_reward": rr_result["mean_reward"], "std_reward": rr_result["std_reward"]},
            "least_load":  {"mean_reward": ll_result["mean_reward"], "std_reward": ll_result["std_reward"]},
        },
        output_dir=output_dir,
    )
    logger.info(f"\nBiểu đồ đã lưu tại: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 2: Dynamic Topology")
    parser.add_argument("--model", default="models/multiagent/dqn_controller_0.zip")
    parser.add_argument("--output", default="data/scenario2")
    parser.add_argument("--episodes", type=int, default=N_EPISODES)
    args = parser.parse_args()

    N_EPISODES = args.episodes
    main(model_path=args.model, output_dir=args.output)