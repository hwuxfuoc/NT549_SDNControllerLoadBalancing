import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from baselines.round_robin import RoundRobinBalancer
from baselines.least_load import LeastLoadBalancer
from utils.visualizer import plot_scenario_summary, plot_comparison

logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario1")

# ------------------------------------------------------------------
# Cấu hình kịch bản
# ------------------------------------------------------------------

NUM_CONTROLLERS = 3
NUM_SWITCHES = 12
STABLE_STEPS = 50       # bước chạy với traffic thấp
BURST_STEPS = 150       # bước chạy sau khi burst xảy ra
TOTAL_STEPS = STABLE_STEPS + BURST_STEPS
N_EPISODES = 20         # số episode để lấy trung bình

BURST_CONTROLLER = 0    # controller bị tấn công (0-indexed)
BURST_MAGNITUDE = 0.85  # CPU giả lập khi burst (normalize 0-1)
STABLE_CPU = 0.3        # CPU nền khi traffic thấp

class BurstScenarioEnv(SDNLoadBalancingEnv):
    """
    Override SDNLoadBalancingEnv để inject burst traffic vào controller 0
    sau STABLE_STEPS bước đầu tiên.

    Thay vì load tỷ lệ thuần túy với switch_count, burst env thêm
    một spike CPU cố định vào controller bị tấn công.
    """

    def __init__(self, stable_steps: int = STABLE_STEPS, **kwargs):
        super().__init__(**kwargs)
        self.stable_steps = stable_steps
        self._burst_active = False

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed = seed, options = options)
        self._burst_active = False
        return obs, info

    def _get_state_mock(self) -> np.ndarray:
        """
        Giai đoạn stable: load thấp đều.
        Giai đoạn burst: controller BURST_CONTROLLER bị spike CPU.
        """
        load = np.zeros((self.num_controllers, 3), dtype = np.float32)
        switch_counts = np.bincount(self.switch_assignment, minlength = self.num_controllers)

        # Kích hoạt burst sau stable_steps
        if self.step_count >= self.stable_steps:
            self._burst_active = True

        for i in range(self.num_controllers):
            base_cpu = switch_counts[i] / self.num_switches

            if self._burst_active and i == BURST_CONTROLLER:
                # Controller bị burst: CPU vọt lên cao bất kể switch count
                burst_cpu = BURST_MAGNITUDE + np.random.normal(0, 0.03)
                load[i, 0] = np.clip(burst_cpu, 0.0, 1.0)
                load[i, 1] = np.clip(burst_cpu * 0.85 + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 2] = np.clip(burst_cpu + np.random.normal(0, 0.05), 0.0, 1.0)
            else:
                # Traffic thấp: load tỷ lệ với switch count + noise nhỏ
                load[i, 0] = np.clip(base_cpu * STABLE_CPU / 0.5 + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 1] = np.clip(base_cpu * 0.25 + np.random.normal(0, 0.02), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu * STABLE_CPU / 0.5 + np.random.normal(0, 0.04), 0.0, 1.0)

        return load.flatten()

# ------------------------------------------------------------------
# Runner cho từng policy
# ------------------------------------------------------------------

def run_rl_agent(model_path: str, n_episodes: int = N_EPISODES) -> Dict:
    """Chạy DQN agent trên BurstScenarioEnv, thu thập metrics."""
    model = DQN.load(model_path)

    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []

    for ep in range(n_episodes):
        env = BurstScenarioEnv(
            stable_steps = STABLE_STEPS,
            num_controllers = NUM_CONTROLLERS,
            num_switches = NUM_SWITCHES,
            use_mock = True,
        )
        obs, _ = env.reset()
        ep_rewards, ep_variances, ep_latencies = [], [], []
        done = truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_rewards.append(float(reward))
            ep_variances.append(float(info.get("variance_cpu", 0.0)))
            ep_latencies.append(float(info.get("latency", 0.0)))

        all_rewards.append(sum(ep_rewards))
        all_variances.extend(ep_variances)
        all_latencies.extend(ep_latencies)
        all_migrations.append(info.get("migration_count", 0))
        logger.info(f"[RL] Episode {ep+1:2d}: reward={sum(ep_rewards):.3f} | "
                    f"migrations={info.get('migration_count', 0)}")

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu": float(np.mean(all_variances)),
        "variance_stable": float(np.mean(all_variances[:STABLE_STEPS * n_episodes])),
        "variance_burst":  float(np.mean(all_variances[STABLE_STEPS * n_episodes:])),
        "mean_latency":  float(np.mean(all_latencies)),
        "mean_migrations": float(np.mean(all_migrations)),
        "rewards_per_ep": all_rewards,
        "variances": all_variances[:TOTAL_STEPS],   # 1 episode đại diện
        "latencies": all_latencies[:TOTAL_STEPS],
        "migrations": all_migrations,
    }

def run_baseline(baseline_name: str, n_episodes: int = N_EPISODES) -> Dict:
    """Chạy Round-Robin hoặc Least-Load trên BurstScenarioEnv."""
    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []

    for ep in range(n_episodes):
        env = BurstScenarioEnv(stable_steps = STABLE_STEPS, num_controllers = NUM_CONTROLLERS, num_switches = NUM_SWITCHES, use_mock = True)

        if baseline_name == "round_robin":
            balancer = RoundRobinBalancer(NUM_CONTROLLERS, NUM_SWITCHES)

        else:
            balancer = LeastLoadBalancer(NUM_CONTROLLERS, NUM_SWITCHES)

        obs, _ = env.reset()
        # Đồng bộ switch_assignment với balancer
        balancer.switch_assignment = env.switch_assignment.copy()

        ep_rewards, ep_variances, ep_latencies = [], [], []
        migration_count = 0
        done = truncated = False

        while not done and not truncated:
            # Baseline quyết định dựa trên load hiện tại
            load_matrix = obs.reshape(NUM_CONTROLLERS, 3)
            result = balancer.decide_migration(load_matrix)

            if result is not None:
                sw_id, tgt_ctrl = result
                # Map sang action index để step env (đảm bảo env và balancer đồng bộ)
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
                action = 0  # no-op: chọn action đầu nhưng không thay đổi gì đáng kể

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

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(model_path: str, output_dir: str = "data/scenario1"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 1: BURST TRAFFIC")
    logger.info(f"  Stable: {STABLE_STEPS} bước | Burst: {BURST_STEPS} bước")
    logger.info(f"  Controller bị tấn công: C{BURST_CONTROLLER + 1} | Burst CPU: {BURST_MAGNITUDE}")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy DQN Agent...")
    rl_result = run_rl_agent(model_path, n_episodes = N_EPISODES)

    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes = N_EPISODES)

    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes = N_EPISODES)

    # In kết quả
    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 1")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Mean Reward':>12} {'Var CPU':>10} {'Latency':>10} {'Migrations':>12}")
    logger.info("-" * 68)
    for name, res in [("DQN Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(
            f"{name:<20} {res['mean_reward']:>12.4f} "
            f"{res['mean_variance_cpu']:>10.4f} "
            f"{res['mean_latency']:>10.4f} "
            f"{res['mean_migrations']:>12.1f}"
        )

    # Kiểm tra mục tiêu
    var_reduction = (rr_result["mean_variance_cpu"] - rl_result["mean_variance_cpu"]) \
                    / max(rr_result["mean_variance_cpu"], 1e-6) * 100
    logger.info(f"\n✓ Variance CPU giảm so với Round-Robin: {var_reduction:.1f}% "
                f"(mục tiêu ≥ 30%)")

    # Vẽ biểu đồ
    plot_scenario_summary(
        rewards=rl_result["rewards_per_ep"],
        variances=rl_result["variances"],
        latencies=rl_result["latencies"],
        migration_counts=rl_result["migrations"],
        output_dir=output_dir,
        scenario_name="Burst Traffic",
    )

    plot_comparison(
        agent_metrics = {"mean_reward": rl_result["mean_reward"], "std_reward": rl_result["std_reward"]},
        baseline_metrics = {
            "round_robin": {"mean_reward": rr_result["mean_reward"], "std_reward": rr_result["std_reward"]},
            "least_load":  {"mean_reward": ll_result["mean_reward"], "std_reward": ll_result["std_reward"]},
        },
        output_dir = output_dir,
    )
    
    logger.info(f"\nBiểu đồ đã lưu tại: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Scenario 1: Burst Traffic")
    parser.add_argument("--model", default = "models/dqn_best.zip")
    parser.add_argument("--output", default = "data/scenario1")
    parser.add_argument("--episodes", type = int, default = N_EPISODES)
    args = parser.parse_args()

    N_EPISODES = args.episodes
    main(model_path=args.model, output_dir = args.output)