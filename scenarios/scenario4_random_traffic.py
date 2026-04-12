import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import DQN
from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from baselines.round_robin import RoundRobinBalancer
from baselines.least_load import LeastLoadBalancer
from utils.visualizer import plot_scenario_summary, plot_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario4")

# ------------------------------------------------------------------
# Cấu hình kịch bản
# ------------------------------------------------------------------

NUM_CONTROLLERS = 3
NUM_SWITCHES = 12
TOTAL_STEPS = 200
N_EPISODES = 20

# Tham số Poisson cho từng controller
# lambda = tốc độ trung bình của packet-in rate (normalize 0-1)
POISSON_LAMBDA_BASE = 0.35      # mức nền trung bình
POISSON_LAMBDA_PEAK = 0.75      # mức peak khi bị overload
PEAK_PROBABILITY = 0.08         # xác suất peak xảy ra mỗi bước
MULTI_PEAK_PROBABILITY = 0.03   # xác suất nhiều controllers peak cùng lúc

# Ngưỡng RAM để đánh giá "RAM ổn định"
RAM_THRESHOLD = 0.70
MAX_LATENCY_TARGET = 0.30       # latency normalize (~ 30ms nếu max=100ms)


def _generate_poisson_load(switch_counts: np.ndarray, num_switches: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sinh load theo Poisson cho từng controller.

    Logic:
    1. Base load: tỷ lệ với switch_count (như các env khác).
    2. Poisson spike: với xác suất PEAK_PROBABILITY, thêm spike lớn.
    3. Multi-peak: với xác suất MULTI_PEAK_PROBABILITY, nhiều controllers cùng spike.

    Returns:
        np.ndarray shape (num_controllers, 3) — [cpu, ram, packet_in]
    """
    n_ctrl = len(switch_counts)
    load = np.zeros((n_ctrl, 3), dtype = np.float32)

    # Xác định controller nào bị peak
    peak_mask = np.zeros(n_ctrl, dtype = bool)

    # Multi-peak event
    if rng.random() < MULTI_PEAK_PROBABILITY:
        n_peak = rng.integers(2, n_ctrl + 1)  # 2 đến tất cả controllers
        peak_controllers = rng.choice(n_ctrl, size = n_peak, replace = False)
        peak_mask[peak_controllers] = True

    # Single peak event (độc lập với multi-peak)
    for i in range(n_ctrl):
        if not peak_mask[i] and rng.random() < PEAK_PROBABILITY:
            peak_mask[i] = True

    for i in range(n_ctrl):
        base_cpu = switch_counts[i] / max(num_switches, 1)

        if peak_mask[i]:
            # Poisson spike: random theo phân phối Poisson xung quanh LAMBDA_PEAK
            spike = rng.poisson(lam = POISSON_LAMBDA_PEAK * 10) / 10.0
            cpu = np.clip(base_cpu + spike + rng.normal(0, 0.04), 0.0, 1.0)
        else:
            # Poisson nền: random xung quanh LAMBDA_BASE
            base = rng.poisson(lam = POISSON_LAMBDA_BASE * 10) / 10.0
            cpu = np.clip(base_cpu * 0.5 + base * 0.5 + rng.normal(0, 0.03), 0.0, 1.0)

        load[i, 0] = cpu                                                          # cpu
        load[i, 1] = np.clip(cpu * 0.85 + rng.normal(0, 0.03), 0.0, 1.0)       # ram
        load[i, 2] = np.clip(cpu + rng.normal(0, 0.05), 0.0, 1.0)               # packet_in

    return load

class PoissonTrafficEnv(SDNLoadBalancingEnv):
    """
    Override env để sinh load theo Poisson distribution.
    Mỗi episode dùng một random seed khác nhau → agent phải học policy
    tổng quát, không overfit vào pattern cố định.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng()

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed = seed, options = options)
        # Seed mới mỗi episode để đảm bảo đa dạng traffic
        self._rng = np.random.default_rng(seed)
        return obs, info

    def _get_state_mock(self) -> np.ndarray:
        switch_counts = np.bincount(self.switch_assignment, minlength = self.num_controllers)
        load = _generate_poisson_load(switch_counts, self.num_switches, self._rng)
        return load.flatten()

# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def run_rl_agent(model_path: str, n_episodes: int = N_EPISODES) -> Dict:
    model = DQN.load(model_path)
    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []
    all_max_latencies, all_max_rams = [], []

    for ep in range(n_episodes):
        env = PoissonTrafficEnv(num_controllers = NUM_CONTROLLERS, num_switches = NUM_SWITCHES, use_mock = True)
        obs, _ = env.reset(seed=ep * 42)
        ep_rewards, ep_variances, ep_latencies = [], [], []
        ep_max_ram = 0.0
        done = truncated = False

        while not done and not truncated:
            action, _ = model.predict(obs, deterministic = True)
            obs, reward, done, truncated, info = env.step(int(action))
            ep_rewards.append(float(reward))
            ep_variances.append(float(info.get("variance_cpu", 0.0)))
            lat = float(info.get("latency", 0.0))
            ep_latencies.append(lat)
            # Đo RAM max trong episode
            ram_values = [obs[i * 3 + 1] for i in range(NUM_CONTROLLERS)]
            ep_max_ram = max(ep_max_ram, max(ram_values))

        all_rewards.append(sum(ep_rewards))
        all_variances.extend(ep_variances)
        all_latencies.extend(ep_latencies)
        all_migrations.append(info.get("migration_count", 0))
        all_max_latencies.append(max(ep_latencies))
        all_max_rams.append(ep_max_ram)
        logger.info(f"[RL] Episode {ep+1:2d}: reward = {sum(ep_rewards):.3f} | "
                    f"max_latency = {max(ep_latencies):.3f} | max_ram = {ep_max_ram:.3f}")

    # % episodes có max RAM dưới ngưỡng
    ram_ok_pct = sum(1 for r in all_max_rams if r <= RAM_THRESHOLD) / len(all_max_rams) * 100

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu":  float(np.mean(all_variances)),
        "mean_latency":  float(np.mean(all_latencies)),
        "mean_max_latency": float(np.mean(all_max_latencies)),
        "mean_max_ram": float(np.mean(all_max_rams)),
        "ram_stable_pct": ram_ok_pct,
        "mean_migrations": float(np.mean(all_migrations)),
        "rewards_per_ep": all_rewards,
        "variances":  all_variances[:TOTAL_STEPS],
        "latencies":  all_latencies[:TOTAL_STEPS],
        "migrations": all_migrations,
    }

def run_baseline(baseline_name: str, n_episodes: int = N_EPISODES) -> Dict:
    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []
    all_max_latencies, all_max_rams = [], []

    for ep in range(n_episodes):
        env = PoissonTrafficEnv(num_controllers = NUM_CONTROLLERS, num_switches = NUM_SWITCHES, use_mock = True)
        obs, _ = env.reset(seed=ep * 42)

        if baseline_name == "round_robin":
            balancer = RoundRobinBalancer(NUM_CONTROLLERS, NUM_SWITCHES)
        else:
            balancer = LeastLoadBalancer(NUM_CONTROLLERS, NUM_SWITCHES)
        balancer.switch_assignment = env.switch_assignment.copy()

        ep_rewards, ep_variances, ep_latencies = [], [], []
        migration_count = 0
        ep_max_ram = 0.0
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
            lat = float(info.get("latency", 0.0))
            ep_latencies.append(lat)
            ram_values = [obs[i * 3 + 1] for i in range(NUM_CONTROLLERS)]
            ep_max_ram = max(ep_max_ram, max(ram_values))

        all_rewards.append(sum(ep_rewards))
        all_variances.extend(ep_variances)
        all_latencies.extend(ep_latencies)
        all_migrations.append(migration_count)
        all_max_latencies.append(max(ep_latencies))
        all_max_rams.append(ep_max_ram)

    ram_ok_pct = sum(1 for r in all_max_rams if r <= RAM_THRESHOLD) / len(all_max_rams) * 100

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu": float(np.mean(all_variances)),
        "mean_latency":  float(np.mean(all_latencies)),
        "mean_max_latency": float(np.mean(all_max_latencies)),
        "mean_max_ram":  float(np.mean(all_max_rams)),
        "ram_stable_pct": ram_ok_pct,
        "mean_migrations": float(np.mean(all_migrations)),
    }

def main(model_path: str, output_dir: str = "data/scenario4"):
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 4: TRAFFIC NGẪU NHIÊN (POISSON)")
    logger.info(f"  Lambda base: {POISSON_LAMBDA_BASE} | Lambda peak: {POISSON_LAMBDA_PEAK}")
    logger.info(f"  Peak prob: {PEAK_PROBABILITY} | Multi-peak prob: {MULTI_PEAK_PROBABILITY}")
    logger.info(f"  Total steps: {TOTAL_STEPS} | Episodes: {N_EPISODES}")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy DQN Agent...")
    rl_result = run_rl_agent(model_path, n_episodes = N_EPISODES)

    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes = N_EPISODES)

    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes = N_EPISODES)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 4")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Reward':>10} {'Var CPU':>10} "
                f"{'MaxLat':>10} {'MaxRAM':>10} {'RAM<70%':>10}")
    logger.info("-" * 75)

    for name, res in [("DQN Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(
            f"{name:<20} {res['mean_reward']:>10.4f} "
            f"{res['mean_variance_cpu']:>10.4f} "
            f"{res['mean_max_latency']:>10.4f} "
            f"{res['mean_max_ram']:>10.4f} "
            f"{res['ram_stable_pct']:>9.1f}%"
        )

    # Kiểm tra mục tiêu
    latency_reduction = (ll_result["mean_max_latency"] - rl_result["mean_max_latency"]) / max(ll_result["mean_max_latency"], 1e-6) * 100
    logger.info(f"\n✓ Max latency giảm so với Least-Load: {latency_reduction:.1f}% "
                f"(mục tiêu: từ ~1.0 xuống ~0.3)")
    logger.info(f"✓ RAM < 70% trong {rl_result['ram_stable_pct']:.1f}% episodes "
                f"(mục tiêu ≥ 80%)")

    plot_scenario_summary(
        rewards = rl_result["rewards_per_ep"],
        variances = rl_result["variances"],
        latencies = rl_result["latencies"],
        migration_counts = rl_result["migrations"],
        output_dir = output_dir,
        scenario_name = "Poisson Traffic",
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
    parser = argparse.ArgumentParser(description="Scenario 4: Poisson Random Traffic")
    parser.add_argument("--model", default = "models/dqn_best.zip")
    parser.add_argument("--output", default = "data/scenario4")
    parser.add_argument("--episodes", type = int, default = N_EPISODES)
    args = parser.parse_args()

    N_EPISODES = args.episodes
    main(model_path = args.model, output_dir = args.output)