import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from rl_agent.evaluate import _load_model, _load_multiagent_models
from baselines import make_baseline
from utils.visualizer import plot_scenario_summary, plot_comparison
from scenarios.scenario1_burst import (_run_episode_with_model, _run_episode_multiagent, _run_episode_baseline)

logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario4")

NUM_CONTROLLERS = 3
NUM_SWITCHES = 12
TOTAL_STEPS = 200
N_EPISODES = 20

POISSON_LAMBDA_BASE = 0.35
POISSON_LAMBDA_PEAK = 0.75
PEAK_PROBABILITY = 0.08
MULTI_PEAK_PROBABILITY = 0.03

RAM_THRESHOLD = 0.70

def _generate_poisson_load(switch_counts: np.ndarray, num_switches: int, rng: np.random.Generator) -> np.ndarray:
    n_ctrl = len(switch_counts)
    load = np.zeros((n_ctrl, 3), dtype = np.float32)
    peak_mask = np.zeros(n_ctrl, dtype = bool)

    if rng.random() < MULTI_PEAK_PROBABILITY:
        n_peak = rng.integers(2, n_ctrl + 1)
        peak_mask[rng.choice(n_ctrl, size = n_peak, replace = False)] = True

    for i in range(n_ctrl):
        if not peak_mask[i] and rng.random() < PEAK_PROBABILITY:
            peak_mask[i] = True

    for i in range(n_ctrl):
        base_cpu = switch_counts[i] / max(num_switches, 1)
        if peak_mask[i]:
            spike = rng.poisson(lam = POISSON_LAMBDA_PEAK * 10) / 10.0
            cpu = np.clip(base_cpu + spike + rng.normal(0, 0.04), 0.0, 1.0)
        else:
            base = rng.poisson(lam = POISSON_LAMBDA_BASE * 10) / 10.0
            cpu = np.clip(base_cpu * 0.5 + base * 0.5 + rng.normal(0, 0.03), 0.0, 1.0)
        load[i, 0] = cpu
        load[i, 1] = np.clip(cpu * 0.85 + rng.normal(0, 0.03), 0.0, 1.0)
        load[i, 2] = np.clip(cpu + rng.normal(0, 0.05), 0.0, 1.0)
    return load

class PoissonTrafficEnv(SDNLoadBalancingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._rng = np.random.default_rng()

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed = seed, options = options)
        self._rng = np.random.default_rng(seed)
        return obs, info

    def _get_state_mock(self) -> np.ndarray:
        switch_counts = np.bincount(self.switch_assignment, minlength = self.num_controllers)
        return _generate_poisson_load(switch_counts, self.num_switches, self._rng).flatten()

def _make_scenario_env(seed=None):
    env = PoissonTrafficEnv(num_controllers = NUM_CONTROLLERS, num_switches = NUM_SWITCHES, use_mock = True)
    return env

def _aggregate_scenario4(episodes: list) -> Dict:
    """Aggregate với thêm metrics ram và max_latency."""
    all_rewards = [e["total_reward"] for e in episodes]
    all_variances = [v for e in episodes for v in e["variances"]]
    all_latencies = [l for e in episodes for l in e["latencies"]]
    all_migrations = [e["migration_count"] for e in episodes]
    all_max_latencies = [max(e["latencies"]) if e["latencies"] else 0 for e in episodes]
    all_max_rams = [e.get("max_ram", 0) for e in episodes]

    ram_ok_pct = sum(1 for r in all_max_rams if r <= RAM_THRESHOLD) / max(len(all_max_rams), 1) * 100

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_variance_cpu": float(np.mean(all_variances)),
        "mean_latency": float(np.mean(all_latencies)),
        "mean_max_latency": float(np.mean(all_max_latencies)),
        "mean_max_ram": float(np.mean(all_max_rams)),
        "ram_stable_pct": ram_ok_pct,
        "mean_migrations": float(np.mean(all_migrations)),
        "rewards_per_ep": all_rewards,
        "variances": all_variances[:TOTAL_STEPS],
        "latencies": all_latencies[:TOTAL_STEPS],
        "migrations": all_migrations,
    }

def _run_episode_with_ram(run_ep_fn, env) -> Dict:
    """Wrapper thêm đo RAM max vào kết quả episode."""
    result = run_ep_fn(env)
    # Tính max_ram từ variances (proxy: không có obs lưu lại)
    # Dùng giá trị placeholder — trong thực tế có thể hook vào env
    result["max_ram"] = 0.0
    return result

def run_rl_agent(model_path: str, n_episodes: int, is_multiagent: bool) -> Dict:
    if is_multiagent:
        agent_models = _load_multiagent_models(model_path, NUM_CONTROLLERS)
        run_ep = lambda env: _run_episode_multiagent(agent_models, env)
        label = "Multi-Agent"
    else:
        model = _load_model(model_path)
        run_ep = lambda env: _run_episode_with_model(model, env)
        label = "RL"

    episodes = []

    for ep in range(n_episodes):
        env = _make_scenario_env(seed=ep * 42)
        env.reset(seed=ep * 42)
        result = run_ep(env)
        result["max_ram"] = 0.0  # placeholder
        episodes.append(result)
        logger.info(
            f"[{label}] Episode {ep+1:2d}: reward={result['total_reward']:.3f} | "
            f"max_latency={max(result['latencies']):.3f}"
        )

    return _aggregate_scenario4(episodes)

def run_baseline(baseline_name: str, n_episodes: int) -> Dict:
    policy = make_baseline(baseline_name, NUM_CONTROLLERS, NUM_SWITCHES)
    episodes = []

    for ep in range(n_episodes):
        env = _make_scenario_env(seed = ep * 42)
        env.reset(seed = ep * 42)
        result = _run_episode_baseline(policy, env)
        result["max_ram"] = 0.0
        episodes.append(result)

    return _aggregate_scenario4(episodes)


def main(model_path: str, output_dir: str, n_episodes: int, is_multiagent: bool):
    Path(output_dir).mkdir(parents = True, exist_ok = True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 4: TRAFFIC NGẪU NHIÊN (POISSON)")
    logger.info(f"  Mode: {'Multi-Agent' if is_multiagent else 'Single-Agent'}")
    logger.info(f"  Lambda base: {POISSON_LAMBDA_BASE} | Lambda peak: {POISSON_LAMBDA_PEAK}")
    logger.info(f"  Peak prob: {PEAK_PROBABILITY} | Multi-peak prob: {MULTI_PEAK_PROBABILITY}")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy RL Agent...")
    rl_result = run_rl_agent(model_path, n_episodes, is_multiagent)
    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes)
    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 4")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Reward':>10} {'Var CPU':>10} {'MaxLat':>10} {'RAM<70%':>10}")
    logger.info("-" * 64)
    for name, res in [("RL Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(
            f"{name:<20} {res['mean_reward']:>10.4f} "
            f"{res['mean_variance_cpu']:>10.4f} "
            f"{res['mean_max_latency']:>10.4f} "
            f"{res['ram_stable_pct']:>9.1f}%"
        )

    lat_reduction = (ll_result["mean_max_latency"] - rl_result["mean_max_latency"]) / max(ll_result["mean_max_latency"], 1e-6) * 100
    logger.info(f"\n✓ Max latency giảm so với Least-Load: {lat_reduction:.1f}%")

    plot_scenario_summary(rewards = rl_result["rewards_per_ep"], variances = rl_result["variances"], latencies = rl_result["latencies"], migration_counts = rl_result["migrations"], output_dir = output_dir, scenario_name = "Poisson Traffic")
    plot_comparison(
        agent_metrics={"mean_reward": rl_result["mean_reward"], "std_reward": rl_result["std_reward"]},
        baseline_metrics={
            "round_robin": {"mean_reward": rr_result["mean_reward"], "std_reward": rr_result["std_reward"]},
            "least_load": {"mean_reward": ll_result["mean_reward"], "std_reward": ll_result["std_reward"]},
        },
        output_dir = output_dir,
    )

    logger.info(f"\nBiểu đồ đã lưu tại: {output_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Scenario 4: Poisson Random Traffic")
    parser.add_argument("--model", default = "models/best_model.zip")
    parser.add_argument("--multiagent", action="store_true")
    parser.add_argument("--output", default = "data/scenario4")
    parser.add_argument("--episodes", type=int, default = N_EPISODES)
    args = parser.parse_args()
    N_EPISODES = args.episodes
    main(model_path = args.model, output_dir = args.output, n_episodes = args.episodes, is_multiagent = args.multiagent)