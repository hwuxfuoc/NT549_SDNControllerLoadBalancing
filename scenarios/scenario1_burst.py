import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from rl_agent.evaluate import _load_model, _load_multiagent_models
from baselines import make_baseline
from utils.visualizer import plot_scenario_summary, plot_comparison

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario1")

# ------------------------------------------------------------------
# Cấu hình kịch bản
# ------------------------------------------------------------------

NUM_CONTROLLERS   = 3
NUM_SWITCHES      = 12
STABLE_STEPS      = 50
BURST_STEPS       = 150
TOTAL_STEPS       = STABLE_STEPS + BURST_STEPS
N_EPISODES        = 20
BURST_CONTROLLER  = 0
BURST_MAGNITUDE   = 0.85
STABLE_CPU        = 0.3


class BurstScenarioEnv(SDNLoadBalancingEnv):
    """Inject burst traffic vào controller 0 sau STABLE_STEPS bước."""

    def __init__(self, stable_steps: int = STABLE_STEPS, **kwargs):
        super().__init__(**kwargs)
        self.stable_steps  = stable_steps
        self._burst_active = False

    def reset(self, seed=None, options=None):
        obs, info          = super().reset(seed=seed, options=options)
        self._burst_active = False
        return obs, info

    def _get_state_mock(self) -> np.ndarray:
        load          = np.zeros((self.num_controllers, 3), dtype=np.float32)
        switch_counts = np.bincount(self.switch_assignment, minlength=self.num_controllers)

        if self.step_count >= self.stable_steps:
            self._burst_active = True

        for i in range(self.num_controllers):
            base_cpu = switch_counts[i] / self.num_switches
            if self._burst_active and i == BURST_CONTROLLER:
                burst_cpu    = BURST_MAGNITUDE + np.random.normal(0, 0.03)
                load[i, 0]   = np.clip(burst_cpu, 0.0, 1.0)
                load[i, 1]   = np.clip(burst_cpu * 0.85 + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 2]   = np.clip(burst_cpu + np.random.normal(0, 0.05), 0.0, 1.0)
            else:
                load[i, 0]   = np.clip(base_cpu * STABLE_CPU / 0.5 + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 1]   = np.clip(base_cpu * 0.25 + np.random.normal(0, 0.02), 0.0, 1.0)
                load[i, 2]   = np.clip(base_cpu * STABLE_CPU / 0.5 + np.random.normal(0, 0.04), 0.0, 1.0)

        return load.flatten()

# ------------------------------------------------------------------
# Runner helpers
# ------------------------------------------------------------------

def _make_scenario_env():
    return BurstScenarioEnv(
        stable_steps=STABLE_STEPS,
        num_controllers=NUM_CONTROLLERS,
        num_switches=NUM_SWITCHES,
        use_mock=True,
    )


def _run_episode_with_model(model, env) -> Dict:
    obs, _  = env.reset()
    rewards, variances, latencies = [], [], []
    done = truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action[0]))
        rewards.append(float(reward))
        variances.append(float(info.get("variance_cpu", 0.0)))
        latencies.append(float(info.get("latency", 0.0)))
    return {
        "total_reward":    sum(rewards),
        "variances":       variances,
        "latencies":       latencies,
        "migration_count": info.get("migration_count", 0),
    }


def _run_episode_multiagent(agent_models: Dict, env) -> Dict:
    obs, _  = env.reset()
    rewards, variances, latencies = [], [], []
    done = truncated = False
    while not done and not truncated:
        cpu_loads         = obs[::3]
        most_loaded_idx   = int(np.argmax(cpu_loads))
        agent_id          = f"controller_{most_loaded_idx}"
        action, _         = agent_models[agent_id].predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, done, truncated, info = env.step(int(action[0]))
        rewards.append(float(reward))
        variances.append(float(info.get("variance_cpu", 0.0)))
        latencies.append(float(info.get("latency", 0.0)))
    return {
        "total_reward":    sum(rewards),
        "variances":       variances,
        "latencies":       latencies,
        "migration_count": info.get("migration_count", 0),
    }


def _run_episode_baseline(policy, env) -> Dict:
    obs, _ = env.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "switch_assignment"):
        policy.switch_assignment = env.switch_assignment.copy()
    rewards, variances, latencies = [], [], []
    migration_count = 0
    done = truncated = False
    while not done and not truncated:
        action             = policy.select_action(obs, env)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        variances.append(float(np.var(obs[::3])))
        latencies.append(float(info.get("latency", 0.0)))
        migration_count   += 1 if info.get("migration_success", False) else 0
    return {
        "total_reward":    sum(rewards),
        "variances":       variances,
        "latencies":       latencies,
        "migration_count": migration_count,
    }

# ------------------------------------------------------------------
# Aggregator
# ------------------------------------------------------------------

def _aggregate(episodes: list) -> Dict:
    all_rewards    = [e["total_reward"]   for e in episodes]
    all_variances  = [v for e in episodes for v in e["variances"]]
    all_latencies  = [l for e in episodes for l in e["latencies"]]
    all_migrations = [e["migration_count"] for e in episodes]
    return {
        "mean_reward":      float(np.mean(all_rewards)),
        "std_reward":       float(np.std(all_rewards)),
        "mean_variance_cpu": float(np.mean(all_variances)),
        "mean_latency":     float(np.mean(all_latencies)),
        "mean_migrations":  float(np.mean(all_migrations)),
        "rewards_per_ep":   all_rewards,
        "variances":        all_variances[:TOTAL_STEPS],
        "latencies":        all_latencies[:TOTAL_STEPS],
        "migrations":       all_migrations,
    }

# ------------------------------------------------------------------
# Run helpers
# ------------------------------------------------------------------

def run_rl_agent(model_path: str, n_episodes: int, is_multiagent: bool) -> Dict:
    if is_multiagent:
        agent_models = _load_multiagent_models(model_path, NUM_CONTROLLERS)
        episodes = []
        for ep in range(n_episodes):
            result = _run_episode_multiagent(agent_models, _make_scenario_env())
            episodes.append(result)
            logger.info(f"[Multi-Agent] Episode {ep+1:2d}: reward={result['total_reward']:.3f}")
    else:
        # FIX: _load_model trả về (model, algo) tuple
        model, algo = _load_model(model_path)
        logger.info(f"Auto-detected: {algo}")
        episodes = []
        for ep in range(n_episodes):
            result = _run_episode_with_model(model, _make_scenario_env())
            episodes.append(result)
            logger.info(f"[RL-{algo}] Episode {ep+1:2d}: reward={result['total_reward']:.3f}")

    return _aggregate(episodes)


def run_baseline(baseline_name: str, n_episodes: int) -> Dict:
    policy   = make_baseline(baseline_name, NUM_CONTROLLERS, NUM_SWITCHES)
    episodes = [_run_episode_baseline(policy, _make_scenario_env()) for _ in range(n_episodes)]
    return _aggregate(episodes)

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(model_path: str, output_dir: str, n_episodes: int, is_multiagent: bool):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 1: BURST TRAFFIC")
    logger.info(f"  Mode: {'Multi-Agent' if is_multiagent else 'Single-Agent'}")
    logger.info(f"  Stable: {STABLE_STEPS} bước | Burst: {BURST_STEPS} bước")
    logger.info(f"  Controller bị tấn công: C{BURST_CONTROLLER+1} | Burst CPU: {BURST_MAGNITUDE}")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy RL Agent...")
    rl_result = run_rl_agent(model_path, n_episodes, is_multiagent)
    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes)
    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 1")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Mean Reward':>12} {'Var CPU':>10} {'Latency':>10} {'Migrations':>12}")
    logger.info("-" * 68)
    for name, res in [("RL Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(
            f"{name:<20} {res['mean_reward']:>12.4f} "
            f"{res['mean_variance_cpu']:>10.4f} "
            f"{res['mean_latency']:>10.4f} "
            f"{res['mean_migrations']:>12.1f}"
        )

    var_reduction = (
        (rr_result["mean_variance_cpu"] - rl_result["mean_variance_cpu"])
        / max(rr_result["mean_variance_cpu"], 1e-6) * 100
    )
    logger.info(f"\n✓ Variance CPU giảm so với Round-Robin: {var_reduction:.1f}% (mục tiêu ≥ 30%)")

    plot_scenario_summary(
        rewards=rl_result["rewards_per_ep"], variances=rl_result["variances"],
        latencies=rl_result["latencies"],    migration_counts=rl_result["migrations"],
        output_dir=output_dir, scenario_name="Burst Traffic",
    )

    # FIX: plot_comparison nhận dict gộp
    rl_label = "Multi-Agent DQN" if is_multiagent else "DQN Best"
    plot_comparison(
        all_results={
            rl_label:      {"mean_reward": rl_result["mean_reward"], "std_reward": rl_result["std_reward"]},
            "round_robin": {"mean_reward": rr_result["mean_reward"], "std_reward": rr_result["std_reward"]},
            "least_load":  {"mean_reward": ll_result["mean_reward"], "std_reward": ll_result["std_reward"]},
        },
        output_dir=output_dir,
    )
    logger.info(f"\nBiểu đồ đã lưu tại: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 1: Burst Traffic")
    parser.add_argument("--model",      default="models/best_model.zip")
    parser.add_argument("--multiagent", action="store_true")
    parser.add_argument("--output",     default="data/scenario1")
    parser.add_argument("--episodes",   type=int, default=N_EPISODES)
    args       = parser.parse_args()
    N_EPISODES = args.episodes
    main(model_path=args.model, output_dir=args.output, n_episodes=args.episodes, is_multiagent=args.multiagent)