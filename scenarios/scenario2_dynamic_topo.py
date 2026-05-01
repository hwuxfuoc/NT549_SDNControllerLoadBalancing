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

# Tái sử dụng helpers từ scenario1
from scenarios.scenario1_burst import (
    _run_episode_with_model, _run_episode_multiagent,
    _run_episode_baseline, _aggregate,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario2")

NUM_CONTROLLERS  = 3
INITIAL_SWITCHES = 9
ADDED_SWITCHES   = 5
REMOVED_SWITCHES = 3
MAX_SWITCHES     = INITIAL_SWITCHES + ADDED_SWITCHES   # 14

PHASE1_STEPS      = 60
PHASE2_STEPS      = 100
PHASE3_STEPS      = 40
TOTAL_STEPS       = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS
N_EPISODES        = 20
TRAFFIC_RAMP_RATE = 0.003


class DynamicTopoEnv(SDNLoadBalancingEnv):
    def __init__(self, **kwargs):
        kwargs["num_switches"] = MAX_SWITCHES
        super().__init__(**kwargs)
        self._active_switches = INITIAL_SWITCHES
        self._traffic_level   = 0.2
        self._phase           = 1

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._active_switches = INITIAL_SWITCHES
        self._traffic_level   = 0.2
        self._phase           = 1
        self.switch_assignment = np.array([i % NUM_CONTROLLERS for i in range(MAX_SWITCHES)])
        self.switch_assignment[INITIAL_SWITCHES:] = 0
        return self._get_state(), info

    def _get_state_mock(self) -> np.ndarray:
        load          = np.zeros((self.num_controllers, 3), dtype=np.float32)
        active_counts = np.zeros(NUM_CONTROLLERS, dtype=int)
        for sw_idx in range(self._active_switches):
            ctrl = int(self.switch_assignment[sw_idx])
            active_counts[ctrl] += 1
        for i in range(NUM_CONTROLLERS):
            base_cpu   = (active_counts[i] / max(self._active_switches, 1)) * self._traffic_level
            load[i, 0] = np.clip(base_cpu + np.random.normal(0, 0.03), 0.0, 1.0)
            load[i, 1] = np.clip(base_cpu * 0.8 + np.random.normal(0, 0.02), 0.0, 1.0)
            load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.04), 0.0, 1.0)
        return load.flatten()

    def step(self, action):
        self._update_phase()
        return super().step(action)

    def _update_phase(self):
        step = self.step_count
        if step < PHASE1_STEPS:
            self._phase         = 1
            self._traffic_level = 0.2
        elif step == PHASE1_STEPS:
            self._phase           = 2
            self._active_switches = MAX_SWITCHES
            self.switch_assignment[INITIAL_SWITCHES:] = 0
        elif PHASE1_STEPS <= step < PHASE1_STEPS + PHASE2_STEPS:
            self._phase         = 2
            self._traffic_level = min(0.9, 0.2 + TRAFFIC_RAMP_RATE * (step - PHASE1_STEPS))
        elif step == PHASE1_STEPS + PHASE2_STEPS:
            self._phase           = 3
            self._active_switches = MAX_SWITCHES - REMOVED_SWITCHES
        else:
            self._phase         = 3
            self._traffic_level = 0.7


def _make_scenario_env():
    return DynamicTopoEnv(num_controllers=NUM_CONTROLLERS, use_mock=True)


def run_rl_agent(model_path: str, n_episodes: int, is_multiagent: bool) -> Dict:
    if is_multiagent:
        agent_models = _load_multiagent_models(model_path, NUM_CONTROLLERS)
        run_ep       = lambda env: _run_episode_multiagent(agent_models, env)
        label        = "Multi-Agent"
    else:
        # FIX: _load_model trả về (model, algo) tuple
        model, algo = _load_model(model_path)
        logger.info(f"Auto-detected: {algo}")
        run_ep = lambda env: _run_episode_with_model(model, env)
        label  = f"RL-{algo}"

    episodes = []
    for ep in range(n_episodes):
        result = run_ep(_make_scenario_env())
        episodes.append(result)
        logger.info(f"[{label}] Episode {ep+1:2d}: reward={result['total_reward']:.3f}")

    return _aggregate(episodes)


def run_baseline(baseline_name: str, n_episodes: int) -> Dict:
    policy   = make_baseline(baseline_name, NUM_CONTROLLERS, MAX_SWITCHES)
    episodes = [_run_episode_baseline(policy, _make_scenario_env()) for _ in range(n_episodes)]
    return _aggregate(episodes)


def main(model_path: str, output_dir: str, n_episodes: int, is_multiagent: bool):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 2: TOPOLOGY ĐỘNG")
    logger.info(f"  Mode: {'Multi-Agent' if is_multiagent else 'Single-Agent'}")
    logger.info(f"  Phase 1: {PHASE1_STEPS} | Phase 2: {PHASE2_STEPS} | Phase 3: {PHASE3_STEPS} bước")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy RL Agent...")
    rl_result = run_rl_agent(model_path, n_episodes, is_multiagent)
    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes)
    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 2")
    logger.info("=" * 60)
    for name, res in [("RL Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(f"{name:<20} reward={res['mean_reward']:.4f}  var={res['mean_variance_cpu']:.4f}")

    ratio = rr_result["mean_variance_cpu"] / max(rl_result["mean_variance_cpu"], 1e-6)
    logger.info(f"\n✓ Throughput imbalance ratio RL/RR: {ratio:.2f}x tốt hơn")

    plot_scenario_summary(
        rewards=rl_result["rewards_per_ep"], variances=rl_result["variances"],
        latencies=rl_result["latencies"],    migration_counts=rl_result["migrations"],
        output_dir=output_dir, scenario_name="Dynamic Topology",
    )

    # FIX: plot_comparison nhận dict gộp
    rl_label = "Multi-Agent DQN" if is_multiagent else "Best Model"
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
    parser = argparse.ArgumentParser(description="Scenario 2: Dynamic Topology")
    parser.add_argument("--model",      default="models/best_model.zip")
    parser.add_argument("--multiagent", action="store_true")
    parser.add_argument("--output",     default="data/scenario2")
    parser.add_argument("--episodes",   type=int, default=N_EPISODES)
    args       = parser.parse_args()
    N_EPISODES = args.episodes
    main(model_path=args.model, output_dir=args.output, n_episodes=args.episodes, is_multiagent=args.multiagent)