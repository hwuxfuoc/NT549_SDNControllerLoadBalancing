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
from scenarios.scenario1_burst import (_run_episode_with_model, _run_episode_multiagent, _run_episode_baseline, _aggregate)

logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scenario3")

NUM_CONTROLLERS = 3
NUM_SWITCHES = 12
FAULT_CONTROLLER = 1
FAULT_CPU = 0.95
PHASE1_STEPS = 50
PHASE2_STEPS = 100
PHASE3_STEPS = 50
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS
UPTIME_VARIANCE_THRESHOLD = 0.15
N_EPISODES = 20

class ControllerFaultEnv(SDNLoadBalancingEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fault_active = False
        self._recovery_active = False
        self._recovery_progress = 0.0

    def reset(self, seed = None, options = None):
        obs, info = super().reset(seed = seed, options = options)
        self._fault_active = False
        self._recovery_active = False
        self._recovery_progress = 0.0
        return obs, info

    def step(self, action):
        self._update_fault_state()
        return super().step(action)

    def _update_fault_state(self):
        step = self.step_count

        if step < PHASE1_STEPS:
            self._fault_active = False
            self._recovery_active = False

        elif step == PHASE1_STEPS:
            self._fault_active = True
            self._recovery_active = False

        elif PHASE1_STEPS <= step < PHASE1_STEPS + PHASE2_STEPS:
            self._fault_active = True

        elif step == PHASE1_STEPS + PHASE2_STEPS:
            self._fault_active = False
            self._recovery_active = True
            self._recovery_progress = 0.0

        else:
            self._fault_active = False
            self._recovery_active = True
            self._recovery_progress = min(1.0, (step - PHASE1_STEPS - PHASE2_STEPS) / PHASE3_STEPS)

    def _get_state_mock(self) -> np.ndarray:
        load = np.zeros((self.num_controllers, 3), dtype = np.float32)
        switch_counts = np.bincount(self.switch_assignment, minlength = self.num_controllers)

        for i in range(self.num_controllers):
            base_cpu = switch_counts[i] / self.num_switches
            if i == FAULT_CONTROLLER and self._fault_active:
                cpu = FAULT_CPU + np.random.normal(0, 0.02)
                load[i, 0] = np.clip(cpu, 0.0, 1.0)
                load[i, 1] = np.clip(cpu * 0.9 + np.random.normal(0, 0.02), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.05), 0.0, 1.0)
            elif i == FAULT_CONTROLLER and self._recovery_active:
                recovering_cpu = FAULT_CPU * (1 - self._recovery_progress) + base_cpu * self._recovery_progress
                load[i, 0] = np.clip(recovering_cpu + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 1] = np.clip(recovering_cpu * 0.8 + np.random.normal(0, 0.02), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.04), 0.0, 1.0)
            else:
                load[i, 0] = np.clip(base_cpu + np.random.normal(0, 0.04), 0.0, 1.0)
                load[i, 1] = np.clip(base_cpu * 0.8 + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.05), 0.0, 1.0)

        return load.flatten()

def _compute_uptime(variances: List[float], threshold: float = UPTIME_VARIANCE_THRESHOLD) -> float:
    if not variances:
        return 0.0
    
    return sum(1 for v in variances if v <= threshold) / len(variances) * 100.0

def _make_scenario_env():
    return ControllerFaultEnv(num_controllers = NUM_CONTROLLERS, num_switches = NUM_SWITCHES, use_mock = True)

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
        env = _make_scenario_env()
        result = run_ep(env)
        episodes.append(result)
        logger.info(f"[{label}] Episode {ep+1:2d}: reward={result['total_reward']:.3f}")

    agg = _aggregate(episodes)
    all_variances = [v for e in episodes for v in e["variances"]]
    agg["uptime_percent"] = _compute_uptime(all_variances)

    return agg

def run_baseline(baseline_name: str, n_episodes: int) -> Dict:
    policy = make_baseline(baseline_name, NUM_CONTROLLERS, NUM_SWITCHES)
    episodes = []

    for ep in range(n_episodes):
        env = _make_scenario_env()
        result = _run_episode_baseline(policy, env)
        episodes.append(result)

    agg = _aggregate(episodes)
    all_variances = [v for e in episodes for v in e["variances"]]
    agg["uptime_percent"] = _compute_uptime(all_variances)

    return agg

def main(model_path: str, output_dir: str, n_episodes: int, is_multiagent: bool):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 3: LỖI CONTROLLER TẠM THỜI")
    logger.info(f"  Mode: {'Multi-Agent' if is_multiagent else 'Single-Agent'}")
    logger.info(f"  Controller bị lỗi: C{FAULT_CONTROLLER+1} | CPU fault: {FAULT_CPU}")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy RL Agent...")
    rl_result = run_rl_agent(model_path, n_episodes, is_multiagent)
    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes)
    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 3")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Mean Reward':>12} {'Var CPU':>10} {'Uptime%':>10}")
    logger.info("-" * 56)
    for name, res in [("RL Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(
            f"{name:<20} {res['mean_reward']:>12.4f} "
            f"{res['mean_variance_cpu']:>10.4f} "
            f"{res.get('uptime_percent', 0):>9.1f}%"
        )
    logger.info(f"\n✓ Uptime RL: {rl_result['uptime_percent']:.1f}% (mục tiêu ≥ 99%)")

    plot_scenario_summary(rewards = rl_result["rewards_per_ep"], variances = rl_result["variances"], latencies = rl_result["latencies"], migration_counts = rl_result["migrations"], output_dir = output_dir, scenario_name = "Controller Fault")
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
    parser = argparse.ArgumentParser(description = "Scenario 3: Controller Fault")
    parser.add_argument("--model", default = "models/best_model.zip")
    parser.add_argument("--multiagent", action="store_true")
    parser.add_argument("--output", default = "data/scenario3")
    parser.add_argument("--episodes", type = int, default = N_EPISODES)
    args = parser.parse_args()
    N_EPISODES = args.episodes
    main(model_path = args.model, output_dir = args.output, n_episodes = args.episodes, is_multiagent = args.multiagent)