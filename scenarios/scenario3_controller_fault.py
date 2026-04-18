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
logger = logging.getLogger("scenario3")

# ------------------------------------------------------------------
# Cấu hình kịch bản
# ------------------------------------------------------------------

NUM_CONTROLLERS = 3
NUM_SWITCHES = 12

FAULT_CONTROLLER = 1        # controller bị lỗi (0-indexed)
FAULT_CPU = 0.95            # CPU khi controller lỗi
RECOVERY_STEPS_DELAY = 10   # bước C1 phục hồi sau khi agent đã migrate xong

PHASE1_STEPS = 50           # hoạt động bình thường
PHASE2_STEPS = 100          # C1 bị lỗi
PHASE3_STEPS = 50           # C1 hồi phục
TOTAL_STEPS = PHASE1_STEPS + PHASE2_STEPS + PHASE3_STEPS  # 200

# Ngưỡng variance để coi là "uptime ok"
UPTIME_VARIANCE_THRESHOLD = 0.15
N_EPISODES = 20

class ControllerFaultEnv(SDNLoadBalancingEnv):
    """
    Override env để inject lỗi vào FAULT_CONTROLLER tại phase 2,
    và phục hồi ở phase 3.

    Phase 1: load bình thường tỷ lệ switch count.
    Phase 2: FAULT_CONTROLLER bị spike CPU lên FAULT_CPU bất kể switch count.
    Phase 3: FAULT_CONTROLLER hồi phục, CPU giảm dần về mức bình thường.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fault_active = False
        self._recovery_active = False
        self._recovery_progress = 0.0   # 0.0 = vừa hồi phục, 1.0 = hoàn toàn bình thường

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
            logger.info(f"[Fault] Bước {step}: C{FAULT_CONTROLLER+1} bị lỗi! CPU → {FAULT_CPU}")

        elif PHASE1_STEPS <= step < PHASE1_STEPS + PHASE2_STEPS:
            self._fault_active = True

        elif step == PHASE1_STEPS + PHASE2_STEPS:
            self._fault_active = False
            self._recovery_active = True
            self._recovery_progress = 0.0
            logger.info(f"[Fault] Bước {step}: C{FAULT_CONTROLLER+1} bắt đầu hồi phục")

        else:
            self._fault_active = False
            self._recovery_active = True
            # Tăng dần recovery (linear trong PHASE3_STEPS bước)
            self._recovery_progress = min(1.0, (step - PHASE1_STEPS - PHASE2_STEPS) / PHASE3_STEPS)

    def _get_state_mock(self) -> np.ndarray:
        load = np.zeros((self.num_controllers, 3), dtype=np.float32)
        switch_counts = np.bincount(self.switch_assignment, minlength=self.num_controllers)

        for i in range(self.num_controllers):
            base_cpu = switch_counts[i] / self.num_switches

            if i == FAULT_CONTROLLER and self._fault_active:
                # Controller bị lỗi: CPU rất cao, RAM cao
                cpu = FAULT_CPU + np.random.normal(0, 0.02)
                load[i, 0] = np.clip(cpu, 0.0, 1.0)
                load[i, 1] = np.clip(cpu * 0.9 + np.random.normal(0, 0.02), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.05), 0.0, 1.0)

            elif i == FAULT_CONTROLLER and self._recovery_active:
                # Hồi phục dần: nội suy từ FAULT_CPU về base_cpu
                recovering_cpu = FAULT_CPU * (1 - self._recovery_progress) + base_cpu * self._recovery_progress
                load[i, 0] = np.clip(recovering_cpu + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 1] = np.clip(recovering_cpu * 0.8 + np.random.normal(0, 0.02), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.04), 0.0, 1.0)

            else:
                # Hoạt động bình thường
                load[i, 0] = np.clip(base_cpu + np.random.normal(0, 0.04), 0.0, 1.0)
                load[i, 1] = np.clip(base_cpu * 0.8 + np.random.normal(0, 0.03), 0.0, 1.0)
                load[i, 2] = np.clip(base_cpu + np.random.normal(0, 0.05), 0.0, 1.0)

        return load.flatten()

# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

def _compute_uptime(variances: List[float], threshold: float = UPTIME_VARIANCE_THRESHOLD) -> float:
    """Tính % bước có variance CPU dưới ngưỡng (= uptime)."""
    if not variances:
        return 0.0
    ok = sum(1 for v in variances if v <= threshold)

    return ok / len(variances) * 100.0


def run_rl_agent(model_path: str, n_episodes: int = N_EPISODES) -> Dict:
    model = DQN.load(model_path)
    all_rewards, all_variances, all_latencies, all_migrations = [], [], [], []
    phase_variances = {1: [], 2: [], 3: []}

    for ep in range(n_episodes):
        env = ControllerFaultEnv(num_controllers=NUM_CONTROLLERS, num_switches=NUM_SWITCHES, use_mock=True)
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

            step = env.step_count
            if step <= PHASE1_STEPS:
                phase_variances[1].append(v)
            elif step <= PHASE1_STEPS + PHASE2_STEPS:
                phase_variances[2].append(v)
            else:
                phase_variances[3].append(v)

        all_rewards.append(sum(ep_rewards))
        all_variances.extend(ep_variances)
        all_latencies.extend(ep_latencies)
        all_migrations.append(info.get("migration_count", 0))
        logger.info(f"[RL] Episode {ep+1:2d}: reward={sum(ep_rewards):.3f} | "
                    f"migrations={info.get('migration_count', 0)}")

    uptime = _compute_uptime(all_variances)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu":  float(np.mean(all_variances)),
        "variance_phase1": float(np.mean(phase_variances[1])) if phase_variances[1] else 0.0,
        "variance_phase2": float(np.mean(phase_variances[2])) if phase_variances[2] else 0.0,
        "variance_phase3": float(np.mean(phase_variances[3])) if phase_variances[3] else 0.0,
        "uptime_percent": uptime,
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
        env = ControllerFaultEnv(num_controllers=NUM_CONTROLLERS, num_switches=NUM_SWITCHES, use_mock=True)
        obs, _ = env.reset()

        if baseline_name == "round_robin":
            balancer = RoundRobinBalancer(NUM_CONTROLLERS, NUM_SWITCHES)
        else:
            balancer = LeastLoadBalancer(NUM_CONTROLLERS, NUM_SWITCHES)
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

    uptime = _compute_uptime(all_variances)

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward":  float(np.std(all_rewards)),
        "mean_variance_cpu": float(np.mean(all_variances)),
        "uptime_percent": uptime,
        "mean_latency":  float(np.mean(all_latencies)),
        "mean_migrations": float(np.mean(all_migrations)),
    }


def main(model_path: str, output_dir: str = "data/scenario3"):
    Path(output_dir).mkdir(parents=True, exist_ok = True)
    logger.info("=" * 60)
    logger.info("KỊCH BẢN 3: LỖI CONTROLLER TẠM THỜI")
    logger.info(f"  Controller bị lỗi: C{FAULT_CONTROLLER + 1} | CPU fault: {FAULT_CPU}")
    logger.info(f"  Phase 1 (normal):   {PHASE1_STEPS} bước")
    logger.info(f"  Phase 2 (fault):    {PHASE2_STEPS} bước")
    logger.info(f"  Phase 3 (recovery): {PHASE3_STEPS} bước")
    logger.info("=" * 60)

    logger.info("\n[1/3] Chạy DQN Agent...")
    rl_result = run_rl_agent(model_path, n_episodes = N_EPISODES)

    logger.info("\n[2/3] Chạy Round-Robin baseline...")
    rr_result = run_baseline("round_robin", n_episodes = N_EPISODES)

    logger.info("\n[3/3] Chạy Least-Load baseline...")
    ll_result = run_baseline("least_load", n_episodes = N_EPISODES)

    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ KỊCH BẢN 3")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Mean Reward':>12} {'Var CPU':>10} {'Uptime%':>10} {'Migrations':>12}")
    logger.info("-" * 68)
    for name, res in [("DQN Agent", rl_result), ("Round-Robin", rr_result), ("Least-Load", ll_result)]:
        logger.info(
            f"{name:<20} {res['mean_reward']:>12.4f} "
            f"{res['mean_variance_cpu']:>10.4f} "
            f"{res['uptime_percent']:>9.1f}% "
            f"{res['mean_migrations']:>12.1f}"
        )

    logger.info(f"\n[RL] Variance theo phase:")
    logger.info(f"  Phase 1 (normal):   {rl_result['variance_phase1']:.4f}")
    logger.info(f"  Phase 2 (fault):    {rl_result['variance_phase2']:.4f}")
    logger.info(f"  Phase 3 (recovery): {rl_result['variance_phase3']:.4f}")
    logger.info(f"\n✓ Uptime RL: {rl_result['uptime_percent']:.1f}% (mục tiêu ≥ 99%)")

    plot_scenario_summary(
        rewards=rl_result["rewards_per_ep"],
        variances=rl_result["variances"],
        latencies=rl_result["latencies"],
        migration_counts=rl_result["migrations"],
        output_dir=output_dir,
        scenario_name="Controller Fault",
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
    parser = argparse.ArgumentParser(description = "Scenario 3: Controller Fault")
    parser.add_argument("--model", default = "models/multiagent/dqn_controller_0.zip")
    parser.add_argument("--output", default = "data/scenario3")
    parser.add_argument("--episodes", type = int, default = N_EPISODES)
    args = parser.parse_args()

    N_EPISODES = args.episodes
    main(model_path = args.model, output_dir = args.output)