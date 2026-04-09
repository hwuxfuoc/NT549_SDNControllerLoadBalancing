import logging
import numpy as np
import sys
from pathlib import Path
from typing import Dict

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.envs.sdn_env import SDNLoadBalancingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("evaluate")

def make_env(num_controllers: int = 3, num_switches: int = 12, use_mock: bool = True):
    """Helper tạo env đúng tham số."""
    return SDNLoadBalancingEnv(
        num_controllers=num_controllers,
        num_switches=num_switches,
        use_mock=use_mock,
    )

def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
    deterministic: bool = True,
    use_mock: bool = True,
) -> Dict[str, float]:
    """Đánh giá trained DQN agent."""

    logger.info(f"Loading model từ {model_path}")
    model = DQN.load(model_path)

    env = DummyVecEnv([lambda: make_env(num_controllers, num_switches, use_mock)])

    episode_rewards = []
    episode_lengths = []
    episode_variances = []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_variance = 1.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)

            total_reward += float(reward[0])
            done = bool(done[0])
            steps += 1

            # Lấy variance từ info nếu có
            if info and "variance_cpu" in info[0]:
                last_variance = info[0]["variance_cpu"]

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_variances.append(last_variance)
        logger.info(f"Episode {episode+1:2d}: reward={total_reward:.3f} | steps={steps} | variance={last_variance:.4f}")

    env.close()

    metrics = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_variance_cpu": float(np.mean(episode_variances)),
    }
    return metrics

def evaluate_baseline(
    baseline_name: str,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
) -> Dict[str, float]:
    """
    Đánh giá một baseline policy.
    Baselines:
        - random: chọn action ngẫu nhiên.
        - round_robin: lần lượt chọn switch để migrate (không dựa trên state).
        - least_load: chọn switch của controller có CPU cao nhất để migrate sang controller ít tải nhất.
    """
    env = make_env(num_controllers, num_switches, use_mock=True)
    episode_rewards = []
    rr_counter = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not done and not truncated:
            if baseline_name == "random":
                action = env.action_space.sample()

            elif baseline_name == "round_robin":
                action = rr_counter % env.action_space.n
                rr_counter += 1

            elif baseline_name == "least_load":
                # Chọn switch của controller có CPU cao nhất → migrate đi
                cpu_loads = obs[::3]  # CPU của từng controller
                overloaded_ctrl = int(np.argmax(cpu_loads))
                # Tìm switch đang thuộc controller đó
                candidates = [
                    sw for sw in range(num_switches)
                    if sw % num_controllers == overloaded_ctrl
                ]
                if candidates:
                    # Chọn switch đầu tiên của controller bị overload
                    switch_id = candidates[0]
                    target_ctrl = int(np.argmin(cpu_loads))
                    if target_ctrl == overloaded_ctrl:
                        target_ctrl = (overloaded_ctrl + 1) % num_controllers
                    # Map (switch_id, target_ctrl) → action index
                    candidates_ctrl = [c for c in range(num_controllers) if c != overloaded_ctrl]
                    offset = candidates_ctrl.index(target_ctrl) if target_ctrl in candidates_ctrl else 0
                    action = switch_id * (num_controllers - 1) + offset
                else:
                    action = env.action_space.sample()
            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    env.close()

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }

def compare_all(
    model_path: str,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
    output_dir: str = "data/",
):
    """So sánh DQN agent với Round-Robin và Least-Load."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("ĐÁNH GIÁ DQN AGENT")
    logger.info("=" * 60)
    agent_metrics = evaluate_agent(
        model_path, n_episodes=n_episodes,
        num_controllers=num_controllers, num_switches=num_switches
    )

    baseline_results = {}
    for baseline in ["random", "round_robin", "least_load"]:
        logger.info(f"\nĐÁNH GIÁ BASELINE: {baseline.upper()}")
        baseline_results[baseline] = evaluate_baseline(
            baseline, n_episodes=n_episodes,
            num_controllers=num_controllers, num_switches=num_switches
        )

    # In kết quả so sánh
    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ SO SÁNH")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<20} {'Mean Reward':>12} {'Std Reward':>12}")
    logger.info("-" * 46)
    logger.info(f"{'DQN Agent':<20} {agent_metrics['mean_reward']:>12.4f} {agent_metrics['std_reward']:>12.4f}")
    for name, m in baseline_results.items():
        logger.info(f"{name:<20} {m['mean_reward']:>12.4f} {m['std_reward']:>12.4f}")

    # Lưu CSV
    import csv
    csv_path = Path(output_dir) / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "mean_reward", "std_reward"])
        writer.writerow(["dqn_agent", agent_metrics["mean_reward"], agent_metrics["std_reward"]])
        for name, m in baseline_results.items():
            writer.writerow([name, m["mean_reward"], m["std_reward"]])
    logger.info(f"\nKết quả đã lưu: {csv_path}")

    # Vẽ biểu đồ nếu có visualizer
    try:
        from utils.visualizer import plot_comparison
        plot_comparison(agent_metrics, baseline_results, output_dir=output_dir)
    except ImportError:
        logger.info("(visualizer chưa sẵn sàng — bỏ qua vẽ biểu đồ)")

    return agent_metrics, baseline_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SDN RL Agent")
    parser.add_argument("--model", default="models/dqn_best", help="Đường dẫn model")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--switches", type=int, default=12)
    parser.add_argument("--output", default="data/")
    args = parser.parse_args()

    compare_all(
        model_path=args.model,
        n_episodes=args.episodes,
        num_controllers=args.controllers,
        num_switches=args.switches,
        output_dir=args.output,
    )