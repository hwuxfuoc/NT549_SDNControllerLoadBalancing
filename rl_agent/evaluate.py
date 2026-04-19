import logging
import numpy as np
import sys
import csv
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from baselines import make_baseline

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(num_controllers: int = 3, num_switches: int = 12, use_mock: bool = True):
    return SDNLoadBalancingEnv(num_controllers = num_controllers, num_switches = num_switches, use_mock = use_mock)

def _load_model(model_path: str):
    """
    Load SB3 model, tự detect DQN hay PPO từ file.
    SB3 lưu metadata trong .zip nên dùng thử-bắt để detect.
    """
    try:
        model = DQN.load(model_path)
        logger.info(f"Loaded DQN model từ {model_path}")
        return model
    except Exception:
        pass

    try:
        model = PPO.load(model_path)
        logger.info(f"Loaded PPO model từ {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Không load được model từ {model_path}: {e}")


def _load_multiagent_models(model_dir: str, num_controllers: int = 3) -> Dict[str, object]:
    """
    Load tất cả agent models từ thư mục multi-agent.
    Tìm file dạng: models/multiagent/dqn_controller_0.zip, dqn_controller_1.zip, ...

    Returns:
        Dict {agent_id: model} — ví dụ {"controller_0": <DQN>, "controller_1": <DQN>, ...}
    """
    model_dir = Path(model_dir)
    models = {}

    for i in range(num_controllers):
        agent_id = f"controller_{i}"
        model_path = model_dir / f"dqn_{agent_id}.zip"

        if not model_path.exists():
            logger.warning(f"Không tìm thấy model: {model_path} — dùng controller_0 thay thế")
            fallback = model_dir / "dqn_controller_0.zip"
            if fallback.exists():
                models[agent_id] = _load_model(str(fallback))
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy model cho {agent_id} tại {model_path}. "
                    f"Chạy train_multiagent.py trước."
                )
        else:
            models[agent_id] = _load_model(str(model_path))

    logger.info(f"Đã load {len(models)} agent models từ {model_dir}")
    return models

# ------------------------------------------------------------------
# Evaluate single-agent RL
# ------------------------------------------------------------------

def evaluate_agent(model_path: str, n_episodes: int = 10, num_controllers: int = 3, num_switches: int = 12, deterministic: bool = True, use_mock: bool = True) -> Dict[str, float]:
    """Đánh giá single-agent DQN hoặc PPO."""

    model = _load_model(model_path)
    env = DummyVecEnv([lambda: make_env(num_controllers, num_switches, use_mock)])

    episode_rewards, episode_lengths, episode_variances = [], [], []

    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_variance = 1.0

        while not done:
            action, _ = model.predict(obs, deterministic = deterministic)
            obs, reward, done, info = env.step(action)
            total_reward += float(reward[0])
            done = bool(done[0])
            steps += 1
            if info and "variance_cpu" in info[0]:
                last_variance = info[0]["variance_cpu"]

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_variances.append(last_variance)
        logger.info(
            f"Episode {episode+1:2d}: reward={total_reward:.3f} | "
            f"steps={steps} | variance={last_variance:.4f}"
        )

    env.close()
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "max_reward": float(np.max(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "mean_variance_cpu": float(np.mean(episode_variances)),
    }

# ------------------------------------------------------------------
# Evaluate multi-agent RL
# ------------------------------------------------------------------

def evaluate_multiagent(model_dir: str, n_episodes: int = 10, num_controllers: int = 3, num_switches: int = 12) -> Dict[str, float]:
    """
    Đánh giá multi-agent: load tất cả agent models, chạy phối hợp
    trên SDNLoadBalancingEnv (single-agent env dùng làm proxy).

    Chiến lược: Mỗi step, lấy obs, hỏi tất cả agents, chọn action
    của agent có controller bị overload nhất (CPU cao nhất).
    Đây là heuristic đơn giản để kết hợp 3 independent agents.
    """
    agent_models = _load_multiagent_models(model_dir, num_controllers)
    env = make_env(num_controllers, num_switches, use_mock = True)

    episode_rewards, episode_variances = [], []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        last_variance = 1.0
        done = truncated = False

        while not done and not truncated:
            # Tìm controller có CPU cao nhất → ưu tiên action của agent đó
            cpu_loads = obs[::3]  # index 0, 3, 6 = cpu của c0, c1, c2
            most_loaded_agent_idx = int(np.argmax(cpu_loads))
            most_loaded_agent_id = f"controller_{most_loaded_agent_idx}"

            # Lấy action từ agent của controller bị overload nhất
            agent_model = agent_models[most_loaded_agent_id]
            action, _ = agent_model.predict(obs.reshape(1, -1), deterministic = True)
            action = int(action[0])

            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            last_variance = float(info.get("variance_cpu", last_variance))

        episode_rewards.append(total_reward)
        episode_variances.append(last_variance)
        logger.info(
            f"[Multi-Agent] Episode {episode + 1:2d}: "
            f"reward={total_reward:.3f} | variance={last_variance:.4f}"
        )

    env.close()
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_variance_cpu": float(np.mean(episode_variances)),
    }

# ------------------------------------------------------------------
# Evaluate baselines
# ------------------------------------------------------------------

def evaluate_baseline(baseline_name: str, n_episodes: int = 10, num_controllers: int = 3, num_switches: int = 12) -> Dict[str, float]:
    """
    Đánh giá baseline policy dùng select_action() — interface chuẩn,
    không cần map action thủ công.
    Baselines: random, round_robin, least_load.
    """
    env = make_env(num_controllers, num_switches, use_mock = True)
    policy = make_baseline(baseline_name, num_controllers, num_switches)
    episode_rewards = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        done = truncated = False
        total_reward = 0.0

        while not done and not truncated:
            action = policy.select_action(obs, env)
            obs, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)

        episode_rewards.append(total_reward)

    env.close()
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
    }

# ------------------------------------------------------------------
# Compare all
# ------------------------------------------------------------------

def compare_all(model_path: str, n_episodes: int = 10, num_controllers: int = 3, num_switches: int = 12, output_dir: str = "data/", is_multiagent: bool = False) -> tuple:
    """
    So sánh RL agent với tất cả baselines.
    Nếu is_multiagent=True thì load từ thư mục multi-agent.
    """
    Path(output_dir).mkdir(parents = True, exist_ok = True)

    # Evaluate RL agent
    logger.info("=" * 60)
    if is_multiagent:
        logger.info("ĐÁNH GIÁ MULTI-AGENT DQN")
        logger.info("=" * 60)
        agent_metrics = evaluate_multiagent(model_dir = model_path, n_episodes = n_episodes, num_controllers = num_controllers, num_switches = num_switches)
        agent_label = "Multi-Agent DQN"
    else:
        logger.info("ĐÁNH GIÁ SINGLE-AGENT RL")
        logger.info("=" * 60)
        agent_metrics = evaluate_agent(model_path = model_path, n_episodes = n_episodes, num_controllers = num_controllers, num_switches = num_switches)
        agent_label = "RL Agent"

    # Evaluate baselines
    baseline_results = {}
    for baseline in ["random", "round_robin", "least_load"]:
        logger.info(f"\nĐÁNH GIÁ BASELINE: {baseline.upper()}")
        baseline_results[baseline] = evaluate_baseline(baseline, n_episodes = n_episodes, num_controllers = num_controllers, num_switches = num_switches)

    # In kết quả so sánh
    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ SO SÁNH")
    logger.info("=" * 60)
    logger.info(f"{'Policy':<22} {'Mean Reward':>12} {'Std Reward':>12} {'Var CPU':>10}")
    logger.info("-" * 60)
    logger.info(
        f"{agent_label:<22} {agent_metrics['mean_reward']:>12.4f} "
        f"{agent_metrics['std_reward']:>12.4f} "
        f"{agent_metrics.get('mean_variance_cpu', 0):>10.4f}"
    )

    for name, m in baseline_results.items():
        logger.info(f"{name:<22} {m['mean_reward']:>12.4f} {m['std_reward']:>12.4f}")

    # Lưu CSV
    csv_path = Path(output_dir) / "comparison_results.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "mean_reward", "std_reward", "mean_variance_cpu"])
        writer.writerow([agent_label, agent_metrics["mean_reward"], agent_metrics["std_reward"], agent_metrics.get("mean_variance_cpu", "")])
        for name, m in baseline_results.items():
            writer.writerow([name, m["mean_reward"], m["std_reward"], ""])

    logger.info(f"\nKết quả đã lưu: {csv_path}")

    # Vẽ biểu đồ
    try:
        from utils.visualizer import plot_comparison
        plot_comparison(agent_metrics, baseline_results, output_dir = output_dir)
    except ImportError:
        logger.info("(visualizer chưa sẵn sàng — bỏ qua vẽ biểu đồ)")

    return agent_metrics, baseline_results

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SDN RL Agent")
    parser.add_argument("--model", default = "models/best_model.zip", help = "Đường dẫn model (.zip) hoặc thư mục multiagent (models/multiagent/)")
    parser.add_argument("--multiagent", action = "store_true", help = "Load multi-agent models từ thư mục --model")
    parser.add_argument("--episodes", type = int, default = 10)
    parser.add_argument("--controllers", type=int, default = 3)
    parser.add_argument("--switches", type=int, default = 12)
    parser.add_argument("--output", default = "data/")
    args = parser.parse_args()

    compare_all(
        model_path = args.model,
        n_episodes = args.episodes,
        num_controllers = args.controllers,
        num_switches = args.switches,
        output_dir = args.output,
        is_multiagent = args.multiagent,
    )