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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("evaluate")

# Phải khớp với EVAL_SEED trong train.py để kết quả nhất quán
EVAL_SEED = 123

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_env(num_controllers: int = 3, num_switches: int = 12, use_mock: bool = True):
    return SDNLoadBalancingEnv(num_controllers=num_controllers, num_switches=num_switches, use_mock=use_mock)


def _make_seeded_vec_env(num_controllers: int, num_switches: int, use_mock: bool = True) -> DummyVecEnv:
    """Tạo DummyVecEnv với seed cố định — khớp với eval_env trong train.py."""
    def _factory():
        env = make_env(num_controllers, num_switches, use_mock)
        env.reset(seed=EVAL_SEED)
        return env
    vec = DummyVecEnv([_factory])
    vec.seed(EVAL_SEED)
    return vec


def _load_model(model_path: str):
    """
    Load SB3 model, tự detect DQN hay PPO từ file.
    SB3 lưu metadata trong .zip nên dùng thử-bắt để detect.
    Trả về (model, algo_name).
    """
    try:
        model = DQN.load(model_path)
        logger.info(f"Loaded DQN model từ {model_path}")
        return model, "DQN"
    except Exception:
        pass

    try:
        model = PPO.load(model_path)
        logger.info(f"Loaded PPO model từ {model_path}")
        return model, "PPO"
    except Exception as e:
        raise ValueError(f"Không load được model từ {model_path}: {e}")


def _load_model_strict(model_path: str, algo: str):
    """
    Load model với algo cụ thể (DQN hoặc PPO). Raise lỗi rõ ràng nếu sai.
    """
    algo = algo.upper()
    try:
        if algo == "DQN":
            model = DQN.load(model_path)
        elif algo == "PPO":
            model = PPO.load(model_path)
        else:
            raise ValueError(f"Algo không hợp lệ: {algo}. Chỉ hỗ trợ DQN hoặc PPO.")
        logger.info(f"Loaded {algo} model từ {model_path}")
        return model
    except Exception as e:
        raise ValueError(f"Không load được {algo} model từ {model_path}: {e}")


def _load_multiagent_models(model_dir: str, num_controllers: int = 3) -> Dict[str, object]:
    """
    Load tất cả agent models từ thư mục multi-agent.
    Tìm file dạng: models/multiagent/dqn_controller_0.zip, dqn_controller_1.zip, ...

    Returns:
        Dict {agent_id: model}
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
                model, _ = _load_model(str(fallback))
                models[agent_id] = model
            else:
                raise FileNotFoundError(
                    f"Không tìm thấy model cho {agent_id} tại {model_path}. "
                    f"Chạy train_multiagent.py trước."
                )
        else:
            model, _ = _load_model(str(model_path))
            models[agent_id] = model

    logger.info(f"Đã load {len(models)} agent models từ {model_dir}")
    return models

# ------------------------------------------------------------------
# Evaluate single-agent RL (dùng chung cho Best / DQN / PPO)
# ------------------------------------------------------------------

def evaluate_agent(
    model_path: str,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
    deterministic: bool = True,
    use_mock: bool = True,
    algo: Optional[str] = None,       # None = auto-detect, "DQN" / "PPO" = strict
) -> Dict[str, float]:
    """
    Đánh giá single-agent.
    - algo=None  → auto-detect (dùng cho --model best_model.zip)
    - algo="DQN" → load đúng DQN (dùng cho --dqn-model)
    - algo="PPO" → load đúng PPO (dùng cho --ppo-model)
    """
    if algo is None:
        model, detected_algo = _load_model(model_path)
        logger.info(f"Auto-detected: {detected_algo}")
    else:
        model = _load_model_strict(model_path, algo)

    # Dùng seeded env để kết quả nhất quán với lúc EvalCallback chọn best_model
    env = _make_seeded_vec_env(num_controllers, num_switches, use_mock)

    episode_rewards, episode_lengths, episode_variances = [], [], []

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
        "mean_reward":       float(np.mean(episode_rewards)),
        "std_reward":        float(np.std(episode_rewards)),
        "mean_length":       float(np.mean(episode_lengths)),
        "max_reward":        float(np.max(episode_rewards)),
        "min_reward":        float(np.min(episode_rewards)),
        "mean_variance_cpu": float(np.mean(episode_variances)),
    }

# ------------------------------------------------------------------
# Evaluate multi-agent RL
# ------------------------------------------------------------------

def evaluate_multiagent(
    model_dir: str,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
) -> Dict[str, float]:
    """
    Đánh giá multi-agent: load tất cả agent models, chạy phối hợp
    trên SDNLoadBalancingEnv (single-agent env dùng làm proxy).

    Chiến lược: Mỗi step, lấy obs, hỏi tất cả agents, chọn action
    của agent có controller bị overload nhất (CPU cao nhất).
    """
    agent_models = _load_multiagent_models(model_dir, num_controllers)
    env = make_env(num_controllers, num_switches, use_mock=True)

    episode_rewards, episode_variances = [], []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        last_variance = 1.0
        done = truncated = False

        while not done and not truncated:
            cpu_loads = obs[::3]
            most_loaded_agent_idx = int(np.argmax(cpu_loads))
            most_loaded_agent_id = f"controller_{most_loaded_agent_idx}"

            agent_model = agent_models[most_loaded_agent_id]
            action, _ = agent_model.predict(obs.reshape(1, -1), deterministic=True)
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
        "mean_reward":       float(np.mean(episode_rewards)),
        "std_reward":        float(np.std(episode_rewards)),
        "mean_variance_cpu": float(np.mean(episode_variances)),
    }

# ------------------------------------------------------------------
# Evaluate baselines
# ------------------------------------------------------------------

def evaluate_baseline(
    baseline_name: str,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
) -> Dict[str, float]:
    """
    Đánh giá baseline policy: random, round_robin, least_load.
    """
    env = make_env(num_controllers, num_switches, use_mock=True)
    policy = make_baseline(baseline_name, num_controllers, num_switches)
    episode_rewards, episode_variances = [], []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        if hasattr(policy, "reset"):
            policy.reset()
        done = truncated = False
        total_reward = 0.0
        last_variance = 1.0

        while not done and not truncated:
            action = policy.select_action(obs, env)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += float(reward)
            if isinstance(info, dict):
                last_variance = float(info.get("variance_cpu", last_variance))

        episode_rewards.append(total_reward)
        episode_variances.append(last_variance)

    env.close()
    return {
        "mean_reward":       float(np.mean(episode_rewards)),
        "std_reward":        float(np.std(episode_rewards)),
        "mean_variance_cpu": float(np.mean(episode_variances)),
    }

# ------------------------------------------------------------------
# Compare all — 6 policies
# ------------------------------------------------------------------

def compare_all(
    best_model_path: str,
    dqn_model_path:  Optional[str] = None,
    ppo_model_path:  Optional[str] = None,
    n_episodes: int = 10,
    num_controllers: int = 3,
    num_switches: int = 12,
    output_dir: str = "data/",
    is_multiagent: bool = False,
) -> tuple:
    """
    So sánh đúng 6 dòng cố định:

    SINGLE-AGENT mode:
      1. DQN Best   (models/dqn_best/best_model.zip)
      2. DQN Final  (models/dqn_final.zip)
      3. PPO Best   (models/ppo_best/best_model.zip)
      4. PPO Final  (models/ppo_final.zip)
      5-7. Random / Round-Robin / Least-Load

    MULTIAGENT mode:
      1. Multi-Agent DQN  (models/multiagent/)
      2-4. Random / Round-Robin / Least-Load
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ordered dict giữ thứ tự chèn
    rl_results: Dict[str, Dict[str, float]] = {}

    def _eval(label: str, path: str, algo: str):
        """Eval 1 model, thêm vào rl_results. Skip nếu file không tồn tại."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Bỏ qua '{label}' — không tìm thấy: {path}")
            return
        logger.info("\n" + "=" * 60)
        logger.info(f"ĐÁNH GIÁ: {label.upper()}")
        logger.info("=" * 60)
        rl_results[label] = evaluate_agent(
            model_path=str(p),
            n_episodes=n_episodes,
            num_controllers=num_controllers,
            num_switches=num_switches,
            algo=algo,
        )

    if is_multiagent:
        # ── MULTIAGENT: 1 RL + 3 baselines ───────────────────────────
        logger.info("=" * 60)
        logger.info("ĐÁNH GIÁ MULTI-AGENT DQN")
        logger.info("=" * 60)
        rl_results["Multi-Agent DQN"] = evaluate_multiagent(
            model_dir=best_model_path,
            n_episodes=n_episodes,
            num_controllers=num_controllers,
            num_switches=num_switches,
        )

    else:
        # ── SINGLE-AGENT: suy ra thư mục từ best_model_path ──────────
        # Nếu truyền vào models/best_model.zip thì base = models/
        # Nếu truyền vào models/best_model.zip, tìm:
        #   models/dqn_best/best_model.zip  và  models/dqn_final.zip
        #   models/ppo_best/best_model.zip  và  models/ppo_final.zip
        base = Path(best_model_path).parent  # models/

        # Ưu tiên dùng path tường minh nếu được truyền vào,
        # ngược lại tự suy từ base
        _dqn_best  = str(base / "dqn_best"  / "best_model.zip")
        _dqn_final = dqn_model_path or str(base / "dqn_final.zip")
        _ppo_best  = str(base / "ppo_best"  / "best_model.zip")
        _ppo_final = ppo_model_path or str(base / "ppo_final.zip")

        _eval("DQN Best",  _dqn_best,  "DQN")
        _eval("DQN Final", _dqn_final, "DQN")
        _eval("PPO Best",  _ppo_best,  "PPO")
        _eval("PPO Final", _ppo_final, "PPO")

    # ── Baselines ────────────────────────────────────────────────────
    baseline_results: Dict[str, Dict[str, float]] = {}
    for baseline in ["random", "round_robin", "least_load"]:
        logger.info(f"\nĐÁNH GIÁ BASELINE: {baseline.upper()}")
        baseline_results[baseline] = evaluate_baseline(
            baseline,
            n_episodes=n_episodes,
            num_controllers=num_controllers,
            num_switches=num_switches,
        )

    # ── In bảng kết quả ──────────────────────────────────────────────
    total_rows = len(rl_results) + len(baseline_results)
    logger.info("\n" + "=" * 65)
    logger.info(f"KẾT QUẢ SO SÁNH ({total_rows} POLICIES)")
    logger.info("=" * 65)
    logger.info(f"{'Policy':<26} {'Mean Reward':>12} {'Std Reward':>12} {'Var CPU':>10}")
    logger.info("-" * 65)

    for label, m in rl_results.items():
        logger.info(
            f"{label:<26} {m['mean_reward']:>12.4f} "
            f"{m['std_reward']:>12.4f} "
            f"{m.get('mean_variance_cpu', 0):>10.4f}"
        )
    for name, m in baseline_results.items():
        logger.info(
            f"{name:<26} {m['mean_reward']:>12.4f} "
            f"{m['std_reward']:>12.4f} "
            f"{m.get('mean_variance_cpu', 0):>10.4f}"
        )

    # ── Lưu CSV ──────────────────────────────────────────────────────
    csv_path = Path(output_dir) / "comparison_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "mean_reward", "std_reward", "mean_variance_cpu"])
        for label, m in rl_results.items():
            writer.writerow([label, m["mean_reward"], m["std_reward"], m.get("mean_variance_cpu", "")])
        for name, m in baseline_results.items():
            writer.writerow([name, m["mean_reward"], m["std_reward"], m.get("mean_variance_cpu", "")])
    logger.info(f"\nKết quả đã lưu: {csv_path}")

    # ── Vẽ biểu đồ ───────────────────────────────────────────────────
    try:
        from utils.visualizer import plot_comparison
        # Truyền thẳng all_results — visualizer không cần agent_metrics riêng nữa
        all_results = {**rl_results, **baseline_results}
        plot_comparison(all_results, output_dir=output_dir)
    except ImportError:
        logger.info("(visualizer chưa sẵn sàng — bỏ qua vẽ biểu đồ)")

    return rl_results, baseline_results

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate SDN RL Agent — so sánh 6 policies",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/best_model.zip",
        help=(
            "Đường dẫn tới best model (.zip) hoặc thư mục multiagent.\n"
            "Ví dụ: models/best_model.zip"
        ),
    )
    parser.add_argument(
        "--dqn-model",
        default="models/dqn_final.zip",
        help="Đường dẫn tới DQN final model (mặc định: models/dqn_final.zip)",
    )
    parser.add_argument(
        "--ppo-model",
        default="models/ppo_final.zip",
        help="Đường dẫn tới PPO final model (mặc định: models/ppo_final.zip)",
    )
    parser.add_argument("--multiagent", action="store_true", help="Load multi-agent models từ thư mục --model")
    parser.add_argument("--episodes",    type=int, default=10)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--switches",    type=int, default=12)
    parser.add_argument("--output",      default="data/")
    args = parser.parse_args()

    compare_all(
        best_model_path=args.model,
        dqn_model_path=args.dqn_model,
        ppo_model_path=args.ppo_model,
        n_episodes=args.episodes,
        num_controllers=args.controllers,
        num_switches=args.switches,
        output_dir=args.output,
        is_multiagent=args.multiagent,
    )