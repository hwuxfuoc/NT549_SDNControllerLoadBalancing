import os
import sys
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from rl_agent.algorithms.dqn_builder import build_dqn
from rl_agent.algorithms.ppo_builder import build_ppo

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")

SUPPORTED_ALGOS = ["dqn", "ppo"]

_BUILDERS = {
    "dqn": build_dqn,
    "ppo": build_ppo,
}

_MODEL_NAMES = {
    "dqn": "dqn_final",
    "ppo": "ppo_final",
}

EVAL_SEED = 123   # seed cố định cho eval_env — nhất quán giữa mọi lần train & evaluate

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def make_env(num_controllers: int, num_switches: int, use_mock: bool) -> SDNLoadBalancingEnv:
    return SDNLoadBalancingEnv(num_controllers=num_controllers, num_switches=num_switches, use_mock=use_mock)


def _make_seeded_eval_env(num_controllers: int, num_switches: int, use_mock: bool) -> DummyVecEnv:
    """
    Tạo eval_env với seed cố định EVAL_SEED.
    Dùng cả trong EvalCallback (training) lẫn evaluate.py để kết quả nhất quán.
    """
    def _factory():
        env = make_env(num_controllers, num_switches, use_mock)
        env.reset(seed=EVAL_SEED)   # fix seed ngay từ đầu
        return env

    vec = DummyVecEnv([_factory])
    vec.seed(EVAL_SEED)
    return vec

# ------------------------------------------------------------------
# Training chính
# ------------------------------------------------------------------

def train(
    algo: str = "dqn",
    total_timesteps: int = 200000,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    buffer_size: int = 100000,
    exploration_fraction: float = 0.15,
    num_controllers: int = 3,
    num_switches: int = 12,
    model_dir: str = "models/",
    log_dir: str = "logs/",
    use_mock: bool = True,
):
    algo = algo.lower().strip()
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Thuật toán '{algo}' không được hỗ trợ. Chọn: {SUPPORTED_ALGOS}")

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"BẮT ĐẦU HUẤN LUYỆN SINGLE-AGENT — {algo.upper()}")
    logger.info(f"Controllers: {num_controllers} | Switches: {num_switches} | Timesteps: {total_timesteps}")
    logger.info(f"Mode: {'MOCK' if use_mock else 'REAL (Ryu + Mininet)'}")
    logger.info("=" * 60)

    # ── Môi trường ────────────────────────────────────────────────────
    env = DummyVecEnv([
        lambda nc=num_controllers, ns=num_switches, um=use_mock: make_env(nc, ns, um)
    ])

    # FIX 1: eval_env seed cố định → EvalCallback chọn best_model nhất quán
    eval_env = _make_seeded_eval_env(num_controllers, num_switches, use_mock)

    # ── Xây dựng model ───────────────────────────────────────────────
    builder = _BUILDERS[algo]

    if algo == "dqn":
        model = builder(
            env=env,
            log_dir=log_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
            buffer_size=buffer_size,
            exploration_fraction=exploration_fraction,
        )
    elif algo == "ppo":
        model = builder(
            env=env,
            log_dir=log_dir,
            learning_rate=learning_rate,
            batch_size=batch_size,
        )

    # ── EvalCallback ─────────────────────────────────────────────────
    # FIX 2: lưu best_model theo algo riêng → DQN/PPO không đè nhau
    #   models/dqn_best_model.zip
    #   models/ppo_best_model.zip
    algo_best_dir = Path(model_dir) / f"{algo}_best"
    algo_best_dir.mkdir(parents=True, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(algo_best_dir),   # → algo_best/best_model.zip
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # ── Training ─────────────────────────────────────────────────────
    logger.info(f"Bắt đầu training {algo.upper()}...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback, progress_bar=True)

    # ── Lưu final model ───────────────────────────────────────────────
    final_name   = _MODEL_NAMES[algo]
    final_path   = os.path.join(model_dir, final_name)
    model.save(final_path)
    logger.info(f"Final model đã lưu: {final_path}.zip")

    # ── FIX 3: so sánh final vs best, ghi best_model.zip tổng ────────
    best_zip      = algo_best_dir / "best_model.zip"
    final_zip     = Path(f"{final_path}.zip")
    global_best   = Path(model_dir) / "best_model.zip"

    # Tạo eval_env mới (seed cố định) để đánh giá công bằng
    cmp_env = _make_seeded_eval_env(num_controllers, num_switches, use_mock)

    # Nạp lại từ file để đảm bảo so sánh đúng weights đã lưu
    from stable_baselines3 import DQN, PPO
    _cls = DQN if algo == "dqn" else PPO

    final_model = _cls.load(str(final_zip).replace(".zip", ""), env=cmp_env)
    final_mean, _ = evaluate_policy(final_model, cmp_env, n_eval_episodes=10, deterministic=True)
    cmp_env.seed(EVAL_SEED)   # reset seed trước khi eval best

    best_mean = float("-inf")
    best_model_zip_src = None

    if best_zip.exists():
        best_model = _cls.load(str(best_zip).replace(".zip", ""), env=cmp_env)
        best_mean, _ = evaluate_policy(best_model, cmp_env, n_eval_episodes=10, deterministic=True)

    logger.info(f"So sánh — Final: {final_mean:.4f} | Best checkpoint: {best_mean:.4f}")

    if final_mean >= best_mean:
        winner_src  = final_zip
        winner_name = f"{algo.upper()} Final"
    else:
        winner_src  = best_zip
        winner_name = f"{algo.upper()} Best checkpoint"

    shutil.copy2(str(winner_src), str(global_best))
    logger.info(f"✓ best_model.zip ← {winner_name} ({max(final_mean, best_mean):.4f})")

    cmp_env.close()
    env.close()
    eval_env.close()

    logger.info("=" * 60)
    logger.info(f"Training hoàn tất!")
    logger.info(f"  {algo}_final.zip   : {final_path}.zip  ({final_mean:.4f})")
    logger.info(f"  {algo}_best/best_model.zip : {best_zip}  ({best_mean:.4f})")
    logger.info(f"  best_model.zip (tổng)      : {global_best}  ({max(final_mean, best_mean):.4f})")
    logger.info(f"  TensorBoard: tensorboard --logdir {log_dir}")
    logger.info("=" * 60)

    return model

# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL Agent for SDN Load Balancing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python rl_agent/train.py --algo dqn
  python rl_agent/train.py --algo ppo --timesteps 300000 --learning-rate 3e-4
        """
    )
    parser.add_argument("--algo",                 choices=SUPPORTED_ALGOS, default="dqn")
    parser.add_argument("--timesteps",            type=int,   default=200000)
    parser.add_argument("--learning-rate",        type=float, default=1e-3)
    parser.add_argument("--batch-size",           type=int,   default=64)
    parser.add_argument("--buffer-size",          type=int,   default=100000)
    parser.add_argument("--exploration-fraction", type=float, default=0.15)
    parser.add_argument("--controllers",          type=int,   default=3)
    parser.add_argument("--switches",             type=int,   default=12)
    parser.add_argument("--model-dir",            default="models/")
    parser.add_argument("--log-dir",              default="logs/")
    parser.add_argument("--real",                 action="store_true")
    args = parser.parse_args()

    train(
        algo=args.algo,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        exploration_fraction=args.exploration_fraction,
        num_controllers=args.controllers,
        num_switches=args.switches,
        model_dir=args.model_dir,
        log_dir=args.log_dir,
        use_mock=not args.real,
    )