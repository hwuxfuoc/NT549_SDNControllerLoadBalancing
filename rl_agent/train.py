import os
import sys
import logging
import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))

from rl_agent.envs.sdn_env import SDNLoadBalancingEnv
from rl_agent.algorithms.dqn_builder import build_dqn
from rl_agent.algorithms.ppo_builder import build_ppo

logging.basicConfig(level = logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")

# ------------------------------------------------------------------
# Các thuật toán được hỗ trợ
# ------------------------------------------------------------------

SUPPORTED_ALGOS = ["dqn", "ppo"]

# Mapping tên algo → builder function
_BUILDERS = {
    "dqn":  build_dqn,
    "ppo":  build_ppo,
}

# Tên file model lưu theo algo
_MODEL_NAMES = {
    "dqn":  "dqn_final",
    "ppo":  "ppo_final",
}

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------

def make_env(num_controllers: int, num_switches: int, use_mock: bool) -> SDNLoadBalancingEnv:
    """Tạo một instance SDNLoadBalancingEnv với tham số cho trước."""
    return SDNLoadBalancingEnv(num_controllers = num_controllers, num_switches = num_switches, use_mock = use_mock)

# ------------------------------------------------------------------
# Hàm training chính
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
    """
    Train một RL agent với thuật toán được chỉ định.

    Args:
        algo:                 Tên thuật toán — "dqn" hoặc "ppo".
        total_timesteps:      Tổng số environment steps để train.
        learning_rate:        Learning rate của optimizer.
        batch_size:           Số samples mỗi lần update network.
        buffer_size:          Kích thước replay buffer (chỉ dùng cho DQN).
        exploration_fraction: Tỉ lệ timesteps dành cho epsilon-greedy exploration (chỉ DQN).
        num_controllers:      Số Ryu controllers trong cluster.
        num_switches:         Số switches trong mạng.
        model_dir:            Thư mục lưu model (.zip).
        log_dir:              Thư mục lưu TensorBoard logs.
        use_mock:             True = dùng mock state, False = kết nối Ryu thật.

    Returns:
        Trained SB3 model object.
    """
    algo = algo.lower().strip()
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Thuật toán '{algo}' không được hỗ trợ. Chọn: {SUPPORTED_ALGOS}")

    Path(model_dir).mkdir(parents = True, exist_ok = True)
    Path(log_dir).mkdir(parents = True, exist_ok = True)

    logger.info("=" * 60)
    logger.info(f"BẮT ĐẦU HUẤN LUYỆN SINGLE-AGENT — {algo.upper()}")
    logger.info(f"Controllers: {num_controllers} | Switches: {num_switches} | Timesteps: {total_timesteps}")
    logger.info(f"Mode: {'MOCK' if use_mock else 'REAL (Ryu + Mininet)'}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Tạo môi trường
    # ------------------------------------------------------------------

    # DummyVecEnv bọc env thành vectorized format mà SB3 yêu cầu.
    # Lambda dùng default argument để tránh closure bug trong Python.
    env = DummyVecEnv([lambda nc = num_controllers, ns = num_switches, um = use_mock: make_env(nc, ns, um)])
    eval_env = DummyVecEnv([lambda nc = num_controllers, ns = num_switches, um = use_mock: make_env(nc, ns, um)])

    # ------------------------------------------------------------------
    # Xây dựng model theo algo
    # ------------------------------------------------------------------
    # Mỗi builder nhận (env, log_dir, learning_rate, ...) và trả về SB3 model.
    # DDPG yêu cầu continuous action space — nếu env là discrete, cần wrap.
    # Với bài toán này discrete action là phù hợp nhất, DDPG là optional.

    builder = _BUILDERS[algo]

    if algo == "dqn":
        model = builder(
            env = env,
            log_dir = log_dir,
            learning_rate = learning_rate,
            batch_size = batch_size,
            buffer_size = buffer_size,
            exploration_fraction = exploration_fraction,
        )

    elif algo == "ppo":
        model = builder(
            env = env,
            log_dir = log_dir,
            learning_rate = learning_rate,
            batch_size = batch_size,
        )

    # ------------------------------------------------------------------
    # EvalCallback: lưu best model định kỳ
    # ------------------------------------------------------------------

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = model_dir,          # lưu vào models/best_model.zip
        log_path = log_dir,
        eval_freq = 10000,                          # eval mỗi 10,000 training steps
        n_eval_episodes = 5,
        deterministic = True,
        verbose = 1,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    logger.info(f"Bắt đầu training {algo.upper()}...")
    model.learn(total_timesteps = total_timesteps, callback = eval_callback, progress_bar = True)

    # Lưu final model (best model đã được EvalCallback lưu riêng)
    final_name = _MODEL_NAMES[algo]
    final_path = os.path.join(model_dir, final_name)
    model.save(final_path)

    logger.info(f"Training hoàn tất!")
    logger.info(f"  Final model: {final_path}.zip")
    logger.info(f"  Best model:  {os.path.join(model_dir, 'best_model')}.zip")
    logger.info(f"  TensorBoard: tensorboard --logdir {log_dir}")

    env.close()
    eval_env.close()
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
    parser.add_argument("--algo", choices = SUPPORTED_ALGOS, default = "dqn", help = "Thuật toán RL (default: dqn)")
    parser.add_argument("--timesteps", type = int, default = 200000, help = "Tổng timesteps training (default: 200000)")
    parser.add_argument("--learning-rate", type = float, default = 1e-3, help = "Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type = int, default = 64, help = "Batch size (default: 64)")
    parser.add_argument("--buffer-size", type = int, default = 100000, help = "Replay buffer size — chỉ DQN (default: 100000)")
    parser.add_argument("--exploration-fraction", type = float, default = 0.15, help = "Exploration fraction — chỉ DQN (default: 0.15)")
    parser.add_argument("--controllers", type = int, default = 3, help = "Số controllers (default: 3)")
    parser.add_argument("--switches", type = int, default = 12, help = "Số switches (default: 12)")
    parser.add_argument("--model-dir", default = "models/", help = "Thư mục lưu model (default: models/)")
    parser.add_argument("--log-dir", default = "logs/", help = "Thư mục TensorBoard logs (default: logs/)")
    parser.add_argument("--real", action = "store_true", help = "Kết nối Ryu thật thay vì mock")
    args = parser.parse_args()

    train(
        algo = args.algo,
        total_timesteps = args.timesteps,
        learning_rate = args.learning_rate,
        batch_size = args.batch_size,
        buffer_size = args.buffer_size,
        exploration_fraction = args.exploration_fraction,
        num_controllers = args.controllers,
        num_switches = args.switches,
        model_dir = args.model_dir,
        log_dir = args.log_dir,
        use_mock = not args.real,
    )