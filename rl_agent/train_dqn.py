import os
import sys
import logging
from pathlib import Path

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_agent.envs.sdn_env import SDNLoadBalancingEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("train_dqn")

def make_env(num_controllers, num_switches, use_mock):
    return SDNLoadBalancingEnv(
        num_controllers=num_controllers,
        num_switches=num_switches,
        use_mock=use_mock,
    )

def train_dqn(
    total_timesteps: int = 200000,
    learning_rate: float = 1e-3,
    exploration_fraction: float = 0.15,
    batch_size: int = 64,
    buffer_size: int = 100000,
    num_controllers: int = 3,
    num_switches: int = 12,
    model_dir: str = "models/",
    log_dir: str = "logs/",
    use_mock: bool = True,
):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU HUẤN LUYỆN SINGLE-AGENT DQN")
    logger.info(f"Controllers: {num_controllers} | Switches: {num_switches} | Timesteps: {total_timesteps}")
    logger.info("=" * 60)

    env = DummyVecEnv([lambda nc=num_controllers, ns=num_switches, um=use_mock: make_env(nc, ns, um)])
    eval_env = DummyVecEnv([lambda nc=num_controllers, ns=num_switches, um=use_mock: make_env(nc, ns, um)])

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        batch_size=batch_size,
        buffer_size=buffer_size,
        train_freq=4,
        target_update_interval=5000,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    logger.info("Bắt đầu training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True,
    )

    final_path = os.path.join(model_dir, "dqn_final")
    model.save(final_path)
    logger.info(f"Training hoàn tất! Model lưu tại: {final_path}.zip")
    logger.info(f"Best model lưu tại: {os.path.join(model_dir, 'best_model')}.zip")

    env.close()
    eval_env.close()
    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train DQN for SDN Load Balancing")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--switches", type=int, default=12)
    parser.add_argument("--real", action="store_true", help="Dùng Ryu thật thay vì mock")
    args = parser.parse_args()

    train_dqn(
        total_timesteps=args.timesteps,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        num_controllers=args.controllers,
        num_switches=args.switches,
        use_mock=not args.real,
    )