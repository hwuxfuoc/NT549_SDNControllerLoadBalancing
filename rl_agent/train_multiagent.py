import sys
import shutil
import logging
from pathlib import Path
from typing import Dict
import numpy as np
import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_agent.envs.sdn_multiagent_env import SDNMultiAgentEnv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_multiagent")

# Phải khớp với EVAL_SEED trong train.py và evaluate.py
EVAL_SEED = 123

# ------------------------------------------------------------------
# Single-agent wrapper
# ------------------------------------------------------------------

class SingleAgentWrapper(gym.Env):
    """
    Wrap SDNMultiAgentEnv thành single-agent Gymnasium env để train từng agent
    với Stable-Baselines3 (SB3 không hỗ trợ multi-agent trực tiếp).
    """

    def __init__(self, agent_id: str, num_controllers: int = 3, num_switches: int = 12):
        super().__init__()

        self.env = SDNMultiAgentEnv(
            num_controllers=num_controllers,
            num_switches=num_switches,
            use_mock=True,
        )

        self.my_agent      = agent_id
        self.num_switches  = num_switches
        self.my_idx        = int(agent_id.split("_")[1])

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_controllers * 3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(num_switches + 1)

    def reset(self, seed=None, options=None):
        obs_dict, _ = self.env.reset(seed=seed, options=options)
        return obs_dict[self.my_agent], {}

    def step(self, action):
        actions = {agent: self.num_switches for agent in self.env.agents}
        actions[self.my_agent] = int(action)

        obs_dict, rew_dict, term_dict, trunc_dict, _ = self.env.step(actions)

        obs       = obs_dict.get(self.my_agent, np.zeros(self.observation_space.shape, dtype=np.float32))
        reward    = rew_dict.get(self.my_agent, 0.0)
        terminated = term_dict.get(self.my_agent, False)
        truncated  = trunc_dict.get(self.my_agent, False)

        return obs, reward, terminated, truncated, {}


def _make_seeded_wrapper(agent_id: str, num_controllers: int, num_switches: int) -> DummyVecEnv:
    """DummyVecEnv với seed cố định cho eval — khớp với evaluate.py."""
    def _factory():
        env = SingleAgentWrapper(agent_id, num_controllers, num_switches)
        env.reset(seed=EVAL_SEED)
        return env
    vec = DummyVecEnv([_factory])
    vec.seed(EVAL_SEED)
    return vec

# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class MultiAgentSDNTrainer:
    def __init__(
        self,
        num_controllers: int = 3,
        num_switches: int = 12,
        model_dir: str = "models/multiagent/",
    ):
        self.num_controllers = num_controllers
        self.num_switches    = num_switches
        self.model_dir       = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Eval env gốc để đánh giá phối hợp
        self.eval_env = SDNMultiAgentEnv(
            num_controllers=num_controllers,
            num_switches=num_switches,
            use_mock=True,
        )

        # Tạo 1 DQN model cho mỗi agent (Independent DQN)
        self.agents: Dict[str, dict] = {}
        agent_ids = [f"controller_{i}" for i in range(num_controllers)]

        for agent_id in agent_ids:
            vec_env = DummyVecEnv([
                lambda aid=agent_id, nc=num_controllers, ns=num_switches:
                    SingleAgentWrapper(aid, nc, ns)
            ])

            model = DQN(
                "MlpPolicy",
                vec_env,
                learning_rate=1e-3,
                exploration_fraction=0.12,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                batch_size=64,
                buffer_size=80000,
                train_freq=4,
                target_update_interval=5000,
                verbose=0,
                tensorboard_log=str(self.model_dir / f"logs_{agent_id}"),
                seed=42,
            )

            self.agents[agent_id] = {"model": model, "env": vec_env}
            logger.info(f"Khởi tạo DQN agent cho {agent_id}")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------

    def train(self, total_timesteps_per_agent: int = 100000):
        """
        Independent training: mỗi agent train riêng.
        Sau khi train xong mỗi agent: so sánh final vs best checkpoint,
        lưu cái tốt hơn làm model chính thức cho agent đó.
        """
        logger.info("=" * 60)
        logger.info("HUẤN LUYỆN MULTI-AGENT (INDEPENDENT DQN)")
        logger.info(f"Controllers: {self.num_controllers} | Switches: {self.num_switches}")
        logger.info(f"Timesteps/agent: {total_timesteps_per_agent}")
        logger.info("=" * 60)

        for agent_id, info in self.agents.items():
            logger.info(f"\n--- Training: {agent_id} ---")

            # FIX 1: eval_env seed cố định cho EvalCallback
            eval_vec = _make_seeded_wrapper(agent_id, self.num_controllers, self.num_switches)

            # FIX 2: lưu best checkpoint riêng theo agent
            best_dir = self.model_dir / f"best_{agent_id}"
            best_dir.mkdir(parents=True, exist_ok=True)

            from stable_baselines3.common.callbacks import EvalCallback
            eval_callback = EvalCallback(
                eval_vec,
                best_model_save_path=str(best_dir),   # → best_{agent_id}/best_model.zip
                log_path=str(self.model_dir / f"eval_logs_{agent_id}"),
                eval_freq=10000,
                n_eval_episodes=5,
                deterministic=True,
                verbose=0,
            )

            info["model"].learn(
                total_timesteps=total_timesteps_per_agent,
                callback=eval_callback,
                progress_bar=True,
            )

            # Lưu final model
            final_path = self.model_dir / f"dqn_{agent_id}"
            info["model"].save(str(final_path))
            logger.info(f"Final model đã lưu: {final_path}.zip")

            # FIX 3: so sánh final vs best checkpoint, giữ cái tốt hơn
            self._pick_best_for_agent(
                agent_id=agent_id,
                final_zip=Path(f"{final_path}.zip"),
                best_zip=best_dir / "best_model.zip",
                num_controllers=self.num_controllers,
                num_switches=self.num_switches,
            )

            eval_vec.close()

        logger.info("\nHUẤN LUYỆN HOÀN TẤT!")

    def _pick_best_for_agent(
        self,
        agent_id: str,
        final_zip: Path,
        best_zip: Path,
        num_controllers: int,
        num_switches: int,
    ):
        """
        Load lại final và best checkpoint, eval 10 episodes với seed cố định.
        Copy cái có mean_reward cao hơn vào models/multiagent/dqn_{agent_id}.zip
        (đây là file mà evaluate.py sẽ đọc).
        """
        cmp_env = _make_seeded_wrapper(agent_id, num_controllers, num_switches)

        final_model = DQN.load(str(final_zip).replace(".zip", ""), env=cmp_env)
        final_mean, _ = evaluate_policy(final_model, cmp_env, n_eval_episodes=10, deterministic=True)

        best_mean = float("-inf")
        if best_zip.exists():
            cmp_env.seed(EVAL_SEED)
            best_model = DQN.load(str(best_zip).replace(".zip", ""), env=cmp_env)
            best_mean, _ = evaluate_policy(best_model, cmp_env, n_eval_episodes=10, deterministic=True)

        logger.info(
            f"[{agent_id}] So sánh — Final: {final_mean:.4f} | "
            f"Best checkpoint: {best_mean:.4f}"
        )

        winner_src  = final_zip if final_mean >= best_mean else best_zip
        winner_name = "Final" if final_mean >= best_mean else "Best checkpoint"
        dest        = self.model_dir / f"dqn_{agent_id}.zip"

        shutil.copy2(str(winner_src), str(dest))
        logger.info(
            f"[{agent_id}] ✓ dqn_{agent_id}.zip ← {winner_name} "
            f"({max(final_mean, best_mean):.4f})"
        )

        cmp_env.close()

    # ------------------------------------------------------------------
    # Evaluate phối hợp
    # ------------------------------------------------------------------

    def evaluate(self, num_episodes: int = 10):
        """
        Đánh giá phối hợp: tất cả agents chạy cùng lúc trên SDNMultiAgentEnv.
        """
        logger.info("=" * 60)
        logger.info("ĐÁNH GIÁ MULTI-AGENT SYSTEM")
        logger.info("=" * 60)

        all_rewards = {agent_id: [] for agent_id in self.agents}

        for episode in range(num_episodes):
            observations, _ = self.eval_env.reset(seed=EVAL_SEED + episode)
            episode_rewards  = {agent_id: 0.0 for agent_id in self.eval_env.possible_agents}
            steps = 0

            while self.eval_env.agents:
                actions = {}
                for agent_id in self.eval_env.agents:
                    if agent_id in self.agents:
                        obs = observations[agent_id].reshape(1, -1)
                        action, _ = self.agents[agent_id]["model"].predict(obs, deterministic=True)
                        actions[agent_id] = int(action[0])
                    else:
                        actions[agent_id] = self.num_switches  # no-op

                observations, rewards, terminated, truncated, _ = self.eval_env.step(actions)

                for agent_id, reward in rewards.items():
                    episode_rewards[agent_id] = episode_rewards.get(agent_id, 0.0) + reward

                steps += 1

            for agent_id in self.agents:
                all_rewards[agent_id].append(episode_rewards.get(agent_id, 0.0))

            total = sum(episode_rewards.values())
            logger.info(f"Episode {episode+1:2d} | Steps: {steps:3d} | Total Reward: {total:.3f}")

        logger.info("\n" + "-" * 60)
        logger.info("KẾT QUẢ ĐÁNH GIÁ")
        logger.info("-" * 60)
        for agent_id, rewards in all_rewards.items():
            logger.info(f"{agent_id:15s} | Mean: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
        logger.info("=" * 60)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Training for SDN Load Balancing")
    parser.add_argument("--train",       action="store_true")
    parser.add_argument("--evaluate",    action="store_true")
    parser.add_argument("--timesteps",   type=int, default=100000)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--switches",    type=int, default=12)
    args = parser.parse_args()

    trainer = MultiAgentSDNTrainer(
        num_controllers=args.controllers,
        num_switches=args.switches,
        model_dir="models/multiagent/",
    )

    if args.train or (not args.train and not args.evaluate):
        trainer.train(total_timesteps_per_agent=args.timesteps)

    if args.evaluate or (not args.train and not args.evaluate):
        trainer.evaluate(num_episodes=10)


if __name__ == "__main__":
    main()