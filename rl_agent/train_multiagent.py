import sys
import logging
from pathlib import Path
from typing import Dict
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))
from rl_agent.envs.sdn_multiagent_env import SDNMultiAgentEnv

logging.basicConfig(level = logging.INFO, format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger("train_multiagent")

class SingleAgentWrapper(SDNMultiAgentEnv):
    """
    Wrap SDNMultiAgentEnv thành single-agent Gymnasium env để train từng agent
    với Stable-Baselines3 (SB3 không hỗ trợ multi-agent trực tiếp).
        - Mỗi instance của SingleAgentWrapper tương ứng với 1 agent (controller).
        - Observation: vẫn là global state (mỗi controller biết load của tất cả).
    """

    def __init__(self, agent_id: str, num_controllers: int = 3, num_switches: int = 12):
        super().__init__(num_controllers = num_controllers, num_switches = num_switches, use_mock = True)
        self.my_agent = agent_id
        self.my_idx = int(agent_id.split("_")[1])

        # Override spaces theo Gymnasium
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(low = 0.0, high = 1.0, shape=(num_controllers * 3,), dtype = np.float32)
        self.action_space = gym.spaces.Discrete(num_switches + 1)

    def reset(self, seed = None, options = None):
        obs_dict, info_dict = super().reset(seed = seed, options = options)
        return obs_dict[self.my_agent], info_dict.get(self.my_agent, {})

    def step(self, action):
        # Tạo actions dict: agent mình dùng action thật, agents khác no-op
        actions = {agent: self.num_switches for agent in self.agents}  # no-op mặc định
        actions[self.my_agent] = int(action)

        obs_dict, rew_dict, term_dict, trunc_dict, info_dict = super().step(actions)

        obs = obs_dict.get(self.my_agent, np.zeros(self.observation_space.shape, dtype = np.float32))
        reward = rew_dict.get(self.my_agent, 0.0)
        terminated = term_dict.get(self.my_agent, False)
        truncated = trunc_dict.get(self.my_agent, False)
        info = info_dict.get(self.my_agent, {})

        return obs, reward, terminated, truncated, info

class MultiAgentSDNTrainer:
    def __init__(
        self,
        num_controllers: int = 3,
        num_switches: int = 12,
        model_dir: str = "models/multiagent/",
    ):
        self.num_controllers = num_controllers
        self.num_switches = num_switches
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents = True, exist_ok = True)

        # Dùng env gốc để evaluate phối hợp
        self.eval_env = SDNMultiAgentEnv(
            num_controllers = num_controllers,
            num_switches = num_switches,
            use_mock = True
        )

        # Tạo 1 DQN model cho mỗi agent (Independent DQN)
        self.agents: Dict[str, dict] = {}
        agent_ids = [f"controller_{i}" for i in range(num_controllers)]

        for agent_id in agent_ids:
            wrapper = SingleAgentWrapper(agent_id, num_controllers, num_switches)
            vec_env = DummyVecEnv([lambda aid=agent_id, nc=num_controllers, ns=num_switches: SingleAgentWrapper(aid, nc, ns)])

            model = DQN(
                "MlpPolicy",
                vec_env,
                learning_rate = 1e-3,
                exploration_fraction = 0.12,
                exploration_initial_eps = 1.0,
                exploration_final_eps = 0.05,
                batch_size = 64,
                buffer_size = 80000,
                train_freq = 4,
                target_update_interval = 5000,
                verbose = 0,
                tensorboard_log = str(self.model_dir / f"logs_{agent_id}"),
                seed = 42,
            )

            self.agents[agent_id] = {"model": model, "env": vec_env}
            logger.info(f"Khởi tạo DQN agent cho {agent_id}")

    def train(self, total_timesteps_per_agent: int = 100000):
        """
        Independent training: mỗi agent train riêng.
        Đây là cách đơn giản nhất — không có communication giữa agents.
        """
        logger.info("=" * 60)
        logger.info("HUẤN LUYỆN MULTI-AGENT (INDEPENDENT DQN)")
        logger.info(f"Controllers: {self.num_controllers} | Switches: {self.num_switches}")
        logger.info(f"Timesteps/agent: {total_timesteps_per_agent}")
        logger.info("=" * 60)

        for agent_id, info in self.agents.items():
            logger.info(f"\n--- Training: {agent_id} ---")
            info["model"].learn(total_timesteps = total_timesteps_per_agent, progress_bar = True)
            save_path = self.model_dir / f"dqn_{agent_id}"
            info["model"].save(str(save_path))
            logger.info(f"Saved: {save_path}.zip")

        logger.info("\nHUẤN LUYỆN HOÀN TẤT!")

    def evaluate(self, num_episodes: int = 10):
        """
        Đánh giá phối hợp: tất cả agents chạy cùng lúc trên SDNMultiAgentEnv.
            - Mỗi agent chọn action dựa trên observation toàn cục.
            - Tính reward tổng hợp và log kết quả.
        """
        logger.info("=" * 60)
        logger.info("ĐÁNH GIÁ MULTI-AGENT SYSTEM")
        logger.info("=" * 60)

        all_rewards = {agent_id: [] for agent_id in self.agents}

        for episode in range(num_episodes):
            observations, _ = self.eval_env.reset()
            episode_rewards = {agent_id: 0.0 for agent_id in self.eval_env.possible_agents}
            steps = 0

            while self.eval_env.agents:
                actions = {}
                for agent_id in self.eval_env.agents:
                    if agent_id in self.agents:
                        obs = observations[agent_id].reshape(1, -1)  # FIX: reshape cho SB3
                        action, _ = self.agents[agent_id]["model"].predict(obs, deterministic = True)
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

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Training for SDN Load Balancing")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--controllers", type=int, default=3)
    parser.add_argument("--switches", type=int, default=12)
    args = parser.parse_args()

    trainer = MultiAgentSDNTrainer(
        num_controllers = args.controllers,
        num_switches = args.switches,
        model_dir = "models/multiagent/",
    )

    if args.train or (not args.train and not args.evaluate):
        trainer.train(total_timesteps_per_agent = args.timesteps)

    if args.evaluate or (not args.train and not args.evaluate):
        trainer.evaluate(num_episodes = 10)

if __name__ == "__main__":
    main()