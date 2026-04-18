from stable_baselines3 import DQN

def build_dqn(env, log_dir, learning_rate, batch_size, buffer_size, exploration_fraction):
    return DQN(
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