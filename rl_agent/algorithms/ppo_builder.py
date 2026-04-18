from stable_baselines3 import PPO

def build_ppo(env, log_dir, learning_rate, batch_size):
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=2048,
        gamma=0.99,
        verbose=1,
        tensorboard_log=log_dir,
        seed=42,
    )