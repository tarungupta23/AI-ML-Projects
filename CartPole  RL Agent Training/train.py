"""
Assignment 1: Train PPO on CartPole
------------------------------------
Trains a PPO agent (Stable-Baselines3) on the CartPole-v1 Gymnasium
environment, logs every episode's reward via the Monitor wrapper so we
can later plot a learning curve, and saves the final model.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

LOG_DIR = "logs"
MODEL_DIR = "models"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 100_000

def main():
    # 1. Create environment, wrapped with Monitor so episode rewards are logged
    env = gym.make("CartPole-v1")
    env = Monitor(env, filename=os.path.join(LOG_DIR, "monitor.csv"))

    # Separate eval env used by EvalCallback to track + save the best model
    eval_env = gym.make("CartPole-v1")
    eval_env = Monitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor.csv"))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best_model"),
        log_path=LOG_DIR,
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # 2. Select algorithm: PPO with the default MLP policy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        batch_size=64,
        verbose=1,
    )

    # 3. Train model
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    # 4. Save final model
    model.save(os.path.join(MODEL_DIR, "cartpole_ppo"))
    print("\nTraining complete. Model saved to models/cartpole_ppo.zip")


if __name__ == "__main__":
    main()
