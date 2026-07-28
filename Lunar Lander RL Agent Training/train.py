"""
train.py
--------
Trains a PPO agent to solve the Gymnasium LunarLander-v3 environment
using Stable-Baselines3.

Usage:
    python train.py                     # default 500,000 timesteps
    python train.py --timesteps 100000  # custom length
"""

import argparse
import os

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback


MODEL_DIR = "models"
LOG_DIR = "logs"


class ProgressCallback(BaseCallback):
    """Prints periodic progress and tracks episode rewards for plotting."""

    def __init__(self, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            print(f"[train] timesteps={self.num_timesteps}")
        return True


def build_env(log_path):
    env = gym.make("LunarLander-v3")
    env = Monitor(env, filename=log_path)
    return env


def main():
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--timesteps", type=int, default=500_000,
                         help="Total training timesteps (default: 500000)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--model-name", type=str, default="ppo_lunarlander")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    monitor_path = os.path.join(LOG_DIR, "monitor.csv")
    env = build_env(monitor_path)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        clip_range=args.clip_range,
        verbose=1,
    )

    callback = ProgressCallback(print_freq=max(args.n_steps, 10000))

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    save_path = os.path.join(MODEL_DIR, args.model_name)
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
