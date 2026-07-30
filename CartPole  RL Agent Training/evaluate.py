"""
Evaluate the trained PPO CartPole agent over N episodes and report
mean/std reward, as required by Section 10 / Assignment 1.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

MODEL_PATH = "models/cartpole_ppo"
N_EVAL_EPISODES = 20


def main():
    env = gym.make("CartPole-v1")
    model = PPO.load(MODEL_PATH)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=N_EVAL_EPISODES
    )

    print(f"Evaluated over {N_EVAL_EPISODES} episodes")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward:  {std_reward:.2f}")

    with open("logs/evaluation_result.txt", "w") as f:
        f.write(f"Episodes: {N_EVAL_EPISODES}\n")
        f.write(f"Mean reward: {mean_reward:.2f}\n")
        f.write(f"Std reward: {std_reward:.2f}\n")


if __name__ == "__main__":
    main()
