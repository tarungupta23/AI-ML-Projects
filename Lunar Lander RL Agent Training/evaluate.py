"""
evaluate.py
-----------
Loads a trained PPO model and evaluates it over N episodes, reporting
the mean/std reward and a qualitative performance rating.

Usage:
    python evaluate.py --model models/ppo_lunarlander --episodes 20
"""

import argparse

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def rate_performance(mean_reward: float) -> str:
    if mean_reward < 0:
        return "Poor"
    elif mean_reward < 100:
        return "Moderate"
    elif mean_reward < 200:
        return "Good"
    else:
        return "Excellent"


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO LunarLander model")
    parser.add_argument("--model", type=str, default="models/ppo_lunarlander")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--deterministic", action="store_true", default=True)
    args = parser.parse_args()

    env = gym.make("LunarLander-v3")
    env = Monitor(env)

    model = PPO.load(args.model)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=args.episodes,
        deterministic=args.deterministic,
    )

    print(f"Episodes evaluated : {args.episodes}")
    print(f"Mean reward        : {mean_reward:.2f}")
    print(f"Std reward         : {std_reward:.2f}")
    print(f"Performance rating : {rate_performance(mean_reward)}")

    env.close()


if __name__ == "__main__":
    main()
