"""
test.py
-------
Runs a trained PPO agent on LunarLander-v3 and records MP4 videos of
its episodes (headless-safe, uses render_mode="rgb_array" so it works
on servers without a display). For a live on-screen window instead,
pass --human (requires a local display).

Usage:
    python test.py --model models/ppo_lunarlander --episodes 5
    python test.py --model models/ppo_lunarlander --human   # live window
"""

import argparse
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO


def run_headless_recording(model, episodes, video_folder):
    os.makedirs(video_folder, exist_ok=True)
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder,
                       episode_trigger=lambda ep: True,
                       name_prefix="lander")

    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()
    print(f"Videos saved in {video_folder}/")


def run_live_window(model, episodes):
    import time
    env = gym.make("LunarLander-v3", render_mode="human")
    for ep in range(episodes):
        obs, info = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            time.sleep(0.02)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Test/record a trained PPO LunarLander model")
    parser.add_argument("--model", type=str, default="models/ppo_lunarlander")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--video-folder", type=str, default="videos")
    parser.add_argument("--human", action="store_true",
                         help="Render in a live window instead of recording video")
    args = parser.parse_args()

    model = PPO.load(args.model)

    if args.human:
        run_live_window(model, args.episodes)
    else:
        run_headless_recording(model, args.episodes, args.video_folder)


if __name__ == "__main__":
    main()
