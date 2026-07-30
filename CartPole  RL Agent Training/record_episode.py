"""
Run the trained PPO agent in CartPole-v1 with rgb_array rendering and
save every frame + state/action/reward info, so we can build an
interactive, step-through visualization in the browser (no live display
is available in this sandboxed environment, so we pre-record frames).
"""

import os
import json
import base64
from io import BytesIO

import numpy as np
import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO

MODEL_PATH = "models/cartpole_ppo"
OUT_DIR = "plots/episode_frames"
DATA_PATH = "plots/episode_data.json"
MAX_STEPS = 500

os.makedirs(OUT_DIR, exist_ok=True)


def frame_to_base64_jpeg(frame, quality=70, scale=0.6):
    img = Image.fromarray(frame)
    if scale != 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.BILINEAR)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    model = PPO.load(MODEL_PATH)

    obs, info = env.reset(seed=42)
    frames_b64 = []
    steps_info = []

    for step in range(MAX_STEPS):
        frame = env.render()
        frames_b64.append(frame_to_base64_jpeg(frame))

        action, _ = model.predict(obs, deterministic=True)
        action_int = int(action)
        next_obs, reward, terminated, truncated, info = env.step(action)

        steps_info.append({
            "step": step,
            "cart_position": float(obs[0]),
            "cart_velocity": float(obs[1]),
            "pole_angle": float(obs[2]),
            "pole_angular_velocity": float(obs[3]),
            "action": action_int,  # 0 = left, 1 = right
            "reward": float(reward),
        })

        obs = next_obs
        if terminated or truncated:
            break

    env.close()

    total_reward = sum(s["reward"] for s in steps_info)
    print(f"Episode finished after {len(steps_info)} steps, total reward = {total_reward}")

    with open(DATA_PATH, "w") as f:
        json.dump({
            "steps": steps_info,
            "total_reward": total_reward,
            "num_frames": len(frames_b64),
        }, f)

    # Save frames as a single JS-friendly array file (base64 strings), chunked
    with open("plots/episode_frames_b64.json", "w") as f:
        json.dump(frames_b64, f)

    print(f"Saved {len(frames_b64)} frames and step data.")


if __name__ == "__main__":
    main()
