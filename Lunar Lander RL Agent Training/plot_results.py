"""
plot_results.py
----------------
Reads the Monitor log produced during training (logs/monitor.csv) and
plots episode reward over time, plus a rolling average trend line.

Usage:
    python plot_results.py --log logs/monitor.csv --out graphs/training_curve.png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt


def load_monitor_csv(path):
    rewards, lengths = [], []
    with open(path, "r") as f:
        lines = f.readlines()
    # First line is a JSON header comment, second line is the CSV header
    for line in lines[2:]:
        parts = line.strip().split(",")
        if len(parts) >= 2:
            rewards.append(float(parts[0]))
            lengths.append(float(parts[1]))
    return np.array(rewards), np.array(lengths)


def rolling_mean(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Plot PPO LunarLander training curve")
    parser.add_argument("--log", type=str, default="logs/monitor.csv")
    parser.add_argument("--out", type=str, default="graphs/training_curve.png")
    parser.add_argument("--window", type=int, default=20)
    args = parser.parse_args()

    rewards, lengths = load_monitor_csv(args.log)
    episodes = np.arange(1, len(rewards) + 1)
    smoothed = rolling_mean(rewards, args.window)
    smoothed_x = np.arange(args.window, len(rewards) + 1) if len(rewards) >= args.window else episodes

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.plot(episodes, rewards, alpha=0.3, label="Episode reward", color="#7aa2f7")
    plt.plot(smoothed_x, smoothed, label=f"{args.window}-episode rolling average", color="#f7768e", linewidth=2)
    plt.axhline(200, color="green", linestyle="--", alpha=0.5, label="Solved threshold (200)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO LunarLander-v3 Training Progress")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()
