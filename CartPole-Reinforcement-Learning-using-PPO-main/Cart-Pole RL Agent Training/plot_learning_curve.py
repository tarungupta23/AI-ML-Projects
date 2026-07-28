"""
Plot the PPO CartPole learning curve (episode reward vs episode number,
with a rolling average) from the Monitor log produced during training.
"""

import pandas as pd
import matplotlib.pyplot as plt

MONITOR_CSV = "logs/monitor.csv"
OUT_PATH = "plots/learning_curve.png"
ROLLING_WINDOW = 20


def main():
    df = pd.read_csv(MONITOR_CSV, skiprows=1)  # skip the json comment header
    df["episode"] = range(1, len(df) + 1)
    df["rolling_mean"] = df["r"].rolling(ROLLING_WINDOW, min_periods=1).mean()

    plt.figure(figsize=(9, 5.5))
    plt.plot(df["episode"], df["r"], color="#8ecae6", linewidth=1, label="Episode reward")
    plt.plot(
        df["episode"],
        df["rolling_mean"],
        color="#023047",
        linewidth=2.2,
        label=f"{ROLLING_WINDOW}-episode rolling average",
    )
    plt.axhline(500, color="#e76f51", linestyle="--", linewidth=1, label="Max reward (500)")
    plt.xlabel("Episode")
    plt.ylabel("Episode reward")
    plt.title("PPO on CartPole-v1 — Learning Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150)
    print(f"Saved learning curve to {OUT_PATH}")
    print(f"Total episodes logged: {len(df)}")
    print(f"Final rolling-average reward: {df['rolling_mean'].iloc[-1]:.1f}")
    print(f"Episode where reward first hit 500: "
          f"{df[df['r'] >= 500]['episode'].min() if (df['r'] >= 500).any() else 'never'}")


if __name__ == "__main__":
    main()
