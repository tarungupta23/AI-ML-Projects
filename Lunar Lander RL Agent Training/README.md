# LunarLander-PPO

Autonomous spacecraft landing trained with **Proximal Policy Optimization (PPO)**
on Gymnasium's `LunarLander-v3` environment, using Stable-Baselines3.

This project was fully run end-to-end: a model was trained, evaluated, recorded
on video, and visualized in an interactive mission-console interface. All
artifacts in this folder (`models/`, `logs/`, `videos/`, `graphs/`) are real
outputs from that run, not placeholders.

## Folder structure

```
LunarLander-PPO/
├── train.py              # Train a PPO agent
├── test.py                # Run/record a trained agent (video or live window)
├── evaluate.py             # Statistical evaluation (mean/std reward)
├── plot_results.py         # Plot the training reward curve
├── requirements.txt
│
├── models/
│   └── ppo_lunarlander.zip     # Trained policy (200,000 timesteps)
├── logs/
│   ├── monitor.csv             # Per-episode reward/length log
│   └── train_output.log        # Console output from the training run
├── videos/
│   └── lander-episode-*.mp4    # 3 recorded test episodes
├── graphs/
│   └── training_curve.png      # Reward-over-time plot
│
└── interface/
    ├── index.html               # Mission-console visualization (open in a browser)
    └── assets/                  # Videos + chart used by index.html
```

## Setup

```bash
pip install -r requirements.txt
```

(On Debian/Ubuntu you also need `swig` installed system-wide for Box2D:
`apt-get install swig`.)

## Usage

Train a new model (defaults to 500,000 timesteps as in a full production run;
this repo's included model was trained for 200,000 as a faster demonstration
run that already reaches "Good" performance):

```bash
python train.py --timesteps 500000
```

Evaluate a trained model over 20 episodes:

```bash
python evaluate.py --model models/ppo_lunarlander --episodes 20
```

Record video of the trained agent (headless-safe — no display required):

```bash
python test.py --model models/ppo_lunarlander --episodes 5
```

Watch the agent live in a window (requires a local display):

```bash
python test.py --model models/ppo_lunarlander --human
```

Plot the training curve from the monitor log:

```bash
python plot_results.py --log logs/monitor.csv --out graphs/training_curve.png
```

## Visual interface

Open `interface/index.html` in any browser. It shows:

- **Descent Playback** — the 3 recorded landing episodes, switchable by tab,
  with the actual reward and outcome (landed / crashed) for each.
- **Training Curve** — the real reward-per-episode plot from this run.
- **Evaluation panel** — mean/std reward over 20 episodes and a performance
  rating bar (Poor / Moderate / Good / Excellent).
- **PPO hyperparameters** and the **8-dimensional observation vector**
  reference table.

## Results from this run

| Metric | Value |
|---|---|
| Training timesteps | 200,000 |
| Mean evaluation reward (20 episodes) | 111.78 |
| Std deviation | 107.87 |
| Performance rating | **Good** |

Reward variance is expected — LunarLander is stochastic (random start
position/velocity each episode) and 200k steps is a fast demo run. Training
to the full 500k–1M timesteps in `train.py`'s default typically pushes the
mean reward past 200 (Excellent) with lower variance.

## Environment reference

- **Observation space:** `Box(8,)` — x, y, vx, vy, angle, angular velocity,
  left-leg contact, right-leg contact
- **Action space:** `Discrete(4)` — do nothing, fire left engine, fire main
  engine, fire right engine
- **Reward:** positive for approaching/landing on the pad upright and gently;
  negative for crashing, drifting away, tilting excessively, or wasting fuel
- **Episode ends on:** successful landing, crash, leaving the simulation
  bounds, or timeout
