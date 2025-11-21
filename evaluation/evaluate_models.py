# evaluation/evaluate_models.py – CLEAN VERSION (A2C, PPO, DQN only)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import A2C, PPO, DQN

from environment.custom_env import PeerTechEnv

RESULT_DIR = "evaluation/results"
PLOT_DIR = "evaluation/plots"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# --------------------------------------------------------
# Run evaluation one model
# --------------------------------------------------------
def evaluate_model(name, model, env, n_episodes=50):
    rewards = []
    lengths = []

    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)

    df = pd.DataFrame({"reward": rewards, "length": lengths})
    df.to_csv(f"{RESULT_DIR}/{name}_episodes.csv", index=False)
    return df


# --------------------------------------------------------
# Main evaluation
# --------------------------------------------------------
def run_all_evaluations():
    print("\n==================== MODEL EVALUATION ====================\n")

    env = PeerTechEnv(render_mode=None)

    MODEL_PATHS = {
        "A2C": "models/pg/a2c_pro_best/best_model.zip",
        "PPO": "models/pg/ppo_pro_best/best_model.zip",
        "DQN": "models/dqn_pro_best/best_model.zip",
    }

    loaded = {}

    # Load models
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            loaded[name] = {"A2C": A2C, "PPO": PPO, "DQN": DQN}[name].load(path)
            print(f" Loaded {name}")
        else:
            print(f" Missing {name}: {path}")

    summary = {}

    for name, model in loaded.items():
        print(f"\nEvaluating {name}...")
        df = evaluate_model(name, model, env)

        summary[name] = {
            "avg_reward": df["reward"].mean(),
            "std_reward": df["reward"].std(),
            "avg_len": df["length"].mean(),
        }

        print(f"{name} Results:")
        print(f"  Avg Reward: {summary[name]['avg_reward']:.2f}")
        print(f"  Std Dev:    {summary[name]['std_reward']:.2f}")
        print(f"  Avg Length: {summary[name]['avg_len']:.2f}")

    # Save summary
    pd.DataFrame(summary).T.to_csv(f"{RESULT_DIR}/summary.csv")
    print("\n Saved summary.csv")

    # ---------- PLOTS ----------
    for name in loaded.keys():
        df = pd.read_csv(f"{RESULT_DIR}/{name}_episodes.csv")
        plt.plot(df["reward"].rolling(5).mean(), label=name)

    plt.title("Smoothed Rewards (rolling=5)")
    plt.legend()
    plt.savefig(f"{PLOT_DIR}/rewards_smoothed.png")
    plt.clf()

    # Boxplot
    box_data = [pd.read_csv(f"{RESULT_DIR}/{name}_episodes.csv")["reward"] for name in loaded.keys()]
    plt.boxplot(box_data, labels=list(loaded.keys()))
    plt.title("Reward Distribution")
    plt.savefig(f"{PLOT_DIR}/reward_boxplot.png")
    plt.clf()

    print("\n Plots generated in evaluation/plots/")
    print("\n==================== DONE ====================\n")


if __name__ == "__main__":
    run_all_evaluations()
