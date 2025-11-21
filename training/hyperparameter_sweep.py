# training/hyperparameter_sweep.py
"""
Hyperparameter Sweep Engine (PRO Version)

Runs:
- 10× DQN variants
- 10× A2C variants
- 10× PPO variants
- 10× REINFORCE variants

Saves:
- Models under models/pg/<algo>_sweep/
- Per-run CSVs under evaluation/sweeps/
- Global summary CSV under evaluation/sweeps/sweep_summary.csv
"""

import os
import sys
import json
import random
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.common.env_util import make_vec_env

# ------------------------------------------------------------------
# PATH SETUP
# ------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from environment.custom_env import PeerTechEnv  # noqa: E402

# Directories
MODELS_BASE = os.path.join(PROJECT_ROOT, "models", "pg")
SWEEP_MODELS_DIR = {
    "DQN": os.path.join(MODELS_BASE, "dqn_sweep"),
    "A2C": os.path.join(MODELS_BASE, "a2c_sweep"),
    "PPO": os.path.join(MODELS_BASE, "ppo_sweep"),
    "REINFORCE": os.path.join(MODELS_BASE, "reinforce_sweep"),
}
SWEEP_RESULTS_DIR = os.path.join(PROJECT_ROOT, "evaluation", "sweeps")

os.makedirs(MODELS_BASE, exist_ok=True)
for d in SWEEP_MODELS_DIR.values():
    os.makedirs(d, exist_ok=True)
os.makedirs(SWEEP_RESULTS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# REINFORCE IMPLEMENTATION (same style as your tuned one)
# ------------------------------------------------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


def run_reinforce_episode(
    env: PeerTechEnv,
    policy: PolicyNetwork,
    gamma: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, float]:
    obs, _ = env.reset()
    done = False
    truncated = False

    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []

    while not (done or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        dist = policy(obs_t)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        obs, reward, done, truncated, _ = env.step(action.item())

        log_probs.append(log_prob)
        rewards.append(float(reward))

    # Compute discounted returns
    returns: List[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns_np = np.array(returns, dtype=np.float32)
    returns_np = (returns_np - returns_np.mean()) / (returns_np.std() + 1e-8)
    returns_t = torch.tensor(returns_np, dtype=torch.float32, device=device)

    log_probs_t = torch.stack(log_probs)
    loss = -(log_probs_t * returns_t).sum()

    return loss, float(sum(rewards))


def evaluate_reinforce_policy(
    policy: PolicyNetwork,
    n_eval_episodes: int = 15,
    device: str = "cpu",
) -> Dict[str, float]:
    env = PeerTechEnv(render_mode=None)
    rewards = []
    lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        ep_len = 0

        while not (done or truncated):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                dist = policy(obs_t)
                # greedy action for evaluation
                logits = dist.logits
                action = torch.argmax(logits, dim=-1).item()

            obs, r, done, truncated, _ = env.step(action)
            ep_rew += r
            ep_len += 1

        rewards.append(ep_rew)
        lengths.append(ep_len)

    rewards_np = np.array(rewards, dtype=np.float32)
    lengths_np = np.array(lengths, dtype=np.float32)

    return {
        "avg_reward": float(rewards_np.mean()),
        "std_reward": float(rewards_np.std()),
        "avg_length": float(lengths_np.mean()),
    }


# ------------------------------------------------------------------
# SB3 EVALUATION
# ------------------------------------------------------------------
def evaluate_sb3_model(
    model,
    n_eval_episodes: int = 15,
    deterministic: bool = True,
) -> Dict[str, float]:
    env = PeerTechEnv(render_mode=None)
    rewards = []
    lengths = []

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        ep_rew = 0.0
        ep_len = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, done, truncated, _ = env.step(int(action))
            ep_rew += r
            ep_len += 1

        rewards.append(ep_rew)
        lengths.append(ep_len)

    rewards_np = np.array(rewards, dtype=np.float32)
    lengths_np = np.array(lengths, dtype=np.float32)

    return {
        "avg_reward": float(rewards_np.mean()),
        "std_reward": float(rewards_np.std()),
        "avg_length": float(lengths_np.mean()),
    }


# ------------------------------------------------------------------
# DQN SWEEP
# ------------------------------------------------------------------
def dqn_param_grid() -> List[Dict[str, Any]]:
    grid = []
    learning_rates = [1e-3, 5e-4, 3e-4]
    gammas = [0.98, 0.99]
    batch_sizes = [32, 64]
    buffer_sizes = [50_000, 100_000]
    exploration_fracs = [0.1, 0.2]
    exploration_finals = [0.02, 0.05]

    for lr in learning_rates:
        for g in gammas:
            for bs in batch_sizes:
                for buf in buffer_sizes:
                    for ef in exploration_fracs:
                        for eps in exploration_finals:
                            grid.append(
                                {
                                    "learning_rate": lr,
                                    "gamma": g,
                                    "batch_size": bs,
                                    "buffer_size": buf,
                                    "exploration_fraction": ef,
                                    "exploration_final_eps": eps,
                                }
                            )
    random.shuffle(grid)
    return grid[:10]  # 10 configs


def run_dqn_sweep(
    total_timesteps: int = 120_000,
    n_eval_episodes: int = 15,
) -> List[Dict[str, Any]]:
    results = []
    configs = dqn_param_grid()

    print(f"\n🔵 DQN SWEEP: {len(configs)} runs...\n")

    for idx, cfg in enumerate(configs, start=1):
        print(f"[DQN Run {idx}/{len(configs)}] hyperparams: {cfg}")

        def make_env_fn():
            return PeerTechEnv(render_mode=None)

        vec_env = make_vec_env(make_env_fn, n_envs=1)

        model = DQN(
            "MlpPolicy",
            vec_env,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            batch_size=cfg["batch_size"],
            buffer_size=cfg["buffer_size"],
            exploration_fraction=cfg["exploration_fraction"],
            exploration_final_eps=cfg["exploration_final_eps"],
            verbose=0,
            tensorboard_log=None,
        )

        model.learn(total_timesteps=total_timesteps)

        stats = evaluate_sb3_model(model, n_eval_episodes=n_eval_episodes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"dqn_sweep_run{idx:02d}_{timestamp}"
        save_path = os.path.join(SWEEP_MODELS_DIR["DQN"], model_name)
        model.save(save_path)

        row = {
            "algo": "DQN",
            "run_id": idx,
            "model_path": save_path,
            **cfg,
            **stats,
        }
        results.append(row)

        print(
            f"    -> avg_reward={stats['avg_reward']:.2f}, "
            f"std={stats['std_reward']:.2f}, len={stats['avg_length']:.2f}"
        )

    # Save CSV
    csv_path = os.path.join(SWEEP_RESULTS_DIR, "dqn_sweep.csv")
    _write_csv(csv_path, results)
    print(f" DQN sweep results saved to {csv_path}")
    return results


# ------------------------------------------------------------------
# A2C SWEEP
# ------------------------------------------------------------------
def a2c_param_grid() -> List[Dict[str, Any]]:
    grid = []
    learning_rates = [7e-4, 5e-4, 3e-4]
    gammas = [0.98, 0.99]
    n_steps_list = [5, 10, 20]
    ent_coefs = [0.0, 0.01, 0.02]

    for lr in learning_rates:
        for g in gammas:
            for ns in n_steps_list:
                for ent in ent_coefs:
                    grid.append(
                        {
                            "learning_rate": lr,
                            "gamma": g,
                            "n_steps": ns,
                            "ent_coef": ent,
                        }
                    )
    random.shuffle(grid)
    return grid[:10]


def run_a2c_sweep(
    total_timesteps: int = 120_000,
    n_eval_episodes: int = 15,
) -> List[Dict[str, Any]]:
    results = []
    configs = a2c_param_grid()

    print(f"\n A2C SWEEP: {len(configs)} runs...\n")

    for idx, cfg in enumerate(configs, start=1):
        print(f"[A2C Run {idx}/{len(configs)}] hyperparams: {cfg}")

        def make_env_fn():
            return PeerTechEnv(render_mode=None)

        vec_env = make_vec_env(make_env_fn, n_envs=1)

        model = A2C(
            "MlpPolicy",
            vec_env,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            n_steps=cfg["n_steps"],
            ent_coef=cfg["ent_coef"],
            verbose=0,
            tensorboard_log=None,
        )

        model.learn(total_timesteps=total_timesteps)

        stats = evaluate_sb3_model(model, n_eval_episodes=n_eval_episodes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"a2c_sweep_run{idx:02d}_{timestamp}"
        save_path = os.path.join(SWEEP_MODELS_DIR["A2C"], model_name)
        model.save(save_path)

        row = {
            "algo": "A2C",
            "run_id": idx,
            "model_path": save_path,
            **cfg,
            **stats,
        }
        results.append(row)

        print(
            f"    -> avg_reward={stats['avg_reward']:.2f}, "
            f"std={stats['std_reward']:.2f}, len={stats['avg_length']:.2f}"
        )

    csv_path = os.path.join(SWEEP_RESULTS_DIR, "a2c_sweep.csv")
    _write_csv(csv_path, results)
    print(f" A2C sweep results saved to {csv_path}")
    return results


# ------------------------------------------------------------------
# PPO SWEEP
# ------------------------------------------------------------------
def ppo_param_grid() -> List[Dict[str, Any]]:
    grid = []
    learning_rates = [3e-4, 1e-4, 5e-5]
    gammas = [0.98, 0.99]
    n_steps_list = [512, 1024, 2048]
    clip_ranges = [0.1, 0.2, 0.3]
    ent_coefs = [0.0, 0.01]

    for lr in learning_rates:
        for g in gammas:
            for ns in n_steps_list:
                for cr in clip_ranges:
                    for ent in ent_coefs:
                        grid.append(
                            {
                                "learning_rate": lr,
                                "gamma": g,
                                "n_steps": ns,
                                "clip_range": cr,
                                "ent_coef": ent,
                            }
                        )
    random.shuffle(grid)
    return grid[:10]


def run_ppo_sweep(
    total_timesteps: int = 120_000,
    n_eval_episodes: int = 15,
) -> List[Dict[str, Any]]:
    results = []
    configs = ppo_param_grid()

    print(f"\n PPO SWEEP: {len(configs)} runs...\n")

    for idx, cfg in enumerate(configs, start=1):
        print(f"[PPO Run {idx}/{len(configs)}] hyperparams: {cfg}")

        def make_env_fn():
            return PeerTechEnv(render_mode=None)

        vec_env = make_vec_env(make_env_fn, n_envs=1)

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=cfg["learning_rate"],
            gamma=cfg["gamma"],
            n_steps=cfg["n_steps"],
            clip_range=cfg["clip_range"],
            ent_coef=cfg["ent_coef"],
            verbose=0,
            tensorboard_log=None,
        )

        model.learn(total_timesteps=total_timesteps)

        stats = evaluate_sb3_model(model, n_eval_episodes=n_eval_episodes)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"ppo_sweep_run{idx:02d}_{timestamp}"
        save_path = os.path.join(SWEEP_MODELS_DIR["PPO"], model_name)
        model.save(save_path)

        row = {
            "algo": "PPO",
            "run_id": idx,
            "model_path": save_path,
            **cfg,
            **stats,
        }
        results.append(row)

        print(
            f"    -> avg_reward={stats['avg_reward']:.2f}, "
            f"std={stats['std_reward']:.2f}, len={stats['avg_length']:.2f}"
        )

    csv_path = os.path.join(SWEEP_RESULTS_DIR, "ppo_sweep.csv")
    _write_csv(csv_path, results)
    print(f" PPO sweep results saved to {csv_path}")
    return results


# ------------------------------------------------------------------
# REINFORCE SWEEP
# ------------------------------------------------------------------
def reinforce_param_grid() -> List[Dict[str, Any]]:
    learning_rates = [1e-3, 5e-4, 3e-4, 1e-4]
    gammas = [0.98, 0.99]
    hidden_dims = [64, 128]
    runs = []

    for lr in learning_rates:
        for g in gammas:
            for hd in hidden_dims:
                runs.append(
                    {
                        "learning_rate": lr,
                        "gamma": g,
                        "hidden_dim": hd,
                    }
                )
    random.shuffle(runs)
    return runs[:10]


def run_reinforce_sweep(
    n_episodes: int = 800,
    n_eval_episodes: int = 15,
) -> List[Dict[str, Any]]:
    results = []
    configs = reinforce_param_grid()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n REINFORCE SWEEP: {len(configs)} runs... (device={device})\n")

    for idx, cfg in enumerate(configs, start=1):
        print(f"[REINFORCE Run {idx}/{len(configs)}] hyperparams: {cfg}")

        env = PeerTechEnv(render_mode=None)
        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        policy = PolicyNetwork(obs_dim, cfg["hidden_dim"], n_actions).to(device)
        optimizer = optim.Adam(policy.parameters(), lr=cfg["learning_rate"])

        all_rewards: List[float] = []

        for ep in range(1, n_episodes + 1):
            optimizer.zero_grad()
            loss, ep_reward = run_reinforce_episode(
                env, policy, cfg["gamma"], device=device
            )
            loss.backward()
            optimizer.step()
            all_rewards.append(ep_reward)

        stats = evaluate_reinforce_policy(
            policy, n_eval_episodes=n_eval_episodes, device=device
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"reinforce_sweep_run{idx:02d}_{timestamp}.pt"
        save_path = os.path.join(SWEEP_MODELS_DIR["REINFORCE"], model_name)
        torch.save(policy.state_dict(), save_path)

        row = {
            "algo": "REINFORCE",
            "run_id": idx,
            "model_path": save_path,
            **cfg,
            **stats,
        }
        results.append(row)

        print(
            f"    -> avg_reward={stats['avg_reward']:.2f}, "
            f"std={stats['std_reward']:.2f}, len={stats['avg_length']:.2f}"
        )

    csv_path = os.path.join(SWEEP_RESULTS_DIR, "reinforce_sweep.csv")
    _write_csv(csv_path, results)
    print(f"💾 REINFORCE sweep results saved to {csv_path}")
    return results


# ------------------------------------------------------------------
# CSV UTIL
# ------------------------------------------------------------------
def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    import csv

    #  FIX: get ALL keys across ALL rows
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    keys = sorted(all_keys)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)



def aggregate_summary(*all_results: List[Dict[str, Any]]) -> None:
    merged: List[Dict[str, Any]] = []
    for res in all_results:
        merged.extend(res)
    if not merged:
        return

    # Save combined CSV
    summary_path = os.path.join(SWEEP_RESULTS_DIR, "sweep_summary.csv")
    _write_csv(summary_path, merged)
    print(f" Global sweep summary saved to {summary_path}")

    # Also save JSON for quick inspection
    json_path = os.path.join(SWEEP_RESULTS_DIR, "sweep_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f" Global sweep summary JSON saved to {json_path}")


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def run_all_sweeps():
    print("\n================== HYPERPARAMETER SWEEP ENGINE (PRO) ==================\n")

    dqn_results = run_dqn_sweep()
    a2c_results = run_a2c_sweep()
    ppo_results = run_ppo_sweep()
    reinforce_results = run_reinforce_sweep()

    aggregate_summary(dqn_results, a2c_results, ppo_results, reinforce_results)

    print("\n=========================== SWEEP COMPLETE =============================\n")


if __name__ == "__main__":
    run_all_sweeps()
