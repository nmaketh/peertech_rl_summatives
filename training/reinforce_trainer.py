# training/reinforce_trainer.py
"""
REINFORCE Training Script for PeerTech Environment (PRO Version)

- Custom policy gradient (REINFORCE)
- Hyperparameters from best run in sweep
- Saves final policy to: models/pg/peertech_reinforce_pro_<timestamp>.pt
"""

import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from environment.custom_env import PeerTechEnv


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


def train_reinforce_pro(
    n_episodes: int = 1500,
    learning_rate: float = 1e-3,   #  from best sweep (Run 6)
    gamma: float = 0.99,           # 
    hidden_dim: int = 64,          # 
    run_name: str = "reinforce_pro",
) -> str:
    """
    Train REINFORCE policy gradient agent.

    Returns
    -------
    save_path : str
        Path to saved REINFORCE policy weights.
    """

    model_dir = os.path.join("models", "pg")
    os.makedirs(model_dir, exist_ok=True)

    env = PeerTechEnv(render_mode=None)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy = PolicyNetwork(obs_dim, hidden_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

    all_rewards: List[float] = []

    print("\n Starting REINFORCE PRO training...")
    for episode in range(1, n_episodes + 1):
        optimizer.zero_grad()
        loss, ep_reward = run_reinforce_episode(env, policy, gamma, device=device)
        loss.backward()
        optimizer.step()

        all_rewards.append(ep_reward)

        if episode % 50 == 0:
            avg_last = float(np.mean(all_rewards[-50:]))
            print(
                f"[REINFORCE] Ep {episode}/{n_episodes} | "
                f"Reward: {ep_reward:.2f} | Avg last 50: {avg_last:.2f}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(model_dir, f"peertech_reinforce_pro_{run_name}_{timestamp}.pt")
    torch.save(policy.state_dict(), save_path)
    print(f" Saved REINFORCE policy to {save_path}")

    return save_path


if __name__ == "__main__":
    train_reinforce_pro()
