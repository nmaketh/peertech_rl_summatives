# training/dqn_trainer.py
"""
DQN Training Script for PeerTech Environment (PRO Version)

- Uses Stable-Baselines3 DQN
- Hyperparameters set from best hyperparameter sweep run
- Saves best model to: models/dqn_pro_best/best_model.zip
- Also saves a timestamped final model under models/pg/
"""

import os
from datetime import datetime
from typing import Optional

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import PeerTechEnv


def make_peertech_env(render_mode: Optional[str] = None):
    """Factory for PeerTechEnv (used by make_vec_env)."""
    def _init():
        return PeerTechEnv(render_mode=render_mode)
    return _init


def train_dqn_pro(
    total_timesteps: int = 200_000,
    learning_rate: float = 1e-3,          #  from best sweep (Run 10)
    gamma: float = 0.99,                  # 
    buffer_size: int = 100_000,           # 
    batch_size: int = 64,                 # 
    exploration_fraction: float = 0.2,    # 
    exploration_final_eps: float = 0.02,  # 
    train_freq: int = 4,
    target_update_interval: int = 10_000,
    run_name: str = "dqn_pro_best",
) -> str:
    """
    Train a DQN agent on the PeerTechEnv and save the best model.

    Returns
    -------
    best_model_path : str
        Path to the best saved model (models/dqn_pro_best/best_model.zip).
    """

    # ------------------- FOLDERS -------------------
    logs_root = os.path.join("logs", "dqn", run_name)
    os.makedirs(logs_root, exist_ok=True)

    model_best_dir = os.path.join("models", "dqn_pro_best")
    os.makedirs(model_best_dir, exist_ok=True)

    # ------------------- ENVIRONMENTS -------------------
    vec_env = make_vec_env(make_peertech_env(render_mode=None), n_envs=1)

    eval_env = PeerTechEnv(render_mode=None)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_best_dir,
        log_path=logs_root,
        eval_freq=5_000,
        deterministic=True,
        render=False,
    )

    # ------------------- MODEL -------------------
    model = DQN(
        "MlpPolicy",              #  correct for vector observations
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        train_freq=train_freq,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        target_update_interval=target_update_interval,
        verbose=1,
        tensorboard_log=logs_root,
    )

    print("\n Starting DQN PRO training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(" DQN training finished.")

    # Save timestamped full model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join("models", "pg", f"peertech_dqn_pro_{run_name}_{timestamp}.zip")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    print(f" Saved final DQN model to {final_model_path}")

    best_model_path = os.path.join(model_best_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f" Best DQN model saved to {best_model_path}")
    else:
        print(" best_model.zip not found – maybe eval_freq too high / too few timesteps?")

    return best_model_path


if __name__ == "__main__":
    train_dqn_pro()
