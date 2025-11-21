# training/ppo_trainer.py
"""
PPO Training Script for PeerTech Environment (PRO Version)

- Uses Stable-Baselines3 PPO
- Hyperparameters set from best sweep run (Run 5)
- Saves best model to: models/pg/ppo_pro_best/best_model.zip
"""

import os
from datetime import datetime
from typing import Optional

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import PeerTechEnv


def make_peertech_env(render_mode: Optional[str] = None):
    def _init():
        return PeerTechEnv(render_mode=render_mode)
    return _init


def train_ppo_pro(
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,   # 0.0003
    gamma: float = 0.99,           # 
    n_steps: int = 1024,           # 
    clip_range: float = 0.3,       # 
    ent_coef: float = 0.01,        # 
    run_name: str = "ppo_pro_best",
) -> str:
    """
    Train a PPO agent and save best model.

    Returns
    -------
    best_model_path : str
        Path to best saved PPO model.
    """

    logs_root = os.path.join("logs", "ppo", run_name)
    os.makedirs(logs_root, exist_ok=True)

    model_best_dir = os.path.join("models", "pg", "ppo_pro_best")
    os.makedirs(model_best_dir, exist_ok=True)

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

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=logs_root,
    )

    print("\n Starting PPO PRO training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(" PPO training finished.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join("models", "pg", f"peertech_ppo_pro_{run_name}_{timestamp}.zip")
    model.save(final_model_path)
    print(f" Saved final PPO model to {final_model_path}")

    best_model_path = os.path.join(model_best_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f" Best PPO model saved to {best_model_path}")
    else:
        print(" best_model.zip not found – maybe eval_freq too high / too few timesteps?")

    return best_model_path


if __name__ == "__main__":
    train_ppo_pro()
