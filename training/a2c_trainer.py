# training/a2c_trainer.py
"""
A2C Training Script for PeerTech Environment (PRO Version)

- Uses Stable-Baselines3 A2C
- Hyperparameters set from best hyperparameter sweep run (Run 7)
- Saves best model to: models/pg/a2c_pro_best/best_model.zip
"""

import os
from datetime import datetime
from typing import Optional

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from environment.custom_env import PeerTechEnv


def make_peertech_env(render_mode: Optional[str] = None):
    def _init():
        return PeerTechEnv(render_mode=render_mode)
    return _init


def train_a2c_pro(
    total_timesteps: int = 200_000,
    learning_rate: float = 7e-4,   #  best: 0.0007
    gamma: float = 0.99,           # 
    n_steps: int = 10,             # 
    ent_coef: float = 0.02,        # 
    run_name: str = "a2c_pro_best",
) -> str:
    """
    Train an A2C agent and save best model.

    Returns
    -------
    best_model_path : str
        Path to best saved A2C model.
    """

    logs_root = os.path.join("logs", "a2c", run_name)
    os.makedirs(logs_root, exist_ok=True)

    model_best_dir = os.path.join("models", "pg", "a2c_pro_best")
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

    model = A2C(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        gamma=gamma,
        n_steps=n_steps,
        ent_coef=ent_coef,
        verbose=1,
        tensorboard_log=logs_root,
    )

    print("\n Starting A2C PRO training...")
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    print(" A2C training finished.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join("models", "pg", f"peertech_a2c_pro_{run_name}_{timestamp}.zip")
    model.save(final_model_path)
    print(f" Saved final A2C model to {final_model_path}")

    best_model_path = os.path.join(model_best_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f" Best A2C model saved to {best_model_path}")
    else:
        print(" best_model.zip not found – maybe eval_freq too high / too few timesteps?")

    return best_model_path


if __name__ == "__main__":
    train_a2c_pro()
