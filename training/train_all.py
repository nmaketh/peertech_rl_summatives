# training/train_all.py
"""
Master Training Script

Trains:
- DQN (PRO, best hyperparams)
- A2C (PRO, best hyperparams)
- PPO (PRO, best hyperparams)
- REINFORCE (PRO, best hyperparams)

Run:
    python -m training.train_all
"""

from training.dqn_trainer import train_dqn_pro
from training.a2c_trainer import train_a2c_pro
from training.ppo_trainer import train_ppo_pro
from training.reinforce_trainer import train_reinforce_pro


def main():
    print("\n==================== TRAINING ALL MODELS (PRO) ====================\n")

    print(" Training DQN (PRO)...")
    dqn_path = train_dqn_pro()
    print(f" DQN done → {dqn_path}\n")

    print(" Training A2C (PRO)...")
    a2c_path = train_a2c_pro()
    print(f" A2C done → {a2c_path}\n")

    print(" Training PPO (PRO)...")
    ppo_path = train_ppo_pro()
    print(f" PPO done → {ppo_path}\n")

    print(" Training REINFORCE (PRO)...")
    reinforce_path = train_reinforce_pro()
    print(f" REINFORCE done → {reinforce_path}\n")

    print("==================== ALL TRAINING COMPLETE ====================")


if __name__ == "__main__":
    main()
