# main.py – Cleaned Version (Only A2C / PPO / DQN)

import os
import numpy as np
import pygame
from stable_baselines3 import A2C, PPO, DQN

from environment.custom_env import PeerTechEnv

# --------------------------------------------------------
# MODEL PATHS
# --------------------------------------------------------
BEST_A2C_PATH = "models/pg/a2c_pro_best/best_model.zip"
BEST_PPO_PATH = "models/pg/ppo_pro_best/best_model.zip"
BEST_DQN_PATH = "models/dqn_pro_best/best_model.zip"

START_MODE = "HUMAN"
START_AGENT = "A2C"

# --------------------------------------------------------
# SAFE LOADERS
# --------------------------------------------------------
def safe_load_sb3(cls, path: str):
    if not os.path.exists(path):
        print(f"⚠️ {cls.__name__} NOT found: {path}")
        return None
    try:
        print(f"✅ Loaded {cls.__name__}")
        return cls.load(path)
    except Exception as e:
        print(f"❌ Failed to load {cls.__name__}: {e}")
        return None


def load_all_agents(env):
    return {
        "A2C": safe_load_sb3(A2C, BEST_A2C_PATH),
        "PPO": safe_load_sb3(PPO, BEST_PPO_PATH),
        "DQN": safe_load_sb3(DQN, BEST_DQN_PATH)
    }

# --------------------------------------------------------
# ACTION SELECTION
# --------------------------------------------------------
def select_action_ai(agent_name, agents, env, obs):
    if agents[agent_name] is not None:
        action, _ = agents[agent_name].predict(obs, deterministic=True)
        return int(action)

    return int(env.action_space.sample())


def poll_human_action(events):
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:  return 0
            if event.key == pygame.K_DOWN:  return 1
            if event.key == pygame.K_RIGHT: return 2
            if event.key == pygame.K_SPACE: return 3
            if event.key == pygame.K_a:     return 4
            if event.key == pygame.K_s:     return 5
    return None

# --------------------------------------------------------
# GAME LOOP
# --------------------------------------------------------
def run_game():
    pygame.init()
    env = PeerTechEnv(render_mode="human")

    agents = load_all_agents(env)
    control_mode = START_MODE
    agent_name = START_AGENT

    obs, info = env.reset()
    episode = 1
    ep_reward = 0

    print("\n🎮 Game Running… Human + AI Mode Enabled\n")

    while env.renderer and env.renderer.running:
        events = pygame.event.get()

        for e in events:
            if e.type == pygame.QUIT:
                env.close()
                return

        # Switching
        keys = pygame.key.get_pressed()
        if keys[pygame.K_h]: control_mode = "HUMAN"
        if keys[pygame.K_j]: control_mode = "AI"

        if keys[pygame.K_1]: agent_name = "A2C"
        if keys[pygame.K_2]: agent_name = "PPO"
        if keys[pygame.K_3]: agent_name = "DQN"

        # Decide action
        if control_mode == "AI":
            action = select_action_ai(agent_name, agents, env, obs)
        else:
            action = poll_human_action(events)

        if control_mode == "HUMAN" and action is None:
            env.render()
            pygame.time.delay(40)
            continue

        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        if terminated or truncated:
            print(f"Episode {episode} | Reward={ep_reward:.2f} | Mode={control_mode}")
            ep_reward = 0
            episode += 1
            obs, info = env.reset()

        pygame.time.delay(40)

    env.close()


if __name__ == "__main__":
    run_game()
