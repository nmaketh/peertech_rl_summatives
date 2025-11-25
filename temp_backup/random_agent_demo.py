print("demo started")

from environment.custom_env import PeerTechEnv
import time

env = PeerTechEnv(render_mode="human")
print("env created")

obs, info = env.reset()
print("env reset")

for i in range(50):
    print("step", i)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.05)
    if terminated or truncated:
        print("terminated or truncated, resetting")
        obs, info = env.reset()

env.close()
print("done")
