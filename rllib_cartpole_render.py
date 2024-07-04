import gymnasium as gym
import time
from ray.rllib.algorithms.algorithm import Algorithm

algo = Algorithm.from_checkpoint("C:/Users/brand/AppData/Local/Temp/tmp0yv4lu5g")

env = gym.make("CartPole-v1", render_mode="human")
state, info = env.reset()
terminated = truncated = False
duration = 0

while not terminated:
    action = algo.compute_single_action(state)
    state, reward, terminated, truncated, _ = env.step(action.item())
    env.render()
    duration += 1
    time.sleep(0.01)
print(duration)