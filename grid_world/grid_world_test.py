import gymnasium as gym
import grid_world

env = gym.make("grid_world:GridWorld-v0", render_mode="human")
env.reset()
env.render()