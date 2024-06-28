import numpy as np
import gymnasium as gym
import recycler

# create the gym environment
env = gym.make("recycler:Recycler-v0", render_mode=None)

trials = 10000
rewards = np.zeros(trials)
for trial in range(trials):
    # print("Doing trial %d..." % (trial))
    state, info = env.reset()
    rewards[trial] = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = 2 if state == 0 else 0 # charge when low, search when high policy [98.876 102.632]
        # action = np.random.randint(0,3) # random policy [54.912 57.83]
        # action = 1 # always wait [10 10]
        # action = 0 # always search [11.112 11.254]
        state, reward, terminated, truncated, _ = env.step(action)
        rewards[trial] += reward
me = np.mean(rewards)
ci = 1.96*np.std(rewards)/np.sqrt(trials)
print(np.round(me-ci,3), np.round(me+ci,3))