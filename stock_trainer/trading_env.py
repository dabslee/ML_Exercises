import numpy as np
import gymnasium as gym
import pandas as pd

'''
We create a TradingEnv where the agent must decide at the
close of each day whether to ACT: At the beginning of the
run, choosing to ACT will make the agent take a long
position, while if it has already taken a long position,
choosing to ACT will make the agent close the position.
All long positions are automatically closed at the training
date end.

We seek to use this to create a simple, time-independent
trading strategy. The state observed by the agent at any
given time step is the open, high, low, and close prices
of a number of previous days set by the user.
'''

class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}
    
    def __init__(self, render_mode=None, memory_length=10):
        self.memory_length = memory_length
        
        # observation space has first element (number of shares in long position) and 4*memory_length elements
        # describing the open, high, low, and close prices of the last memory_length days
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4*memory_length+1,))
        self.action_space = gym.spaces.Discrete(2) # 0 = don't act, 1 = act

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        prices_memory = np.zeros(4*self.memory_length)
        for days_ago in range(self.memory_length):
            day_row = self.training_data.iloc[self.day_number - days_ago]
            prices_memory[4*days_ago] = day_row["open"]
            prices_memory[4*days_ago+1] = day_row["high"]
            prices_memory[4*days_ago+2] = day_row["low"]
            prices_memory[4*days_ago+3] = day_row["close"]
        return np.insert(prices_memory, 0, self.long_position)
    
    def reset(self, seed=None, options={"starting_equity": 100}):
        super().reset(seed=seed)

        self.training_data = pd.read_csv("BATS_QQQ_1D.csv")
        samplelength = 90
        startindex = np.random.randint(0,self.training_data.shape[0]-samplelength)
        self.training_data = self.training_data.iloc[startindex:startindex+samplelength]

        self.day_number = self.memory_length
        self.long_position = 0
        self.buy_price = None
        self.cash = options["starting_equity"]
        return self._get_obs(), {}
    
    def step(self, action):
        reward = 0
        if action == True:
            if self.long_position != 0:
                # close position at open of the next day
                next_open_price = self.training_data.iloc[self.day_number+1]["open"]
                self.cash = next_open_price*self.long_position
                reward = self.cash - self.long_position*self.buy_price
                if self.render_mode == "human":
                    print(f"Sold {self.long_position} shares at price {next_open_price} on open of day number {self.day_number+1}.")
                self.long_position = 0
            else:
                # invest all cash in long at open of the next day
                self.buy_price = self.training_data.iloc[self.day_number+1]["open"]
                self.long_position = self.cash / self.buy_price
                self.cash = 0
                if self.render_mode == "human":
                    print(f"Bought {self.long_position} shares at price {self.buy_price} on open of day number {self.day_number+1}.")
        
        terminated = (self.day_number == self.training_data.shape[0]-2)
        if terminated:
            reward += self.cash + self.long_position*self.training_data.iloc[self.day_number]["close"]
        self.day_number += 1
        return self._get_obs(), reward, terminated, False, {}
    
gym.envs.registration.register(
     id="TradingEnv-v0",
     entry_point="trading_env:TradingEnv",
)