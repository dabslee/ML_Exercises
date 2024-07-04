import numpy as np
import gymnasium as gym

'''
We create an environment that behaves according to the
game described in David Silver's Easy21 assignment:
https://www.davidsilver.uk/wp-content/uploads/2020/03/Easy21-Johannes.pdf
'''

class Easy21Env(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, config=None):
        # observation space: [player_total, dealer_total]
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.int32)
        # action space: stick (0) or hit (1)
        self.action_space = gym.spaces.Discrete(2)
        # render settings
        render_mode = config.get("render_mode")
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return np.array([self.player_total, self.dealer_total])
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_total = np.random.randint(1,10+1)
        self.dealer_total = np.random.randint(1,10+1)
        
        self.render(f"Player: {self.player_total}, Dealer: {self.dealer_total}")
        return self._get_obs(), {}
    
    def step(self, action):
        reward = 0
        terminated = False

        # player hits
        if action == 1:
            card_sign = -1 if np.random.random() < 1/3 else 1
            self.player_total += card_sign * np.random.randint(1,10+1)
            self.render(f"Player hits; new value {self.player_total}.")
            # check for player bust
            if self.player_total > 21 or self.player_total < 1:
                self.render("Player busts.")
                reward = -1
                terminated = True
        
        # dealer takes turns only if player sticks
        if action == 0:
            self.render("Player sticks.")
            while not terminated:
                # dealer hits if their total is <17
                if self.dealer_total < 17:
                    card_sign = -1 if np.random.random() < 1/3 else 1
                    self.dealer_total += card_sign * np.random.randint(1,10+1)
                    self.render(f"Dealer hits; new value {self.dealer_total}")
                    # check for dealer bust
                    if self.dealer_total > 21 or self.dealer_total < 1:
                        self.render("Dealer busts.")
                        reward = 1
                        terminated = True
                # dealer sticks, game ends
                else:
                    self.render(f"Dealer sticks.")
                    terminated = True
        
            # if no one went bust, then update the reward based on highest sum
            if reward == 0:
                reward = np.sign(self.player_total - self.dealer_total)
                if reward == 0:
                    self.render("End result: Tie.")
                else:
                    self.render(("Player" if reward == 1 else "Dealer") + " wins.")
        
        # render status and return outcome
        self.render(f"Player: {self.player_total}, Dealer: {self.dealer_total}")
        return self._get_obs(), reward, terminated, False, {}
    
    def render(self, msg):
        if self.render_mode == "human":
            print(msg)
    
gym.envs.registration.register(
     id="Easy21-v0",
     entry_point="easy21_env:Easy21Env",
)