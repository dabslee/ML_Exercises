import numpy as np
import gymnasium as gym

class RecyclerEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(self, render_mode=None):
        # Two possible states: low (0) and high (1)
        self.observation_space = gym.spaces.Discrete(2)
        # Three possible actions: search (0), wait (1), and charge (2)
        self.action_space = gym.spaces.Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_state = 0
        if self.render_mode == "human":
            self.render()
        return self._agent_state, {}
    
    def step(self, action):
        p_h = 0.9 # probability of remaining high battery upon searching
        p_l = 0.1 # probability of remaining low battery upon searching
        # (x, x_next, action) -> probability
        prob_matrix = np.array([
                [
                    [p_l,    1.0,    0.0], # low to low
                    [1-p_l,  0.0,    1.0], # low to high
                ], [
                    [1-p_h,  0.0,    0.0], # high to low
                    [p_h,    1.0,    1.0], # high to high
                ],
            ])
        next_state = int(np.random.choice(
            [0,1], p=prob_matrix[self._agent_state,:,action]
        ))
        terminated = (action == 0) and (next_state != self._agent_state)
        self._agent_state = next_state
        reward = 0
        if action == 0:
            reward = 10
        elif action == 1:
            reward = 0.1
        if self.render_mode == "human":
            print("Action taken:", action)
            self.render()

        return self._agent_state, reward, terminated, False, {}
    
    def render(self):
        return print("Agent state:", self._agent_state)

gym.envs.registration.register(
     id="Recycler-v0",
     entry_point="recycler:RecyclerEnv",
     max_episode_steps=100,
)