import gymnasium as gym
import torch
import pickle
import time

# create the gym environment
env = gym.make("CartPole-v1", render_mode="human")

class DQN(torch.nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, 128)
        self.layer2 = torch.nn.Linear(128, 128)
        self.layer3 = torch.nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer1(x))
        x = torch.nn.functional.relu(self.layer2(x))
        return self.layer3(x)

state, info = env.reset()
policy_net = DQN(len(state), env.action_space.n)
with open("policy_net_state_dict.pkl", "rb") as inp:
    policy_net.load_state_dict(pickle.load(inp))

state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
duration = 0
while state is not None:
    action = policy_net(state).max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    state = None if terminated else torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
    env.render()
    duration += 1
    time.sleep(0.01)
print(duration)