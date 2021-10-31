import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.optimize_model import optimize_model
from src.utils.ReplayMemory import ReplayMemory
from src.utils.ReplayMemory import Transition
from src.utils.select_action import select_action
from src.environments.hitorstandcontinuous import hitorstandcontinuous

#PLACEHOLDER ENVIRONMENT
env = hitorstandcontinuous()
m = env.action_space.n

class DQN(nn.Module):
    def __init__(self, sigma=0.4, hidden=16):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)
        self.sigma = sigma
        self.memory = None
        self.total_reward = 0

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.head(x)
        if self.sigma > 0:
            eps = [self.nb.sample(float(state)) for state in s]
            eps = torch.Tensor(eps)
            return x + eps
        else:
            return x

    # init ReplayMemory with specified capacity
    def agent_set_memory(self, capacity):
        self.memory = ReplayMemory(capacity)

    # getters for ReplayMemory and total reward
    def agent_get_memory(self):
        return self.memory

    def agent_get_total_reward(self):
        return self.total_reward

    # RL Glue methods:
    def agent_start(self, state, net):
        action = select_action(state, net, m)
        return action

    # added replaymemory as a parameter here since Nidhi's update loop expects the memory as a global variable
    def agent_step(self, state, reward, net, target_net):

        # choose action from current state
        action = select_action(state, net, m)
        # extract next state info
        next_state, next_reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state).unsqueeze(0)

        # manually putting in device='cpu' here to avoid having to pass it in
        next_reward = torch.tensor([reward], device='cpu')

        # store transition in memory
        memory = self.agent_get_memory()
        memory.push(state, action, next_state, next_reward)

        # recieve reward
        self.total_reward += float(reward.squeeze(0).data)

        # move to next state
        state = next_state

        # optimize model
        optimize_model(memory, net, target_net, Transition, BATCH_SIZE=128, GAMMA=0.99, device='cpu')

        return action

    def agent_end(self, reward):
        pass