import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect
from src.environments.hitorstandcontinuous import hitorstandcontinuous
from src.utils.optimize_model import optimize_model
from src.utils.ReplayMemory import ReplayMemory
from src.utils.ReplayMemory import Transition
from src.utils.select_action import select_action

#PLACEHOLDER ENV (NIDHI'S)
env = hitorstandcontinuous()
m = env.action_space.n


#NOISEBUFFER && NOISEBUFFER HELPER METHODS
def kk(x, y):
    return np.exp(-abs(x-y))

def rho(x, y):
    return np.exp(abs(x-y)) - np.exp(-abs(x-y))

class noisebuffer:
    def __init__(self, m, sigma):
        self.buffer = []
        self.base = {}
        self.m = m
        self.sigma = sigma

    def sample(self, s):
        buffer = self.buffer
        sigma = self.sigma
            
        if len(buffer) == 0:
            v0 = np.random.normal(0, sigma)
            v1 = np.random.normal(0, sigma)
            self.buffer.append((s, v0, v1))
            return (v0, v1)
        else:
            idx = bisect.bisect(buffer, (s, 0, 0))
            if len(buffer) == 1:
                if buffer[0][0] == s:
                    return (buffer[0][1], buffer[0][2])
            else:
                if (idx <= len(buffer)-1) and (buffer[idx][0] == s):
                    return (buffer[idx][1], buffer[idx][2])
                elif (idx >= 1) and (buffer[idx-1][0] == s):
                    return (buffer[idx-1][1], buffer[idx-1][2])
                elif (idx <= len(buffer)-2) and (buffer[idx+1][0] == s):
                    return (buffer[idx+1][1], buffer[idx+1][2])
            
        if s < buffer[0][0]:
            mean0 = kk(s, buffer[0][0]) * buffer[0][1]
            mean1 = kk(s, buffer[0][0]) * buffer[0][2]
            var0 = 1 - kk(s, buffer[0][0]) ** 2
            var1 = 1 - kk(s, buffer[0][0]) ** 2
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(0, (s, v0, v1))
        elif s > buffer[-1][0]:
            mean0 = kk(s, buffer[-1][0]) * buffer[0][1]
            mean1 = kk(s, buffer[-1][0]) * buffer[0][2]
            var0 = 1 - kk(s, buffer[-1][0]) ** 2
            var1 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(len(buffer), (s, v0, v1))
        else:
            idx = bisect.bisect(buffer, (s, None, None))
            sminus, eminus0, eminus1 = buffer[idx-1]
            splus, eplus0, eplus1 = buffer[idx]
            mean0 = (rho(splus, s)*eminus0 + rho(sminus, s)*eplus0) / rho(sminus, splus)
            mean1 = (rho(splus, s)*eminus1 + rho(sminus, s)*eplus1) / rho(sminus, splus)
            var0 = 1 - (kk(sminus, s)*rho(splus, s) + kk(splus, s)*rho(sminus, s)) / rho(sminus, splus)
            var1 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(idx, (s, v0, v1))
        return (v0, v1)

    def reset(self):
        self.buffer = []

#AGENT
class DQN(nn.Module):
    def __init__(self, sigma=0.4, hidden=16):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)
        self.sigma = sigma
        self.nb = noisebuffer(2, sigma)
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

    #init ReplayMemory with specified capacity
    def agent_set_memory(self, capacity):
        self.memory = ReplayMemory(capacity)

    #getters for ReplayMemory and total reward
    def agent_get_memory(self):
        return self.memory

    def agent_get_total_reward(self):
        return self.total_reward

    #RL Glue methods:
    def agent_start(self, state, net):
        action = select_action(state, net, m)
        return action

    #added replaymemory as a parameter here since Nidhi's update loop expects the memory as a global variable
    def agent_step(self, state, reward, net, target_net):

        #choose action from current state
        action = select_action(state, net, m)
        #extract next state info
        next_state, next_reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state).unsqueeze(0)

        #manually putting in device='cpu' here to avoid having to pass it in
        next_reward = torch.tensor([reward], device='cpu')

        #store transition in memory
        memory = self.agent_get_memory()
        memory.push(state, action, next_state, next_reward)

        # recieve reward
        self.total_reward += float(reward.squeeze(0).data)

        #move to next state
        state = next_state

        #optimize model
        optimize_model(memory, net, target_net, Transition, BATCH_SIZE=128, GAMMA=0.99, device='cpu')

        return action

    def agent_end(self, reward):
        pass

