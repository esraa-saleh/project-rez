import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import bisect
import random
from collections import namedtuple
import math
from src.environments.hitorstandcontinuous import hitorstandcontinuous

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = hitorstandcontinuous()
m = env.action_space.n

steps_done = 0

#REPLAY (in separate file instead?)
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#NOISEBUFFER && NOISEBUFFER HELPER METHODS (in separate file instead?)
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

    #MODIFICATIONS MADE: converted from global method by requiring policy_net & eps variables as arguments
    def select_action(self, state, policy_net, EPS_START, EPS_END, EPS_DECAY):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest value for column of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(m)]], dtype=torch.long)
