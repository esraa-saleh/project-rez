import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import bisect
import random
import math
from src.environments.hitorstandcontinuous import hitorstandcontinuous
from src.utils.ReplayMemory import ReplayMemory
from src.utils.ReplayMemory import Transition

#TODO: have seeding for neural net init to make results reproducable

#NOISEBUFFER
class NoiseBuffer:
    def __init__(self, m, sigma):
        self.buffer = []
        self.base = {}
        self.m = m
        self.sigma = sigma

    def kk(self, x, y):
        return np.exp(-abs(x - y))

    def rho(self, x, y):
        return np.exp(abs(x - y)) - np.exp(-abs(x - y))

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
            mean0 = self.kk(s, buffer[0][0]) * buffer[0][1]
            mean1 = self.kk(s, buffer[0][0]) * buffer[0][2]
            var0 = 1 - self.kk(s, buffer[0][0]) ** 2
            var1 = 1 - self.kk(s, buffer[0][0]) ** 2
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(0, (s, v0, v1))
        elif s > buffer[-1][0]:
            mean0 = self.kk(s, buffer[-1][0]) * buffer[0][1]
            mean1 = self.kk(s, buffer[-1][0]) * buffer[0][2]
            var0 = 1 - self.kk(s, buffer[-1][0]) ** 2
            var1 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(len(buffer), (s, v0, v1))
        else:
            idx = bisect.bisect(buffer, (s, None, None))
            sminus, eminus0, eminus1 = buffer[idx-1]
            splus, eplus0, eplus1 = buffer[idx]
            mean0 = (self.rho(splus, s)*eminus0 + self.rho(sminus, s)*eplus0) / self.rho(sminus, splus)
            mean1 = (self.rho(splus, s)*eminus1 + self.rho(sminus, s)*eplus1) / self.rho(sminus, splus)
            var0 = 1 - (self.kk(sminus, s)*self.rho(splus, s) + self.kk(splus, s)*self.rho(sminus, s)) / self.rho(sminus, splus)
            var1 = var0
            v0 = np.random.normal(mean0, np.sqrt(var0) * sigma)
            v1 = np.random.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(idx, (s, v0, v1))
        return (v0, v1)

    def reset(self):
        self.buffer = []

#NETWORK
class PrivateDQN(nn.Module):
    def __init__(self, m, hidden=16, sigma=0.4):
        super(PrivateDQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)
        self.sigma = sigma
        self.nb = NoiseBuffer(m, sigma)

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

#AGENT
class PrivateDQNAgent():
    def __init__(self, seed_bundle, m, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, TARGET_UPDATE=10, BATCH_SIZE=128, GAMMA=0.99):
        self.memory = ReplayMemory(10000)
        # self.total_reward = None
        self.state = None
        self.action = None
        self.episode = 1
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.STEPS_DONE = 0
        self.m = m
        self.policy_net = PrivateDQN(self.m)
        self.target_net = PrivateDQN(self.m)
        #self.target_net.load_state_dict(self.policy_net.parameters())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.action_rng = np.random.RandomState(seed_bundle.action_seed)


    #getters for ReplayMemory and total reward
    def agent_get_memory(self):
        return self.memory

    # def agent_get_total_reward(self):
    #     return self.total_reward

    def select_action(self, state):
        # sample = random.random()
        # instead of python's random we'll use numpy's to be able to have a generator with its own seed
        sample = self.action_rng.uniform()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
                        math.exp(-1. * self.STEPS_DONE / self.EPS_DECAY)
        self.STEPS_DONE += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest value for column of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.action_rng.randint(self.m)]], dtype=torch.long)

    def optimize_model(self, device="cpu"):

        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    #RL Glue methods:
    def agent_start(self, state):
        # self.total_reward = 0
        self.state = torch.Tensor(state).unsqueeze(0)
        action = self.select_action(state)
        self.action = action
        return action

    def agent_step(self, reward, next_state):

        #manually putting in device='cpu' here to avoid having to pass it in
        reward = torch.tensor([reward], device='cpu')
        reform_next_state = torch.Tensor(next_state).unsqueeze(0)
        #store transition in memory
        memory = self.agent_get_memory()
        memory.push(self.state, self.action, reform_next_state, reward)

        # recieve reward
        # self.total_reward += float(reward.squeeze(0).data)

        #move to next state
        self.state = reform_next_state

        #optimize model
        self.optimize_model()

        #select next action here
        next_action = self.select_action(reform_next_state)
        self.action = next_action

        return next_action

    def agent_end(self, reward):
        if self.episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.memory.push(self.state, self.action, None, reward)
        # reward = torch.tensor([reward], device='cpu')
        # self.total_reward += float(reward.squeeze(0).data)
        self.state = None

        self.optimize_model()
        self.episode += 1



