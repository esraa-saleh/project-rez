import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import bisect
import random

from src.utils.ReplayMemory import ReplayMemory
from src.utils.select_action import select_action
from src.utils.optimize_model import optimize_model

'''
The code for this private agent is a modified version of Wang & Hegde [2019]
'''

#NOISEBUFFER
class NoiseBuffer:
    def __init__(self, m, sigma, nb_seeds):
        self.buffer = []
        self.base = {}
        self.m = m
        self.sigma = sigma
        self.nb_seeds = nb_seeds

    def kk(self, x, y):
        return np.exp(-abs(x - y))

    def rho(self, x, y):
        return np.exp(abs(x - y)) - np.exp(-abs(x - y))

    def sample(self, s):

        buffer = self.buffer
        sigma = self.sigma
            
        if len(buffer) == 0:
            v0 = self.nb_seeds.normal(0, sigma)
            v1 = self.nb_seeds.normal(0, sigma)
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
            v0 = self.nb_seeds.normal(mean0, np.sqrt(var0) * sigma)
            v1 = self.nb_seeds.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(0, (s, v0, v1))
        elif s > buffer[-1][0]:
            mean0 = self.kk(s, buffer[-1][0]) * buffer[0][1]
            mean1 = self.kk(s, buffer[-1][0]) * buffer[0][2]
            var0 = 1 - self.kk(s, buffer[-1][0]) ** 2
            var1 = var0
            v0 = self.nb_seeds.normal(mean0, np.sqrt(var0) * sigma)
            v1 = self.nb_seeds.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(len(buffer), (s, v0, v1))
        else:
            idx = bisect.bisect(buffer, (s, None, None))
            sminus, eminus0, eminus1 = buffer[idx-1]
            splus, eplus0, eplus1 = buffer[idx]
            mean0 = (self.rho(splus, s)*eminus0 + self.rho(sminus, s)*eplus0) / self.rho(sminus, splus)
            mean1 = (self.rho(splus, s)*eminus1 + self.rho(sminus, s)*eplus1) / self.rho(sminus, splus)
            var0 = 1 - (self.kk(sminus, s)*self.rho(splus, s) + self.kk(splus, s)*self.rho(sminus, s)) / self.rho(sminus, splus)
            var1 = var0
            v0 = self.nb_seeds.normal(mean0, np.sqrt(var0) * sigma)
            v1 = self.nb_seeds.normal(mean1, np.sqrt(var1) * sigma)
            self.buffer.insert(idx, (s, v0, v1))
        return (v0, v1)

    def reset(self):
        self.buffer = []

#NETWORK
class PrivateDQN(nn.Module):
    def __init__(self, m, nb_rng, torch_rng, hidden=16, sigma=0.4):
        super(PrivateDQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)
        self.sigma = sigma
        self.nb = NoiseBuffer(m, sigma, nb_rng)
        #sets global PyTorch seeds (np & random seeds required for NN backend)
        torch_seed = torch_rng.randint(1000)
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(torch_seed)
        random.seed(torch_seed)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = self.head(x)
        if self.sigma > 0:
            #adds Gaussian process noise to output
            eps = [self.nb.sample(float(state)) for state in s]
            eps = torch.Tensor(eps)
            return x + eps
        else:
            return x

#AGENT
class PrivateDQNAgent():
    def __init__(self, seed_bundle, m, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, TARGET_UPDATE=10, BATCH_SIZE=128, GAMMA=0.99):
        #sets independent generators for selecting actions, the noisebuffer, the replay memory, and the agent itself
        #by keeping the generators separate with distinct seeds, results are more reproducible.
        self.action_rng = np.random.RandomState(seed_bundle.action_seed)
        self.nb_rng = np.random.RandomState(seed_bundle.noisebuffer_seed)
        self.replay_rng = np.random.RandomState(seed_bundle.replay_seed)
        self.torch_rng = np.random.RandomState(seed_bundle.parameter_init_seed)

        self.memory = ReplayMemory(10000, self.replay_rng)
        self.state = None
        self.action = None
        self.episode = 0
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TARGET_UPDATE = TARGET_UPDATE
        self.STEPS_DONE = 0
        self.m = m
        self.policy_net = PrivateDQN(self.m, self.nb_rng, self.torch_rng)
        self.target_net = PrivateDQN(self.m, self.nb_rng, self.torch_rng)
        #set target network to eval mode
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    #RL Glue methods:
    def agent_start(self, state):
        self.state = torch.Tensor(state).unsqueeze(0)
        action, step_count = select_action(self.state, self.policy_net, self.m, self.action_rng, self.EPS_START, self.EPS_END, self.EPS_DECAY, self.STEPS_DONE)
        self.action = action
        self.STEPS_DONE = step_count
        return action

    def agent_step(self, reward, next_state):
        #manually putting in device='cpu' here to avoid having to pass it in
        reward_tensor = torch.tensor([reward], device='cpu')
        reform_next_state = torch.Tensor(next_state).unsqueeze(0)
        #store transition in memory
        self.memory.push(self.state, self.action, reform_next_state, reward_tensor)

        #move to next state
        self.state = reform_next_state

        #optimize model
        optimized_policy_net = optimize_model(self.memory, self.optimizer, self.policy_net, self.target_net, self.GAMMA, self.BATCH_SIZE)

        #copy optimized policy net to self.policy_net
        self.policy_net = optimized_policy_net
        #select next action here
        next_action, step_count = select_action(reform_next_state, self.policy_net, self.m, self.action_rng, self.EPS_START, self.EPS_END, self.EPS_DECAY, self.STEPS_DONE)
        self.action = next_action
        self.STEPS_DONE = step_count

        return next_action

    def agent_end(self, reward):
        if self.episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        reward_tensor = torch.tensor([reward], device='cpu')
        self.memory.push(self.state, self.action, None, reward_tensor)

        self.state = None

        optimized_policy_net = optimize_model(self.memory, self.optimizer, self.policy_net, self.target_net, self.GAMMA, self.BATCH_SIZE)
        # copy optimized policy net to self.policy_net
        self.policy_net = optimized_policy_net
        self.episode += 1

    def check_nan_weights(self):

        policy_net_layer1_nan = torch.isnan(self.policy_net.linear1.weight).any()
        policy_net_layer2_nan = torch.isnan(self.policy_net.linear2.weight).any()
        target_net_layer1_nan = torch.isnan(self.target_net.linear1.weight).any()
        target_net_layer2_nan = torch.isnan(self.target_net.linear2.weight).any()

        return (policy_net_layer1_nan or policy_net_layer2_nan or target_net_layer1_nan or target_net_layer2_nan)




