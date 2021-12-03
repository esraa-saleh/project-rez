import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import random
import math
import numpy as np

from src.utils.ReplayMemory import ReplayMemory
from src.utils.ReplayMemory import Transition
from src.utils.select_action import select_action
from src.utils.optimize_model import optimize_model

'''
The code for this non-private agent is a modifed version of Wang & Hegde [2019] and the PyTorch DQN tutorial found at:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''
#NETWORK
class DQN(nn.Module):
    def __init__(self, m, torch_rng, hidden=16):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)

        # sets global PyTorch seeds (np & random seeds required for NN backend)
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
        return x

#AGENT
class DQNAgent:
    def __init__(self, seed_bundle, m, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=200, TARGET_UPDATE=10, BATCH_SIZE=128, GAMMA=0.99):
        # sets independent generators for selecting actions, the replay memory, and the agent itself
        # by keeping the generators separate with distinct seeds, results are more reproducible.
        self.action_rng = np.random.RandomState(seed_bundle.action_seed)
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
        self.policy_net = DQN(self.m, self.torch_rng)
        self.target_net = DQN(self.m, self.torch_rng)
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    # RL Glue methods:
    def agent_start(self, state):
        self.state = torch.Tensor(state).unsqueeze(0)
        action, step_count = select_action(self.state, self.policy_net, self.m, self.action_rng, self.EPS_START,
                                           self.EPS_END, self.EPS_DECAY, self.STEPS_DONE)
        self.action = action
        self.STEPS_DONE = step_count
        return action

    def agent_step(self, reward, next_state):
        # manually putting in device='cpu' here to avoid having to pass it in
        reward_tensor = torch.tensor([reward], device='cpu')
        reform_next_state = torch.Tensor(next_state).unsqueeze(0)
        # store transition in memory
        self.memory.push(self.state, self.action, reform_next_state, reward_tensor)

        # move to next state
        self.state = reform_next_state

        # optimize model
        optimized_policy_net = optimize_model(self.memory, self.optimizer, self.policy_net, self.target_net, self.GAMMA,
                                              self.BATCH_SIZE)

        # copy optimized policy net to self.policy_net
        self.policy_net = optimized_policy_net
        # select next action here
        next_action, step_count = select_action(reform_next_state, self.policy_net, self.m, self.action_rng,
                                                self.EPS_START, self.EPS_END, self.EPS_DECAY, self.STEPS_DONE)
        self.action = next_action
        self.STEPS_DONE = step_count

        return next_action

    def agent_end(self, reward):
        if self.episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        reward_tensor = torch.tensor([reward], device='cpu')
        self.memory.push(self.state, self.action, None, reward_tensor)

        self.state = None

        optimized_policy_net = optimize_model(self.memory, self.optimizer, self.policy_net, self.target_net, self.GAMMA,
                                              self.BATCH_SIZE)
        # copy optimized policy net to self.policy_net
        self.policy_net = optimized_policy_net
        self.episode += 1

    def check_nan_weights(self):

        policy_net_layer1_nan = torch.isnan(self.policy_net.linear1.weight).any()
        policy_net_layer2_nan = torch.isnan(self.policy_net.linear2.weight).any()
        target_net_layer1_nan = torch.isnan(self.target_net.linear1.weight).any()
        target_net_layer2_nan = torch.isnan(self.target_net.linear2.weight).any()

        return (policy_net_layer1_nan or policy_net_layer2_nan or target_net_layer1_nan or target_net_layer2_nan)

