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

#TODO: have seeding for neural net init to make results reproducable


#NETWORK
class DQN(nn.Module):
    def __init__(self, m, torch_rng, hidden=16):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(1, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, m)
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
        self.action_rng = np.random.RandomState(seed_bundle.action_seed)
        self.replay_rng = np.random.RandomState(seed_bundle.replay_seed)
        self.torch_rng = np.random.RandomState(seed_bundle.parameter_init_seed)
        self.memory = ReplayMemory(10000, self.replay_rng)
        # self.total_reward = None
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
        #self.target_net.load_state_dict(self.policy_net.parameters())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters())

    #getters for ReplayMemory and total reward
    def agent_get_memory(self):
        return self.memory

    # def agent_get_total_reward(self):
    #     return self.total_reward
    '''
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
    '''
    '''
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
    '''

    # RL Glue methods:
    def agent_start(self, state):
        # self.total_reward = 0
        self.state = torch.Tensor(state).unsqueeze(0)
        action, step_count = select_action(self.state, self.policy_net, self.m, self.action_rng, self.EPS_START,
                                           self.EPS_END, self.EPS_DECAY, self.STEPS_DONE)
        self.action = action
        self.STEPS_DONE = step_count
        return action.item()

    def agent_step(self, reward, next_state):
        # manually putting in device='cpu' here to avoid having to pass it in
        reward_tensor = torch.tensor([reward], device='cpu')
        reform_next_state = torch.Tensor(next_state).unsqueeze(0)
        # store transition in memory
        self.memory.push(self.state, self.action, reform_next_state, reward_tensor)

        # recieve reward
        # self.total_reward += float(reward.squeeze(0).data)

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

        return next_action.item()

    def agent_end(self, reward):
        if self.episode % self.TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        reward_tensor = torch.tensor([reward], device='cpu')
        self.memory.push(self.state, self.action, None, reward_tensor)
        # reward = torch.tensor([reward], device='cpu')
        # self.total_reward += float(reward.squeeze(0).data)
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

