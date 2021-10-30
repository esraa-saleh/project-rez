import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from os import path
import math


import random
from src.agents.qlearning import ReplayMemory
from src.environments.hitorstandcontinuous import hitorstandcontinuous

env = hitorstandcontinuous()
m = env.action_space.n

def test_replay_push():
    rp = ReplayMemory(1)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    #random actions
    action = torch.tensor([[random.randrange(m)]], dtype=torch.long)
    next_state, reward, done, info = env.step(action.item())
    reward = torch.Tensor([reward])
    #push a random action to rp
    rp.push(state, action, next_state, reward)
    #testing if single item was pushed && rp at capacity
    assert rp.capacity == len(rp.memory)
