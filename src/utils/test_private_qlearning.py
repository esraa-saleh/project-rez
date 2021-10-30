import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from os import path
import math
import random
from src.agents.private_qlearning import ReplayMemory, noisebuffer
from src.agents.private_qlearning import DQN as private_qlearning
from src.environments.hitorstandcontinuous import hitorstandcontinuous

env = hitorstandcontinuous()
m = env.action_space.n

def test_noisebuffer_reset():
    nb = noisebuffer(2, 1.)
    nb.sample(0.5)
    nb.reset()
    assert len(nb.buffer) == 0

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

if __name__ == "main":
    test_noisebuffer_reset()
    test_replay_push()