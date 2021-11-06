import pytest
import torch
import random

from src.agents.private_qlearning import DQNAgent
from src.environments.hitorstandcontinuous import hitorstandcontinuous

env = hitorstandcontinuous()
m = env.action_space.n
episode = 10 #arbitrary

#AGENT ATTRIBUTE TESTING BLOCK

def test_inner_DQN():
    agent = DQNAgent(episode, m)
    policy_net = agent.policy_net
    target_net = agent.target_net
    assert policy_net is not None
    assert target_net is not None
    assert policy_net.sigma is not None
    assert target_net.sigma is not None
    assert policy_net.nb is not None
    assert target_net.nb is not None


def test_agent_attributes():
    agent = DQNAgent(episode,m)
    assert agent.GAMMA is not None
    assert agent.BATCH_SIZE is not None
    assert agent.EPS_START is not None
    assert agent.EPS_END is not None
    assert agent.EPS_DECAY is not None
    assert agent.TARGET_UPDATE is not None
    assert agent.STEPS_DONE is not None
    assert agent.memory is not None
    assert agent.m is not None
    assert agent.optimizer is not None
    assert agent.memory is not None

#AGENT METHOD TESTING BLOCK
def test_agent_start():
    agent = DQNAgent(episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    action = agent.agent_start(state)
    assert action is not None
    assert agent.action is not None
    assert agent.state is not None
    assert agent.total_reward == 0

def test_agent_step():
    agent = DQNAgent(episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    reward = 10 #arbitrary reward
    action = agent.agent_start(state)
    next_action = agent.agent_step(state, reward)
    assert next_action is not None
    assert agent.state is not None
    assert agent.action is not None
    assert agent.total_reward is not None
    assert agent.total_reward > 0

def test_agent_end():
    agent = DQNAgent(episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    action = agent.agent_start(state)
    reward = 10 #arbitrary
    agent.agent_end(reward)
    assert agent.state is None
    assert agent.total_reward > 0

def test_noisebuffer_reset():
    agent = DQNAgent(episode,m)
    agent.policy_net.nb.sample(0.5)
    agent.policy_net.nb.reset()
    assert len(agent.policy_net.nb.buffer) == 0

def test_replay_push():
    agent = DQNAgent(episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    action = agent.agent_start(state)
    reward = 10 #arbitrary
    next_action = agent.agent_step(state, reward)
    #nonsense pushing nonsense transition (same state)
    agent.memory.push(state, action, state, reward)
    #not sure what else to check here since memory pre-allocated
    assert len(agent.memory) > 0


if __name__ == "main":
    test_inner_DQN()
    test_agent_attributes()
    test_agent_start()
    test_agent_step()
    test_agent_end()
    test_noisebuffer_reset()
    test_replay_push()