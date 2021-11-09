import pytest
import torch
import random
from src.environments.hitorstandcontinuous import hitorstandcontinuous
from src.agents.qlearning import DQNAgent
from src.utils.SeedsHolder import SeedsHolder
from collections import namedtuple

env = hitorstandcontinuous()
m = env.action_space.n
episode = 10 #arbitrary

seed_bundle = namedtuple('SeedBundle', 'action_seed parameter_init_seed')
bundle = seed_bundle(1234, 5678)

#AGENT ATTRIBUTE TESTING BLOCK

def test_inner_DQN():
    agent = DQNAgent(bundle, episode, m)
    policy_net = agent.policy_net
    target_net = agent.target_net
    assert policy_net is not None
    assert target_net is not None


def test_agent_attributes():
    agent = DQNAgent(bundle, episode,m)
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
    agent = DQNAgent(bundle, episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    action = agent.agent_start(state)
    assert action is not None
    assert agent.action is not None
    assert agent.state is not None

def test_agent_step():
    agent = DQNAgent(bundle, episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    reward = 10 #arbitrary reward
    action = agent.agent_start(state)
    next_action = agent.agent_step(state, reward)
    assert next_action is not None
    assert agent.state is not None
    assert agent.action is not None


def test_agent_end():
    agent = DQNAgent(bundle, episode,m)
    state = torch.Tensor(env.reset()).unsqueeze(0)
    action = agent.agent_start(state)
    reward = 10 #arbitrary
    agent.agent_end(reward)
    assert agent.state is None


def test_replay_push():
    agent = DQNAgent(bundle, episode,m)
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
    test_replay_push()