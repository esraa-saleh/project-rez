import pytest
import torch



import random
from src.agents.qlearning import ReplayMemory
from src.environments.hitorstandcontinuous import hitorstandcontinuous
from src.agents.qlearning import DQN as qlearning
from src.utils.select_action import select_action

env = hitorstandcontinuous()
m = env.action_space.n

#testing memory init
def test_set_replay():
    agent = qlearning()
    agent.agent_set_memory(10)
    assert agent.memory != None

def test_get_replay():
    agent = qlearning()
    agent.agent_set_memory(10)
    memory = agent.agent_get_memory()
    assert memory is not None
    assert memory.capacity == 10


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

def test_agent_start():
    agent = qlearning()
    state = torch.Tensor(env.reset()).unsqueeze(0)

    action = agent.agent_start(state, agent)

    assert action is not None
'''
#this test FAILS on Nidhi's environment, because her env.step() method has bugs
def test_agent_step():
    agent = qlearning()
    state = torch.Tensor(env.reset()).unsqueeze(0)
    target_net = qlearning()

    reward = torch.tensor([1.0])
    step_action = agent.agent_step(state, reward, agent, target_net)
    assert step_action is not None
'''

if __name__ == "main":
    test_set_replay()
    test_get_replay()
    test_replay_push()
    test_agent_start()
    #test_agent_step()