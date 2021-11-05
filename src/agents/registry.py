from src.agents.SARSA import SARSA
from src.agents.qlearning import DQN as qlearning
from src.agents.private_qlearning import DQN as private_qlearning

def getAgent(name):
    if(name == 'qlearning'):
        return qlearning
    if(name == 'private_qlearning'):
        return private_qlearning
    if name == 'SARSA':
        return SARSA

    raise NotImplementedError()

