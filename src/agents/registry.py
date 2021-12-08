from src.agents.qlearning import DQNAgent as qlearning
from src.agents.private_qlearning import PrivateDQNAgent as private_qlearning

def getAgent(name):
    if(name == 'qlearning'):
        return qlearning
    if(name == 'private_qlearning'):
        return private_qlearning

    raise NotImplementedError()

