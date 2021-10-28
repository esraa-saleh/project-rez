from src.agents.SARSA import SARSA

def getAgent(name):
    if(name == 'qlearning'):
        pass
        #return QLearning
    if(name == 'private_qlearning'):
        pass
        #return PrivateQLearning
    if name == 'SARSA':
        return SARSA

    raise NotImplementedError()

