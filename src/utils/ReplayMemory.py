from collections import namedtuple

'''
This code is a modified version of Wang & Hegde [2019] and the PyTorch DQN tutorial found at:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
'''

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):
    def __init__(self, capacity, replay_seeds):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        seed = replay_seeds.randint(1000)
        random.seed(seed)

    def push(self, *args):
        #adds a transition to memory
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    #samples from the memory using the set seed
    def sample(self, batch_size):
        return random.sample(self.memory, size=batch_size)

    def __len__(self):
        return len(self.memory)