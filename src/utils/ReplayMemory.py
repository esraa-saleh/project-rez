from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayMemory(object):

#change to replay_rng
    def __init__(self, capacity, replay_rng):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.replay_rng = replay_rng


    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return self.replay_rng.choice(self.memory, size=batch_size)

    def __len__(self):
        return len(self.memory)