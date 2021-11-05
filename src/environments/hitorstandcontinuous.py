from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np

class hitorstandcontinuous:
    def __init__(self):
        self.observation_space = Box(0*np.ones(1), 1.0*np.ones(1), dtype=np.float64)
        self.action_space = Discrete(2)
        self.num_envs = 1
        self.cnt = 0
        self.length = 50

    def step(self, action):
        self.cnt += 1
        #if action == 1:
        #    if self.state > 0.5:
        #        self.state = np.random.uniform(0.0, self.state, (1,))
        #    else:
        #        self.state = np.random.uniform(self.state, 1.0, (1,))
        #reward = 0.0
        if action == 0:
            self.state = min(self.state + np.random.uniform(0.0, 0.25, (1,)), np.array([1.0]))
        elif action == 1:
            self.state = max(self.state - np.random.uniform(0.0, 0.25, (1,)), np.array([0.0]))
        done = np.array([self.cnt == self.length])
        reward = float(0.5 - np.abs(self.state-0.5))
        if done:
            self.cnt = 0
        #    reward = float(self.state)
        return self.state, reward, done, None

    def reset(self):
        self.cnt = 0
        self.state = np.random.uniform(0.0, 1.0, (1,))
        return self.state

    def render(self):
        raise NotImplementedError

    def seed(self, seed_value):
        np.random.seed(seed_value)
        print('numpy seed is changed to {} globally'.format(seed_value))