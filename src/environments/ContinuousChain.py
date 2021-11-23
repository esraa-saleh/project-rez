#from src.environments.env import BaseEnvironment
from copy import deepcopy
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
# from mypy_extensions import TypedDict



class ContinuousChain:


    def __init__(self, sparsity, seed, max_episode_len):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        self.observation_space = Box(0*np.ones(1), 1.0*np.ones(1), dtype=np.float64)
        self.action_space = Discrete(2)
        self.cnt = 0
        self.length = max_episode_len
        self.sparsity = sparsity
        self.rng = np.random.RandomState(seed)
        self.num_actions = 2

    def get_num_actions(self):
        return self.num_actions


    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        self.cnt = 0
        self.state = self.rng.uniform(0.0, 1.0, (1,))
        return self.state
        


    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """
        self.cnt += 1
        if action == 1:
            next__state = min(self.state + self.rng.uniform(0.0, 0.25, (1,)), np.array([1.0]))
            if next__state != np.array([1.0]):
                reward = -1 * float(next__state//self.sparsity - self.state//self.sparsity)
            self.state = next__state
            
        elif action == 0:
            self.state = max(self.state - self.rng.uniform(0.0, 0.25, (1,)), np.array([0.0]))
            reward = -1 * float(next__state//self.sparsity - self.state//self.sparsity)
        else:
            raise NotImplementedError
        
        if self.state == np.array([1.0]):
            done = np.array([True])
            reward = 0
        else:
            done = np.array([self.cnt == self.length])
        
        if done:
            self.cnt = 0
        
        return reward, self.state, done

        


    
