#from src.environments.env import BaseEnvironment
from copy import deepcopy
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import numpy as np
# from mypy_extensions import TypedDict



class ContinuousChain:


    def __init__(self, sparsity, seed, max_episode_len, extra_properties):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        self.cnt = 0
        self.length = max_episode_len
        self.interval_end = extra_properties["interval_end"]
        self.sparsity = sparsity*self.interval_end
        self.min_inc = extra_properties["reward_mult"]
        self.max_dist = extra_properties["max_dist"]
        self.min_dist = extra_properties["min_dist"]
        if(self.sparsity < self.min_inc):
            raise NotImplementedError
        self.rng = np.random.RandomState(seed)
        self.num_actions = 2

    def get_num_actions(self):
        return self.num_actions
    def get_actions(self):
        return [0,1]


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
        # s = self.state
        if action == 1:
            next__state = min(self.state + self.rng.uniform(self.min_dist, self.max_dist, (1,)), np.array([self.interval_end]))
            reward = float(((next__state//self.sparsity) - (self.state//self.sparsity)))*self.min_inc
            self.state = next__state
            
        elif action == 0:
            self.state = max(self.state - self.rng.uniform(self.min_dist, self.max_dist, (1,)), np.array([0.0]))
            reward = 0.0
        else:
            raise NotImplementedError
        
        if self.state == np.array([self.interval_end]):
            done = np.array([True])
            reward = 100.0
        else:
            done = np.array([self.cnt == self.length])
        
        if done:
            self.cnt = 0
        # print("action: ", action, "r: ", reward, "s:", s , " s_prime: ", self.state)
        return reward, self.state, done

        


    
