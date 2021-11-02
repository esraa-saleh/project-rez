#from src.environments.env import BaseEnvironment
from copy import deepcopy
# from mypy_extensions import TypedDict



class ContinuousChain():


    def __init__(self, env_info={}):
        """Setup for the environment called when the experiment first starts.
        Note:
            Initialize a tuple with the reward, first state, boolean
            indicating if it's terminal.
        """
        # a term idx is a 1d indexing of the grid that includes terminal states
        # a nonterm idx is the same but it skips terminal states in its counting

        super().__init__()
        pass

    def env_start(self):
        """The first method called when the episode starts, called before the
        agent starts.

        Returns:
            The first state from the environment.
        """
        pass


    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        pass


    def env_cleanup(self):
        """Cleanup done after the environment ends"""
        # usually after the end of an episode the agent location is reset
        # self.agent_loc = self.start_loc
        pass

