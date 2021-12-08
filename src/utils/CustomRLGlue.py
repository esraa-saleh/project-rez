"""Glues together an experiment, agent, and environment.
Reference: https://github.com/andnp/coursera-rl-glue/blob/master/RLGlue/rl_glue.py
"""


from __future__ import print_function



class CustomRLGlue:
    """RLGlue class
    args:
        env_name (string): the name of the module where the Environment class can be found
        agent_name (string): the name of the module where the Agent class can be found
    """

    def __init__(self, env_obj, agent_obj):
        self.environment = env_obj
        self.agent = agent_obj

        self.last_action = None
        self.total_reward = 0.0
        self.num_steps = 0
        self.num_episodes = 0
        self.action_counts = self.__empty_action_counts()

    def __empty_action_counts(self):
        actions = self.environment.get_actions()
        action_counts = {}
        for a in actions:
            action_counts[a] = 0
        return action_counts


    def __add_action_to_counts(self, action):
        self.action_counts[action] += 1

    def __rl_start(self):
        """Starts RLGlue experiment
        Returns:
            tuple: (state, action)
        """
        self.action_counts = self.__empty_action_counts()
        self.total_reward = 0.0
        self.num_steps = 0
        last_state = self.environment.env_start()
        self.last_action = self.agent.agent_start(last_state)
        self.__add_action_to_counts(self.last_action)
        observation = (last_state, self.last_action)

        return observation



    def __rl_step(self):
        """Step taken by RLGlue, takes environment step and either step or
            end by agent.
        Returns:
            (float, state, action, Boolean): reward, last state observation,
                last action, boolean indicating termination
        """

        (reward, last_state, term) = self.environment.env_step(self.last_action)

        self.total_reward += reward

        if term:
            self.num_episodes += 1
            self.agent.agent_end(reward)
            roat = (reward, last_state, None, term)
        else:
            self.num_steps += 1
            self.last_action = self.agent.agent_step(reward, last_state)
            self.__add_action_to_counts(self.last_action)
            roat = (reward, last_state, self.last_action, term)

        return roat



    def rl_episode(self, max_steps_this_episode):
        """Runs an RLGlue episode
        Args:
            max_steps_this_episode (Int): the maximum steps for the experiment to run in an episode
        Returns:
            Boolean: if the episode should terminate
        """
        is_terminal = False

        self.__rl_start()

        while (not is_terminal) and ((max_steps_this_episode == 0) or
                                     (self.num_steps < max_steps_this_episode)):
            rl_step_result = self.__rl_step()
            is_terminal = rl_step_result[3]

        return is_terminal

    def rl_return(self):
        """The total reward
        Returns:
            float: the total reward
        """
        return self.total_reward

    def rl_num_steps(self):
        """The total number of steps taken
        Returns:
            Int: the total number of steps taken
        """
        return self.num_steps

    def rl_num_episodes(self):
        """The number of episodes
        Returns
            Int: the total number of episodes
        """
        return self.num_episodes

    def rl_episode_action_proportion(self, action):
        if(not(action in self.action_counts)):
            raise NotImplementedError
        return self.action_counts[action]/float(self.num_steps+1)

    def check_nan_agent_weights(self):
        return self.agent.check_nan_weights()
