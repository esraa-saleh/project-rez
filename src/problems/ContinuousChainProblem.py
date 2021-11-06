# This is where we should have our actual environemnt

from src.environments.ContinuousChain import ContinuousChain
from src.agents.registry import getAgent

# We will create a parent class if needed. Keeping it simple for now.

class ContinuousChainProblem:
    def __init__(self, exp, idx, agent_seed_bundle, env_seed_bundle):
        self.exp = exp
        self.idx = idx
        perm = exp.getPermutation(idx)
        self.params = perm['metaParameters']

        self.env = ContinuousChain(self.params["sparsity"], env_seed_bundle.env_reaction_seed, exp.max_steps)
        m = self.env.get_num_actions()
        Agent = getAgent(self.exp.agent)

        self.agent = Agent(seed_bundle=agent_seed_bundle, m=m,
                           EPS_START=self.params["EPS_START"], EPS_END=self.params["EPS_END"],
                           EPS_DECAY=self.params["EPS_DECAY"], TARGET_UPDATE=self.params["TARGET_UPDATE"],
                           BATCH_SIZE=self.params["BATCH_SIZE"], GAMMA=self.params["GAMMA"])

    def getAgent(self):
        return self.agent

    def getEnvironment(self):
        return self.env