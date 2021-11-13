import json
from collections import namedtuple

class SeedsHolder:
    def __init__(self, max_required_seeds):
        filename = "./src/utils/seeds.json"
        with open(filename) as json_file:
            data = json.load(json_file)
            self.action_seeds = data["action_seeds"]
            self.parameter_init_seeds = data["parameter_init_seeds"]
            self.env_reaction_seeds = data["env_reaction_seeds"]
            self.noisebuffer_seeds = data["noisebuffer_seeds"]
            self.replay_seeds = data["replay_seeds"]

        if(len(self.action_seeds) < max_required_seeds
                and len(self.parameter_init_seeds) < max_required_seeds
                and len(self.noisebuffer_seeds) < max_required_seeds
                and len(self.replay_seeds) < max_required_seeds
                and len(self.env_reaction_seeds) < max_required_seeds):
            raise NotImplementedError

        self.agent_seed_bundle_class = namedtuple('AgentSeedBundle', 'action_seed parameter_init_seed noisebuffer_seed replay_seed')
        self.env_seed_bundle_class = namedtuple('EnvSeedBundle', 'env_reaction_seed')


    def get_seed_for_parallel_run(self, run):
        seed_action = self.action_seeds[run]
        seed_param_init = self.parameter_init_seeds[run]
        seed_env_reaction = self.env_reaction_seeds[run]
        return self.agent_seed_bundle_class(seed_action, seed_param_init), self.env_seed_bundle_class(seed_env_reaction)