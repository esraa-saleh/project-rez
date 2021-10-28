import json
from collections import namedtuple

class SeedsHolder:
    def __init__(self, max_required_seeds):
        filename = "./src/utils/seeds.json"
        with open(filename) as json_file:
            data = json.load(json_file)
            self.real_action_seeds = data["real_action_seeds"]
            self.parameter_init_seeds = data["parameter_init_seeds"]

        if(len(self.real_action_seeds) < max_required_seeds
                or len(self.parameter_init_seeds) < max_required_seeds):
            raise NotImplementedError

        self.seed_bundle_class = namedtuple('SeedBundle', 'real_action_seed parameter_init_seed')


    def get_seed_for_parallel_run(self, run):
        seed_real_action = self.real_action_seeds[run]
        seed_param_init = self.parameter_init_seeds[run]
        return self.seed_bundle_class(seed_real_action, seed_param_init)