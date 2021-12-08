import numpy as np
import sys
import os
sys.path.append(os.getcwd())

# from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
# from src.utils.rl_glue import OneStepWrapper
from src.utils.CustomRLGlue import CustomRLGlue
from src.utils.SeedsHolder import SeedsHolder


if len(sys.argv) < 3:
    print('run again with:')
    print('python3 src/main.py <runs> <path/to/description.json> <idx>')
    exit(1)

runs = int(sys.argv[1])
exp = ExperimentModel.load(sys.argv[2])
idx = int(sys.argv[3])

max_steps = exp.max_steps
curr_run = exp.getRun(idx=idx)
seeds_holder = SeedsHolder(max_required_seeds=runs)
agent_seed_bundle, env_seed_bundle = seeds_holder.get_seed_for_parallel_run(run=curr_run)
print(agent_seed_bundle, env_seed_bundle)

# np.random.seed(agent_seed_bundle.real_action_seed)

Problem = getProblem(exp.problem)
problem = Problem(exp, idx, agent_seed_bundle, env_seed_bundle)
# problem = Problem(exp, idx, agent_seed_bundle)

agent = problem.getAgent()
env = problem.getEnvironment()

glue = CustomRLGlue(agent_obj=agent, env_obj=env)

# Run the experiment
rewards = []
collector = Collector()
for episode in range(exp.episodes):
    # glue.total_reward = 0
    glue.rl_episode(max_steps_this_episode=max_steps)
    # if the weights diverge to nan, just quit. This run doesn't matter to me anyways now.
    if(glue.check_nan_agent_weights()):
        collector.fillRest(np.nan, exp.episodes)
        break

    collector.collect('episodic_rewards', glue.total_reward)
    collector.collect('right_action_proportion', glue.rl_episode_action_proportion(1))

return_data = collector.getCurrentRunData('episodic_rewards')
r_prop_data = collector.getCurrentRunData('right_action_proportion')

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('episodic_rewards.npy'), return_data)
np.save(save_context.resolve('right_action_proportion.npy'), r_prop_data)
