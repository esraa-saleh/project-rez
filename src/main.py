import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from RlGlue import RlGlue
from src.experiment import ExperimentModel
from src.problems.registry import getProblem
from src.utils.Collector import Collector
from src.utils.rl_glue import OneStepWrapper
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
agent_seed_bundle = seeds_holder.get_seed_for_parallel_run(run=curr_run)

np.random.seed(agent_seed_bundle.real_action_seed)

Problem = getProblem(exp.problem)
problem = Problem(exp, idx)
# TODO: adjust problem so that it accepts seed, instead of seeding with numpy seed
# problem = Problem(exp, idx, agent_seed_bundle)

agent = problem.getAgent()
env = problem.getEnvironment()

wrapper = OneStepWrapper(agent, problem.getGamma(), problem.rep)
glue = RlGlue(wrapper, env)

# Run the experiment
rewards = []
collector = Collector()
for episode in range(exp.episodes):
    glue.total_reward = 0
    glue.runEpisode(max_steps)

    # if the weights diverge to nan, just quit. This run doesn't matter to me anyways now.
    if np.isnan(np.sum(agent.w)):
        collector.fillRest(np.nan, exp.episodes)
        broke = True
        break

    collector.collect('return', glue.total_reward)

return_data = collector.getCurrentRunData('return')

# collector.reset()

# return_data = collector.getStats('return')

# import matplotlib.pyplot as plt
# from src.utils.plotting import plot
# fig, ax1 = plt.subplots(1)

# plot(ax1, return_data)
# ax1.set_title('Return')

# plt.show()
# exit()

# save results to disk
save_context = exp.buildSaveContext(idx, base="./")
save_context.ensureExists()

np.save(save_context.resolve('returns.npy'), return_data)
