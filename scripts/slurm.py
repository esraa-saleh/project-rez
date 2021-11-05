import time
import sys
import os
sys.path.append(os.getcwd())

import src.experiment.ExperimentModel as Experiment
from PyExpUtils.results.paths import listResultsPaths
from PyExpUtils.utils.generator import group
import src.utils.MySlurm as MySlurm
import uuid
from datetime import datetime

if len(sys.argv) < 4:
    print('Please run again using')
    print('python -m env-folder-name scripts.scriptName [path/to/slurm-def] [src/executable.py] [base_path] [runs] [paths/to/descriptions]...')
    exit(0)

# -------------------------------
# Generate scheduling bash script
# -------------------------------

# the contents of the string below will be the bash script that is scheduled on compute canada
# change the script accordingly (e.g. add the necessary `module load X` commands)
cwd = os.getcwd()
def getJobScript(parallel, env_folder_name):
    return f"""#!/bin/bash
cd {cwd}
. {env_folder_name}/bin/activate
{parallel}
    """

# --------------------------
# Get command-line arguments
# --------------------------
env_folder_name = sys.argv[1]
slurm_path = sys.argv[2]
executable = sys.argv[3]
base_path = sys.argv[4]
runs = int(sys.argv[5])
experiment_paths = sys.argv[6:]

# generates a list of indices whose results are missing
def generateMissing(paths):
    for i, p in enumerate(paths):
        summary_path = p + '/returns.npy' # <-- TODO: change this to match the filename where you save your results
        if not os.path.exists(summary_path):
            yield i

# prints a progress bar
def printProgress(size, it):
    for i, _ in enumerate(it):
        print(f'{i + 1}/{size}', end='\r')
        if i - 1 == size:
            print()
        yield _

# ----------------
# Scheduling logic
# ----------------
for path in experiment_paths:
    print(path)
    # load the experiment json file
    exp = Experiment.load(path)
    # load the slurm config file. This is using MySlurm because I needed to have
    # custom functionality where I could have an options object that
    # could parse through gpu options
    slurm = MySlurm.fromFile(slurm_path)
    print(slurm)

    # figure out how many indices to use
    size = exp.numPermutations()

    # get a list of all expected results paths
    paths = listResultsPaths(exp, runs=runs)
    paths = printProgress(size, paths)
    # get a list of the indices whose results paths are missing
    #TODO: Warn team about how "generateMissing" works
    indices = generateMissing(paths)

    # compute how many "tasks" to clump into each job

    groupSize = slurm.tasks * slurm.tasksPerNode

    for g in group(indices, groupSize):
        l = list(g)
        print("scheduling:", path, l)

        # build the executable string
        runner = f'python {executable} {runs} {path}'
        # generate the gnu-parallel command for dispatching to many CPUs across server nodes
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%d-%b-%Y__%H:%M:%S.%f")
        unique_log_file_name = timestampStr+"_"+str(uuid.uuid4())+".log"

        slurm.tasks = min([slurm.tasks, len(l)])


        srun_prallel_specs = {
            'ntasks': slurm.tasks,
        }


        parallel = MySlurm.buildParallel(runner, l, srun_prallel_specs, log_file=unique_log_file_name, with_gpu=slurm.gpu_specs_present())

        # generate the bash script which will be scheduled
        script = getJobScript(parallel, env_folder_name)
        print(script)
        print("-----------------------")

        ## uncomment for debugging the scheduler to see what bash script would have been scheduled
        # print(script)
        # exit()

        # make sure to only request the number of CPU cores necessary
        MySlurm.schedule(script, slurm)

        # DO NOT REMOVE. This will prevent you from overburdening the slurm scheduler. Be a good citizen.
        time.sleep(2)
