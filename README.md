# project-rez

# Setup Instructions

Make sure you're using Python 3.9.

Create a virtual environment "rez-env" with the command :
```
virtualenv rez-env
```
Activate your virtual environment.
```
source rez-env/bin/activate
```

Now, install dependencies.
```
pip install -r requirements.txt
```

## Running Locally
If you would like to run local example experiments use this (IMPORTANT: make sure your working directory is project-rez,
I would usually set that up on my PyCharm IDE):

```
python local.py src/main.py 10 experiments/example_mountain_car/SARSA_MountainCar.json
```

local.py is the entry point, which then launches main.py programs, one for each run. In this case, there are
10 runs specified. The json file given is where the experiment specification is. It tells the program what
environment to use, what agent to use, and what hyperparams to sweep over.
What happens internally is that main.py will recieve an index that tells it which hyperparam config
from the sweeps it will be doing and which run it is at. What is bieng saved now in the example
is the returns per run.

json files for experiments in the report are given in these folders inside the experiments folder:
continuous_chain_finalized_small_pos_big_end_lenline1_sparsity_0.05
continuous_chain_finalized_small_pos_big_end_lenline1_sparsity_1.0
continuous_chain_finalized_small_pos_big_end_lenline3_sparsity_0.1
continuous_chain_finalized_small_pos_big_end_lenline3_sparsity_0.9

## Running on Compute Canada

### First Time Setup

```
module --force purge
module load nixpkgs/16.09 python/3.6 gcc/7.3.0 cuda/10.2 cudacore/.10.1.243 cudnn/7.6.5
virtualenv rez-env
source rez-env/bin/activate
pip install --upgrade --no-index pip
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH

pip install -r requirements.txt
```
Note: Not sure why but Narval's highest Pytorch is 1.9.
If you can't install something with requirements.txt,
 because a version isn't available you'll need to install manually with pip and try out
 whatever version is available.

### Running after setup was done once
The following command will submit jobs with parallel tasks automatically.

Note that you can have an arbitrary number of files that are json experiment files in the command
The "./" is just the path to where the parent directory for the results folder is. Usually results are under
the project's directory.

```
module --force purge
module load nixpkgs/16.09 python/3.6 gcc/7.3.0 cuda/10.2 cudacore/.10.1.243 cudnn/7.6.5
source rez-env/bin/activate
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CUDA_PATH
python scripts/slurm.py <env_name> ./clusters/beluga.json src/main.py ./ <num_runs> <json_experiment_file1> <json_experiment_file2>
```

### Plotting
To plot data after it is generated, use:
```
python learning_curve_from_runs.py <number_of_runs_to_plot> ./ right_action_proportion.npy <path/to/experiment_file1.json> <path/to/experiment_file2.json>

```
You can add an arbitrary number of json files to plot their experiments.

# Citations:
This codebase is built with a customization of the rl-control-template repo by Andy Patterson and the forked repo by Kirby Banman.
The agents are based on code from "Privacy-preserving Q-Learning with Functional Noise in Continuous State Spaces"
 by Wang and Hegde (2019): https://arxiv.org/abs/1901.10634
