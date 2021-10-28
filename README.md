# project-rez

# Setup Instructions

Make sure you're using Python 3.9 (that's Esra'a initially tested with).

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



# Citations:
This codebase is built with a customization of the rl-control-template repo by Andy Patterson and the forked repo by Kirby Banman.