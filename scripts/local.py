import sys
import os
sys.path.append(os.getcwd())

import subprocess
from functools import partial
from multiprocessing.pool import Pool

from PyExpUtils.runner import Args
from PyExpUtils.results.indices import listIndices
from PyExpUtils.results.paths import listResultsPaths
import src.experiment.ExperimentModel as Experiment

# TODO: change name of results file
def generateMissing(paths):
    for i, p in enumerate(paths):
        summary_path = p + '/return_summary.npy'
        if not os.path.exists(summary_path):
            yield i

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Please run again using')
        print('python scripts/local.py [entry.py] [runs] [paths/to/descriptions]...')
        exit(0)

    pool = Pool()
    experiment_paths = sys.argv[3:]
    runs = int(sys.argv[2])
    print("runs: ", runs)
    executable = "python " + sys.argv[1]
    retry = True

    cmds = []
    for path in experiment_paths:
        exp = Experiment.load(path)

        paths = listResultsPaths(exp, runs)
        # get all of the indices corresponding to missing results
        if retry:
            indices = listIndices(exp, runs)
        else:
            indices = generateMissing(paths)

        for idx in indices:
            exe = f'{executable} {runs} {path} {idx}'
            cmds.append(exe)

    print(len(cmds))
    res = pool.imap_unordered(partial(subprocess.run, shell=True, stdout=subprocess.PIPE), cmds)
    for i, _ in enumerate(res):
        sys.stderr.write(f'\r{i}/{len(cmds)}')
    sys.stderr.write('\n')
