from PyExpUtils.results.results import loadResults
import matplotlib.pyplot as plt
import numpy as np
from src.experiment.ExperimentModel import load

if __name__ == '__main__':
    demo_exp_path = 'experiments/example_mountain_car/SARSA_MountainCar.json'
    num_runs = 10
    exp = load(demo_exp_path)

    num_results = exp.numPermutations()*num_runs
    for i in range(num_results):
        path = exp.interpolateSavePath(idx=i)
        return_arr = np.load(path + "/returns.npy", allow_pickle=True)
        plt.title(path)
        plt.plot(return_arr)
        plt.show()
    # plt.plot()
    # plt.show()
