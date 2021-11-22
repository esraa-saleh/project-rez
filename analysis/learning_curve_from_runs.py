from PyExpUtils.results.results import loadResults
import matplotlib.pyplot as plt
import numpy as np
from src.experiment.ExperimentModel import load
import sys
from collections import namedtuple
from scipy.stats import binned_statistic
from src.utils.Collector import Collector
from src.experiment import ExperimentModel
from PyExpUtils.utils.dict import hyphenatedStringify
from pathlib import Path
from scipy.integrate import simps
from src.analysis.colors import colors

DataClass = namedtuple("BestCurveData", "exp bestparams mean, stderr auc")

# this is if we want smoothing.
def bin_series(arr, bin_size):
    num_bins = len(arr)//bin_size
    x = np.linspace(1, len(arr), num=len(arr))
    return binned_statistic(x, arr, statistic='sum', bins=num_bins).statistic


# gets the averaged curve given a certain hyperparam result directory containing run folders
# that contain the run data in numpy files
def averagedResultsFromFilePathsForRuns(numRuns, runDirPath, fileName, bin_size=1):
    collector = Collector()
    for currRun in range(numRuns):
        runPath = runDirPath + "/" + str(currRun) + "/" + fileName
        print(runPath)
        try:
            arr = np.array(np.load(runPath, allow_pickle=True), dtype = 'float64')
        except (FileNotFoundError, EOFError, OSError) as e :
            # an empty array is interpreted as less than the length expected so auc is inf
            arr = np.array([np.inf]*10)
            print("For path", runPath)
            print(e)

        if(bin_size > 1):
            arr = bin_series(arr, bin_size)
        collector.collectFullRun('squared_value_error', arr)
    mean, stderr, _ = collector.getStats('squared_value_error')
    return mean, stderr



# assume that the "best" hyperparam config is the one that results in the max AUC
def createPlottingData(expPaths, numRuns, resultsFileName, curveGranularity= "episodic"):
    data = []
    for expPath in expPaths:
        print(expPath)
        exp = ExperimentModel.load(expPath)
        numPerms = exp.numPermutations()
        expDir = Path(exp.path).parent.name
        if(curveGranularity == "episodic"):
            expectedCurveLen = exp.episodes
        else:
            raise NotImplementedError(curveGranularity)

        bestAUC = float('inf')
        bestParams = None
        bestCurve = None
        for perm_num in range(numPerms):
            perm = exp.getPermutation(perm_num)['metaParameters']
            runsDirPath = "results/" + expDir + "/" + exp.agent + "/" + hyphenatedStringify(perm)
            mean, stderr = averagedResultsFromFilePathsForRuns(numRuns, runsDirPath, resultsFileName, bin_size=1)
            if (mean.shape[0] < expectedCurveLen):  # this is a diverging run
                aucOther = float('inf')
            else:
                aucOther = simps(mean, dx=1)

            if aucOther < bestAUC:
                bestAUC = aucOther
                bestCurve = (mean, stderr)
                bestParams = perm
        data.append(DataClass(exp, bestParams, bestCurve[0], bestCurve[1], bestAUC))
    return data

def confidenceInterval(mean, stderr):
    return (mean - stderr, mean + stderr)

def plotData(dataList):
    alpha =0.4
    alphaMain = 1

    f, ax = plt.subplots(1)
    for data in dataList:
        sparsity_str = str(data.exp.permutable()["metaParameters"]["sparsity"])
        label = data.exp.agent + "_" + sparsity_str + "-sparsity"
        ax.plot(data.mean, linestyle=None, label=label, alpha=alphaMain, linewidth=2)
        (low_ci, high_ci) = confidenceInterval(data.mean, data.stderr)
        ax.fill_between(range(data.mean.shape[0]), low_ci, high_ci,
                        alpha=alpha * alphaMain)
        ax.legend()
        print("plotted:", data.exp.agent, ", AUC: ", data.auc)
        print("hyperparameters:", data.bestparams)
    plt.ylim([0,50])
    plt.show()


if __name__ == '__main__':

    runs_to_plot = int(sys.argv[1])
    path_to_results_folder = sys.argv[2]
    resultsFileName = sys.argv[3]
    expPaths = sys.argv[4:]
    data = createPlottingData(expPaths, runs_to_plot, resultsFileName, curveGranularity="episodic")
    plotData(data)

# demo_exp_path = 'experiments/continuous_chain/QLearning_ContinuousChain.json'
    # num_runs = 5
    # exp = load(demo_exp_path)
    #
    # num_results = exp.numPermutations()*num_runs
    # for i in range(num_results):
    #     path = exp.interpolateSavePath(idx=i)
    #     return_arr = np.load(path + "/episodic_rewards.npy", allow_pickle=True)
    #     plt.title(path)
    #     plt.plot(return_arr)
    #     plt.show()
    # plt.plot()
    # plt.show()