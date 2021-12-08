from src.problems.ContinuousChainProblem import ContinuousChainProblem

def getProblem(name):
    if(name == "ContinuousChain"):
        return ContinuousChainProblem

    raise NotImplementedError(name)
