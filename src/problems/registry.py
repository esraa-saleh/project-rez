from src.problems.ContinuousChain import ContinuousChain
from src.problems.MountainCar import MountainCar

def getProblem(name):
    if(name == "ContinuousChain"):
        return ContinuousChain
    elif(name == 'MountainCar'):
        return MountainCar

    raise NotImplementedError(name)
