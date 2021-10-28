# This is where we should have our actual environemnt

from src.problems.BaseProblem import BaseProblem
from src.environments.ContinuousChain import ContinuousChain as Env
from PyFixedReps.TileCoder import TileCoder

#This is just an example, it should be replaced by our actual code

class ScaledTileCoder(TileCoder):
    def encode(self, s):
        p = s[0]
        v = s[1]

        p = (p + 1.2) / 1.7
        v = (v + 0.07) / 0.14
        return super().encode((p, v)) / float(self.num_tiling)

class ContinuousChain(BaseProblem):
    def __init__(self, exp, idx):
        super().__init__(exp, idx)
        self.env = Env()
        self.actions = 2

        self.rep = ScaledTileCoder({
            'dims': 1,
            'tiles': 4,
            'tilings': 16,
        })

        self.features = self.rep.features()
        self.gamma = 1.0