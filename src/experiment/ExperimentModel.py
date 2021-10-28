import sys
from PyExpUtils.models.ExperimentDescription import ExperimentDescription
import json
from PyExpUtils.utils.dict import merge, hyphenatedStringify, pick
from PyExpUtils.utils.str import interpolate
from PyExpUtils.models.Config import getConfig
# type checking
from typing import Optional, Union, List
Keys = Union[str, List[str]]

class ExperimentModel(ExperimentDescription):
    def __init__(self, d, path):
        super().__init__(d, path)
        self.agent = d['agent']
        self.problem = d['problem']

        self.max_steps = d.get('max_steps', 0)
        self.episodes = d.get('episodes')

def load(path=None):
    path = path if path is not None else sys.argv[1]
    with open(path, 'r') as f:
        d = json.load(f)
    print(d)
    exp = ExperimentModel(d, path)
    return exp