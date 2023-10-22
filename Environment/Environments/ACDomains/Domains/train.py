import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def track(objects):
    objects["Track"].attribute = objects["Switch"].attribute if objects["Break"].attribute != 1 else 2

def arrive(objects):
    objects["Arrive"].attribute = 1 if objects["Track"].attribute != 2 else 0

class Train(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False, cf_states=False):
        self.all_names = ["Break", "Switch", "Track", "Arrive"]
        self.objects = {"Break": ACObject("Break", 2),
                        "Switch": ACObject("Switch", 2),
                        "Track": ACObject("Track", 3),
                        "Arrive": ACObject("Arrive", 2)} # dict of name to value
        self.binary_relations = [track, arrive] # must get set prior to calling super (), the order follows the order of operations
        self.relation_outcome = ["Track", "Arrive"]
        self.passive_mask = np.array([0,0,0])
        self.outcome_variable = "Arrive"
        super().__init__(frameskip, variant, fixed_limits, cf_states=cf_states)
