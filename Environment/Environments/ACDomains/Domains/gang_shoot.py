import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def gang(objects):
    objects["Gang"].attribute = objects["Leader"].attribute

def shoot(objects):
    objects["Death"].attribute = int(objects["Gang"].attribute or objects["Leader"].attribute)

class GangShoot(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False, cf_states=False):
        self.all_names = ["Gang", "Leader", "Death"]
        self.objects = {"Gang": ACObject("Gang", 2),
                        "Leader": ACObject("Leader", 2),
                        "Death": ACObject("Death", 2)} # dict of name to value
        self.binary_relations = [gang, shoot] # must get set prior to calling super (), the order follows the order of operations
        self.relation_outcome = ["Gang", "Death"]
        self.passive_mask = np.array([0,0])
        self.outcome_variable = "Death"
        super().__init__(frameskip, variant, fixed_limits, cf_states=cf_states)
