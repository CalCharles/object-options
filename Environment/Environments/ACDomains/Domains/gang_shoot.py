import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def shoot(objects):
    objects["Death"].attribute = int(objects["Gang"].attribute or objects["Leader"].attribute)

def gang(objects):
    objects["Gang"].attribute = Leader

class HaltCharge(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False):
        self.all_names = ["Gang", "Leader", "Death"]
        self.objects = {"Gang": ACObject("Gang", 2),
                        "Leader": ACObject("Leader", 2),
                        "Death": ACObject("Death", 2)} # dict of name to value
        self.binary_relations = [gang, shoot] # must get set prior to calling super (), the order follows the order of operations
        super().__init__(frameskip, variant, fixed_limits)
