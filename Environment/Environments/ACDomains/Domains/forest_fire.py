import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def check_fire(objects):
    if objects["RainApril"].attribute == 0 and objects["ElectricalMay"].attribute == 1:
        objects["Fire"].attribute = 2
    elif objects["ElectricalJune"].attribute == 1:
        objects["Fire"].attribute = 1
    else:
        objects["Fire"].attribute = 0

class ForestFire(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False):
        self.all_names = ["Fire", "RainApirl", "ElectricalMay", "EletricalJune"]
        self.objects = {"Fire": ACObject("Fire", 3),
                        "RainApril": ACObject("RainApril", 2),
                        "ElectricalMay": ACObject("ElectricalMay", 2),
                        "ElectricalJune": ACObject("ElectricalJune", 2)} # dict of name to value
        self.binary_relations = [check_fire] # must get set prior to calling super (), the order follows the order of operations
        self.outcome_variable = "Fire"
        super().__init__(frameskip, variant, fixed_limits)
