import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def outcome(objects):
    if objects["A1"].attribute == objects["A2"].attribute:
        objects["Outcome"].attribute = objects["A2"].attribute
    elif objects["A1"].attribute != objects["A2"].attribute and (objects["A2"].attribute == objects["A3"].attribute == objects["A4"].attribute == objects["A5"].attribute):
        objects["Outcome"].attribute = objects["A1"].attribute
    else:
        objects["Outcome"].attribute = 0 if np.sum([objects["A1"].attribute, objects["A2"].attribute, objects["A3"].attribute, objects["A4"].attribute, objects["A5"].attribute]) < 3 else 1

class Voting(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False):
        self.all_names = ["A1", "A2", "A3", "A4", "A5", "Outcome"]
        self.objects = {"A1": ACObject("A1", 2),
                        "A2": ACObject("A2", 2),
                        "A3": ACObject("A3", 2),
                        "A4": ACObject("A4", 2),
                        "A5": ACObject("A5", 2),
                        "Outcome": ACObject("Outcome", 2)} # dict of name to value
        self.binary_relations = [suzy, billy, bottle] # must get set prior to calling super (), the order follows the order of operations
        super().__init__(frameskip, variant, fixed_limits)
