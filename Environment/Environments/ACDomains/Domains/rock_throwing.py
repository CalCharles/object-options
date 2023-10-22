import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def suzy(objects):
    objects["SuzyHit"].attribute = objects["SuzyThrow"].attribute

def billy(objects):
    objects["BillyHit"].attribute = int(objects["SuzyHit"].attribute == 0 and objects["BillyThrow"].attribute == 1)

def bottle(objects):
    objects["Bottle"].attribute = int(objects["SuzyHit"].attribute or objects["BillyHit"].attribute)

class RockThrowing(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False, cf_states=False):
        self.all_names = ["SuzyHit", "SuzyThrow", "BillyHit", "BillyThrow", "Bottle"]
        self.objects = {"SuzyHit": ACObject("SuzyHit", 2),
                        "SuzyThrow": ACObject("SuzyThrow", 2),
                        "BillyHit": ACObject("BillyHit", 2),
                        "BillyThrow": ACObject("BillyThrow", 2),
                        "Bottle": ACObject("Bottle", 2)} # dict of name to value
        self.binary_relations = [suzy, billy, bottle] # must get set prior to calling super (), the order follows the order of operations
        self.relation_outcome = ["SuzyHit", "BillyHit", "Bottle"]
        self.passive_mask = np.array([0,0,0,0])
        self.outcome_variable = "Bottle"
        super().__init__(frameskip, variant, fixed_limits, cf_states=cf_states)
