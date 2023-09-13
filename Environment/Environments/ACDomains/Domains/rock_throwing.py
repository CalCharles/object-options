import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def suzy(objects):
    objects["SuzyHit"].attribute = objects["SuzyThrow"].attribute

def billy(objects):
    objects["BillyHit"].attribute = int(objects["SuzyHit"].attribute or objects["BillyThrow"].attribute)

def bottle(objects):
    objects["Bottle"].attribute = int(objects["SuzyHit"].attribute or objects["BillyHit"].attribute)

class RockThrowing(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False):
        self.all_names = ["SuzyHit", "SuzyThrow", "BillyHit", "BillyThrow", "Bottle"]
        self.objects = {"SuzyHit": ACObject("SuzyHit", 2),
                        "SuzyThrow": ACObject("SuzyThrow", 2),
                        "BillyHit": ACObject("BillyHit", 2),
                        "BillyThrow": ACObject("BillyThrow", 2),
                        "Bottle": ACObject("Bottle", 2)} # dict of name to value
        self.binary_relations = [suzy, billy, bottle] # must get set prior to calling super (), the order follows the order of operations
        super().__init__(frameskip, variant, fixed_limits)
