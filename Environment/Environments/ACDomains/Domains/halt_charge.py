import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

def corporal(objects):
    objects["Corporal"].attribute = objects["Major"].attribute if objects["Major"].attribute != 2 else objects["Sergeant"].attribute

class HaltCharge(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False, cf_states=False):
        self.all_names = ["Major", "Sergeant", "Corporal"]
        self.objects = {"Major": ACObject("Major", 3),
                        "Sergeant": ACObject("Sergeant", 2),
                        "Corporal": ACObject("Corporal", 2)} # dict of name to value
        self.binary_relations = [corporal] # must get set prior to calling super (), the order follows the order of operations
        self.relation_outcome = ["Corporal"]
        self.passive_mask = np.array([0,0])
        self.outcome_variable = "Corporal"
        super().__init__(frameskip, variant, fixed_limits, cf_states=cf_states)
