import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

mod_variants = {
    "and": [3, [[[0,1],2, "mul", []]], 2],
    "or": [3, [[[0,1],2, "ineq", [1, True, True]]], 2],
    "xor": [3, [[[0,1],2, "add", []]], 2],
} # parents, targets, relation, hyperparams


def create_add_relation(names, modval, params):
    def add_relation(objects):
        objects[names[1]].attribute = sum([objects[name].attribute for name in names[0]]) % modval
    return add_relation

def create_mul_relation(names, modval, params):
    def mul_relation(objects):
        objects[names[1]].attribute = int(np.prod([objects[name].attribute for name in names[0]])) % modval
    return mul_relation

def create_sub_relation(names, modval, params):
    def sub_relation(objects):
        objects[names[1]].attribute = sum([-objects[name].attribute for name in names[0]]) % modval
    return sub_relation

def create_ineq_relation(names, modval, params):
    threshold, equality, greater = params
    def ineq_relation(objects): # only returns binary outputs
        if equality and greater:
            objects[names[1]].attribute = int(sum([objects[name].attribute for name in names[0]]) >= threshold)
        elif greater:
            objects[names[1]].attribute = int(sum([objects[name].attribute for name in names[0]]) > threshold)
        elif equality and not greater:
            objects[names[1]].attribute = int(sum([objects[name].attribute for name in names[0]]) <= threshold)
        elif not greater:
            objects[names[1]].attribute = int(sum([objects[name].attribute for name in names[0]]) < threshold)
    return ineq_relation

def create_neg_relation(names, modval, params):
    def neg_relation(objects):
        objects[names[1]].attribute = (-sum([objects[name].attribute for name in names[0]])) % modval
    return neg_relation


class ModDAG(ACDomain):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False, cf_states=False):
        num, relations, maxval = mod_variants[variant]
        self.maxval = maxval
        
        self.all_names = [chr(ord('@')+ i + 1) for i in range(num)]
        self.objects = {n: ACObject(n, maxval) for n in self.all_names} # dict of name to value
        print(relations[0])
        self.binary_relations = [self.create_relation(*rel) for rel in relations] # must get set prior to calling super (), the order follows the order of operations
        self.relation_outcome = [self.all_names[rel[1]] for rel in relations]
        self.passive_mask = np.zeros(len(self.all_names)-1)
        self.outcome_variable = self.all_names[-1]
        super().__init__(frameskip, variant, fixed_limits, cf_states=cf_states)

    def create_relation(self, parents, target, relation, hyperparams):
        print(parents, self.all_names[parents[0]])
        names = ([self.all_names[p] for p in parents], self.all_names[target]) 
        if relation == "add":
            relation = create_add_relation(names, self.maxval, hyperparams)
        if relation == "sub":
            relation = create_sub_relation(names, self.maxval, hyperparams)
        if relation == "mul":
            relation = create_mul_relation(names, self.maxval, hyperparams)
        if relation == "ineq":
            relation = create_ineq_relation(names, self.maxval, hyperparams)
        if relation == "neg":
            relation = create_neg_relation(names, self.maxval, hyperparams)
        return relation
