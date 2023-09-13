# Algorithm 0:
from itertools import combinations, chain
import numpy as np
from Environment.Environments.Pusher1D.pusher1D import Pusher1D
from Environment.Environments.ACDomains.Domains.forest_fire import ForestFire
import sys

def all_subsets(n):
    return list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))

def hash_vector(vals): # handles state values up to 10
    tv = 0
    for i, v in enumerate(vals):
        tv += v * np.power(10, i)
    return tv

def compute_possible(environment):
    all_binaries = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,environment.num_objects)
    all_states = environment.all_states
    outcomes = environment.outcomes
    
    
    # get all valid binary assignments, a binary is invalid if the same input maps to different outcomes
    subsets = all_subsets(len(all_states))
    all_state_combinations = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,environment.num_objects)
    valid_subsets = list()
    for binary in all_binaries:
        bin_valid = list()
        for subset in subsets:
            outcome_map = dict()
            valid = True
            inval = None
            for i in subset:
                inval = hash_vector(all_states[i] * binary)
                if inval in outcome_map:
                    if outcome_map[inval] != outcomes[i]:
                        valid=False
                        break
                else:
                    outcome_map[inval] = outcomes[i]
            if valid: print(binary, valid, outcome_map, subset, inval, )
            if valid:
                bin_valid.append(subset)
        valid_subsets.append(bin_valid)
    # print(valid_subsets, len(valid_subsets), [len(vss) for vss in valid_subsets])
    subset_indices = [np.arange(len(vss)) for vss in valid_subsets]
    # print(subset_indices)
    all_combinations = np.array(np.meshgrid(*valid_subsets)).T.reshape(-1,len(valid_subsets))
    # all_combinations = np.array(np.meshgrid(*subset_indices)).T.reshape(-1,len(valid_subsets))
    # print(all_combinations)

    def check_valid(comb, num): # checks if a combination of binary assignments is valid
        print(comb)
        for i in range(len(comb)-1):
            for j in range(i+1,len(comb)):
                # print(i, j, comb[i], comb[j], all(np.isin(np.array(comb[i]),np.array(comb[j]))))
                if any(np.isin(comb[i],comb[j])): # overlapping subsets are invalid
                    return False
        all_covered_states = list(set(np.array(comb).flatten()))
        all_covered_states = sum([list(ac) for ac in all_covered_states if len(ac) > 0], start=list())
        all_covered_states.sort()
        # if comb[0] == (0,2,3): print(comb, all_covered_states)
        # print(all_covered_states)
        return len(all_covered_states) == len(all_states)


    valid_combinations = list()
    print("all combinations", len(all_combinations))
    for comb in all_combinations:
        if check_valid(comb, len(all_states)):
            valid_combinations.append(comb)
            print(len(valid_combinations), len(all_combinations))
    cost = list()
    for valid_combination in valid_combinations:
        cost.append(np.sum(np.array([np.sum(bin) for bin in all_binaries]) * np.array([len(c) for c in valid_combination])))
        # print(valid_combination, cost[-1])
    min_cost = min(cost)
    print("min cost combinations")
    for valid_combination, c in zip(valid_combinations, cost):
        if c == min_cost:
            print(valid_combination, c)


if __name__ == '__main__':
    env_name = sys.argv[1]
    print(env_name)
    if env_name == "Pusher1D":
        env = Pusher1D()
    elif env_name == "ForestFire":
        env = ForestFire()
    compute_possible(env) 