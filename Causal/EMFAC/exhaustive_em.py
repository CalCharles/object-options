# Algorithm 0:
from itertools import combinations, chain
import numpy as np
from Environment.Environments.Pusher1D.pusher1D import Pusher1D 

def allsubsets(n):
    return list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))

def hash_vector(vals): # handles state values up to 10
    tv = 0
    for i, v in enumerate(vals):
        tv += v * np.pow(10, i)
    return tv

def compute_possible(environment):
    all_binaries = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,3)
    all_states = environment.all_states
    outcomes = environment.outcomes
    
    
    # get all valid binary assignments, a binary is invalid if the same input maps to different outcomes
    subsets = all_subsets(len(all_states))
    all_state_combinations = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,3)
    for binary in all_binaries:
        bin_valid = list()
        for subset in all_subsets:
            outcome_map = set()
            valid = True
            for i in subset:
                inval = hash_vector(all_states[i] * binary)
                if inval in outcome_map:
                    if outcome_map[inval] != outcomes[i]:
                        valid=False
                        break
                else:
                    outcome_map[inval] = outcomes[i]
            if valid:
                bin_valid.append(subset)
        valid_subsets.append(bin_valid)
    
    all_combinations = np.array(np.meshgrid(*[ss for ss in valid_subsets])).T.reshape(-1,3)

    def check_valid(comb, num): # checks if a combination of binary assignments is valid
        for i in range(len(comb)):
            for j in range(i,len(comb)):
                if all(np.isin(comb[i],comb[j])): # overlapping subsets are invalid
                    return False
        all_covered_states = list(set(np.array(comb).flatten()))
        all_covered_states.sort()
        return np.linalg.norm(all_covered_states - np.arange(num)) == 0


    valid_combinations = list()
    for comb in all_combinations:
        if check_valid(comb, len(all_states)):
            valid_combinations.append(comb)
    print(valid_combinations)    

if __name__ == '__main__':
    env = Pusher1D()
    compute_possible(env) 