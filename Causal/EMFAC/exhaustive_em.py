# Algorithm 0:
from itertools import combinations, chain
import collections
import numpy as np
from Environment.Environments.Pusher1D.pusher1D import Pusher1D
from Environment.Environments.ACDomains.Domains.forest_fire import ForestFire
from Environment.Environments.ACDomains.Domains.rock_throwing import RockThrowing
from Environment.Environments.ACDomains.Domains.gang_shoot import GangShoot
from Environment.Environments.ACDomains.Domains.halt_charge import HaltCharge
from Environment.Environments.ACDomains.Domains.train import Train
from Environment.Environments.ACDomains.Domains.voting import Voting
from Environment.Environments.ACDomains.Domains.mod_DAG import ModDAG
import sys
import time

def get_all_subsets(n):
    # returns all combinations up to length n, including the empty set
    return list(chain(*[combinations(range(n), ni) for ni in range(n+1)]))

def hash_vector(vals): # handles state values up to 10
    tv = 0
    for i, v in enumerate(vals):
        tv += v * np.power(10, i)
    return tv

def compute_possible(environment):
    # a binary includes the object, or does not
    all_binaries = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,environment.num_objects)
    all_states = environment.all_states
    outcomes = environment.outcomes
    
    
    # get all valid binary assignments, a binary is invalid if the same input maps to different outcomes
    subsets = get_all_subsets(len(all_states))
    # all_state_combinations = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,environment.num_objects)
    valid_subsets = list()
    # try out every binary
    for binary in all_binaries:
        bin_valid = list()
        # try out every subset of states
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
        # print(comb)
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
    for i, comb in enumerate(all_combinations):
        start = time.time()
        if check_valid(comb, len(all_states)):
            valid_combinations.append(comb)
            print(i, len(valid_combinations), len(all_combinations))
    cost = list()
    for valid_combination in valid_combinations:
        cost.append(np.sum(np.array([np.sum(bin) for bin in all_binaries]) * np.array([len(c) for c in valid_combination])))
        # print(valid_combination, cost[-1])
    min_cost = min(cost)
    print("min cost combinations")
    for valid_combination, c in zip(valid_combinations, cost):
        if c == min_cost:
            print(valid_combination, c)

def partition(list_, k):
    """
    partition([1, 2, 3, 4], 2) -> [[1], [2, 3, 4]], [[1, 2], [3, 4]], ..., [[1, 3], [2, 4]], [[3], [1, 2, 4]]
    """
    if k == 1:  # base case 1: k == 1, just yield itself as a list
        yield [list_]
    elif k == len(list_):  # base case 2: k == len(list_), yield each item in the list wrapped in a list
        yield [[s] for s in list_]
    else:
        head, *tail = list_  # head = the first element, tail = the rest
        for p in partition(tail, k-1):  # case 1: head -> 1, partition(tail, k-1) -> k-1.
                                        # head + partition(tail, k-1) -> 1+k-1 -> k
            yield [[head], *p]
        for p in partition(tail, k):  # case 2: head -> 1, partition(tail, k) -> k.
                                      # bloat x to [e1, e2, e3] -> [[x+e1, e2, e3], [e1, x+e2, e3], [e1, e2, x+e3]]
            for i in range(len(p)):
                yield p[:i] + [[head] + p[i]] + p[i+1:]  # bloat head to partition(tail, k) -> k

def get_all_disjoint_sets(iterable):
    l = list(iterable)
    return chain.from_iterable(list(partition(l, i)) for i in range(1, len(l)+1))

global counter

def binary_state_compatibility(all_binaries, all_states, environment):
    # returns the compatibility (measure of necessity) between the
    # binary and every other state
    cost = 0
    compatibility = dict()
    for i, binary in enumerate(all_binaries):
        compatibility[i] = list()
        for j, state in enumerate(all_states):
            pos_comp, neg_comp, cf_cost = environment.evaluate_split_counterfactuals(binary, state)
            compatibility[i].append((pos_comp, neg_comp))
            cost += cf_cost
    print(cost)
    return compatibility

def compute_possible_efficient(environment, compatibility_constant):
    # a binary includes the object, or does not
    all_binaries = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,environment.num_objects)
    use_zero = environment.use_zero
    if not use_zero: all_binaries = all_binaries[1:]
    all_states = environment.all_states
    outcomes = environment.outcomes
    passive_mask = environment.passive_mask
    all_subsets = get_all_subsets(len(all_states))
    
    compatibility = binary_state_compatibility(all_binaries, all_states, environment)
    for k in compatibility.keys():
        for s, c in enumerate(compatibility[k]):
            print(all_binaries[int(k)], all_states[s], c)

    print(np.concatenate([np.array(all_states), np.expand_dims(np.array(outcomes), axis=-1)], axis=-1))

    def check_valid(subset, binaries):
        # check with binaries are compatible with the given subset
        valid_binaries = list()
        for i, binary in enumerate(all_binaries):
            # check if binary is compatible with the following subset
            binary_check = dict()
            invalid = False
            for s in subset:
                # state of the form [k_0, \hdots, k_n] where n is the number of factors, k_i is the number of discrete values factor i can
                factored_state = all_states[s]
                # convert to a tuple where the binary sets certain values to 0
                masked_factored_state = tuple((factored_state * binary).tolist())
                if masked_factored_state in binary_check:
                    if binary_check[masked_factored_state] != outcomes[s]:
                        # invalid if the same masked state has at least two different outcomes
                        invalid = True
                        break
                else: # assign the masked state
                    binary_check[masked_factored_state] = outcomes[s]
            if not invalid: # append the index of the binary
                valid_binaries.append(i)
        return valid_binaries
    
    def check_compatible(subset, binaries, compatibility, compatibility_constant):
        if compatibility_constant < 0: return binaries # don't use compatibility if constant negative
        valid_binaries = list()
        for i in binaries:# binaries identified by index
            compatible = compatibility[i]
            subset_compatible = True
            for s in subset:
                if compatible[s][0] < compatibility_constant or compatible[s][1] > (1-compatibility_constant):
                    subset_compatible = False
                    break
            if subset_compatible: valid_binaries.append(i)
        return valid_binaries
    # create a mapping of every subset to each of its valid binaries
    subset_binary = dict()
    subset_index_mapping = dict()
    print("num subsets", len(list(all_subsets)))
    print("num binaries", len(all_binaries))
    for i, subset in enumerate(all_subsets):
        subset_binary[i] = check_compatible(subset, check_valid(subset, all_binaries), compatibility, compatibility_constant)
        subset_index_mapping[tuple(subset)] = i
    # create all disjoint, complete partitionings of the subsets
    disjoint_sets = list(get_all_disjoint_sets(range(len(all_states))))
    print("num disjoint", len(disjoint_sets))
    print("num valid binaries", len(list(subset_binary.keys())))

    def all_assignments(disjoint_subset, unusable_binaries): # returns all mappings of unusable binaries to a particular disjoint subset
        if len(disjoint_subset) == 1:
            subset_valid = subset_binary[subset_index_mapping[tuple(disjoint_subset[0])]]
            usable_binaries = set(subset_valid) - set(unusable_binaries)
            return [[ub] for ub in usable_binaries], [[subset_index_mapping[tuple(disjoint_subset[0])]] for _ in usable_binaries], len(usable_binaries)
        else:
            counter = 0
            subset_valid = subset_binary[subset_index_mapping[tuple(disjoint_subset[0])]]
            usable_binaries = set(subset_valid) - set(unusable_binaries)
            # print("ss", subset_valid, usable_binaries, unusable_binaries)
            binary_assn = list()
            subset_assn = list()
            for b in usable_binaries: # append each usable bin to the front
                rem_bin, rem_subset, ctr = all_assignments(disjoint_subset[1:], unusable_binaries + [b])
                counter += ctr
                for rb, rs in zip(rem_bin, rem_subset):
                    counter += 1
                    binary_assn.append([b] + rb)
                    subset_assn.append([subset_index_mapping[tuple(disjoint_subset[0])]] + rs)
                    # print([b] + assign)
            return binary_assn, subset_assn, counter

    # for each disjoint set, find all valid assignments of binary to set
    assigned_binaries = list()
    assigned_subsets = list()
    counter = 0
    for ds in disjoint_sets:
        ab, asub, count  = all_assignments(ds, [])
        counter += count
        assigned_binaries += ab
        assigned_subsets += asub
    print("cost counter, number of assigned binaries", counter, len(assigned_binaries))
    
    cost = list()
    for ab, asub in zip(assigned_binaries, assigned_subsets):
        # print(ab, convert_subset(ab, all_binaries), convert_subset(asub, all_subsets), np.sum(np.sum(convert_subset(ab, all_binaries), axis=-1) * np.array([len(all_subsets[c]) for c in asub])))
        cost.append(np.sum(np.array(np.sum(np.abs(convert_subset(ab, all_binaries) - passive_mask), axis=-1)) * np.array([len(all_subsets[c]) for c in asub])))
        # print(valid_combination, cost[-1])
    min_cost = min(cost)
    print("min cost combinations")
    cost_counter = collections.Counter()
    min_cost_strings = list()
    for ab, asub, c in zip(assigned_binaries, assigned_subsets, cost):
        if c == min_cost:
            print(len(ab), np.array(convert_subset(ab, all_binaries)), [(np.array(convert_subset(ss, all_states)), np.array(convert_subset(ss, outcomes))) for ss in convert_subset(asub, all_subsets)], c)
            print(convert_subset(asub, all_subsets))
            print([np.array(convert_subset(ss, outcomes)) for ss in convert_subset(asub, all_subsets)])
            outcomes = [np.array(convert_subset(ss, outcomes)) for ss in convert_subset(asub, all_subsets)]
            states = [np.array(convert_subset(ss, all_states)) for ss in convert_subset(asub, all_subsets)]
            state_outcomes = [s.tolist() + [o] for o in zip(state, outcome)  for state, outcome in zip(states, outcomes)]
            print(state_outcomes)
            # print([ss.tolist() + [0] for ss, o in list(zip(np.array(convert_subset(convert_subset(asub, all_subsets)[0], all_states)), np.array(convert_subset(convert_subset(asub, all_subsets)[0], outcomes))))])
            subset_outcome_strings = [",".join([ss.tolist() + [0] for ss, o in list(zip(np.array(convert_subset(ss, all_states)), np.array(convert_subset(ss, outcomes))))]) for ss in convert_subset(asub, all_subsets)]
            zipped_binary_sso_strings = ["".join(bn) + ssos for bn, ssos in zip(np.array(convert_subset(ab, all_binaries)), subset_outcome_strings)]
            print(";".join(zipped_binary_sso_strings))
            min_cost_strings.append(";".join(zipped_binary_sso_strings))
        cost_counter[c] += 1
    costs = [i for i in cost_counter.items()]
    costs.sort(key=lambda x: x[0])
    print("num per cost", costs)



def compute_normality_binaries(environment):
    all_binaries = np.array(np.meshgrid(*[[0,1] for i in range(environment.num_objects)])).T.reshape(-1,environment.num_objects)
    use_zero = environment.use_zero
    if not use_zero: all_binaries = all_binaries[1:]
    all_states = environment.all_states
    outcomes = environment.outcomes
    passive_mask = environment.passive_mask
    all_subsets = get_all_subsets(len(all_states))
    
    # assigns each binary-state pair with a splitting value, indexed by binary
    compatibility = binary_state_compatibility(all_binaries, all_states, environment)

    # Find a minimum cost valid binary for each state
    


def convert_subset(subset, all_subsets, sort = False):
    if sort: subset.sort()
    return [all_subsets[s] for s in subset]



if __name__ == '__main__':
    env_name = sys.argv[1]
    compatibility_constant = float(sys.argv[2]) if len(sys.argv) == 3 else -1
    variant = sys.argv[3] if len(sys.argv) == 4 else ""
    print(env_name)
    if env_name == "Pusher1D":
        env = Pusher1D()
    elif env_name == "ForestFire":
        env = ForestFire()
    elif env_name == "RockThrowing":
        env = RockThrowing()
    elif env_name == "GangShoot":
        env = GangShoot()
    elif env_name == "HaltCharge":
        env = HaltCharge()
    elif env_name == "Train":
        env = Train()
    elif env_name == "Voting":
        env = Voting()
    elif env_name == "ModDAG":
        env = ModDAG(variant=variant)
    compute_possible_efficient(env, compatibility_constant) 