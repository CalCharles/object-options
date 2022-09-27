import numpy as np
import string
from Record.file_management import load_from_pickle, save_to_pickle, create_directory
from State.feature_selector import construct_object_selector_dict, broadcast
from train_interaction import init_names
from Causal.Utils.interaction_selectors import CausalExtractor
from Environment.Normalization.norm import MappedNorm, NormalizationModule

  # Normalization:
  #   input: random ranges, random state shapes, mapped or not mapped
  #   output: correct normalization

norm_names = ["target", "inter", "parent", "additional", "additional_part", "diff", "dyn", "rel"]

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(letters[np.random.randint(len(letters))] for i in range(length))
    return result_str

def generate_random_dicts(selection): # creates a 5 object module
    all_object_names = [get_random_string(np.random.randint(7) + 3) for i in range(5)]
    object_counts = {n: np.random.randint(5) + 1 for n in all_object_names}
    inter_names = [n for n in np.random.choice(all_object_names, size=3)]
    while inter_names[0] in inter_names[1:len(inter_names)-1]:
        inter_names = [n for n in np.random.choice(all_object_names, size=3)]
    object_counts[inter_names[0]] = 1 # only allow single instanced parents
    dims = {n: np.random.randint(5) + 1 for n in all_object_names}
    dims[inter_names[1]] = 1
    dims[inter_names[0]] = dims[inter_names[-1]]
    object_names = init_names(inter_names)
    lim_dict = {n: (np.random.rand(dims[n])*5 - 10, -np.random.rand(dims[n]) * 5 + 10) for n in all_object_names}
    dynamics_dict = {n: (np.random.rand(dims[n]) - 1, np.random.rand(dims[n]) + 1) for n in all_object_names}
    save_to_pickle(create_directory("Testing/testing_data/norm_random_dicts"+str(selection)+".pkl", drop_last=True), (lim_dict, dynamics_dict, object_names, object_counts, inter_names))
    save_to_pickle(create_directory("Testing/testing_data/norm_random_dims"+str(selection)+".pkl", drop_last=True), dims)
    return all_object_names, (lim_dict, dynamics_dict, object_names, object_counts, inter_names), dims

def generate_mapped_dicts(names, dims, selection):
    name = np.random.choice(names)
    mask = np.array([np.random.randint(2) for i in range(dims[name])])
    while np.sum(mask) == 0:
        mask = np.array([np.random.randint(2) for i in range(dims[name])])

    save_to_pickle(create_directory("Testing/testing_data/mapped_random_dims"+str(selection)+".pkl", drop_last=True), (name, mask))
    return (name, mask)

def create_dummy_data(names, dicts, dims):
    lim_dict = dicts[0]
    full_input = dict()
    for n in names:
        if dicts[3][n] > 1:
            for j in range(dicts[3][n]):
                full_input[n + str(j)] = np.array([np.random.uniform(lim_dict[n][0][k], lim_dict[n][1][k]) for k in range(len(lim_dict[n][0]))])
        else: full_input[n] = np.array([np.random.uniform(lim_dict[n][0][k], lim_dict[n][1][k]) for k in range(len(lim_dict[n][0]))])
    return full_input

class target_select:
    def __init__(self, norm_selectors):
        self.norm_selectors = norm_selectors

    def __call__(self, lx, x):
        return self.norm_selectors["target"](x) - self.norm_selectors["target"](lx)

class rel_select:
    def __init__(self, norm_selectors, target_num):
        self.norm_selectors = norm_selectors
        self.target_num = target_num

    def __call__(self, x):
        return broadcast(self.norm_selectors["parent"](x), self.target_num) - self.norm_selectors["target"](x)

def generate_norm_inputs(names, dicts, dims, target, mask, selection):
    lim_dict = dicts[0]
    mask = np.array([np.random.randint(2) for i in range(dims[name])])
    mapped_norm = MappedNorm(dicts[0], dicts[1], target, mask)
    mapped_selector = construct_object_selector_dict([target], dims, dicts[3], masks=[mask])
    norm = NormalizationModule(*dicts)
    norm_selectors = {"target": construct_object_selector_dict([dicts[-1][-1]], dims, dicts[3]),
                        "inter": construct_object_selector_dict(dicts[-1], dims, dicts[3]),
                        "parent": construct_object_selector_dict([dicts[-1][0]], dims, dicts[3]),
                        "additional": construct_object_selector_dict(dicts[-1][1:len(dicts[-1])-1], dims, dicts[3]),
                        "additional_part": [construct_object_selector_dict([dicts[-1][1+i]], dims, dicts[3]) for i in range(len(dicts[-1]) - 2)]}
    norm_selectors["diff"] = target_select(norm_selectors)
    norm_selectors["dyn"] = target_select(norm_selectors)
    norm_selectors["rel"] = rel_select(norm_selectors, dicts[3][dicts[-1][-1]])
    norm_selectors["map"] = mapped_selector
    last_inputs, full_inputs, targets, mapped_inputs, mapped_normed = list(), list(), {n: list() for n in norm_names}, list(), list()
    print(dims, dicts)
    for i in range(50):
        full_input = create_dummy_data(names, dicts, dims)
        last_input = create_dummy_data(names, dicts, dims)
        full_inputs.append(full_input)
        last_inputs.append(last_input)
        for n in norm_names:
            if n == "additional_part":
                targets[n].append([norm(adisel(full_input), form="additional" + str(i)) for (i, adisel) in enumerate(norm_selectors[n])])
            elif n == "dyn" or n == "diff":
                targets[n].append(norm(norm_selectors[n](last_input, full_input), form=n))                
            else:
                targets[n].append(norm(norm_selectors[n](full_input), form=n))
        ms = mapped_selector(full_input)
        mapped_inputs.append(ms)
        mapped_normed.append(mapped_norm(ms))
    print(targets, mapped_normed)
    save_to_pickle(create_directory("Testing/testing_data/selectors"+str(selection)+".pkl", drop_last=True), norm_selectors)
    save_to_pickle(create_directory("Testing/testing_data/norm_inputs"+str(selection)+".pkl", drop_last=True), (last_inputs, full_inputs, targets))
    save_to_pickle(create_directory("Testing/testing_data/mapped_norm_inputs"+str(selection)+".pkl", drop_last=True), (mapped_inputs, mapped_normed))
    return full_inputs, targets, mapped_inputs, mapped_normed, norm_selectors

def generate_norm_env_inputs(args):
    object_names = init_names(args)
    environment, record = initialize_environment(args.environment, args.record)
    lim_dict = environment.object_range
    mask = np.array([np.random.randint(2) for i in range(environment.object_sizes[name])])
    while np.sum(mask) == 0:
        mask = np.array([np.random.randint(2) for i in range(environment.object_sizes[name])])
    save_to_pickle(create_directory("Testing/testing_data/mapped_random_dims_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl", drop_last=True), (object_names.target, mask))
    mapped_norm = MappedNorm(environment.object_range, environment.object_dynamics, target, mask)
    mapped_selector = construct_object_selector_dict([target], dims, environment.object_instanced, masks=[mask])
    norm = NormalizationModule(environment.object_range, environment.object_dynamics, object_names, environment.object_instanced, mask)
    norm_selectors = {"target": construct_object_selector_dict([object_names.target], dims, environment.object_instanced),
                        "inter": construct_object_selector_dict(object_names.inter_names, dims, environment.object_instanced),
                        "parent": construct_object_selector_dict([object_names.primary_parent], dims, environment.object_instanced),
                        "additional": construct_object_selector_dict(object_names.additional, dims, environment.object_instanced),
                        "additional_part": [construct_object_selector_dict([a], dims, environment.object_instanced) for a in object_names.additional]}
    norm_selectors["diff"] = target_select(norm_selectors)
    norm_selectors["dyn"] = target_select(norm_selectors)
    norm_selectors["rel"] = rel_select(norm_selectors, environment.object_instanced[object_names.inter_names[-1]])
    norm_selectors["map"] = mapped_selector
    last_inputs, full_inputs, targets, mapped_inputs, mapped_normed = list(), list(), {n: list() for n in norm_names}, list(), list()
    print(dims, dicts)
    inp_path, out_path = "Testing/testing_data/trajectory_inputs_"+environment.name+ "_" + environment.variant + "_" + object_names.target +  ".pkl", "Testing/testing_data/trajectory_outputs_"+environment.name+"_" + environment.variant + "_" + object_names.target + ".pkl"
    full_states = load_from_pickle(inp_path)
    for last_input, full_input in zip(full_states, full_states[1:]):
        full_inputs.append(full_input)
        last_inputs.append(last_input)
        for n in norm_names:
            if n == "additional_part":
                targets[n].append([norm(adisel(full_input), form="additional" + str(i)) for (i, adisel) in enumerate(norm_selectors[n])])
            elif n == "dyn" or n == "diff":
                targets[n].append(norm(norm_selectors[n](last_input, full_input), form=n))                
            else:
                targets[n].append(norm(norm_selectors[n](full_input), form=n))
        ms = mapped_selector(full_input)
        mapped_inputs.append(ms)
        mapped_normed.append(mapped_norm(ms))
    print(targets, mapped_normed)
    save_to_pickle(create_directory("Testing/testing_data/selectors_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl", drop_last=True), norm_selectors)
    save_to_pickle(create_directory("Testing/testing_data/norm_inputs_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl", drop_last=True), (last_inputs, full_inputs, targets))
    save_to_pickle(create_directory("Testing/testing_data/mapped_norm_inputs_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl", drop_last=True), (mapped_inputs, mapped_normed))
    return full_inputs, targets, mapped_inputs, mapped_normed, norm_selectors


def get_diffs(inp, cor_norm, norm, norm_name, mapped):
    norm_diff = np.linalg.norm(norm(inp, form=norm_name) - cor_norm) if mapped is None else np.linalg.norm(norm(inp) - cor_norm)
    mpv = inp if mapped is None else mapped
    denorm_diff = np.linalg.norm(norm.reverse(norm(inp, form=norm_name), form=norm_name) - mpv) if mapped is None else np.linalg.norm(norm.reverse(norm(inp)) - mpv)
    return norm_diff, denorm_diff

def check_norm_denorm(norm, norm_name, last_inputs, inputs, correct_normalization, selector, mapped=None):
    norm_diffs, denorm_diffs = list(), list()
    for i, (linp, inp, cor_norm) in enumerate(zip(last_inputs, inputs, correct_normalization)):
        if norm_name == "additional_part":
            for i in range(len(selector)):
                use_name = "additional" + str(i)
                inp = selector[i](inp)
                print(cor_norm)
                norm_diff, denorm_diff = get_diffs(inp, cor_norm, norm, use_name, None)
                norm_diffs.append(norm_diff)
                denorm_diffs.append(denorm_diff)
        else:
            if norm_name == "dyn" or norm_name == "diff": inp = selector(linp, inp)
            else: inp = selector(inp)
            norm_diff, denorm_diff = get_diffs(inp, cor_norm, norm, norm_name, mapped[i] if mapped is not None else None)
            norm_diffs.append(norm_diff)
            denorm_diffs.append(denorm_diff)
    return norm_diffs, denorm_diffs

def test_normalization(selection):
    dims = load_from_pickle("Testing/testing_data/norm_random_dims" + str(selection) + ".pkl")
    random_dicts = load_from_pickle("Testing/testing_data/norm_random_dicts" + str(selection) + ".pkl")
    last_inputs, inputs, correct_normalization = load_from_pickle("Testing/testing_data/norm_inputs" + str(selection) + ".pkl")
    selectors = load_from_pickle("Testing/testing_data/selectors" + str(selection) + ".pkl")
    norm = NormalizationModule(*random_dicts)

    norm_diffs, denorm_diffs = list(), list()
    for norm_name in norm_names:
        # if norm_name == "additional_part" and random_dicts[-1]:
        norm_diff, denorm_diff = check_norm_denorm(norm, norm_name, last_inputs, inputs, correct_normalization[norm_name], selectors[norm_name])
        print(norm_name, norm_diff)
        norm_diffs.append(norm_diff)
        denorm_diffs.append(denorm_diff)

    target, mask = load_from_pickle("Testing/testing_data/mapped_random_dims" + str(selection) + ".pkl")
    mapped_norm = MappedNorm(random_dicts[0], random_dicts[1], target, mask)
    mapped_denorm, mapped_correct_normalization = load_from_pickle("Testing/testing_data/mapped_norm_inputs" + str(selection) + ".pkl")
    mapped_norm_diff, mapped_denorm_diff = check_norm_denorm(mapped_norm, "mapped", last_inputs, inputs, mapped_correct_normalization, selectors["map"], mapped_denorm)
    return np.sum(norm_diffs), np.sum(denorm_diffs), np.sum(mapped_norm_diff), np.sum(mapped_denorm_diff)

def test_normalization_env(args):
    object_names = init_names(args)
    environment, record = initialize_environment(args.environment, args.record)
    last_inputs, inputs, correct_normalization = load_from_pickle("Testing/testing_data/norm_inputs_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl")
    selectors = load_from_pickle("Testing/testing_data/selectors" + str(selection) + ".pkl")
    norm = NormalizationModule(*random_dicts)

    norm_diffs, denorm_diffs = list(), list()
    for norm_name in norm_names:
        # if norm_name == "additional_part" and random_dicts[-1]:
        norm_diff, denorm_diff = check_norm_denorm(norm, norm_name, last_inputs, inputs, correct_normalization[norm_name], selectors[norm_name])
        norm_diffs.append(norm_diff)
        denorm_diffs.append(denorm_diff)

    target, mask = load_from_pickle("Testing/testing_data/mapped_random_dims_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl")
    mapped_norm = MappedNorm(random_dicts[0], random_dicts[1], target, mask)
    mapped_denorm, mapped_correct_normalization = load_from_pickle("Testing/testing_data/mapped_norm_inputs_" + args.environment + "_" + args.variant + "_" + object_names.target + ".pkl")
    mapped_norm_diff, mapped_denorm_diff = check_norm_denorm(mapped_norm, "mapped", last_inputs, inputs, mapped_correct_normalization, selectors["map"], mapped_denorm)
    return np.sum(norm_diffs), np.sum(denorm_diffs), np.sum(mapped_norm_diff), np.sum(mapped_denorm_diff)

