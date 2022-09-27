import numpy as np
import string
from Record.file_management import load_from_pickle, save_to_pickle, create_directory, read_obj_dumps, read_action_dumps
from train_interaction import init_names
from Causal.Utils.interaction_selectors import CausalExtractor
from State.object_dict import ObjDict
from Environment.Normalization.norm import MappedNorm, NormalizationModule
from Testing.normalization_test import create_dummy_data
from Environment.Environments.initialize_environment import initialize_environment

from State.state_extractor import StateExtractor
# State Extractor:
#   input: Randomly constructed state spaces with random state extraction selections
#   output: check that selection and relative values are correct with normalization

def load_trajectory(args):
    names = init_names(args.train.train_edge)
    trajectory_inputs = read_obj_dumps(args.train.load_rollouts, i= 0, rng=1000)
    trajectory_inputs = [{n: np.array(ti[n]) for n in ti} for ti in trajectory_inputs]
    trajectory_inputs = [{"factored_state": ti} for ti in trajectory_inputs]
    action_inputs, _, _ = read_action_dumps(args.train.load_rollouts, i=0, rng=1000, indexed=False)
    params, _, _ = read_action_dumps(args.train.load_rollouts, i=0, rng=1000, filename="param_dumps.txt", indexed=True)
    masks, _, _ = read_action_dumps(args.train.load_rollouts, i=0, rng=1000, filename="mask_dumps.txt", indexed=True)
    save_to_pickle("Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl", trajectory_inputs)
    save_to_pickle("Testing/testing_data/trajectory_action_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl", action_inputs)
    save_to_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + names.target + ".pkl", (params, masks))

def create_mask(dims, target):
    tar_shape = (dims[target], )
    mask = np.random.randint(2, size = tar_shape)
    while np.sum(mask) == 0:
        mask = np.random.randint(2, size = tar_shape)
    return mask

def generate_extractor(args, single_obs_setting, relative_obs_setting, setting, n_setting, normed=True):
    object_sizes = load_from_pickle("Testing/testing_data/norm_random_dims" + str(n_setting) + ".pkl")
    random_dicts = load_from_pickle("Testing/testing_data/norm_random_dicts" + str(n_setting) + ".pkl")
    # random_dicts = lim_dict, dynamics_dict, object_names, object_counts, inter_names
    if normed: 
        environment = ObjDict({"object_sizes": object_sizes, "object_instanced": random_dicts[3]})
        args.object_names = random_dicts[2]
        args.extract.inter_extractor = CausalExtractor(args.object_names, environment)
    else: 
        environment, record = initialize_environment(args.environment, args.record)
        args.object_names = init_names(args.train.train_edge)
        args.extract.inter_extractor = CausalExtractor(args.object_names, environment)
    if normed: args.norm = NormalizationModule(*random_dicts)
    else: args.norm = NormalizationModule(environment.object_range, environment.object_dynamics, args.object_names, environment.object_instanced, args.extract.inter_extractor.active)
    save_to_pickle("Testing/testing_data/norm_module" + str(n_setting) + ".pkl", args.norm)
    args.extract.object_names = args.object_names
    args.target_select, args.full_parent_select, args.additional_select, args.additional_selectors, \
            args.padi_selector, args.parent_select, args.inter_select = args.extract.inter_extractor.get_selectors()

    args.extract.single_obs_setting = single_obs_setting
    args.extract.relative_obs_setting = relative_obs_setting
    args.extract.norm = args.norm
    state_extractor = StateExtractor(args.extract)

    if normed: 
        last_inputs, inputs, normalized_inputs = load_from_pickle("Testing/testing_data/norm_inputs" + str(n_setting) + ".pkl")
        last_inputs = [{'factored_state': inp} for inp in last_inputs]
        inputs = [{'factored_state': inp} for inp in inputs]
    else: 
        last_inputs = load_from_pickle("Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
        last_inputs = [{'factored_state': inp} for inp in last_inputs]
        inputs = last_inputs[1:]
        actions = load_from_pickle("Testing/testing_data/trajectory_action_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
    if normed:
        params, assign_params, masks = list(), list(), list()
        for i in range(len(inputs)):
            select, sze = np.random.randint(random_dicts[3][args.object_names.target]), object_sizes[args.object_names.target]
            params.append(args.target_select(create_dummy_data(list(object_sizes.keys()), random_dicts, object_sizes))[...,select * sze: (select + 1) * sze])
            assign_params.append(args.target_select(create_dummy_data(list(object_sizes.keys()), random_dicts, object_sizes))[...,select * sze: (select + 1) * sze])
        params, assign_params =  np.stack(params, axis=0), np.stack(assign_params, axis=0)
        masks = np.stack([create_mask(object_sizes, args.object_names.target) for _ in range(len(inputs))],axis=0)
        assign_masks = masks
        save_to_pickle("Testing/testing_data/extractor_inputs" + str(setting) +"_" + str(n_setting)+ ".pkl", (params, assign_params, masks, assign_masks))
    else:
        params, masks = load_from_pickle("Testing/testing_data/trajectory_param_mask_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
        params, masks = np.array(params), np.array(masks)
        assign_params = np.stack([np.random.rand(*args.norm.lim_dict[args.object_names.target][0].shape) * args.norm.norm_dict[args.object_names.target][1] + args.norm.lim_dict[args.object_names.target][0] for _ in range(len(params))], axis=0)
        assign_masks = masks
        save_to_pickle("Testing/testing_data/extractor_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl", (params, assign_params, masks, assign_masks))
    values = list()
    for lfs, inp, p, ap, m, am in zip(last_inputs, inputs, params, assign_params, masks, assign_masks):
        obs = state_extractor.get_obs(lfs, inp, p, m)
        assigned = state_extractor.get_obs(lfs, inp, ap, am)
        param = state_extractor.param_mask(p,m)
        target = state_extractor.get_target(inp)
        inter = state_extractor.get_inter(inp)
        diff = state_extractor.get_diff(lfs, inp)
        additional = state_extractor.get_additional(inp)
        parent = state_extractor.get_parent(inp)
        reverse = state_extractor.reverse_obs_norm(state_extractor.get_obs(lfs, inp, p, m))
        values.append((obs, assigned, param, target, inter, diff, additional, parent, reverse))
    if normed: save_to_pickle(create_directory("Testing/testing_data/extractor_outputs" + str(setting) + "_" + str(n_setting)+ ".pkl", drop_last=True), values)
    else: save_to_pickle(create_directory("Testing/testing_data/extractor_outputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + str(setting) + ".pkl", drop_last=True), values)
    return values


def diff_compute(args, norm, environment, single_obs_setting, relative_obs_setting, inp_path, params_path, out_path, normed=False):
    args.extract.inter_extractor = CausalExtractor(args.object_names, environment)
    args.target_select, args.full_parent_select, args.additional_select, args.additional_selectors, \
            args.padi_selector, args.parent_select, args.inter_select = args.extract.inter_extractor.get_selectors()

    args.extract.single_obs_setting = single_obs_setting
    args.extract.relative_obs_setting = relative_obs_setting
    args.extract.norm = norm
    args.extract.object_names = args.object_names
    state_extractor = StateExtractor(args.extract)

    param, assign_param, mask, assign_mask = load_from_pickle(params_path)
    if normed:
        last_full_states, full_states, normalized_inputs = load_from_pickle(inp_path)
    else:
        full_state = load_from_pickle(inp_path)
        last_full_states, full_states = full_state, full_state[1:]
    last_full_states = [{'factored_state': inp} for inp in last_full_states]
    full_states = [{'factored_state': inp} for inp in full_states]
    true_values = load_from_pickle(out_path)

    diff_values = list()
    for lfs, inp, p, ap, m, am, tv in zip(last_full_states, full_states, param, assign_param, mask, assign_mask, true_values):
        o, oa, mp, tar, inter, diff, a, par, dno = tv
        obs_diff = np.linalg.norm(state_extractor.get_obs(lfs, inp, p, m) - o)
        assign_diff = np.linalg.norm(state_extractor.assign_param(inp, state_extractor.get_obs(lfs, inp, p, m), ap, am) - oa)
        param_mask_diff = np.linalg.norm(state_extractor.param_mask(p,m) - mp)
        target_diff = np.linalg.norm(state_extractor.get_target(inp) - tar)
        inter_diff = np.linalg.norm(state_extractor.get_inter(inp) - inter)
        diff_diff = np.linalg.norm(state_extractor.get_diff(lfs, inp) - diff)
        add_diff = np.linalg.norm(state_extractor.get_additional(inp) - a)
        par_diff = np.linalg.norm(state_extractor.get_parent(inp) - par)
        denorm_diff = np.linalg.norm(state_extractor.reverse_obs_norm(state_extractor.get_obs(lfs, inp, p, m)) - dno)
        diff_values.append((obs_diff, assign_diff, param_mask_diff, target_diff, inter_diff, diff_diff, add_diff, par_diff, denorm_diff))
    return tuple(np.sum(np.array(diff_values), axis=0))


def test_state_extraction_single(args, single_obs_setting, relative_obs_setting, setting, n_setting):
    object_sizes = load_from_pickle("Testing/testing_data/norm_random_dims" + str(n_setting) + ".pkl")
    random_dicts = load_from_pickle("Testing/testing_data/norm_random_dicts" + str(n_setting) + ".pkl")
    lim_dict, dynamics_dict, args.object_names, object_instanced, inter_names = random_dicts
    environment = ObjDict({"object_sizes": object_sizes, "object_instanced": object_instanced})

    args.norm = load_from_pickle("Testing/testing_data/norm_module" + str(n_setting) + ".pkl")
    inp_path, params_path, out_path = "Testing/testing_data/norm_inputs" + str(n_setting) + ".pkl",\
    "Testing/testing_data/extractor_inputs" + str(setting) + "_" +str(n_setting)+ ".pkl",\
    "Testing/testing_data/extractor_outputs" + str(setting) + "_" +str(n_setting)+ ".pkl"
    vals = diff_compute(args, args.norm, environment, single_obs_setting, relative_obs_setting, inp_path, params_path, out_path, normed=True)

    return vals

def test_state_extraction_environment(args, single_obs_setting, relative_obs_setting, setting):
    object_names = init_names(args.train.train_edge)
    args.object_names = object_names
    environment, record = initialize_environment(args.environment, args.record)
    args.norm = NormalizationModule(environment.object_range, environment.object_dynamics, object_names, environment.object_instanced, object_names.inter_names)
    inp_path, params_path, out_path = "Testing/testing_data/trajectory_inputs_"+environment.name+ "_" + environment.variant + "_" + object_names.target + ".pkl", \
                                    "Testing/testing_data/extractor_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl", \
                                    "Testing/testing_data/extractor_outputs_"+environment.name+"_" + environment.variant + "_" + object_names.target + str(setting) + ".pkl"
    vals = diff_compute(args, args.norm, environment, single_obs_setting, relative_obs_setting, inp_path, params_path, out_path)
    return vals