  # Sampling
  #   Mask
  #   Sample ranges and truncated ranges from controllable features
  #   History used for sampling
import numpy as np
import string
from Record.file_management import load_from_pickle, save_to_pickle, create_directory
from Causal.Sampling.sampling import samplers
from Causal.Utils.interaction_selectors import CausalExtractor
from State.object_dict import ObjDict
from Environment.Environments.initialize_environment import initialize_environment
from train_interaction import init_names
from Graph.graph import load_graph

  #   Sampling predictor
  #   Sampler parameters
  #   Sampler training occurs properly
sampler_names = ["uni", "hist", 'cent', "exist", "angle", "empty", "target", "dummy"]

def generate_samples(save_path, sampler, full_states):
    np.random.seed(1) # set the seed to ensure sampling will always follow the same pattern
    total_samp_mask = list()
    print(sampler)
    for full_state in full_states:
        sample, mask = sampler.sample(full_state)
        total_samp_mask.append((sample, mask))
    save_to_pickle(save_path, total_samp_mask)
    return total_samp_mask

def create_samples(args):
    environment, record = initialize_environment(args.environment, args.record)
    object_names = init_names(args.train.train_edge)
    mask = np.random.randint(2, size=(environment.object_sizes[object_names.target],))
    args.sample.causal_extractor = CausalExtractor(object_names, environment)
    args.sample.target_select, args.sample.full_parent_select, args.sample.additional_select, args.sample.additional_selectors, \
            args.sample.padi_selector, args.sample.parent_select, args.sample.inter_select = args.sample.causal_extractor.get_selectors()
    inp_path = "Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant +"_" + object_names.target +".pkl"
    param_mask_path = "Testing/testing_data/trajectory_param_mask_"+environment.name+"_" + args.environment.variant +"_" + object_names.target +".pkl"
    full_states = load_from_pickle(inp_path)
    params, masks = load_from_pickle(param_mask_path)
    graph = load_graph(args.record.load_dir, args.torch.gpu) # device is needed to load options properly
    args.sample.mask = ObjDict({"active_mask": np.array(masks[0]), "limits": environment.object_range[object_names.target], 
        "range": environment.object_range[object_names.target][1] - environment.object_range[object_names.target][0],
        "filtered_active_set": graph.nodes[object_names.target].interaction.mask.filtered_active_set})
    args.sample.obj_dim = environment.object_range[object_names.target][1].shape[0]
    args.sample.num_angles = 8
    args.sample.positive= True
    args.sample.epsilon_close = args.option.epsilon_close

    # sample each kind of sampler
    for stxt in sampler_names:
        sampler = samplers[stxt](**args.sample)
        true_samples_path = "Testing/testing_data/sample_outputs_"+environment.name+"_" + environment.variant + "_" + object_names.target + "_" + stxt +".pkl"
        generate_samples(true_samples_path, sampler, full_states)

def sample_states(sampler, full_states, true_samples_path):
    true_sample_mask = load_from_pickle(true_samples_path)
    np.random.seed(1) # set the seed to ensure sampling will always follow the same pattern
    total_diff = list()
    for true_sample_mask, full_state in zip(true_sample_mask, full_states):
        sample, mask = sampler.sample(full_state)
        total_diff.append(np.sum(true_sample_mask[0] * true_sample_mask[1] - sample * mask))
    return np.sum(total_diff)

def test_sample_env(args):
    environment, record = initialize_environment(args.environment, args.record)
    object_names = init_names(args.train.train_edge)
    mask = np.random.randint(2, size=(environment.object_sizes[object_names.target],))
    args.sample.causal_extractor = CausalExtractor(object_names, environment)
    args.sample.target_select, args.sample.full_parent_select, args.sample.additional_select, args.sample.additional_selectors, \
            args.sample.padi_selector, args.sample.parent_select, args.sample.inter_select = args.sample.causal_extractor.get_selectors()
    inp_path = "Testing/testing_data/trajectory_inputs_"+environment.name+"_" + args.environment.variant +"_" + object_names.target +".pkl"
    param_mask_path = "Testing/testing_data/trajectory_param_mask_"+environment.name+"_" + args.environment.variant +"_" + object_names.target +".pkl"
    
    full_states = load_from_pickle(inp_path)
    params, masks = load_from_pickle(param_mask_path)
    graph = load_graph(args.record.load_dir, args.torch.gpu) # device is needed to load options properly
    args.sample.mask = ObjDict({"active_mask": np.array(masks[0]), "limits": environment.object_range[object_names.target], 
        "range": environment.object_range[object_names.target][1] - environment.object_range[object_names.target][0],
        "filtered_active_set": graph.nodes[object_names.target].interaction.mask.filtered_active_set})
    args.sample.obj_dim = environment.object_range[object_names.target][1].shape[0]
    args.sample.num_angles = 8
    args.sample.positive= True
    args.sample.epsilon_close = args.option.epsilon_close

    # sample each kind of sampler

    sampler_names = ["uni", "hist", 'cent', "exist", "angle", "empty", "target", "dummy"]
    for stxt in sampler_names:
        sampler = samplers[stxt](**args.sample)
        true_samples_path = "Testing/testing_data/sample_outputs_"+environment.name+ "_" + environment.variant + "_" + object_names.target + "_" + stxt +".pkl"
        sample_states(sampler, full_states, true_samples_path)