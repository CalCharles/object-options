# generates a reward function based on HyPE
import numpy as np
import torch
import os
from train_interaction import init_names
from arguments import get_args
from Baselines.HyPE.ChangepointDetection.CHAMP import CHAMPDetector
from Baselines.HyPE.Reward.changepointClusterModels import BayesianGaussianMixture
from Baselines.HyPE.Reward.changepointModel import ChangepointModel
from Baselines.HyPE.get_states import get_states
from Baselines.HyPE.Reward.changepointReward import ChangepointDetectionReward
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle, create_directory
from Environment.Environments.initialize_environment import initialize_environment
from Causal.Utils.interaction_selectors import CausalExtractor
from Environment.Normalization.norm import NormalizationModule

def generate_extractor_norm(object_names, environment):
    extractor = CausalExtractor(object_names, environment)
    norm = NormalizationModule(environment.object_range, environment.object_range_true, environment.object_dynamics, object_names,
                                    environment.object_instanced, [object_names.primary_parent, object_names.target])
    return extractor, norm

def save_reward(args, reward):
    print(os.path.join(create_directory(args.record.save_dir), args.object_names.target + "_reward.pkl"))
    save_to_pickle(os.path.join(create_directory(args.record.save_dir), args.object_names.target + "_reward.pkl"), reward)

def load_reward(args):
    print(create_directory(args.record.load_dir) + args.object_names.target+ "_reward.pkl")
    return load_from_pickle(os.path.join(args.record.load_dir, args.object_names.target + "_reward.pkl"))


# necessary args: traj_dim, parameter_minmax, train_edge
def generate_reward_function():
    args = get_args()
    args.object_names = init_names(args.train_edge)
    torch.cuda.set_device(args.torch.gpu)
    environment, record = initialize_environment(args.environment, args.record)
    traj_dim = int(np.sum(environment.position_masks[args.object_names.target])) # position should be the first n dimensions
    extractor, norm = generate_extractor_norm(args.object_names, environment)

    changepoint_model = CHAMPDetector(args.train_edge, args.reward.champ_parameters)
    cluster_model = BayesianGaussianMixture(args.reward.dp_gmm)
    model = ChangepointModel(changepoint_model, args.reward.proximity, cluster_model, traj_dim, skip_one = args.environment.env == "Breakout")
    
    data = read_obj_dumps(args.reward.load_rollouts, i=-1, rng = args.reward.num_frames, filename='object_dumps.txt')
    states = get_states(extractor, data, norm, args.object_names)
    valid_modes = model.fit_modes(states["target_diff"], states["target"], states["parent_state"], states["done"], min_size=args.reward.min_size)
    model.changepoint_model = model.changepoint_model if args.reward.use_changepoint else None # set it to none until after fitting

    reward = ChangepointDetectionReward(args.object_names, model, valid_modes, args.reward.reward_base, args.reward.param_reward, args.reward.changepoint_reward, extractor, norm)
    if args.reward.one_mode: reward.toggle_one_mode(args.reward.one_mode)
    rewards = reward.compute_reward(states["target_diff"], states["target"], states["parent_state"], states["done"])
    print(rewards[:100])
    save_reward(args, reward)