import yaml
import copy
from State.object_dict import ObjDict

network_args = {    
    "net_type": "mlp",
    "use_layer_norm": False,
    "hidden_sizes": list(),
    "init_form": "",
    "activation": "relu",
    "activation_final": "none",
    "pair": {
        "drop_first": False,
        "reduce_function": "mean",
        "post_dim": -1,
    },
    "optimizer": {
        "lr": 1e-4,
        "alt_lr": 1e-5,
        "eps": 1e-5,
        "alpha": 0.99,
        "betas": [0.9, 0.999],
        "weight_decay": 0.00
    },
}


expected_args = {
    "record": {
        "record_rollouts": "",
        "record_recycle": -1,
        'save_dir': "",
        'load_dir': "",
        'load_checkpoint': "",
        'checkpoint_dir': "",
        "pretrain_dir": "",
        'save_action': False, # saves in record_rollouts, saves the whole action chain and the param
        "save_interval": 0,
        "log_filename": "",
        "refresh": False,
    },
    "environment": {
        "env": None,
        "render": False,
        "frameskip": 1,
        "variant": "default",
        "time_cutoff": -1,
        "seed": -1,
        "demonstrate": False,
    },
    "torch": {
        "gpu": 1,
        "cuda": True,
    },
    "train": {
        "dummy": False,
        "train": False,
        "num_frames": 0,
        "train_edge": list(),
        "load_rollouts": "",
        "train_test_ratio": 0.9,
        "train_test_order": "random",
        "num_iters": 0,
        "pretrain_frames": 0,
        "batch_size": 128,
        "num_steps": 0
    },
    "inter": {
        "predict_dynamics": False,
        "load_intermediate": False,
        "save_intermediate": False,
        "interaction_testing": [],
        "proximity_epsilon": -1,
        "compare_trace": False,
        "passive": {
            "passive_iters": 0,
            "passive_log_interval": 1000,
            "pretrain_active": False
        },
        "interaction": {
            "interaction_pretrain": 0,
        },
        "active": {
            "weighting": [0,0,-1,0], # must be length 4
            "active_log_interval": 100,
            "interaction_schedule": -1,
            "inline_iters": [5, 1, 1000],
            "interaction_weighting": [0,0], # must be length 2
            "intrain_passive": False,
        },
    },
    "mask": {
        "min_sample_difference": 1,
        "var_cutoff": [0.1],
        "num_samples": 30,
        "sample_grid": True,
    },
    "sample": { # TODO NEW
        "sample_type": "uni",
        "sample_distance": -1,
        "sample_schedule": -1,
        "sample_raw": False,
        "param_recycle": -1,
    },
    "extract": {
        "obs_setting": [0, 0, 0, 0, 0, 0, 0, 0],
        "combine_param_mask": True
    },
    "option": { # mostly terminate and reward parameters
        "term_form": "param",
        "term_as_done": False,
        "true_done": True,
        "epsilon_close": -1,
        "param_norm": 1,
        "constant_lambda": 0,
        "param_lambda": -1,
        "inter_lambda": -1,
        "interaction_as_termination": False,
        "temporal_extend": -1,
        "time_cutoff": 0,
        "between_terminate": 1
    },
    "action": {
        "use_relative_action": False,
        "relative_action_ratio": -1,
        "min_active_size": 10,
        "discrete_params": False,
    },
    "collect": {
        "buffer_len": 100000,
        "prioritized_replay": list(),
        "test_episode": True,
        "max_steps": 1000,
        "terminate_reset": False,
        "aggregator": {
            "sum_rewards": True,
            "only_termination": False,
        },
    },
    "policy": {
        "learning_type": "dqn",
        "epsilon_random": 0.0,
        "epsilon_schedule": -1,
        "rainbow": {
                    "num_atoms": 51,
        },
        "learn": {
            "grad_epoch": 10,
            "sample_form": "merged",
        },
        "discount_factor": 0.99,
        "lookahead": 2,
        "max_critic": -1,
        "reward_normalization": False,
        "tau": 0.005,
        "sac_alpha": 0.2,
        "logging": {
            "log_interval": 5,
            "train_log_maxlen": 0,
            "test_log_maxlen": 0,
            "initial_trials": 10,
            "test_trials": 10,
            "max_terminate_step": [0,0]
        }
    },
    "hindsight": {
        "use_her": False,
        "resample_timer": -1,
        "select_positive": 0.5,
        "interaction_resample": False,
        "max_hindsight": -1,
        "early_stopping": False,
        "interaction_criteria": 0,
        "min_replay_len": -1
    },
    "inline": {
        "interaction_config": "",
        "inpolicy_iters": 5000,
        "inpolicy_schedule": -1,
        "policy_intrain_passive": False,
        "intrain_weighting": [-13, 1, 1, -1],   
        "save_inline": False,
        "policy_inline_iters": [5, 1, 1000],
    },
    "network": copy.deepcopy(network_args),
}

def read_config(pth):
    # copied from https://pynative.com/python-yaml/
    with open(pth) as file:
        try:
            data = yaml.safe_load(file)
        except yaml.YAMLError as exception:
            print("error: ", exception)
    data = construct_namespace(data)
    return data

def construct_namespace(data):
    args = ObjDict()
    def add_data(add_dict, data_dict, exp_dict):
        if data_dict is None: # if only the name is returned, create a new dict to fill with values
            data_dict = ObjDict()
        for key in exp_dict.keys():
            if type(exp_dict[key]) == dict or type(exp_dict[key]) == ObjDict: # if we have a dict recursively call
                new_add = ObjDict()
                add_dict[key] = add_data(new_add, data_dict[key] if key in data_dict else dict(), exp_dict[key])
            else: # exp_dict contains the default values
                add_dict[key] = data_dict[key] if key in data_dict else exp_dict[key]
                # handling special characters
                if add_dict[key] == "None": add_dict[key] = None
                elif add_dict[key] == "[]": add_dict[key] = list()
                elif type(exp_dict[key]) == list and key in data_dict:
                    if type(data_dict[key]) != str:
                        add_dict[key] = [data_dict[key]]
                    else:
                        try:
                            add_dict[key] = [float(v) for v in add_dict[key].split(" ")]
                        except ValueError as e:
                            add_dict[key] = add_dict[key].split(" ")
        return add_dict
    args = add_data(args, data, expected_args)
    for key in data.keys():
        if key.find("_net") != -1: # _net 
            vargs = add_data(ObjDict(), data[key], args.network)
            args[key] = vargs
            print(vargs)
    return add_data(args, data, expected_args)