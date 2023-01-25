from Hyperparam.Argsets.full_args import network_args
hype_args = {
    "record": {
        "record_rollouts": "",
        "record_graphs": "",
        "record_recycle": -1,
        'save_dir': "",
        'load_dir': "",
        'load_checkpoint': "",
        'checkpoint_dir': "",
        "pretrain_dir": "",
        'save_action': False, # saves in record_rollouts, saves the whole action chain and the param
        "save_interval": 100,
        "log_filename": "",
        "refresh": False,
        "presave_graph": False,
    },
    "environment": {
        "env": None,
        "render": False,
        "frameskip": 1,
        "variant": "default",
        "horizon": -1,
        "seed": -1,
        "demonstrate": False,
        "load_environment": "",
        "fixed_limits": False,
    },
    "torch": {
        "gpu": 1,
        "cuda": True,
        "torch_seed": -1
    },
    "arg_dict": "hype",
    "train_mode": "reward", # either reward or policy
    "train_edge": list(),
    "reward": {
        "num_frames": 10000,
        "champ_parameters": [3, 5, 1, 100, 100, 2, 1e-2, 3], # Ball: [15, 10, 1, 100, 100, 2, 1e-2, 3] 
        "dp_gmm": [10, 6000, 100, 'diag', 1e-10],
        "use_changepoint": True, # use changepoints for rewards
        "proximity": -1.0, # use proximity for rewards
        "reward_base": 0.0,
        "param_reward": 0.0,
        "changepoint_reward": 0.0,
        "load_rollouts": "",
        "min_size": 10, # minimum size to be considered a valid mode
        "one_mode": False,# changes to a single mode version of the code
        "true_reward": False,# uses the true reward
    },
    "skill": {
        "temporal_extend": 5, # the max timestep for an action before a new one is sampled
        "num_iters": 100,
        "policy_iters": 100, # the number of steps to sample for a policy
        "test_policy_iters": 10,
        "policy_iters_schedule": 0, # doubles every n iterations
        "num_repeats_schedule": 0,
        "num_repeats": 1, # number of repeats for each sample, this MUST be a multiple of num_networks in cmaes for proper running
        "epsilon_random": 0.0, # action randomness
        "epsilon_schedule": -1, # number of iters before epsilon reduces by e^-1 from 1
        "num_networks": 1, # number of networks for cmaes
        "learning_type": "",
        "log_interval": 100,
        "test_trials": 10,
        "train_log_maxlen": 50,
        "buffer_len": 1000,
        "prioritized_replay": list(),
        "batch_size": 64, # unnecessary for cmaes
        "demonstrate": False,
        "input_scaling": 1, # multiplies the inputs by this value
        "obs_components": [1.0,1.0,1.0], # which of the there components to use as obs
        "normalized": True, # uses the normalzied values for the obs
        "merge_data": False, # merges together all the buffers, only off policy
        "include_primitive": False, # includes the primitive actions in the action space, pushing primitive actions down the chain
        "learn": {
            "discount_factor": 0.99,
            "lookahead": 2,
            "tau": 0.005,
            "max_min_critic": [-1.0,-1.0],
            "num_atoms": 51,
            "reward_normalization": False,
            "is_dueling": True,
            "init_var": 1,
            "grad_epoch": 1, # only use one epoch for CMAES
            "elitism": 0, # elitism for CMAES
        },
    },
    "network": network_args,
}