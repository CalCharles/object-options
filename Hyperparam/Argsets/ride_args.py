import copy
from Hyperparam.Argsets.full_args import network_args


ride_args = {
    "arg_dict": "ride",
    "record": {
        'save_dir': "",
        'load_dir': "",
        "log_filename": "",
        "record_rollouts": "",
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
        "gym_to_gymnasium": True,
        "flat_obs": True,
        "append_id": False,
    },
    "torch": {
        "gpu": 1,
        "cuda": True,
        "torch_seed": -1
    },
    "train": {
        "train": False,
        "num_iters": 0,
        "pretrain_frames": 0,
        "batch_size": 128,
        "num_steps": 0,
    },
    "RIDE": {
        "lr_scale": 1.0,
        "reward_scale": 0.01,
        "forward_loss_weight": 1.0, 
        "training_num": 16,
        "test_num": 8,
        "pseudocount_lambda": 0.1,
        "pair_network": False,
        "conv_mode": False,
    },
    "collect": {
        "buffer_len": 100000,
        "prioritized_replay": list(),
        "display_frame": 0,
    },
    "policy": {
        "learning_type": "dqn",
        "epsilon_random": 0.0,
        "rainbow": {
            "num_atoms": 51,
            "is_dueling": True,
            "is_noisy": True,
        },
        "learn": {
            "grad_epoch": 0.1, # in RIDE, grad epochs are represented as updates / steps ratio
        },
        "discount_factor": 0.99,
        "lookahead": 1,
        "max_min_critic": [-1.0,-1.0],
        "reward_normalization": False,
        "tau": 0.005,
        "sac_alpha": 0.2,
        "auto_alpha": False,
        "alpha_lr": .0003,
        "deterministic_eval": False,
        "logging": {
            "log_interval": 5,
            "train_log_maxlen": 0,
            "test_log_maxlen": 0,
            "initial_trials": 10,
            "test_trials": 10,
            "max_terminate_step": [0,0]
        }
    },
    "network": copy.deepcopy(network_args),
}
