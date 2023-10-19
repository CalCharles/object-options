import copy

network_args = {    
    "net_type": "mlp",
    "use_layer_norm": False,
    "hidden_sizes": list(), # TODO: must have values if used
    "init_form": "",
    "activation": "leakyrelu",
    "activation_final": "none",
    "scale_logits": -1,
    "scale_final": 1, # scales the final layer in MLPs
    "use_bias": True, # uses a bias in MLPs and 1D conv
    "multi": {
        "num_masks": -1, # should be set outside of network_args
        "embedding_sizes": list(), # number and shape of hidden layers
        "embedding_output": 256,
        "use_embedding": True,
    },
    "pair": {
        "drop_first": False,
        "reduce_function": "max",
        "post_dim": -1,
        "difference_first": False,
        "final_layers": [],
        "num_pair_layers": 1,
        "repeat_layers": False,
        "preencode": False,
        "pre_dropout": 0.0
    },
    "embedpair": {
        "new_embedding": True, # indicates if the network needs a new embedding, or the inputs are already embedded
        "query_aggregate": True, # in keypair networks, aggregates the queries (for prediction)
        "preembed_dim": 64, # the embedding size for the internal pair network
    },
    "embed_inputs": 0, # embeds the inputs, used as the embed_dim in transformers, keypair and pair networks
    "cluster": {
        "cluster_mode": False,
        "inter_pair_layers": 1,
        "num_clusters": 0, # overloaded: for expert models, this is the number of clusters. For interaction selection models, this is the number of interaction masks
        "cluster_inter_hidden": [],
    },
    "comb_embed": {
        "max_hash": -1, # TODO: not implemented
    },
    "mask_attn": {
        "model_dim": 0,
        "num_heads": 0,
        "num_layers": 1,
        "attention_dropout": 0.0,
        "mask_mode": "attn", # where the mask is applied, either 'query' or 'attn'
        "gumbel_attention": -1.0, # if a gumbel softmax is used at the attention level
        "needs_encoding": True, # should be set in init, default value here
        "merge_function": "cat", # the function for merging together the heads
        "append_keys": False, # appends the keys to the values TODO: move this to pair
        "no_hidden": False, # adds no hidden layers to the key, query or value operations
        "renormalize": False, # Normalizes the attention weights after masking and valid
    },
    "input_expand": {
        "include_relative": False,
        "pre_embed": [],
        "first_include": 0, # set in init, this is default initialization
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


full_args = {
    "arg_dict": "full",
    "debug": {
        "run_test": "",
        "collect_mode": "",
    },
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
        "save_interval": 0,
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
        "gym_to_gymnasium": True,
        "flat_obs": False,
        "append_id": False,
        "debug_mode": False, # puts the environment in a debugging mode where attributes are easier to identify. Implemented: RandomDAG
    },
    "torch": {
        "gpu": 1,
        "cuda": True,
        "torch_seed": -1
    },
    "train": {
        "dummy": "",
        "train": False,
        "num_frames": 0,
        "train_edge": list(),
        "override_name": "",
        "load_rollouts": "",
        "train_test_ratio": 0.9,
        "train_test_order": "random",
        "num_iters": 0,
        "pretrain_frames": 0,
        "batch_size": 128,
        "num_steps": 0,
    },
    "full_inter": {
        "object_id": True, # appends a 1 hot identifier of the object class to the object
        "lasso_lambda": [1, 0, 0, -1, -1], # lasso_lambda, open mask forcing, 0.5 mask forcing, one mask schedule, masking schedule
        "lasso_order": 1,
        "adaptive_lasso": [-1.0, -1.0], # adapts the lasso value according to the magnitude of the active interaction loss (multiplied by this hyperparameter), flattens the decay rate (exp(-\|perf diff\| / adaptive[1]))
        "adaptive_lasso_type": "likelihood", # different ways of computing adaptive lasso, uses: likelihood, l2 mean, l1 mean and variance
        "dual_lasso": [0,0],
        "entropy_lambda": [0,0], # penalizes the individual values of the binary mask for having high entropy (close to 0.5)
        "soft_distribution": "Identity",
        "dist_temperature": 1, # distribution temperature for relaxed distributions on the interaction mask
        "selection_temperature": 1, # distribution temperature for relaxed distributions on the selection network
        "mixed_interaction": "weighting",
        "use_active_as_passive": False,
        "proximal_weights": False,
        "reconstruct_embedding": False, # trains the factor-specific embeddings with reconstruction, only implemented for linpair net
        "log_gradients": False,
        "train_full_only": False,
        "lightweight_passive": True,
        "train_names": [], # for debugging, only trains certain names
        "load_forward_only": "", # loads only the forward models
        "selection_mask": False, # uses a selection mask network
        "selection_train": "",
        "nextstate_interaction": False, # uses the outcome for the interaction network
        "predict_next_state": True, # predicts the next state, otherwise, predicts the current state (useful for DAG methods)
        "delay_inter_train": -1, # delays starting interaciton training for this number of batches
    },
    "inter": {
        "predict_dynamics": False,
        "load_intermediate": "",
        "save_intermediate": "",
        "interaction_testing": [],
        "proximity_epsilon": -1,
        "compare_trace": False,
        "passive": {
            "train_passive": True, # trains the passive model (might only want to train the active model)
            "load_passive": "",
            "passive_iters": 0,
            "passive_log_interval": 1000,
            "pretrain_active": False,
        },
        "interaction": {
            "interaction_pretrain": 0,
            "subset_training": 0,
            "soft_train": 0, # adds this constant to the trace values then renormalizes
        },
        "active": {
            "active_steps": 1,
            "no_interaction": 0,
            "weighting": [0,0,-1,0], # must be length 4
            "active_log_interval": 100,
            "log_gradients": False,
            "interaction_schedule": -1, # if negative, uses 0.5 fixed tradeoff, if 0<= is <=1 uses the value at fixed value, if > 1 uses exp(-i/is)
            "inline_iters": [5, 1, 1000],
            "interaction_weighting": [0,0], # must be length 2
            "intrain_passive": 0,
            "error_binary_upweight": 1,
            "adaptive_inter_lambda": -1.0, # adaptive weight for training the forward model with full or inter inputs, still uses interaction schedule for tradeoff 
            "log_timestamps": True,
            "train_true": False, # trains using the true trace values
        },
    },
    "EMFAC": {
        "full_train": "",
        "num_masks": 1,
        "is_EMFAC": False,
        "E_step_iters": 1, # steps of forward model training
        "M_step_iters": 1, # steps of interaction model training
        "refine_iters": 1,
        "binary_cost": 1,
        "model_mask_weights": [0,0,0.4], # weight for the forward model performance, weight for the mask magnitude, weight lambda regularization
        "weight_forward": False, # Weights the losses by the sampling weights, TODO: might need more options
        "train_reconstruction": False # trains the embedding to perform reconstruction
    },
    "multi_inter": {
        "evaluate": False, # evaluation mode, where a trained model is tested for how well it can predict actual cause using null assumption, if 0, no evaluation, if 1, only evaluation, if 2, both train and eval
        "max_combination": 1, # searches for up to this many different parent sets (num factors combinations max_combination) when looking for simultanious interactions
        "dist_epsilon": 1e-1, # the minimum closeness of the distributions to e considered a null component
    },
    "mask": {
        "min_sample_difference": 1,
        "var_cutoff": [0.1],
        "num_samples": 30,
        "sample_grid": True,
        "dynamics_difference": False,
    },
    "sample": { # TODO NEW
        "sample_type": "uni",
        "sample_distance": -1,
        "sample_schedule": -1,
        "sample_raw": False,
        "sample_parent": False,
        "param_recycle": -1,
    },
    "extract": {
        "single_obs_setting": [0, 0, 0, 0, 0, 0],
        "relative_obs_setting": [0, 0, 0, 0, 0],
        "combine_param_mask": True
    },
    "option": { # mostly terminate and reward parameters
        "term_form": "param",
        "term_as_done": False,
        "use_binary": False,
        "true_done": True,
        "trunc_true": False,
        "epsilon_close": [-1.0],
        "param_norm": 1.0,
        "constant_lambda": 0.0,
        "true_lambda": 0.0,
        "param_lambda": -1.0,
        "inter_lambda": -1.0,
        "negative_true": -1.0,
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
        "round_values": False,
    },
    "collect": {
        "buffer_len": 100000,
        "prioritized_replay": list(),
        "test_episode": True,
        "max_steps": 1000,
        "terminate_reset": False,
        "display_frame": 0,
        "save_display": "",
        "stream_print_file": "",
        "demonstrate_option": False,
        "aggregator": {
            "sum_rewards": True,
            "only_termination": False,
        },
        "time_check": False, # returns a miss only if timing check fails
        "omit_done": False, # omits dones (EOE) from the buffer entirely
    },
    "policy": {
        "learning_type": "dqn",
        "epsilon_random": 0.0,
        "epsilon_schedule": -1,
        "rainbow": {
            "num_atoms": 51,
            "is_dueling": True,
            "is_noisy": True,
        },
        "ground_truth": "",
        "learn": {
            "post_random_iters": 0,
            "grad_epoch": 10,
            "sample_form": "merged",
        },
        "primacy": {
            "reset_layers": -1,
            "reset_frequency": -1,
            "primacy_iters": -1,
            "stop_resets": -1
        },
        "discount_factor": 0.99,
        "lookahead": 2,
        "max_min_critic": [-1.0,-1.0],
        "reward_normalization": False,
        "tau": 0.005,
        "sac_alpha": 0.2, # TODO: move sac arguments into a sac specific subcategory
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
    "hindsight": {
        "use_her": False,
        "resample_timer": -1,
        "select_positive": 0.5,
        "interaction_resample": False,
        "max_hindsight": -1,
        "early_stopping": False,
        "interaction_criteria": 0,
        "min_replay_len": -1,
        "num_param_samples": -1,
    },
    "inline": {
        "interaction_config": "",
        "inpolicy_iters": 5000,
        "inpolicy_schedule": -1,
        "inpolicy_times": -1,
        "policy_intrain_passive": False,
        "intrain_weighting": [-13, 1, 1, -1],   
        "save_inline": False,
        "policy_inline_iters": [5, 1, 1000],
        "reset_weights": [0,0,0]
    },
    "testing": {
        "test_type": "",
    },
    "network": copy.deepcopy(network_args),
}
