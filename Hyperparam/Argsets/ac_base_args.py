import copy
from Hyperparam.Argsets.full_args import network_args


ac_base_args = {
    "arg_dict": "ride",
    "record": {
        'save_dir': "",
        'load_dir': "",
        "log_filename": "",
        "record_rollouts": "",
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
    "inter_baselines": {
        "gradient_threshold": -1.0,
        "grad_lasso_lambda": 0.0,
        "attention_threshold": -1.0,
        "attention_lambda": 0.0,
        "dist_distance": "likelihood",
        "counterfactual_threshold": -1.0,
        "counterfactual_lambda": 0.0,
        "num_counterfactual": -1,
        "trace_weighting": -1.0,
    },
    "collect": {
        "buffer_len": 100000,
        "prioritized_replay": list(),
        "display_frame": 0,
        "omit_done": False,
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
    },
    "full_inter": {
        "object_id": True, # appends a 1 hot identifier of the object class to the object
        "lasso_lambda": [1, 0, 0, -1, -1], # lasso_lambda, open mask forcing, 0.5 mask forcing, one mask schedule, masking schedule
        "lasso_order": 1,
        "adaptive_lasso": [-1.0, -1.0], # adapts the lasso value according to the magnitude of the active interaction loss (multiplied by this hyperparameter), flattens the decay rate (exp(-\|perf diff\| / adaptive[1]))
        "adaptive_lasso_bias": [0.0, -1.0], # biases the adaptive lasso baseline constant by negative the adaptive bias, decayed at the schedule
        "adaptive_lasso_type": "likelihood", # different ways of computing adaptive lasso, uses: likelihood, l2 mean, l1 mean and variance
        "reset_caloss": False, # resets the converged active loss after passive training 
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
        "partial_active_reset": [-1,-1,-1], # the number of layers to reset, the frequency of interactions (num iters), the iteration to stop resetting at
        "partial_inter_reset": [-1,-1,-1],
        "cap_probability": 1e-5, # 1-s cannot be higher confidence than this probability in interaction training
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

    "network": copy.deepcopy(network_args),
}
