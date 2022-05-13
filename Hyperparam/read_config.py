import yaml
from State.object_dict import ObjDict

expected_args = {
    "record": {
        "record_rollouts": "",
        "record_recycle": -1,
        'save_dir': ""
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
        "num_frames": 1000,
        "train_edge": [],
        "load_rollouts": "",
        "train_test_ratio": 0.9,
        "num_iters": 10000,
        "batch_size": 128,
    },
    "inter": {
        "predict_dynamics": False,
        "load_intermediate": False,
        "save_intermediate": False,
        "interaction_testing": [0.9, -16, -16, -1],
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
            "weighting": [0,0,0,0], # must be length 4
            "active_log_interval": 100,
            "interaction_schedule": -1,
            "inline_iters": [5, 1, 1000],
            "interaction_weighting": [0,0], # must be length 2
            "intrain_passive": False,
        },
    },
    "network": {
        "net_type": "mlp",
        "use_layer_norm": False,
        "hidden_sizes": [],
        "init_form": "",
        "activation": "relu",
        "activation_final": "none",
        "pair": {
            "drop_first": False,
            "reduce_function": "mean",
        }
    },
    "optimizer": {
        "lr": 1e-4,
        "alt_lr": 1e-5,
        "eps": 1e-5,
        "alpha": 0.99,
        "betas": [0.9, 0.999],
        "weight_decay": 0.00
    }
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
    print(data, expected_args)
    def add_data(add_dict, data_dict, exp_dict):
        for key in exp_dict.keys():
            if type(exp_dict[key]) == dict: # if we have a dict recursively call
                new_add = ObjDict()
                add_dict[key] = add_data(new_add, data_dict[key] if key in data_dict else dict(), exp_dict[key])
            else: # exp_dict contains the default values
                add_dict[key] = data_dict[key] if key in data_dict else exp_dict[key]
                # handling special characters
                if add_dict[key] == "None": add_dict[key] = None
                elif add_dict[key] == "[]": add_dict[key] = list()
                elif type(add_dict[key]) == str and len(add_dict[key].split(" ")) > 1: 
                    try:
                        add_dict[key] = [float(v) for v in add_dict[key].split(" ")]
                    except ValueError as e:
                        add_dict[key] = add_dict[key].split(" ")
        return add_dict
    args = add_data(args, data, expected_args)
    print("args", args, args.train)
    return add_data(args, data, expected_args)