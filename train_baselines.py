import numpy as np
import torch
from arguments import get_args

if __name__ == "__main__":
    args = get_args()
    if args.arg_dict == "hype":
        from Environment.Environments.initialize_environment import initialize_environment
        from Baselines.HyPE.generate_reward_fn import generate_reward_function
        from Baselines.HyPE.train_skill import train_skill
        from Baselines.HyPE.test_skill import test_skill
        if args.train_mode == "reward":
            generate_reward_function()
        elif args.train_mode == "test":
            test_skill(args)
        else:
            train_skill(args)
    if args.arg_dict == "cdl":
        from Baselines.CDL.main_policy import train
        from Baselines.CDL.utils.utils import TrainingParams
        params = TrainingParams(training_params_fname=args.param_config, train=True)
        train(params, args)
    if args.arg_dict == "ride":
        from Baselines.RIDE.train_RIDE import train_RIDE
        train_RIDE(args)