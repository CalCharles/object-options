from train_option import train_option
from config_list import breakout_configs, breakout_variant_configs, robopushing_configs, obstacle_config
from arguments import get_args
from Hyperparam.read_config import read_config
from train_interaction import train_interaction
from train_masking import train_mask
from generate_random import generate_random, generate_args

def breakout_random(config_choice):
    breakout_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = breakout_path
    rand_args.env = "Breakout"
    # generate_random(rand_args)

def robo_random(config_choice):
    robo_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = robo_path
    rand_args.env = "RoboPushing"
    generate_random(rand_args)


if __name__ == "__main__":
    args = get_args()
    if args.main_train == "BreakoutVariants":
        for variant in breakout_variant_configs:
            args = read_config(variant)
            train_option(args)
    else:
        if args.main_train == "BreakoutStack":
            config_choice = breakout_configs
            breakout_random(breakout_configs)
        elif args.main_train == "RoboPushingStack":
            config_choice = robopushing_configs
            robo_random(robopushing_configs)
        for i, config in enumerate(config_choice):
            # if i < 3: continue
            args = read_config(config)
            if i % 3 == 0:
                pass
                # train_interaction(args)
            elif i % 3 == 1:
                pass
                # train_mask(args)
            else:
                train_option(args)
        if args.main_train == "RoboPushingObstacle":
            args = read_config(obstacle_config)
            train_option(obstacle_config)