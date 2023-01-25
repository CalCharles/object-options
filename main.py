from train_option import train_option
from config_list import breakout_configs, breakout_variant_configs, robopushing_configs, obstacle_config


if __name__ == "__main__":
    args = get_args()
    if args.main_train == "BreakoutVariants":
        for variant in breakout_variant_configs:
            args = read_config(variant)
            train_option(args)
    else:
        if args.main_train == "BreakoutStack":
            config_choice = breakout_configs
        elif args.main_train == "RoboPushingStack":
            config_choice = robopushing_configs
        for i, config in enumerate(breakout_variant_configs):
            args = read_config(config)
            if i % 3 == 0:
                train_interaction(args)
            elif i % 2 == 0:
                train_masking(args)
            else:
                train_option(args)
        if args.main_train == "RoboPushingObstacle":
            args = read_config(obstacle_config)
            train_option(obstacle_config)