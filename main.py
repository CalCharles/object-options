from train_option import train_option
from config_list import *
from arguments import get_args
from Hyperparam.read_config import read_config
from train_interaction import train_interaction
from train_masking import train_mask
from generate_random import generate_random, generate_args

def breakout_random(config_choice, uid):
    breakout_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = breakout_path + uid
    rand_args.env = "Breakout"
    generate_random(rand_args)

def robo_random(config_choice, uid):
    robo_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = robo_path + uid
    rand_args.env = "RoboPushing"
    generate_random(rand_args)

def printer_random(config_choice, uid):
    printer_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = printer_path + uid
    print(rand_args.record_rollouts)
    rand_args.env = "MiniBehavior"
    rand_args.variant = "installing_printer"
    generate_random(rand_args)

def thawing_random(config_choice, uid):
    thaw_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = thaw_path + uid
    print(rand_args.record_rollouts)
    rand_args.env = "MiniBehavior"
    rand_args.variant = "thawing"
    generate_random(rand_args)

def cleaning_random(config_choice, uid):
    clean_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = clean_path + uid
    print(rand_args.record_rollouts)
    rand_args.env = "MiniBehavior"
    rand_args.variant = "cleaning_car"
    generate_random(rand_args)

def igibson_random(config_choice, uid):
    igibson_path = config_choice.pop(0)
    rand_args = generate_args()
    rand_args.record_rollouts = igibson_path + uid
    print(rand_args.record_rollouts)
    rand_args.env = "iGibson"
    rand_args.variant = "default"
    generate_random(rand_args)


if __name__ == "__main__":
    args = get_args()
    uid = args.main_uid # should be a number
    if args.main_train == "BreakoutVariants":
        for variant in breakout_variant_configs:
            args = read_config(variant)
            train_option(args)
    elif args.main_train == "RoboPushingObstacle":
        args = read_config(obstacle_config)
        train_option(args)
    elif args.main_train == "InstallingPrinter":
        args = read_config(install_printer_config)
        train_option(args)
    else:
        if args.main_train == "BreakoutStack":
            config_choice = breakout_configs
            breakout_random(breakout_configs, uid)
        elif args.main_train == "RoboPushingStack":
            config_choice = robopushing_configs
            robo_random(robopushing_configs, uid)
        elif args.main_train == "PrinterStack":
            config_choice = printer_configs
            # config_choice.pop(0)
            printer_random(printer_configs, uid)
        elif args.main_train == "ThawStack":
            config_choice = thaw_configs
            # config_choice.pop(0)
            thawing_random(thaw_configs, uid)
        elif args.main_train == "CleanStack":
            config_choice = clean_configs
            # config_choice.pop(0)
            cleaning_random(clean_configs, uid)
        elif args.main_train == "iGibsonStack":
            config_choice = igibson_configs
            # config_choice.pop(0)
            igibson_random(igibson_configs, uid)
            
        for i, config in enumerate(config_choice):
            # if i < 3: continue
            args = read_config(config)
            args.torch.gpu = int(uid) % 4 if len(uid) > 0 else args.torch.gpu# hardcoded for 4-gpu servers to spread load
            if i % 3 == 0:
                args.record.save_dir = args.record.save_dir + uid
                args.train.load_rollouts = args.train.load_rollouts + uid
                train_interaction(args)
            elif i % 3 == 1:
                args.record.save_dir = args.record.save_dir + uid
                args.record.load_dir = args.record.load_dir + uid
                args.train.load_rollouts = args.train.load_rollouts + uid
                train_mask(args)
            else:
                if len(args.record.log_filename) > 0: args.record.log_filename = args.record.log_filename + uid
                if len(args.record.record_graphs) > 0: args.record.record_graphs = args.record.record_graphs + uid
                args.record.save_dir = args.record.save_dir + uid
                args.record.load_dir = args.record.load_dir + uid
                if len(args.record.record_rollouts) > 0: args.record.record_rollouts = args.record.record_rollouts + uid
                args.train.load_rollouts = args.train.load_rollouts + uid
                train_option(args)