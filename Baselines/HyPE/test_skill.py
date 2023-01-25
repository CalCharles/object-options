from Baselines.HyPE.Policy.collector import HyPECollector
from Baselines.HyPE.Policy.skill import load_skill
from Baselines.HyPE.Policy.test_trials import collect_test_trials
from ReinforcementLearning.utils.RL_logger import RLLogger
from State.object_dict import ObjDict
from Environment.Environments.initialize_environment import initialize_environment
from train_interaction import init_names
import torch
import numpy as np


def test_skill(args):
    args.object_names = init_names(args.train_edge)
    torch.cuda.set_device(args.torch.gpu)
    environment, record = initialize_environment(args.environment, args.record)
    skill = load_skill(args.record.load_dir, args.object_names.target, device=args.torch.gpu)

    test_logger = RLLogger(args.object_names.target + "_test", args.record.record_graphs, args.skill.log_interval, args.skill.train_log_maxlen, args.record.log_filename)
    test_collector = HyPECollector(environment, None, skill, skill.extractor, True, record, args.skill.test_policy_iters, use_true_reward = args.reward.true_reward)

    collect_test_trials(test_logger, skill, test_collector, args.skill.test_policy_iters, 0, 0, args.skill.test_trials, False)
    test_logger.print_log(0, force=True)

