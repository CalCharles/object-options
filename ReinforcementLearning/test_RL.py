from collections import deque
import numpy as np
from tianshou.data import Batch
import os
import copy
import psutil
import time
import logging
from ReinforcementLearning.utils.RL_logging import collect_test_trials
from State.observation_extractor import obs_names


def testRL(args, collector, option, graph, logger, full_logger):
    '''
    Run the RL train loop
    '''
    logger.logout("observation: " + str([obs_names[i] for i in range(len(option.state_extractor.obs_setting)) if option.state_extractor.obs_setting[i] == 1]))
    for i in range(args.train.num_iters):  # total step
        full_results = collect_test_trials(logger, option, collector, args.policy.logging.max_terminate_step, i, args.policy.logging.initial_trials, False)

        # only prints if log interval is reached
        logger.print_log(i)
        for res in full_results:
            full_logger.log_results(res)
    full_logger.print_log(0, force=True)