from collections import deque
import numpy as np
from file_management import save_to_pickle
from Rollouts.collector import BufferWrapper
from tianshou.data import Batch
import os
import copy
import psutil
import time

def trainRL(args, train_collector, test_collector, environment, environment_model, option, names, graph,  tensorboard_logger):
    '''
    Run the RL train loop
    '''
    start = time.time()
    # collect initial random actions
    if not len(args.load_pretrain) > 0:
        pretrain_result = train_collector.collect(n_step=args.pretrain_iters, random=True, visualize_param=args.visualize_param, no_fulls=True) # param doesn't matter with random actions
    total_steps, total_episodes = pretrain_result['n/st'], pretrain_result['n/tep']
    
    save_pretrain()

    # collect initial test trials
    initial_perf, initial_suc, initial_hit = _collect_test_trials(args, test_collector, 0, total_steps, total_episodes, list(), list(), list(), list(), list(), list(),list(), list(), random=True, option=option, tensorboard_logger=tensorboard_logger)

    hit_miss_queue_test = deque(maxlen=2000)
    hit_miss_queue_train = deque(maxlen=args.log_interval)
    cumul_losses = deque(maxlen=args.log_interval)
    train_drops = deque(maxlen=1000)

    print("collect", psutil.Process().memory_info().rss / (1024 * 1024 * 1024))
    for i in range(args.num_iters):  # total step
        collect_result = train_collector.collect(n_step=args.num_steps, visualize_param=args.visualize_param) # TODO: make n-episode a usable parameter for collect

        if i % args.log_interval == 0:
            _collect_test_trials(args, test_collector, i, total_steps, total_episodes, test_perf, suc, hit_miss_queue_test, hit_miss_queue_train, assessment_test, assessment_train, drops,  train_drops, option=option, tensorboard_logger=tensorboard_logger)
        # train option.policy with a sampled batch data from buffer

        losses = option.policy.update(args.batch_size, train_collector.buffer)
        record_stats(collect_result, losses,)

        if i % args.log_interval == 0:
            # compute the average loss
            log_stats()
        if args.save_interval > 0 and (i+1) % args.save_interval == 0: full_save(args, option, graph)


    if args.save_interval > 0: full_save(args, option, graph)
    final_perf, final_suc = list(), list()
    final_perf, final_suc, final_hit = _collect_test_trials(args, test_collector, i, total_steps, total_episodes, final_perf, final_suc, list(), list(),list(),list(),list(), list(), option=option, tensorboard_logger=tensorboard_logger)

    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False
