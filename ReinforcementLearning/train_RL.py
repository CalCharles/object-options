from collections import deque
import numpy as np
from tianshou.data import Batch
import os
import copy
import psutil
import time
import logging
from Record.file_management import save_to_pickle, load_from_pickle
from ReinforcementLearning.utils.RL_logging import collect_test_trials, buffer_printouts
from State.observation_extractor import obs_names



def trainRL(args, train_collector, test_collector, option, graph,  loggers):
    '''
    Run the RL train loop
    '''
    train_logger, test_logger, initial_logger = loggers
    initial_logger.logout("observation: " + str([obs_names[i] for i in range(len(option.state_extractor.obs_setting)) if option.state_extractor.obs_setting[i] == 1]))
    start = time.time()
    # collect initial random actions
    if len(args.record.pretrain_dir) == 0: pretrain_result = train_collector.collect(n_step=args.train.pretrain_frames, random=True) # param doesn't matter with random actions
    else: 
        train_collector.load(os.path.join(args.record.pretrain_dir, "pretrain_buffers.bf"))
        pretrain_result = load_from_pickle(os.path.join(args.record.pretrain_dir, "pretrain_result.pkl"))
    train_logger.log_results(pretrain_result)
    buffer_printouts(args, train_collector, option)
    
    if len(args.record.checkpoint_dir) > 0:
        train_collector.save(args.record.checkpoint_dir, "pretrain_buffers.bf")
        save_to_pickle(os.path.join(args.record.checkpoint_dir, "pretrain_result.pkl"), pretrain_result) # directory created by collector.save

    # collect initial test trials
    collect_test_trials(initial_logger, option, test_collector, (1,30), 0, args.policy.logging.initial_trials, True)
    initial_perf, initial_success = initial_logger.print_log(0, force=True)

    for i in range(args.train.num_iters):  # total step
        collect_result = train_collector.collect(n_step=args.train.num_steps) # TODO: make n-episode a usable parameter for collect
        train_logger.log_results(collect_result)

        if i % args.policy.logging.log_interval == 0:
            collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, i, args.policy.logging.initial_trials, False)
        # train option.policy with a sampled batch data from buffer

        losses = option.policy.update(args.train.batch_size, train_collector.buffer, train_collector.her_buffer)
        train_logger.log_losses(losses)
        if option.inline_trainer.train: option.inline_trainer.run_train(i, train_collector.full_buffer)

        # only prints if log interval is reached
        train_logger.print_log(i)
        test_logger.print_log(i)
        if args.record.save_interval > 0 and (i+1) % args.record.save_interval == 0:
            buffer_printouts(args, train_collector, option)
            if len(args.record.save_dir) > 0: graph.save(args.record.save_dir)
            if len(args.record.checkpoint_dir) > 0:
                graph.save(args.record.checkpoint_dir)
                train_collector.save(args.record.checkpoint_dir, "RL_buffers.bf")

    if args.record.save_interval > 0: 
        graph.save(args.record.save_dir)
        if len(args.record.checkpoint_dir) > 0: train_collector.save(args.record.checkpoint_dir, "RL_buffers.bf")
    
    # final logging step to determinie the performance of the final policy
    test_logger.print_log(i, force=True)
    test_logger.reset()
    collect_test_trials(test_logger, option, test_collector, (1,30), i, args.policy.logging.initial_trials, False)
    final_perf, final_success = test_logger.print_log(i, force=True)

    logging.info(f"performance comparison: {initial_perf}, {final_perf}")
    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False
