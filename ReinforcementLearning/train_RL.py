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
from State.observation_extractor import COMPONENT_NAMES



def trainRL(args, train_collector, test_collector, option, graph,  loggers):
    '''
    Run the RL train loop
    '''
    train_logger, test_logger, initial_logger = loggers
    initial_logger.logout("observation: " + str([COMPONENT_NAMES[i] for i in range(len(option.state_extractor.obs_setting)) if option.state_extractor.obs_setting[i] == 1]))
    start = time.time()
    # collect initial random actions
    train_collector.reset_env()
    if len(args.record.pretrain_dir) == 0: pretrain_result = train_collector.collect(n_step=args.train.pretrain_frames, random=True) # param doesn't matter with random actions
    else: 
        train_collector.load(os.path.join(args.record.pretrain_dir, "pretrain_buffers.bf"))
        pretrain_result = load_from_pickle(os.path.join(args.record.pretrain_dir, "pretrain_result.pkl"))
    train_logger.log_results(pretrain_result)
    # buffer_printouts(args, train_collector, option)
    
    if len(args.record.checkpoint_dir) > 0:
        train_collector.save(args.record.checkpoint_dir, "pretrain_buffers.bf")
        save_to_pickle(os.path.join(args.record.checkpoint_dir, "pretrain_result.pkl"), pretrain_result) # directory created by collector.save

    # collect initial test trials
    collect_test_trials(initial_logger, option, test_collector, args.policy.logging.max_terminate_step, 0, args.policy.logging.initial_trials, True)
    initial_perf, initial_success = initial_logger.print_log(0, force=True)

    if args.policy.learn.post_random_iters > 0:
        for i in range(args.policy.learn.post_random_iters):
            losses = option.policy.update(args.train.batch_size, train_collector.buffer, train_collector.her_buffer)
            train_logger.log_losses(losses)
            train_logger.print_losses(i)
            
    for i in range(args.train.num_iters):  # total step
        tc_iter_start = time.time()
        collect_result = train_collector.collect(n_step=args.train.num_steps, demonstrate = args.collect.demonstrate_option) # TODO: make n-episode a usable parameter for collect
        tc_collect = time.time()

        train_logger.log_results(collect_result)
        tc_logging = time.time()
        if i % args.policy.logging.log_interval == 0:
            collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, i, args.policy.logging.initial_trials, False)
        # train option.policy with a sampled batch data from buffer
        tc_test = time.time()
        losses = option.policy.update(args.train.batch_size, train_collector.buffer, train_collector.her_buffer)
        tc_train = time.time()
        train_logger.log_losses(losses)
        if option.inline_trainer.train: option.inline_trainer.run_train(i, train_collector.full_buffer)

        # only prints if log interval is reached
        train_logger.print_log(i)
        test_logger.print_log(i)
        if i % (args.policy.logging.log_interval * 2) == 0:
            buffer_printouts(args, train_collector, option)

        tc_print = time.time()

        if args.record.save_interval > 0 and (i+1) % args.record.save_interval == 0:
            if len(args.record.save_dir) > 0: option.save(args.record.save_dir)
            if len(args.record.checkpoint_dir) > 0:
                graph.save(args.record.checkpoint_dir)
                train_collector.save(args.record.checkpoint_dir, "RL_buffers.bf")

        tc_save = time.time()
        if args.policy.primacy.reset_frequency > 0 and i % args.policy.primacy.reset_frequency == 0 and i != 0 and i < args.policy.primacy.stop_resets + 1:
            print("performing reset")
            option.policy.reset_select_params()
            if args.policy.primacy.primacy_iters > 0:
                for i in range(args.policy.primacy.primacy_iters):
                    losses = option.policy.update(args.train.batch_size, train_collector.buffer, train_collector.her_buffer)
                    train_logger.log_losses(losses)
                    train_logger.print_losses(i)
        tc_primacy = time.time()

        perf_times = collect_result["perf"]
        print(f"times: collect {tc_collect - tc_iter_start}, logging {tc_logging - tc_collect}, test {tc_test - tc_logging}, train \
            {tc_train - tc_test}, print {tc_print - tc_train}, save {tc_save - tc_print}, total {tc_primacy - tc_iter_start}")
        print(f"action {perf_times['action']} step {perf_times['step']} term {perf_times['term']} inline {perf_times['inline']} process {perf_times['process']} record {perf_times['record']} aggregate {perf_times['aggregate']} total {perf_times['total']}")

    if args.record.save_interval > 0: 
        option.save(args.record.save_dir)
        if len(args.record.checkpoint_dir) > 0: train_collector.save(args.record.checkpoint_dir, "RL_buffers.bf")
    
    # final logging step to determinie the performance of the final policy
    test_logger.print_log(i, force=True)
    test_logger.reset()
    collect_test_trials(test_logger, option, test_collector, args.policy.logging.max_terminate_step, i, args.policy.logging.initial_trials, False)
    final_perf, final_success = test_logger.print_log(i, force=True)

    logging.info(f"performance comparison: {initial_perf}, {final_perf}")
    print("performance comparison", initial_perf, final_perf)
    if initial_perf < final_perf - 2:
        return True # trained is true
    return False
