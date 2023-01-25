from Baselines.HyPE.Policy.collector import HyPECollector
from Baselines.HyPE.Policy.skill import load_skill, Skill, Extractor
from Baselines.HyPE.Policy.primitive import PrimitiveSkill
from Baselines.HyPE.Policy.policy import Policy
from Baselines.HyPE.Policy.test_trials import collect_test_trials
from Baselines.HyPE.Policy.HyPE_buffer import HyPEBuffer, HyPEPrioBuffer
from Baselines.HyPE.Policy.temporal_extension_manager import TemporalExtensionManager
from Baselines.HyPE.generate_reward_fn import load_reward, generate_extractor_norm
from Baselines.HyPE.Reward.true_reward import TrueReward
from Baselines.HyPE.Policy.default_extractors import BreakoutExtractor, RoboPushingExtractor
from ReinforcementLearning.utils.RL_logger import RLLogger
from State.object_dict import ObjDict
from Environment.Environments.initialize_environment import initialize_environment
from train_interaction import init_names
import torch
import numpy as np

def train_skill(args):
    args.object_names = init_names(args.train_edge)
    torch.cuda.set_device(args.torch.gpu)
    environment, record = initialize_environment(args.environment, args.record)
    last_skill = load_skill(args.record.load_dir, args.object_names.primary_parent) if args.object_names.primary_parent != "Action" else PrimitiveSkill(environment)
    last_skill.zero_epsilon()
    last_skill.toggle_assignment_mode()
    last_skill.set_cutoff(args.skill.temporal_extend)
    policy_iters_schedule = (lambda i: args.skill.policy_iters * (min(2 ** ( i // args.skill.policy_iters_schedule), 8))) if args.skill.policy_iters_schedule  > 0 else (lambda i: args.skill.policy_iters)
    num_repeats_schedule = (lambda i: args.skill.num_repeats * (min(2 ** ( i // args.skill.num_repeats_schedule), 4))) if args.skill.num_repeats_schedule  > 0 else (lambda i: args.skill.num_repeats)
    epsilon_schedule = (lambda i: max(np.exp(- i / args.skill.epsilon_schedule), args.skill.epsilon_random)) if args.skill.epsilon_schedule > 0 else (lambda i: args.skill.epsilon_random)


    if args.reward.true_reward:
        reward = TrueReward(args.object_names) # actual reward handling in collector
    else:
        reward = load_reward(args)
        reward.set_extractor_norm(*generate_extractor_norm(args.object_names, environment))
        reward.set_params(args.reward.reward_base, args.reward.param_reward, args.reward.changepoint_reward, args.reward.one_mode)
    models = ObjDict()
    models.temporal_extension_manager= TemporalExtensionManager(args)
    if args.reward.true_reward:
        if args.environment.env == "Breakout":
            models.extractor = BreakoutExtractor(args.skill.input_scaling, args.skill.normalized, environment.num_blocks)
            pair_args = models.extractor.pair_args()
        elif args.environment.env == "RoboPushing":
            models.extractor = RoboPushingExtractor(args.skill.input_scaling, args.skill.normalized, environment.num_obstacles)
            pair_args = models.extractor.pair_args()
    else:
        models.extractor = Extractor(reward.extractor, reward.norm, args.skill.obs_components, args.skill.input_scaling, args.skill.normalized)
        pair_args = None # TODO: not supported
    input_shape = models.extractor.get_obs(environment.reset()).shape # get the first state
    models.reward_model = reward
    num_actions = last_skill.num_skills + environment.num_actions if args.skill.include_primitive else last_skill.num_skills
    policies = [Policy(num_actions, input_shape, args, pair_args) for i in range(reward.num_modes)]
    skill = Skill(args, policies, models, last_skill)#, augment_primitive=environment.num_actions if args.skill.augment_primitive else num_actions)
    # train_loggers = [RLLogger(args.object_names.target + "_train" + str(i), args.record.record_graphs, args.skill.log_interval, args.skill.train_log_maxlen, args.record.log_filename) for
    #                         i in range(reward.num_modes)]
    train_logger = RLLogger(args.object_names.target + "_train", args.record.record_graphs, args.skill.log_interval, args.skill.train_log_maxlen, args.record.log_filename)
    test_logger = RLLogger(args.object_names.target + "_test", args.record.record_graphs, args.skill.log_interval, args.skill.train_log_maxlen, args.record.log_filename)

    buffers = [(HyPEBuffer(args.skill.buffer_len) if len(args.skill.prioritized_replay) == 0 else HyPEPrioBuffer(args.skill.buffer_len, *args.skill.prioritized_replay) )for i in range(reward.num_modes)]
    print(policy_iters_schedule(0), policy_iters_schedule)
    train_collector = HyPECollector(environment, buffers, skill, skill.extractor, False, record, policy_iters_schedule(0), args.skill.merge_data, use_true_reward = args.reward.true_reward)
    test_collector = HyPECollector(environment, None, skill, skill.extractor, True, record, policy_iters_schedule(0), use_true_reward = args.reward.true_reward)

    print(args.skill.num_iters)
    for i in range(args.skill.num_iters):  # total step
        if args.skill.epsilon_schedule > 0 and i % args.skill.epsilon_schedule == 0:
            skill.set_epsilon(epsilon_schedule(i))
        policy_iters = policy_iters_schedule(i)
        num_repeats = num_repeats_schedule(i)
        collect_result = train_collector.collect(policy_iters, num_repeats, demonstrate = args.skill.demonstrate) # TODO: make n-episode a usable parameter for collect

        batch_size = policy_iters * num_repeats if args.skill.learning_type == "cmaes" else args.skill.batch_size
        for k, policy in enumerate(skill.policies):
            losses = skill.policies[k].update(batch_size, buffers[k], policy_iters=policy_iters)
        train_logger.log_results(collect_result)
        train_logger.log_losses(losses)
        train_logger.print_log(i)

        if i % args.skill.log_interval == 0:
            print("testing")
            collect_test_trials(test_logger, skill, test_collector, args.skill.test_policy_iters, num_repeats, i, args.skill.test_trials, False)
            test_logger.total_steps, test_logger.total_episodes, test_logger.total_true_episodes = train_logger.total_steps, train_logger.total_episodes, train_logger.total_true_episodes
            test_logger.print_log(i)
        # only prints if log interval is reached
        # if i % (args.skill.log_interval * 2) == 0:
        #     buffer_printouts(args, train_collector, option)


        if args.record.save_interval > 0 and (i+1) % args.record.save_interval == 0:
            if len(args.record.save_dir) > 0: skill.save(args.record.save_dir)
            if len(args.record.checkpoint_dir) > 0:
                skill.save(args.record.checkpoint_dir)
                train_collector.save(args.record.checkpoint_dir, "RL_buffers.bf")
    if len(args.record.save_dir) > 0: skill.save(args.record.save_dir)

        # print(f"times: collect {tc_collect - tc_iter_start}, logging {tc_logging - tc_collect}, test {tc_test - tc_logging}, train \
        #     {tc_train - tc_test}, print {tc_print - tc_train}, save {tc_save - tc_print}, total {tc_primacy - tc_iter_start}")
        # print(f"action {perf_times['action']} step {perf_times['step']} term {perf_times['term']} inline {perf_times['inline']} process {perf_times['process']} record {perf_times['record']} aggregate {perf_times['aggregate']} total {perf_times['total']}")
