# main entry point for RIDE

from ReinforcementLearning.utils.RL_logger import RLLogger
from State.object_dict import ObjDict
from Environment.Environments.initialize_environment import initialize_environment
from train_interaction import init_names
from Baselines.RIDE.ride_module import RIDEPolicy
from Baselines.RIDE.ride_network import DQN, Rainbow, RIDEModule
from Baselines.RIDE.ride_initializer import initialize_data, initialize_ride_continuous, initialize_ride_discrete

import torch
import numpy as np
from tianshou.env import ShmemVectorEnv
from tianshou.trainer import offpolicy_trainer



def initialize_multienv(args):
    env = initialize_environment(args.environment, args.record, no_record=True)
    train_envs = ShmemVectorEnv(
        [
            lambda:
            initialize_environment(args.environment, args.record, no_record=True)
            for _ in range(args.RIDE.training_num)
        ]
    )
    test_envs = ShmemVectorEnv(
        [
            lambda:
            initialize_environment(args.environment, args.record, no_record=True)
            for _ in range(args.RIDE.test_num)
        ]
    )
    return env, train_envs, test_envs


def train_RIDE(args):
    # Similar to code in tianshou.examples
    torch.manual_seed(args.torch.torch_seed)
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    # env, record = initialize_environment(args.environment, args.record)
    env, train_envs, test_envs = initialize_multienv(args)
    state, info = env.reset()
    args.RIDE.state_shape = (state.shape[0] // env.num_objects, ) if args.RIDE.conv_mode else state.shape
    args.RIDE.action_shape = (env.num_actions, ) if env.discrete_actions else env.action_space.shape
    args.RIDE.action_space = env.action_space
    args.num_objects = env.num_objects
    if not env.discrete_actions: args.RIDE.max_action = env.action_space.high
    ride_policy = initialize_ride_discrete(args) if env.discrete_actions else initialize_ride_continuous(args)
    train_collector, test_collector, replay_buffer, logger = initialize_data(args, ride_policy, train_envs, test_envs)
    
    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(args.record.net_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= args.train.reward_threshold

    def train_fn(epoch, env_step):
        # eps annnealing, just a demo
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    print(int(args.train.num_iters // args.policy.logging.log_interval), 
        args.policy.logging.log_interval,
        args.train.num_steps)
    result =  offpolicy_trainer(
        ride_policy,
        train_collector,
        test_collector,
        args.train.num_iters, # epoch
        args.policy.logging.log_interval, # each "epoch" is the number of log intervals, step per epoch
        args.train.num_steps, # steps per collect
        args.policy.logging.test_trials, # test number
        args.train.batch_size,
        save_best_fn=False,
        logger=logger,
        update_per_step=args.policy.learn.grad_epoch,
        test_in_train=False,
    )