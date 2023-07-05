# main entry point for RIDE

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


def train_HRL():
    # Similar to code in tianshou.examples
    torch.manual_seed(args.torch.torch_seed)
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    args = get_args()

    env = initialize_environment(args)
    ride_policy = initialize_ride(args, env)
    train_collector, test_collector, replay_buffer, logger = initialize_data(args, ride_policy, env)
    
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

    result = train_offline(
        policy,
        dynamics,
        train_collector,
        test_collector,
        args.train.epoch,
        args.train.step_per_epoch,
        args.train.repeat_per_collect,
        args.train.dynamics_per_collect,
        args.train.collect_per_update,
        args.train.test_num,
        args.train.batch_size,
        step_per_collect=args.train.step_per_collect,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
    )
