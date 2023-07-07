# main entry point for RIDE

from ReinforcementLearning.utils.RL_logger import RLLogger
from State.object_dict import ObjDict
from Environment.Environments.initialize_environment import initialize_environment
from train_interaction import init_names
import torch
import numpy as np
from tianshou.env import ShmemVectorEnv
from Baselines.RIDE.ride_module import RIDEPolicy
from Baselines.RIDE.ride_module import DQN, Rainbow, RIDEModule

def initialize_data(args, train_envs, test_envs)

    buffer = VectorReplayBuffer(
        args.collect.buffer_len,
        buffer_num=len(train_envs),
        ignore_obs_next=True,
        save_only_last_obs=True,
        stack_num=1
    )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # logger
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "ride"
    log_name = os.path.join(args.environment.env, "RIDE", str(args.environment.seed), now)
    log_path = os.path.join(args.record.log_filename, log_name)

    # if args.logger == "wandb":
    #     logger = WandbLogger(
    #         save_interval=1,
    #         name=log_name.replace(os.path.sep, "__"),
    #         run_id=args.resume_id,
    #         config=args,
    #         project=args.wandb_project,
    #     )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    # if args.logger == "tensorboard":
    logger = TensorboardLogger(writer)
    # else:  # wandb
    #     logger.load(writer)


def initialize_ride_dicrete(args):
    state_shape = args.RIDE.state_shape
    action_shape = args.RIDE.action_shape
    num_atoms = args.policy.rainbow.num_atoms
    noisy_std = 0.5
    device = 'cuda:' + str(args.torch.gpu) if args.torch.gpu >= 0 else 'cpu'
    gamma = args.policy.discount_factor
    v_min = -10
    v_max = 100
    n_step = 1
    target_update_freq = args.policy.tau
    hidden_sizes = args.network.hidden_sizes
    output_dim = args.network.embed_inputs

    net = Rainbow(
        *state_shape,
        action_shape,
        hidden_sizes,
        output_dim,
        num_atoms,
        noisy_std,
        device,
        is_dueling=not args.no_dueling,
        is_noisy=not args.no_noisy
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    # define policy
    policy = RainbowPolicy(
        net,
        optim,
        gamma,
        num_atoms,
        v_min,
        v_max,
        n_step,
        target_update_freq=target_update_freq
    ).to(device)



    feature_net = DQN(
        *state_shape, 
        action_shape,
        hidden_sizes,
        output_dim,
        device, 
        features_only=True
    )
    action_dim = np.prod(action_shape)
    feature_dim = feature_net.output_dim # args.net.embed_dim
    icm_net = IntrinsicCuriosityModule(
        feature_net.net,
        feature_dim,
        action_dim,
        hidden_sizes=[hidden_sizes],
        device=device,
    )

    icm_optim = torch.optim.Adam(icm_net.parameters(), lr=args.network.optimizer.lr)
    policy = ICMPolicy(
        policy, icm_net, icm_optim, args.RIDE.lr_scale, args.RIDE.reward_scale,
        args.RIDE.forward_loss_weight
    ).to(device)

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
    state = env.reset()
    args.RIDE.state_shape = state.shape
    args.RIDE.action_shape = (env.num_actions, ) if env.discrete_actions else env.action_space.shape
    ride_policy = initialize_ride_discrete(args) if args.environment.env == "Breakout" else initialize_ride_continuous(args)
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

    result =  offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        int(args.train.num_iters // args.policy.logging.log_interval),
        args.policy.logging.log_interval, # each "epoch" is the number of log intervals
        args.train.num_steps,
        args.policy.logging.log_interval,
        args.train.batch_size,
        save_best_fn=False,
        logger=logger,
        update_per_step=args.policy.learn.grad_epoch,
        test_in_train=False,
    )