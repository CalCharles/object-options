import torch
import numpy as np
import datetime, os
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.policy import RainbowPolicy, DQNPolicy, SACPolicy
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


from Baselines.RIDE.ride_module import RIDEPolicy
from Baselines.RIDE.ride_network import DQN, Rainbow, RIDEModule



def initialize_data(args, policy, train_envs, test_envs):

    print(train_envs, test_envs)
    if len(args.collect.prioritized_replay) == 0:
        buffer = VectorReplayBuffer(
            args.collect.buffer_len,
            buffer_num=len(train_envs),
            ignore_obs_next=False,
            save_only_last_obs=False,
            stack_num=1
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.collect.buffer_len,
            buffer_num=len(train_envs),
            ignore_obs_next=False,
            save_only_last_obs=False,
            stack_num=1,
            alpha=args.collect.prioritized_replay[0],
            beta=args.collect.prioritized_replay[1],
            weight_norm=False
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
    return train_collector, test_collector, buffer,  logger

# TODO: allow for pointnet implementations

def initialize_ride_continuous(args):
    args.max_action = args.RIDE.action_space.high[0]
    state_shape = args.RIDE.state_shape
    action_shape = args.RIDE.action_shape
    action_space = args.RIDE.action_space
    device = 'cuda:' + str(args.torch.gpu) if args.torch.gpu >= 0 else 'cpu'
    hidden_sizes = [int(hs) for hs in args.network.hidden_sizes]
    actor_lr = args.network.optimizer.lr
    critic_lr = args.network.optimizer.alt_lr
    auto_alpha = args.policy.auto_alpha
    alpha_lr = args.policy.alpha_lr
    alpha = args.policy.sac_alpha
    tau = args.policy.tau # default 0.0005
    gamma = args.policy.discount_factor
    output_dim = args.network.embed_inputs
    conv_mode = args.RIDE.conv_mode
    num_objects = args.num_objects if conv_mode else -1
    n_step = 1

    # model
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    if auto_alpha:
        target_entropy = -np.prod(action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=n_step,
        action_space=action_space,
    )

    feature_net = DQN(
        action_shape[0],
        *state_shape, 
        hidden_sizes,
        output_dim,
        device, 
        features_only=True,
        num_objects = num_objects,
    )
    action_dim = np.prod(action_shape)
    feature_dim = feature_net.output_dim # args.net.embed_dim
    ride_net = RIDEModule(
        feature_net.net,
        feature_dim,
        action_dim,
        hidden_sizes=hidden_sizes,
        device=device,
        num_objects=num_objects,
        discrete_actions = False,
    )

    ride_optim = torch.optim.Adam(ride_net.parameters(), lr=args.network.optimizer.lr)
    policy = RIDEPolicy(
        policy, ride_net, ride_optim, args.RIDE.lr_scale, args.RIDE.reward_scale,
        args.RIDE.forward_loss_weight, args.RIDE.pseudocount_lambda, discrete_actions = False,
    ).to(device)
    return policy

def initialize_ride_discrete(args):
    state_shape = args.RIDE.state_shape
    action_shape = args.RIDE.action_shape[0]
    num_atoms = args.policy.rainbow.num_atoms
    noisy_std = 0.5
    device = 'cuda:' + str(args.torch.gpu) if args.torch.gpu >= 0 else 'cpu'
    gamma = args.policy.discount_factor
    v_min = 0
    v_max = 30
    n_step = args.policy.lookahead
    target_update_freq = args.policy.tau
    hidden_sizes = [int(hs) for hs in args.network.hidden_sizes]
    output_dim = args.network.embed_inputs
    dueling = args.policy.rainbow.is_dueling
    noisy = args.policy.rainbow.is_noisy
    lr = args.network.optimizer.lr
    conv_mode = args.RIDE.conv_mode
    num_objects = args.num_objects if conv_mode else -1

    net = Rainbow(
        action_shape,
        *state_shape,
        hidden_sizes,
        output_dim,
        num_atoms,
        noisy_std,
        device,
        is_dueling=dueling,
        is_noisy=noisy,
        num_objects = num_objects
    )
    # net = DQN(
    #     action_shape,
    #     *state_shape, 
    #     hidden_sizes,
    #     output_dim,
    #     device,
    #     num_objects=num_objects,
    # )
    optim = torch.optim.Adam(net.parameters(), lr=lr)
    # print(net, state_shape)
    # define policy
    policy = RainbowPolicy(
        net,
        optim,
        gamma,
        num_atoms,
        v_min,
        v_max,
        n_step,
        target_update_freq=target_update_freq,
    ).to(device)
    # policy = DQNPolicy(
    #     net,
    #     optim,
    #     gamma,
    #     n_step,
    #     target_update_freq=target_update_freq
    # )



    feature_net = DQN(
        action_shape,
        *state_shape, 
        hidden_sizes,
        output_dim,
        device,
        num_objects=num_objects,
        features_only=True
    )

    action_dim = np.prod(action_shape)
    feature_dim = feature_net.output_dim # args.net.embed_dim
    ride_net = RIDEModule(
        feature_net.net,
        feature_dim,
        action_dim,
        hidden_sizes=hidden_sizes,
        device=device,
        num_objects=num_objects,
        discrete_actions = True,
    )

    ride_optim = torch.optim.Adam(ride_net.parameters(), lr=args.network.optimizer.lr)
    policy = RIDEPolicy(
        policy, ride_net, ride_optim, args.RIDE.lr_scale, args.RIDE.reward_scale,
        args.RIDE.forward_loss_weight, args.RIDE.pseudocount_lambda, discrete_actions = True,
    ).to(device)
    return policy