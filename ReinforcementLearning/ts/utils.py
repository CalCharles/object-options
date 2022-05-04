
def reassign_optim(algo_policy):
    if hasattr(algo_policy, "optim"):
        algo_policy.optim = torch.optim.Adam(algo_policy.model.parameters(), lr=critic_lr)
    if hasattr(algo_policy, "critic_optim"):
        algo_policy.critic_optim = torch.optim.Adam(algo_policy.critic.parameters(), lr=critic_lr)
    if hasattr(algo_policy, "critic1_optim"):
        algo_policy.critic1_optim = torch.optim.Adam(algo_policy.critic1.parameters(), lr=critic_lr)
    if hasattr(algo_policy, "critic2_optim"):
        algo_policy.critic2_optim = torch.optim.Adam(algo_policy.critic2.parameters(), lr=critic_lr)
    if hasattr(algo_policy, "actor_optim"):
        algo_policy.actor_optim = torch.optim.Adam(algo_policy.actor.parameters(), lr=actor_lr)
    if hasattr(algo_policy, "alpha_optim"):
        algo_policy.alpha_optim = torch.optim.Adam(_alpha, lr=1e-4) # TODO alpha learning rate not hardcoded


def assign_device(algo_policy, discrete_actions, device):
    '''
    Tianshou stores the device on a variable inside the internal models. This must be pudated when changing CUDA/CPU devices
    '''
    if type(device) == int:
        device = 'cuda:' + str(device)
    if hasattr(algo_policy, "actor"):
        if not discrete_actions:
            algo_policy.actor.mu.device = device
            algo_policy.actor.sigma.device = device
        else:
            algo_policy.actor.last.device = device
        algo_policy.actor.device = device
    if hasattr(algo_policy, "critic"):
        algo_policy.critic.last.device = device
        algo_policy.critic.device = device
    if hasattr(algo_policy, "critic1"):
        algo_policy.critic1.last.device = device
        algo_policy.critic1.device = device
    if hasattr(algo_policy, "critic2"):
        algo_policy.critic2.last.device = device
        algo_policy.critic2.device = device
    algo_policy.device = device

def _init_critic(args, NetType, discrete_actions, action_shape, input_shape, final_layer, nets_optims):
    # discrete actions have action_shape outputs, while continuous have the actions as input
    # initializes 
    cinp_shape = input_shape if discrete_actions else int(input_shape + action_shape)
    cout_shape = args.hidden_sizes[-1] if discrete_actions else 1
    critic = NetType(num_inputs=cinp_shape, num_outputs=cout_shape, action_dim=int(discrete_actions * action_shape), aggregate_final=True, continuous_critic=not discrete_actions, **args)
    if final_layer:
        if discrete_actions: critic = Critic(critic, last_size=action_shape, device=device).to(device)
        else: critic = Critic(critic, device=device).to(device)
        critic_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    nets_optims += [critic, critic_optim]

def init_networks(args, input_shape, action_shape, discrete_actions):
    '''
    input_shape is the dimension of the input
    action_shape is the integer number of dimensions for the action
    discrete_action: boolean indicates whether it is discrete or not
    '''
    device = 'cpu' if not args.cuda else 'cuda:' + str(args.gpu)
    nets_optims = list()

    # initialize actor
    needs_actor = args.algo_name in ["sac", "ddpg", "ppo"]
    final_layer = args.algo_name in ["sac", "ddpg", "ppo"]
    aout_shape = action_shape if final_layer and discrete_actions else args.hidden_sizes[-1] # no final layer, else has final layer

    # shared actor initialization
    if needs_actor: # only actor-critic or actor algorithms need an actor
        # args.unbounded defines whether the action space is bounded
        NetType = networks[args.policy_type]
        actor = NetType(num_inputs=input_shape, num_outputs=aout_shape, aggregate_final=True, **args)
        if final_layer:
            if discrete_actions: actor = Actor(actor, action_shape, device=device).to(device)
            else: actor = ActorProb(actor, action_shape, device=device, conditioned_sigma=True).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
        nets_optims += [actor, actor_optim]

    # initialize critic
    _init_critic(args, NetType, discrete_actions, action_shape, input_shape, final_layer, nets_optims)
    if args.algo_name == "sac": _init_critic(args, NetType, discrete_actions, action_shape, input_shape, final_layer, nets_optims)

    if args.algo_name == "sac" and args.sac_alpha == -1:
        args.sac_alpha = (-action_shape, torch.zeros(1, requires_grad=True, device=device), torch.optim.Adam([log_alpha], lr=1e-4) )
    return nets_optims

def init_algorithm(args, nets):
    noise = GaussianNoise(sigma=args.epsilon) if args.epsilon > 0 else None
    if args.algo_name == "dqn":
        policy = ts.policy.DQNPolicy(*nets, discount_factor=args.discount_factor, estimation_step=args.lookahead, target_update_freq=int(args.tau))
        policy.set_eps(args.epsilon)
    elif args.algo_name == "rainbow":
        assert args.max_critic != 0
        policy = ts.policy.RainbowPolicy(*nets, discount_factor=args.discount_factor, estimation_step=args.lookahead,
         target_update_freq=int(args.tau), v_min=-args.max_critic, v_max=args.max_critic, num_atoms=args.num_atoms)
        policy.set_eps(args.epsilon)
    elif args.algo_name == "ppo": 
        if args.discrete_actions:
            policy = ts.policy.PPOPolicy(*nets, torch.distributions.Categorical, discount_factor=args.discount_factor, max_grad_norm=None,
                                eps_clip=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95, # parameters hardcoded to defaults
                                reward_normalization=args.reward_normalization, dual_clip=None, value_clip=False,
                                action_space=args.action_space)

        else:
            def dist(*logits):
                return Independent(Normal(*logits), 1)
            policy = ts.policy.PPOPolicy(
                *nets, dist, discount_factor=args.discount_factor, max_grad_norm=None, eps_clip=0.2, vf_coef=0.5, 
                ent_coef=0.01, reward_normalization=args.reward_normalization, advantage_normalization=1, recompute_advantage=0, 
                value_clip=False, gae_lambda=0.95, action_space=args.action_space)
    elif args.algo_name == "ddpg": 
        policy = ts.policy.DDPGPolicy(*nets, tau=args.tau, gamma=args.gamma,
                                        exploration_noise=args.exploration_noise,
                                        estimation_step=args.lookahead, action_space=args.action_space,
                                        action_bound_method='clip')
    elif args.algo_name == "sac":
        print(args.sac_alpha)
        if args.discrete_actions:
            policy = ts.policy.DiscreteSACPolicy(
                    *nets, tau=args.tau, gamma=args.gamma, alpha=args.sac_alpha, estimation_step=args.lookahead,
                    reward_normalization=args.reward_normalization, deterministic_eval=args.deterministic_eval)
        else:
            policy = ts.policy.SACPolicy(*nets, tau=args.tau, gamma=args.gamma, alpha=args.sac_alpha,
                                                exploration_noise=args.exploration_noise,
                                                estimation_step=args.lookahead, action_space=args.action_space,
                                                action_bound_method='clip', deterministic_eval=args.deterministic_eval)
    elif args.algo_name == 'isl':
        policy = IteratedSupervisedPolicy(args.actor, args.actor_optim, label_smoothing=args.label_smoothing)
        args.sample_HER = True
    # support as many algos as possible, at least ddpg, dqn SAC
    return policy