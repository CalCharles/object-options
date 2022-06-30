import numpy as np
import torch
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
cActor, cCritic = Actor, Critic
from tianshou.utils.net.discrete import Actor, Critic
dActor, dCritic = Actor, Critic
import tianshou as ts
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
from Network.network_utils import pytorch_model
from Network.ts_network import networks

rand_actor = ["sac"]

def reassign_optim(algo_policy, critic_lr, actor_lr):
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

def _init_critic(args, NetType, discrete_actions, action_shape, input_shape, final_layer, device, nets_optims):
    # discrete actions have action_shape outputs, while continuous have the actions as input
    # initializes critic network and optimizer
    cinp_shape = int(input_shape) if discrete_actions else int(input_shape + action_shape)
    cout_shape = int(action_shape) if discrete_actions else args.hidden_sizes[-1]
    critic = NetType(num_inputs=cinp_shape, num_outputs=cout_shape, action_dim=int(discrete_actions * action_shape), aggregate_final=True, continuous_critic=not discrete_actions, **args)
    if final_layer:
        if discrete_actions: critic = dCritic(critic, last_size=action_shape, device=device).to(device)
        else: critic = cCritic(critic, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.optimizer.lr)
    nets_optims += [critic, critic_optim]

def init_networks(args, input_shape, action_shape, discrete_actions):
    '''
    input_shape is the dimension of the input
    action_shape is the integer number of dimensions for the action
    discrete_action: boolean indicates whether it is discrete or not
    '''
    device = 'cpu' if not args.torch.cuda else 'cuda:' + str(args.torch.gpu)
    nets_optims = list()

    # initialize actor
    needs_actor = args.policy.learning_type in ["sac", "ddpg", "ppo"]
    final_layer = args.policy.learning_type in ["sac", "ddpg", "ppo"]
    rand_actor = args.policy.learning_type in ["sac"]
    args.actor_net.hidden_sizes = np.array(args.actor_net.hidden_sizes).astype(int)
    aout_shape = action_shape if final_layer and discrete_actions else args.actor_net.hidden_sizes[-1] # no final layer, else has final layer

    NetType = networks[args.actor_net.net_type] # TODO: have two sets of arguments
    args.actor_net.cuda = args.torch.cuda
    # shared actor initialization
    if needs_actor: # only actor-critic or actor algorithms need an actor
        # args.unbounded defines whether the action space is bounded
        actor = NetType(num_inputs=input_shape, num_outputs=aout_shape, aggregate_final=True, **args.actor_net)
        if final_layer:
            if discrete_actions: actor = dActor(actor, action_shape, device=device).to(device)
            else:
                if rand_actor: actor = ActorProb(actor, action_shape, device=device, conditioned_sigma=True).to(device)
                else: actor = cActor(actor, action_shape, device=device, max_action=torch.ones(action_shape).to(device)).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_net.optimizer.lr)
        nets_optims += [actor, actor_optim]

    # initialize critic
    NetType = networks[args.critic_net.net_type] # TODO: have two sets of arguments
    args.critic_net.hidden_sizes = np.array(args.critic_net.hidden_sizes).astype(int)
    args.critic_net.cuda = args.torch.cuda
    _init_critic(args.critic_net, NetType, discrete_actions, action_shape, input_shape, final_layer, device, nets_optims)
    if args.policy.learning_type == "sac": _init_critic(args, NetType, discrete_actions, action_shape, input_shape, final_layer, device, nets_optims)

    if args.policy.learning_type == "sac" and args.sac_alpha == -1:
        args.sac_alpha = (-action_shape, torch.zeros(1, requires_grad=True, device=device), torch.optim.Adam([log_alpha], lr=1e-4) )
    return nets_optims

def init_algorithm(args, nets, action_space, discrete_actions):
    noise = GaussianNoise(sigma=args.policy.epsilon_random) if args.policy.epsilon_random > 0 else None
    if args.policy.learning_type == "dqn":
        policy = ts.policy.DQNPolicy(*nets, discount_factor=args.policy.discount_factor, estimation_step=args.policy.lookahead, target_update_freq=int(args.policy.tau))
        policy.set_eps(args.policy.epsilon_random)
    elif args.policy.learning_type == "rainbow":
        assert args.policy.max_critic != 0
        policy = ts.policy.RainbowPolicy(*nets, discount_factor=args.policy.discount_factor, estimation_step=args.policy.lookahead,
         target_update_freq=int(args.policy.tau), v_min=-args.policy.max_critic, v_max=args.policy.max_critic, num_atoms=args.policy.num_atoms)
    elif args.policy.learning_type == "ppo": 
        if discrete_actions:
            policy = ts.policy.PPOPolicy(*nets, torch.distributions.Categorical, discount_factor=args.policy.discount_factor, max_grad_norm=None,
                                eps_clip=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95, # parameters hardcoded to defaults
                                reward_normalization=args.policy.reward_normalization, dual_clip=None, value_clip=False,
                                action_space=action_space)

        else:
            def dist(*logits):
                return Independent(Normal(*logits), 1)
            policy = ts.policy.PPOPolicy(
                *nets, dist, discount_factor=args.policy.discount_factor, max_grad_norm=None, eps_clip=0.2, vf_coef=0.5, 
                ent_coef=0.01, reward_normalization=args.policy.reward_normalization, advantage_normalization=1, recompute_advantage=0, 
                value_clip=False, gae_lambda=0.95, action_space=action_space)
    elif args.policy.learning_type == "ddpg": 
        policy = ts.policy.DDPGPolicy(*nets, tau=args.policy.tau, gamma=args.policy.discount_factor,
                                        exploration_noise=args.policy.epsilon_random,
                                        estimation_step=args.policy.lookahead, action_space=action_space,
                                        action_bound_method='clip')
    elif args.policy.learning_type == "sac":
        if discrete_actions:
            policy = ts.policy.DiscreteSACPolicy(
                    *nets, tau=args.policy.tau, gamma=args.policy.discount_factor, alpha=args.policy.sac_alpha, estimation_step=args.policy.lookahead,
                    reward_normalization=args.policy.reward_normalization, deterministic_eval=args.policy.deterministic_eval)
        else:
            policy = ts.policy.SACPolicy(*nets, tau=args.policy.tau, gamma=args.policy.policy.discount_factor, alpha=args.policy.sac_alpha,
                                                exploration_noise=args.policy.epsilon_random,
                                                estimation_step=args.policy.lookahead, action_space=action_space,
                                                action_bound_method='clip', deterministic_eval=args.policy.deterministic_eval)
        args.sample_HER = True
    # support as many algos as possible, at least ddpg, dqn SAC
    return policy