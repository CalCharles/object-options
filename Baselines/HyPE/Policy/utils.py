import numpy as np
import torch
import copy
from tianshou.utils.net.continuous import Actor, Critic, ActorProb
cActor, cCritic = Actor, Critic
from tianshou.utils.net.discrete import Actor, Critic
dActor, dCritic = Actor, Critic
import tianshou as ts
from tianshou.exploration import GaussianNoise, OUNoise
from tianshou.data import Batch, ReplayBuffer
from Network.network_utils import pytorch_model, reset_parameters, count_layers
from Network.ts_network import networks
from Baselines.HyPE.Policy.learning_algorithms import CMAES_optimizer

_rand_actor = ["sac"]
_actor_critic = ['ddpg', 'sac']
_double_critic = ['sac']

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

def _assign_device(algo_policy, algo_name, discrete_actions, device):
    '''
    Tianshou stores the device on a variable inside the internal models. This must be updated when changing CUDA/CPU devices
    '''
    if type(device) == int:
        if device < -1: device = 'cpu'
        else: device = 'cuda:' + str(device)
    if algo_name == "cmaes":
        for model in algo_policy.models:
            model.device = device
        algo_policy.best.device = device
        algo_policy.mean.device = device
    if hasattr(algo_policy, "actor"):
        if not discrete_actions and algo_name in _rand_actor:
            algo_policy.actor.mu.device = device
            if hasattr(algo_policy.actor, "sigma"): algo_policy.actor.sigma.device = device
        else:
            algo_policy.actor.last.device = device
        if hasattr(algo_policy.actor, "_max") and type(algo_policy.actor._max) == torch.Tensor:
            algo_policy.actor._max = algo_policy.actor._max.to(device)
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

def reset_params(reset_layers, algo_policy, init_form, algo_name):
    if reset_layers < 0:
        return
    elif reset_layers == 0: # use zero to reset all layers
        reset_layers = 10000 # a large number
    reset_at = reset_layers
    if hasattr(algo_policy, "actor"):
        reset_parameters(algo_policy.actor.last, init_form)
        reset_at -= count_layers(algo_policy.actor.last)
        reset_parameters(algo_policy.actor.preprocess, init_form, reset_at)
    reset_at = reset_layers
    if hasattr(algo_policy, "critic"):
        reset_parameters(algo_policy.critic.last, init_form, reset_at)
        reset_parameters(algo_policy.critic, init_form, reset_at - count_layers(algo_policy.critic.last))
    reset_at = reset_layers
    if hasattr(algo_policy, "critic1"):
        reset_parameters(algo_policy.critic1.last, init_form, reset_at)
        reset_parameters(algo_policy.critic1, init_form, reset_at - count_layers(algo_policy.critic1.last))
    reset_at = reset_layers
    if hasattr(algo_policy, "critic2"):
        reset_parameters(algo_policy.critic2.last, init_form, reset_at)
        reset_parameters(algo_policy.critic2, init_form, reset_at - count_layers(algo_policy.critic2.last))


def _init_critic(args, NetType, action_shape, input_shape, final_layer, device, nets_optims):
    # discrete actions have action_shape outputs, while continuous have the actions as input
    # initializes critic network and optimizer
    cinp_shape = int(input_shape[0])
    last_shape = args.pair.final_layers[-1] if args.net_type == "pair" else args.hidden_sizes[-1]
    cout_shape = int(action_shape)
    critic = NetType(num_inputs=cinp_shape, num_outputs=cout_shape, action_dim=0, aggregate_final=True, continuous_critic=False, **args)
    if final_layer:
        critic = dCritic(critic, last_size=action_shape, device=device).to(device)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.optimizer.lr)
    nets_optims += [critic, critic_optim]

def init_networks(args, input_shape, action_shape, pair_args=None):
    '''
    input_shape is the dimension of the input
    action_shape is the integer number of dimensions for the action
    discrete_action: boolean indicates whether it is discrete or not
    '''
    device = 'cpu' if not args.torch.cuda else 'cuda:' + str(args.torch.gpu)
    nets_optims = list()

    # initialize actor
    if args.critic_net.net_type == "pair":
        args.critic_net.pair.first_obj_dim, args.critic_net.pair.object_dim = pair_args
        args.critic_net.pair.aggregate_final, args.critic_net.pair.post_dim = True, 0 
        args.actor_net.pair.first_obj_dim, args.actor_net.pair.object_dim = pair_args
        args.actor_net.pair.aggregate_final, args.actor_net.pair.post_dim = True, 0 
    needs_actor = args.skill.learning_type in ["ppo", "cmaes"]
    needs_optim = args.skill.learning_type in ["ppo"]
    final_layer = args.skill.learning_type in ["ppo", "cmaes"]
    actor_args = copy.deepcopy(args.actor_net)
    last_shape = actor_args.pair.final_layers[-1] if actor_args.net_type == "pair" else actor_args.hidden_sizes[-1]
    aout_shape = action_shape if not final_layer else actor_args.hidden_sizes[-1]
    actor_args.activation_final = actor_args.activation_final if not final_layer else actor_args.activation
    actor_args.hidden_sizes = np.array(actor_args.hidden_sizes).astype(int) if not final_layer else np.array(actor_args.hidden_sizes).astype(int)[:-1]
    print(args.actor_net.net_type == "pair", args.actor_net.hidden_sizes)

    NetType = networks["basic"] # TODO: have two sets of arguments
    actor_args.cuda = args.torch.cuda
    actor_args.pair.aggregate_final, args.critic_net.pair.aggregate_final = True, True # TODO: object-centric decisionmaking not implemented
    # shared actor initialization
    if needs_actor: # only actor-critic or actor algorithms need an actor
        for i in range(args.skill.num_networks):
            actor = NetType(num_inputs=input_shape[0], num_outputs=aout_shape, aggregate_final=True, **actor_args)
            if final_layer:
                actor = dActor(actor, action_shape, device=device).to(device)
            if needs_optim:
                actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_args.optimizer.lr)
                nets_optims += [actor, actor_optim]
            else: nets_optims += [actor]
    print(args.skill.learning_type, nets_optims)

    # initialize critic
    needs_critic = args.skill.learning_type in ["dqn", "rainbow", "ppo"]
    if needs_critic:
        NetType = networks["basic"] if args.skill.learning_type != "rainbow" else networks["rainbow"] # TODO: have two sets of arguments
        # if args.critic_net.net_type == "pair": args.critic_net.pair.first_obj_dim += action_shape * int(not discrete_actions)
        args.critic_net.hidden_sizes = np.array(args.critic_net.hidden_sizes).astype(int)
        args.critic_net.cuda = args.torch.cuda
        args.critic_net.num_atoms, args.critic_net.is_dueling = args.skill.learn.num_atoms, args.skill.learn.is_dueling
        _init_critic(args.critic_net, NetType, action_shape, input_shape, final_layer, device, nets_optims)
    return nets_optims

def init_algorithm(args, nets, action_space):
    noise = GaussianNoise(sigma=args.skill.epsilon_random) if args.skill.epsilon_random > 0 else None
    if args.skill.learning_type == "dqn":
        policy = ts.policy.DQNPolicy(*nets, discount_factor=args.skill.learn.discount_factor, estimation_step=args.skill.learn.lookahead, target_update_freq=int(args.skill.learn.tau))
        policy.set_eps(args.skill.epsilon_random)
    elif args.skill.learning_type == "rainbow":
        # assert args.policy.max_min_critic != 0
        policy = ts.policy.RainbowPolicy(*nets, discount_factor=args.skill.learn.discount_factor, estimation_step=args.skill.learn.lookahead,
         target_update_freq=int(args.skill.learn.tau), v_min=args.skill.learn.max_min_critic[0], v_max=args.skill.learn.max_min_critic[1], num_atoms=args.skill.learn.num_atoms)
    elif args.skill.learning_type == "ppo": 
        policy = ts.policy.PPOPolicy(*nets, torch.distributions.Categorical, discount_factor=args.skill.learn.discount_factor, max_grad_norm=None,
                            eps_clip=0.2, vf_coef=0.5, ent_coef=0.01, gae_lambda=0.95, # parameters hardcoded to defaults
                            reward_normalization=args.skill.learn.reward_normalization, dual_clip=None, value_clip=False,
                            action_space=action_space)
    elif args.skill.learning_type == "cmaes":
        policy = CMAES_optimizer(args.skill.num_networks, args.skill.learn.init_var, nets, action_space.n, 
            gamma=args.skill.learn.discount_factor, elitism = args.skill.learn.elitism,
            dist_fn=torch.distributions.Categorical, reward_classes = None, needs_init=False) # TODO init Cmaes policy
    # support as many algos as possible, at least ddpg, dqn SAC
    return policy