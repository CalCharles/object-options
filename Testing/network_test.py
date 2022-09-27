import numpy as np
import string
from Record.file_management import load_from_pickle, save_to_pickle, create_directory
from Causal.Utils.interaction_selectors import CausalExtractor
from ReinforcementLearning.ts.utils import init_networks
from State.object_dict import ObjDict
from Environment.Normalization.norm import MappedNorm, NormalizationModule
from Testing.normalization_test import create_dummy_data
from Environment.Environments.initialize_environment import initialize_environment
from State.state_extractor import StateExtractor


  # Network:
  #   check layer resets are resetting properly
  #   check pair and MLP architectures for causal and policy
  #   check pair Slicing

def generate_networks(args, selection, normed=True):
    if normed:
        object_sizes = load_from_pickle("Testing/testing_data/norm_random_dims" + str(selection) + ".pkl")
        random_dicts = load_from_pickle("Testing/testing_data/norm_random_dicts" + str(selection) + ".pkl")
        environment = ObjDict({"object_sizes": object_sizes, "object_instanced": random_dicts[3]})
        names = random_dicts[2]
        args.norm = NormalizationModule(*random_dicts)
    else: 
        environment, record = initialize_environment(args.environment, args.record)
        names = init_names(args.train.train_edge)
        args.norm = NormalizationModule(environment.object_range, environment.object_dynamics, object_names, environment.object_instanced, mask)
    args.inter_extractor = CausalExtractor(names, environment)
    args.target_select, args.full_parent_select, args.additional_select, args.additional_selectors, \
            args.padi_selector, args.parent_select, args.inter_select = args.inter_extractor.get_selectors()

    args.extract.inter_extractor = args.inter_extractor
    args.extract.single_obs_setting = np.ones(6).tolist()
    args.extract.relative_obs_setting = np.ones(4).tolist()
    args.extract.norm = args.norm
    args.extract.object_names = names
    state_extractor = StateExtractor(args.extract)

    if normed:
        last_inputs, inputs, targets = load_from_pickle("Testing/testing_data/norm_inputs" + str(selection) + ".pkl")
        params, assign_param, masks, assign_mask = load_from_pickle("Testing/testing_data/extractor_inputs" + str(2) + "_" + str(selection) + ".pkl")
        inps, outs = list(), list()
        for lobs, obs, p, m in zip(last_inputs, inputs, params, masks):
            obs, lobs = {"factored_state": obs}, {"factored_state": lobs}
            inps.append(state_extractor.get_obs(obs, lobs, p, m))
            outs.append(state_extractor.get_obs(lobs, obs, p, m))
        save_to_pickle("Testing/testing_data/network_io" + str(selection) + ".pkl", (inps, outs, state_extractor))
    else:
        last_inputs = load_from_pickle("Testing/testing_data/trajectory_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
        last_inputs = [{'factored_state': inp} for inp in last_inputs]
        inputs = last_inputs[1:]
        actions = load_from_pickle("Testing/testing_data/trajectory_action_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl")
        params, assign_params, masks, assign_masks = load_from_pickle("Testing/testing_data/extractor_inputs_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl", )
        for lobs, obs, p, m in zip(last_inputs, inputs, params, masks):
            obs, lobs = {"factored_state": obs}, {"factored_state": lobs}
            inps.append(state_extractor.get_obs(lobs, lobs, p, m))
            outs.append(state_extractor.get_obs(lobs, obs, p, m))
        save_to_pickle("Testing/testing_data/network_io_" + args.environment.env + "_" + args.environment.variant + "_" + args.object_names.target + ".pkl", (inps, outs, state_extractor))
    
    args.actor_net.net_type = 'basic'
    args.critic_net.net_type = 'basic'
    args.policy.learning_type = "dqn"
    policy_nets_mlp_discrete_critic, policy_nets_mlp_discrete_critic_optim = init_networks(args, args.target_select(inputs[0]).shape[0], 5, True)
    args.policy.learning_type = "ddpg"
    policy_nets_mlp_cont_actor, policy_nets_mlp_cont_actor_optim, policy_nets_mlp_cont_critic, policy_nets_mlp_cont_critic_optim = init_networks(args, args.target_select(inputs[0]).shape[0], 3, False)
    args.actor_net.net_type = 'pair'
    args.critic_net.net_type = 'pair'
    args.policy.learning_type = "dqn"
    args.actor_net.pair.first_obj_dim, args.actor_net.pair.object_dim = state_extractor.first_obj_dim, state_extractor.obj_dim
    args.critic_net.pair.first_obj_dim, args.critic_net.pair.object_dim = state_extractor.first_obj_dim, state_extractor.obj_dim
    policy_nets_pair_discrete_critic, policy_nets_pair_discrete_critic_optim = init_networks(args, args.target_select(inputs[0]).shape[0], 5, True)
    args.policy.learning_type = "ddpg"
    policy_nets_pair_cont_actor, policy_nets_pair_cont_actor_optim, policy_nets_pair_cont_critic, policy_nets_pair_cont_critic_optim = init_networks(args, args.target_select(inputs[0]).shape[0], 3, False)
    
    save_to_pickle("Testing/testing_data/network_reset_mlpcont.pkl", policy_nets_mlp_cont_actor)
    save_to_pickle("Testing/testing_data/network_reset_paircont.pkl", policy_nets_pair_cont_actor)
    save_to_pickle("Testing/testing_data/network_reset_mlpcont.pkl", policy_nets_mlp_discrete_critic)
    save_to_pickle("Testing/testing_data/network_reset_paircont.pkl", policy_nets_pair_discrete_critic)
    save_to_pickle("Testing/testing_data/network_reset_mlpcont.pkl", policy_nets_mlp_cont_critic)
    save_to_pickle("Testing/testing_data/network_reset_paircont.pkl", policy_nets_pair_cont_critic)

    return policy_nets_mlp_cont_actor, policy_nets_pair_cont_actor, policy_nets_mlp_discrete_critic, policy_nets_pair_discrete_critic, policy_nets_mlp_cont_critic, policy_nets_pair_cont_critic

def policy_constructor(args, inp, act_shape, discrete_actions):
    nets_optims = init_networks(args, inp.squeeze().shape, act_shape, discrete_actions)
    return nets_optims

def test_network_construction(args, setting):
    # network initialization test
    inps, outs, extractor = load_from_pickle("Testing/testing_data/network_io" + str(setting) + ".pkl")
    args.actor_net.net_type = 'mlp'
    args.critic_net.net_type = 'mlp'
    policy_nets_mlp_discrete_actor, policy_nets_mlp_discrete_actor_optim, policy_nets_mlp_discrete_critic, policy_nets_mlp_discrete_critic_optim = init_networks(args, inps[0], 5, True)
    policy_nets_mlp_cont_actor, policy_nets_mlp_cont_actor_optim, policy_nets_mlp_cont_critic, policy_nets_mlp_cont_critic_optim = init_networks(args, inps[0], 3, False)

    args.actor_net.net_type = 'pair'
    args.critic_net.net_type = 'pair'
    policy_nets_pair_discrete_actor, policy_nets_pair_discrete_actor_optim, policy_nets_pair_discrete_critic, policy_nets_pair_discrete_critic_optim = init_networks(args, inps[0], 5, True)
    policy_nets_pair_cont_actor, policy_nets_pair_cont_actor_optim, policy_nets_pair_cont_critic, policy_nets_pair_cont_critic_optim = init_networks(args, inps[0], 3, False)

    extractor = load_from_pickle("Testing/testing_data/network_extractor.pkl")
    model = ObjDict({'extractor': extractor})
    ama, pma, ima = get_params(model,args,False,False)
    mlp_active_model = DiagGaussianForwardNetwork(ama)
    mlp_passive_model = DiagGaussianForwardNetwork(pma)
    mlp_interaction_model = InteractionNetwork(ima)
    ama, pma, ima = get_params(model,args,True,True)
    pair_multi_active_model = DiagGaussianForwardNetwork(ama)
    pair_multi_passive_model = DiagGaussianForwardNetwork(pma)
    pair_multi_interaction_model = InteractionNetwork(ima)
    ama, pma, ima = get_params(model,args,True,False)
    pair_active_model = DiagGaussianForwardNetwork(ama)
    pair_passive_model = DiagGaussianForwardNetwork(pma)
    pair_interaction_model = InteractionNetwork(ima)
    nets = [policy_nets_mlp_discrete_actor, policy_nets_mlp_discrete_critic, policy_nets_mlp_cont_actor, policy_nets_mlp_cont_critic, policy_nets_pair_discrete_actor, policy_nets_pair_discrete_critic, policy_nets_pair_cont_actor, policy_nets_pair_cont_critic, mlp_active_model, mlp_passive_model, mlp_interaction_model, pair_multi_active_model, pair_multi_passive_model, pair_multi_interaction_model, pair_active_model, pair_passive_model, pair_interaction_model]

    true_policy_nets_mlp_discrete_actor, true_policy_nets_mlp_discrete_critic, true_policy_nets_mlp_cont_actor, true_policy_nets_mlp_cont_critic, true_policy_nets_pair_discrete_actor, true_policy_nets_pair_discrete_critic, true_policy_nets_pair_cont_actor, true_policy_nets_pair_cont_critic, true_mlp_active_model, true_mlp_passive_model, true_mlp_interaction_model, true_pair_multi_active_model, true_pair_multi_passive_model, true_pair_multi_interaction_model, true_pair_active_model, true_pair_passive_model, true_pair_interaction_model = \
        load_from_pickle("Testing/testing_data/network_true_strings.pkl")
    true_nets = [true_policy_nets_mlp_discrete_actor, true_policy_nets_mlp_discrete_critic, true_policy_nets_mlp_cont_actor, true_policy_nets_mlp_cont_critic, true_policy_nets_pair_discrete_actor, true_policy_nets_pair_discrete_critic, true_policy_nets_pair_cont_actor, true_policy_nets_pair_cont_critic, true_mlp_active_model, true_mlp_passive_model, true_mlp_interaction_model, true_pair_multi_active_model, true_pair_multi_passive_model, true_pair_multi_interaction_model, true_pair_active_model, true_pair_passive_model, true_pair_interaction_model]

    constructed = [str(n) == tns for (n, tns) in zip(nets, true_nets)]

    # network backpropagation test
    outs = load_from_pickle("Testing/testing_data/mlp_discrete_actor_outs.pkl")
    original = copy.deepcopy(policy_nets_mlp_discrete_actor)
    for i in range(10000):
        idxes = np.random.choice(len(inps), size = 32)
        inp = pytorch_model.wrap(inps[idxes])
        out = pytorch_model.wrap(outs[idxes])
        nout = policy_nets_mlp_discrete_actor(inp)
        loss = (nout - pytorch_model.wrap(out)).abs().sum()
        run_optimizer(policy_nets_mlp_discrete_actor_optim, policy_nets_mlp_discrete_actor, loss)
    mlddiscdiff, mlpdiscdiff_list = compare_networks(policy_nets_mlp_discrete_actor, original)

    outs = load_from_pickle("Testing/testing_data/pair_cont_actor_outs.pkl")
    original = copy.deepcopy(policy_nets_pair_cont_actor)
    for i in range(10000):
        idxes = np.random.choice(len(inps), size = 32)
        inp = pytorch_model.wrap(inps[idxes])
        out = pytorch_model.wrap(outs[idxes])
        nout = policy_nets_pair_cont_actor(inp)
        loss = (nout - pytorch_model.wrap(out)).abs().sum()
        run_optimizer(policy_nets_pair_cont_actor_optim, policy_nets_pair_cont_actor, loss)
    mlddiscdiff, mlpdiscdiff_list = compare_networks(policy_nets_pair_cont_actor, original)


    # network parameter reset test
    true_net = load_from_pickle("Testing/testing_data/network_reset_mlpcont.pkl")
    reset_parameters(policy_nets_mlp_cont, "zero")
    reset_parameters(policy_nets_mlp_cont, "xnorm", 3)
    mlpcontdiff, diff_list = compare_networks(policy_nets_mlp_cont, true_net)

    true_net = load_from_pickle("Testing/testing_data/network_reset_pairdisc.pkl")
    reset_parameters(policy_nets_mlp_discrete_actor, "zero")
    reset_parameters(policy_nets_mlp_discrete_actor, "xnorm", 3)
    mlpdiscdiff, diff_list = compare_networks(policy_nets_mlp_discrete_actor, true_net)

    true_net = load_from_pickle("Testing/testing_data/network_reset_pairdisc.pkl")
    reset_parameters(policy_nets_pair_discrete, "zero")
    reset_parameters(policy_nets_pair_discrete, "xnorm", 3)
    pairdiscdiff, diff_list = compare_networks(policy_nets_pair_discrete, true_net)

    true_net = load_from_pickle("Testing/testing_data/network_reset_paircont.pkl")
    reset_parameters(policy_nets_pair_cont, "zero")
    reset_parameters(policy_nets_pair_cont, "xnorm", 3)
    paircontdiff, diff_list = compare_networks(policy_nets_pair_cont, true_net)

    return constructed, mlpcontdiff, mlpdiscdiff, pairdiscdiff, paircontdiff