import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from tianshou.data import Collector, Batch, ReplayBuffer
from State.object_dict import ObjDict
from Network.distributional_network import DiagGaussianForwardNetwork, InteractionNetwork
from Network.network_utils import pytorch_model
from Causal.interaction_test import InteractionTesting
from Causal.active_mask import ActiveMasking
from Environment.Normalization.norm import NormalizationModule


def assign_distribution(dist):
        if dist == "Gaussian": return torch.distributions.normal.Normal
        # elif dist == "Discrete": return Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        # elif dist == "MultiBinary": return Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
        else: raise NotImplementedError

def load_interaction(pth):
    for name in os.listdir(pth):
        if "inter_model.pt" in name:
            break
    return torch.load(os.path.join(pth, name))

def get_params(model, full_args, is_pair):
    active_model_args = copy.deepcopy(full_args.network)
    active_model_args.num_inputs = model.inter_select.output_size()
    active_model_args.num_outputs = model.target_select.output_size()

    passive_model_args = copy.deepcopy(full_args.network)
    passive_model_args.num_inputs = model.target_select.output_size()
    passive_model_args.num_outputs = model.target_select.output_size()

    interaction_model_args = copy.deepcopy(full_args.network)
    interaction_model_args.num_inputs = model.inter_select.output_size()
    interaction_model_args.num_outputs = 1
    
    if is_pair:
        pair = copy.deepcopy(full_args.network.pair)
        pair.object_dim = model.obj_dim
        pair.first_obj_dim = model.first_obj_dim
        pair.post_dim = -1
        pair.aggregate_final = False
        active_model_args.pair, passive_model_args.pair, interaction_model_args.pair = copy.deepcopy(pair), copy.deepcopy(pair), copy.deepcopy(pair)
    return active_model_args, passive_model_args, interaction_model_args

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, args, environment):
        super().__init__()
        # set input and output
        self.name = args.object_names.primary_parent + "->" + args.object_names.target
        self.names = args.object_names
        self.inter_select = args.inter_select
        self.target_select = args.target_select
        self.parent_selectors = args.parent_selectors
        self.parent_select = args.parent_select
        self.controllable = args.controllable

        # if we are predicting the dynamics
        self.predict_dynamics = args.inter.predict_dynamics
        
        # construct the active model
        self.first_obj_dim = [self.parent_selectors[p].output_size() for p in self.names.parents] # the first object dim is the combined length of the parents
        self.obj_dim = self.target_select.output_size() # the selector gets the size of a single instance
        active_model_args, passive_model_args, interaction_model_args = get_params(self, args, args.network.net_type == "pair")
        self.multi_instanced = environment.object_instanced[self.names.target]

        # set the distributions
        self.dist = assign_distribution("Gaussian") # TODO: only one kind of dist at the moment

        # set the forward model
        self.active_model = DiagGaussianForwardNetwork(active_model_args)

        # set the passive model
        self.passive_model = DiagGaussianForwardNetwork(passive_model_args)

        # construct the interaction model        
        self.interaction_model = InteractionNetwork(interaction_model_args)

        # set the testing module
        self.test = InteractionTesting(args.inter.interaction_testing)

        # set the normalization function
        self.norm = NormalizationModule(environment.object_range, environment.object_dynamics, args.object_names)

        # set the masking module to None as a placeholder
        self.mask = None

        # set values for proximity calculations
        self.proximity_epsilon, self.position_masks = args.inter.proximity_epsilon, environment.position_masks

        # set up cuda
        self.cuda() if args.torch.cuda else self.cpu()

    def save(self, pth):
        try:
            os.mkdir(pth)
        except OSError as e:
            pass
        torch.save(self, os.path.join(pth, self.name + "_inter_model.pt"))

    def cpu(self):
        super().cpu()
        self.active_model.cpu()
        self.interaction_model.cpu()
        self.passive_model.cpu()
        self.iscuda = False
        return self

    def cuda(self):
        super().cuda()
        self.active_model.cuda()
        self.interaction_model.cuda()
        self.passive_model.cuda()
        self.iscuda = True
        return self

    def reset_parameters(self):
        self.active_model.reset_parameters()
        self.interaction_model.reset_parameters()
        self.passive_model.reset_parameters()

    def _wrap_state(self, state):
        # takes in a state, either a full state (factored state dict (name to ndarray)), or tuple of (inter_state, target_state) 
        if type(state) == tuple:
            inter_state, tar_state = state
            inp_state = pytorch_model.wrap(inter_state, cuda=self.iscuda)
            tar_state = pytorch_model.wrap(tar_state, cuda=self.iscuda)
        else: # assumes that state is a batch or dict
            if (type(state) == Batch or type(state) == dict) and ('factored_state' in state): state = state['factored_state'] # use gamma on the factored state
            inp_state = pytorch_model.wrap(self.gamma(state), cuda=self.iscuda)
            tar_state = pytorch_model.wrap(self.delta(state), cuda=self.iscuda)
        return inp_state, tar_state

    def predict_next_state(self, state):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        inp_state, tar_state = self._wrap_state(state)

        rv = self.norm.reverse
        inter = pytorch_model.unwrap(self.interaction_model(inp_state))
        inter_bin = self.test(inter)

        # if predicting dynamics, add the mean of the model to the target state
        if self.predict_dynamics:
            fpred, ppred = tar_state + rv(self.active_model(inp_state)[0]), tar_state + rv(self.passive_model(tar_state)[0])
        else:
            fpred, ppred = rv(self.active_model(inp_state)[0]), rv(self.passive_model(tar_state)[0])
        
        # TODO: remove this conditional with appropriate slicing
        # select active or passive based on inter value
        if len(state.shape) == 1:
            return (inter, fpred) if pytorch_model.unwrap(inter) > self.interaction_prediction else (inter, ppred)
        else:
            pred = torch.stack((ppred, fpred), dim=1)
            intera = pytorch_model.wrap(intera.squeeze().long(), cuda=self.iscuda)
            pred = pred[torch.arange(pred.shape[0]).long(), intera]
        return pytorch_model.unwrap(inter), pytorch_model.unwrap(pred)

    def hypothesize(self, state):
        # takes in a full state (dict with raw_state, factored_state) or tuple of ndarray of input_state, target_state 
        # computes the interaction value, the mean, var of forward model, the mean, var of the passive model
        inter_state, tar_state = self._wrap_state(state)
        rv = self.norm.reverse
        mu_inter, var_inter = self.active_model(inter_state)
        pmu_inter, pvar_inter = self.passive_model(target_state)
        return (pytorch_model.unwrap(self.interaction_model(inp_state)),
            (rv(mu_inter), rv(var_inter)), 
            (rv(pmu_inter), rv(pvar_inter)))

    def check_interaction(self, inter):
        return self.test(inter)

    def get_active_mask(self):
        return self.test.selection_binary

    def interaction(self, batch):
        return self.interaction_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))

    def _target_dists(self, batch, params):
        target = batch.target_diff if self.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=self.iscuda)
        dist = self.dist(*params)
        log_probs = dist.log_prob(target)
        return target, dist, log_probs

    # likelihood functions (below) get the gaussian distributions output by the active and passive models
    def likelihoods(self, batch):
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        passive_params = self.passive_model(pytorch_model.wrap(batch.target, cuda=self.iscuda))
        inter = self.interaction_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        return active_params, passive_params, inter, target, active_dist, passive_dist, active_log_probs, passive_log_probs        

    def weighted_likelihoods(self, batch):
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        inter = self.interaction_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        target, dist, log_probs = self._target_dists(batch, active_params)
        return active_params, inter, dist, log_probs + torch.log(inter + 1e-6)
    
    def passive_likelihoods(self, batch):
        passive_params = self.passive_model(pytorch_model.wrap(batch.target, cuda=self.iscuda))
        target, dist, log_probs = self._target_dists(batch, passive_params)
        return passive_params, dist, log_probs

    def active_likelihoods(self, batch):
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        target, dist, log_probs = self._target_dists(batch, passive_params)
        return active_params, dist, log_probs

interaction_models = {'neural': NeuralInteractionForwardModel}