import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from tianshou.data import Collector, Batch, ReplayBuffer
from Record.file_management import create_directory
from State.object_dict import ObjDict
from Network.distributional_network import DiagGaussianForwardNetwork, InteractionNetwork
from Network.network_utils import pytorch_model, cuda_string
from Causal.interaction_test import InteractionTesting
from Causal.Utils.instance_handling import compute_likelihood
from Causal.active_mask import ActiveMasking
from Environment.Normalization.norm import NormalizationModule


def assign_distribution(dist):
        if dist == "Gaussian": return torch.distributions.normal.Normal
        # elif dist == "Discrete": return Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        # elif dist == "MultiBinary": return Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
        else: raise NotImplementedError

def load_interaction(pth, name, device=-1):
    # loads an interaction model, or returns None if not found
    found = False
    for file in os.listdir(pth):
        if "inter_model.pt" in file and name in file: # looks for inter_model and the object name
            found = True
            break
    if found:
        model= torch.load(os.path.join(pth, name + "_inter_model.pt"))
        if device != -1:
            model.cuda(device=device)
        return model
    return None

def get_params(model, full_args, is_pair):
    active_model_args = copy.deepcopy(full_args.interaction_net)
    active_model_args.num_inputs = model.inter_select.output_size()
    active_model_args.num_outputs = model.target_select.output_size()

    passive_model_args = copy.deepcopy(full_args.interaction_net)
    passive_model_args.num_inputs = model.target_select.output_size()
    passive_model_args.num_outputs = model.target_select.output_size()

    interaction_model_args = copy.deepcopy(full_args.interaction_net)
    interaction_model_args.num_inputs = model.inter_select.output_size()
    interaction_model_args.num_outputs = 1
    
    if is_pair:
        pair = copy.deepcopy(full_args.interaction_net.pair)
        pair.object_dim = model.obj_dim
        pair.first_obj_dim = model.first_obj_dim
        pair.post_dim = -1
        pair.aggregate_final = False
        active_model_args.pair, passive_model_args.pair, interaction_model_args.pair = copy.deepcopy(pair), copy.deepcopy(pair), copy.deepcopy(pair)
        passive_model_args.pair.first_obj_dim = 0
    return active_model_args, passive_model_args, interaction_model_args

def make_name(object_names):
    # return object_names.primary_parent + "->" + object_names.target
    return object_names.target # naming is target centric

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, args, environment):
        super().__init__()
        # set input and output
        self.name = make_name(args.object_names)
        self.names = args.object_names
        self.inter_select = args.inter_select
        self.target_select = args.target_select
        self.parent_selectors = args.parent_selectors
        self.parent_select = args.parent_select
        self.additional_select = args.additional_select
        self.controllable = args.controllable

        # if we are predicting the dynamics
        self.predict_dynamics = args.inter.predict_dynamics
        
        # construct the active model
        self.first_obj_dim = np.sum([self.parent_selectors[p].output_size() for p in self.names.parents]) # the first object dim is the combined length of the parents
        self.obj_dim = self.target_select.output_size() # the selector gets the size of a single instance
        self.additional_dim = environment.object_sizes[self.names.additional[0]] if len(self.names.additional) > 0 else 0# all additional objects must have the same dimension
        active_model_args, passive_model_args, interaction_model_args = get_params(self, args, args.interaction_net.net_type == "pair")
        self.multi_instanced = environment.object_instanced[self.names.target] > 1 # TODO: might need multi-instanced for parents also, but defined differently
        self.multi_parents = environment.object_instanced[self.names.primary_parent] > 1
        self.multi_additional = environment.object_instanced[self.names.additional[0]] > 1 if len(self.names.additional) > 0 else False

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
        self.norm = NormalizationModule(environment.object_range, environment.object_dynamics, args.object_names, environment.object_instanced)

        # set the masking module to None as a placeholder
        self.mask = None
        self.active_mask = None # also a placeholder
        self.active_select = None

        # set values for proximity calculations
        self.proximity_epsilon, self.position_masks = args.inter.proximity_epsilon, environment.position_masks

        # set up cuda
        self.cuda() if args.torch.cuda else self.cpu()

    def regenerate_norm(self, environment):
        self.norm = NormalizationModule(environment.object_range, environment.object_dynamics, self.names, environment.object_instanced)
        if self.mask is not None: self.mask.regenerate_norm(self.norm)
        return self.norm

    def save(self, pth):
        torch.save(self.cpu(), os.path.join(create_directory(pth), self.name + "_inter_model.pt"))

    def cpu(self):
        super().cpu()
        self.active_model.cpu()
        self.interaction_model.cpu()
        self.passive_model.cpu()
        self.iscuda = False
        return self

    def cuda(self, device=-1):
        device = cuda_string(device)
        super().cuda()
        self.active_model.cuda().to(device)
        self.interaction_model.cuda().to(device)
        self.passive_model.cuda().to(device)
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
            inp_state = pytorch_model.wrap(self.norm(self.inter_select(state), form="inter"), cuda=self.iscuda)
            tar_state = pytorch_model.wrap(self.norm(self.target_select(state)), cuda=self.iscuda)
        return inp_state, tar_state

    def predict_next_state(self, state, normalized=False):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        inp_state, tar_state = self._wrap_state(state)

        rv = self.norm.reverse
        inter = pytorch_model.unwrap(self.interaction_model(inp_state))
        inter_bin = self.test(inter)

        # if predicting dynamics, add the mean of the model to the target state
        if self.predict_dynamics:
            fpred, ppred = rv(tar_state) + rv(self.active_model(inp_state)[0], form="dyn"), rv(tar_state) + rv(self.passive_model(tar_state)[0], form="dyn")
        else:
            fpred, ppred = rv(self.active_model(inp_state)[0]), rv(self.passive_model(tar_state)[0])
        
        if normalized: fpred, ppred = self.norm(fpred), self.norm(ppred)

        # TODO: remove this conditional with appropriate slicing
        # select active or passive based on inter value
        if len(fpred.shape) == 1:
            return (inter, fpred) if inter_bin else (inter, ppred)
        else:
            pred = np.stack((ppred, fpred), axis=1)
            intera = inter_bin.squeeze().astype(int)
            pred = pred[np.arange(pred.shape[0]).astype(int), intera]
        return inter, pred

    def hypothesize(self, state):
        # takes in a full state (dict with raw_state, factored_state) or tuple of ndarray of input_state, target_state 
        # computes the interaction value, the mean, var of forward model, the mean, var of the passive model
        inter_state, tar_state = self._wrap_state(state)
        rv = self.norm.reverse
        mu_inter, var_inter = self.active_model(inter_state)
        pmu_inter, pvar_inter = self.passive_model(tar_state)
        return (pytorch_model.unwrap(self.interaction_model(inter_state)),
            (rv(pytorch_model.unwrap(mu_inter)), rv(pytorch_model.unwrap(var_inter))), 
            (rv(pytorch_model.unwrap(pmu_inter)), rv(pytorch_model.unwrap(pvar_inter))))

    def check_interaction(self, inter):
        return self.test(inter)

    def get_active_mask(self):
        return self.test.selection_binary

    def interaction(self, val, prenormalize=False): # val is either a batch, or a ndarray of inter_state. Does NOT unwrap, Does NOT normalize
        if type(val) != np.ndarray: val = val.inter_state # if not an array, assume it is a Batch
        if prenormalize: val = self.norm(val, form="inter")
        return self.interaction_model(pytorch_model.wrap(val, cuda=self.iscuda))

    def _target_dists(self, batch, params):
        target = batch.target_diff if self.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=self.iscuda)
        dist = self.dist(*params)
        log_probs = dist.log_prob(target)
        return target, dist, log_probs

    def normalize_batch(self, batch): # normalizes the components in the batch to be used for likelihoods
        batch.inter_state = self.norm(batch.inter_state, form="inter")
        batch.target = self.norm(batch.target)
        batch.next_target = self.norm(batch.next_target)
        batch.target_diff = self.norm(batch.target_diff, form="dyn")
        return batch

    # likelihood functions (below) get the gaussian distributions output by the active and passive models
    def likelihoods(self, batch, normalize=False):
        if normalize: batch = self.normalize_batch(batch)
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        passive_params = self.passive_model(pytorch_model.wrap(batch.target, cuda=self.iscuda))
        # print(np.concatenate([batch.inter_state, batch.target_diff, pytorch_model.unwrap(active_params[0])], axis=-1))
        # print(np.concatenate([self.norm.reverse(batch.inter_state, form="inter"), self.norm.reverse(batch.target_diff, form="dyn"), self.norm.reverse(pytorch_model.unwrap(active_params[0]), form = 'dyn')], axis=-1))
        inter = self.interaction_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        # error
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
        target, dist, log_probs = self._target_dists(batch, active_params)
        return active_params, dist, log_probs

interaction_models = {'neural': NeuralInteractionForwardModel}