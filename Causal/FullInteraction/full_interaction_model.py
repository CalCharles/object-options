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
from Causal.Utils.interaction_selectors import CausalExtractor
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

def get_params(model, full_args, is_pair, multi_instanced):
    active_model_args = copy.deepcopy(full_args.interaction_net)
    active_model_args.num_inputs = model.extractor.total_inter_size
    active_model_args.num_outputs = model.extractor.total_target_size

    passive_model_args = copy.deepcopy(full_args.interaction_net)
    passive_model_args.num_inputs = model.extractor.total_target_size
    passive_model_args.num_outputs = model.extractor.total_target_size

    interaction_model_args = copy.deepcopy(full_args.interaction_net)
    interaction_model_args.num_inputs = model.extractor.total_inter_size
    interaction_model_args.num_outputs = 1
    
    if is_pair:
        pair = copy.deepcopy(full_args.interaction_net.pair)
        pair.object_dim = model.obj_dim
        pair.first_obj_dim = model.first_obj_dim
        pair.post_dim = -1
        if model.extractor.target_instanced:
            pair.aggregate_final = False # we are multi-instanced in outputs
        else:
            pair.aggregate_final = True # we are multi-instanced, but outputting a single value
        active_model_args.pair, passive_model_args.pair, interaction_model_args.pair = copy.deepcopy(pair), copy.deepcopy(pair), copy.deepcopy(pair)
        if not multi_instanced: # passive model won't be a pairnet TODO: add additional to passive model
            passive_model_args.net_type = "mlp" # TODO: defaults to MLP
            print("passive", passive_model_args)
        passive_model_args.pair.first_obj_dim = 0
        passive_model_args.pair.difference_first = False # difference first cannot be true since there is no first, at least for now
    print("active_model_args", active_model_args)
    return active_model_args, passive_model_args, interaction_model_args

def make_name(object_names):
    # return object_names.primary_parent + "->" + object_names.target
    return object_names.target # naming is target centric

class FullNeuralInteractionForwardModel(nn.Module):
    def __init__(self, args, environment):
        super().__init__()
        # set input and output
        self.name = args.target
        self.all_names = environment.object_names
        self.target_select = np.array([True if n == self.name else False for n in self.all_names])
        self.extractor = CausalExtractor(self.names, environment)
        self.target_select, self.full_select, self.parents_select = self.extractor.get_selectors()
        self.controllable = args.controllable

        # if we are predicting the dynamics
        self.predict_dynamics = args.inter.predict_dynamics
        
        # construct the active model
        self.multi_instanced = environment.object_instanced[self.names.target] > 1 # TODO: might need multi-instanced for parents also, but defined differently
        self.active_model_args, self.passive_model_args, self.interaction_model_args = get_params(self, args, args.interaction_net.net_type == "pair", self.multi_instanced)

        # set the distributions
        self.dist = assign_distribution("Gaussian") # TODO: only one kind of dist at the moment

        # set the forward model
        self.active_model = DiagGaussianMaskForwardNetwork(self.active_model_args)

        # set the passive model
        self.passive_model = DiagGaussianForwardNetwork(self.passive_model_args)

        # construct the interaction model        
        self.interaction_model = InteractionMaskNetwork(self.interaction_model_args)

        # set the testing module
        self.test = InteractionMaskTesting(args.inter.interaction_testing)

        # set the normalization function
        self.norm, self.extractor = self.regenerate(environment)
        # proximity terms
        pad_size
        pos_size
        proximity_epsilon
        object_proximal

        # set the masking module to None as a placeholder
        self.mask = None
        self.active_mask = None # also a placeholder
        self.active_select = None

        # set values for proximity calculations
        self.proximity_epsilon, self.position_masks = args.inter.proximity_epsilon, environment.position_masks

        # set up cuda
        self.cuda() if args.torch.cuda else self.cpu()

    def regenerate(self, environment):
        self.extractor = CausalExtractor(self.name, environment)
        self.target_select, self.parent_select, self.inter_select = self.extractor.get_selectors()
        self.norm = FullNormalizationModule(environment.object_range, environment.object_dynamics, self.name, environment.object_instanced, environment.object_names)
        if hasattr(self, "mask") and self.mask is not None: self.mask.regenerate_norm(self.norm)
        return self.norm, self.extractor

    def reset_network(self, net = None):
        if net == "interaction":
            self.interaction_model.reset_parameters()
            net = self.interaction_model
        elif net == "active": 
            self.active_model.reset_parameters()
            net = self.active_model
        elif net == "passive":
            self.passive_model.reset_parameters()
            net = self.passive_model
        return net

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

    def predict_next_state(self, state, normalized=False, difference=False):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        # @param difference returns the dynamics prediction instead of the active prediction, not used if the full model is not a dynamics predictor
        inp_state, tar_state = self._wrap_state(state)

        rv = self.norm.reverse
        inter = pytorch_model.unwrap(self.interaction_model(inp_state))
        inter_bin = self.test(inter)

        # if predicting dynamics, add the mean of the model to the target state
        if self.predict_dynamics:
            if difference:
                fpred, ppred = rv(self.active_model(inp_state)[0], form="dyn"), rv(self.passive_model(tar_state)[0], form="dyn")
            else:
                fpred, ppred = rv(tar_state) + rv(self.active_model(inp_state)[0], form="dyn"), rv(tar_state) + rv(self.passive_model(tar_state)[0], form="dyn")
        else:
            fpred, ppred = rv(self.active_model(inp_state)[0]), rv(self.passive_model(tar_state)[0])
        
        if normalized: fpred, ppred = self.norm(fpred, form="dyn" if difference else "target"), self.norm(ppred, form="dyn" if difference else "target")

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

    def interaction(self, val, target=None, next_target=None, target_diff=None, prenormalize=False, use_binary=False): # val is either a batch, or a ndarray of inter_state. Does NOT unwrap, Does NOT normalize
        if type(val) != Batch:
            bat = Batch()
            bat.inter_state = val
            bat.target = target
            bat.next_target = next_target
            bat.target_diff = target_diff
        else:
            bat = val
        if type(val) != np.ndarray: val = val.inter_state # if not an array, assume it is a Batch
        if prenormalize: 
            bat = self.normalize_batch(bat)

        if use_binary:
            _, _, inter, _, _, _, active_log_probs, passive_log_probs = self.likelihoods(bat)
            binary = self.test.compute_binary(- active_log_probs.sum(dim=-1),
                                                - passive_log_probs.sum(dim=-1)).unsqueeze(-1)
            return binary
        else:
            val = bat.inter_state
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
        inter = self.interaction_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda), inter)
        passive_params = self.passive_model(pytorch_model.wrap(batch.target, cuda=self.iscuda))
        # print(np.concatenate([batch.inter_state, batch.target_diff, pytorch_model.unwrap(active_params[0])], axis=-1))
        # print(np.concatenate([self.norm.reverse(batch.inter_state, form="inter"), self.norm.reverse(batch.target_diff, form="dyn"), self.norm.reverse(pytorch_model.unwrap(active_params[0]), form = 'dyn')], axis=-1))
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        return active_params, passive_params, inter, target, active_dist, passive_dist, active_log_probs, passive_log_probs        

    # gets the active likelihood without the interaction mask blocking any inputs
    def active_open_likelihood(self, batch, normalize=False):
        if normalize: batch = self.normalize_batch(batch)
        inter = pytorch_model.wrap(torch.ones(len(self.active_model.total_object_sizes)), cuda=self.iscuda)
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda), inter)
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        return active_params, active_dist, active_log_probs

    def passive_likelihoods(self, batch):
        passive_params = self.passive_model(pytorch_model.wrap(batch.target, cuda=self.iscuda))
        target, dist, log_probs = self._target_dists(batch, passive_params)
        return passive_params, dist, log_probs

    def active_likelihoods(self, batch):
        inter = self.interaction_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda))
        active_params = self.active_model(pytorch_model.wrap(batch.inter_state, cuda=self.iscuda), inter)
        target, dist, log_probs = self._target_dists(batch, active_params)
        return active_params, dist, log_probs

interaction_models = {'neural': NeuralInteractionForwardModel}