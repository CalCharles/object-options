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
from Network.distributional_network import DiagGaussianForwardMaskNetwork, DiagGaussianForwardPadMaskNetwork, InteractionMaskNetwork, DiagGaussianForwardNetwork
from Network.network_utils import pytorch_model, cuda_string
from Causal.interaction_test import InteractionTesting
from Causal.Utils.instance_handling import compute_likelihood
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.active_mask import ActiveMasking
from Causal.FullInteraction.interaction_full_extractor import CausalPadExtractor, CausalFullExtractor
from Causal.interaction_model import get_params
from Causal.FullInteraction.full_interaction_testing import InteractionMaskTesting
from Environment.Normalization.norm import NormalizationModule
from Environment.Normalization.full_norm import FullNormalizationModule
from Environment.Normalization.pad_norm import PadNormalizationModule


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

def make_name(object_names):
    # return object_names.primary_parent + "->" + object_names.target
    return object_names.target # naming is target centric

def regenerate(append_id, environment):
    extractor = CausalPadExtractor(environment, append_id)
    # norm = FullNormalizationModule(environment.object_range, environment.object_dynamics, name, environment.object_instanced, environment.object_names)
    pad_size = max(list(environment.object_sizes.values()))
    append_size = len(list(environment.object_sizes.keys())) * int(append_id)
    norm = PadNormalizationModule(environment.object_range, environment.object_dynamics, environment.object_instanced, environment.object_names, pad_size, append_size)
    return extractor, norm


class FullNeuralInteractionForwardModel(nn.Module):
    def __init__(self, args, target, environment, causal_extractor, normalization):
        super().__init__()
        # set input and output
        self.is_full = True
        self.name = target
        self.names = environment.object_names
        self.all_names = environment.all_names
        self.target_select = np.array([True if n == self.name else False for n in self.names])
        self.extractor = causal_extractor
        self.norm = normalization
        self.target_selectors, self.full_select = self.extractor.get_selectors()
        self.target_select = self.target_selectors[self.name]
        # self.controllable = args.controllable

        # if we are predicting the dynamics
        self.predict_dynamics = True
        
        # construct the active model
        self.multi_instanced = environment.object_instanced[self.name] > 1 # TODO: might need multi-instanced for parents also, but defined differently
        self.target_num = environment.object_instanced[self.name]
        self.obj_dim, self.single_obj_dim, self.first_obj_dim = self.extractor._get_dims(self.name)
        args.interaction_net.object_dim = self.obj_dim
        self.active_model_args, self.passive_model_args, self.interaction_model_args = get_params(self, args, args.interaction_net.net_type in ["pair", "keypair"], environment.object_instanced[self.name], self.extractor.total_inter_size, self.extractor.single_object_size)

        # set the distributions
        self.dist = assign_distribution("Gaussian") # TODO: only one kind of dist at the moment

        # set the forward model
        self.active_model = DiagGaussianForwardPadMaskNetwork(self.active_model_args)

        # set the passive model
        self.passive_model = DiagGaussianForwardNetwork(self.passive_model_args)

        # construct the interaction model        
        self.interaction_model = InteractionMaskNetwork(self.interaction_model_args)

        # set the testing module
        self.test = InteractionMaskTesting(args.inter.interaction_testing)

        # set the normalization function
        self.norm, self.extractor = normalization, causal_extractor
        self.target_select, self.inter_select = self.extractor.target_selectors[self.name], self.extractor.inter_selector
        # proximity terms
        self.pad_size = normalization.pad_size + normalization.append_size
        self.pos_size = environment.pos_size
        self.object_proximal = None # not sure what I planned to do with this

        # set the masking module to None as a placeholder
        self.mask = None
        self.active_mask = None # also a placeholder
        self.active_select = None

        # set values for proximity calculations
        self.proximity_epsilon, self.position_masks = args.inter.proximity_epsilon, environment.position_masks

        # set up cuda
        self.cuda() if args.torch.cuda else self.cpu()

    def regenerate(self, extractor, norm):
        self.norm, self.extractor = norm, extractor
        self.target_num = extractor.complete_instances[extractor.names.index(self.name)]
        if hasattr(self, "mask") and self.mask is not None: self.mask.regenerate_norm(norm)

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
            inp_state = pytorch_model.wrap(np.concatenate([tar_state, inter_state], axis=-1), cuda=self.iscuda)
            tar_state = pytorch_model.wrap(tar_state, cuda=self.iscuda)
        else: # assumes that state is a batch or dict
            if (type(state) == Batch or type(state) == dict) and ('factored_state' in state): state = state['factored_state'] # use gamma on the factored state
            tar_state = pytorch_model.wrap(self.norm(self.target_select(state)), cuda=self.iscuda)
            inp_state = torch.cat([tar_state, pytorch_model.wrap(self.norm(self.inter_select(state), form="inter"), cuda=self.iscuda)], dim=-1)
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
        tarinter_state, tar_state = self._wrap_state(state)
        rv = self.norm.reverse
        mu_inter, var_inter = self.active_model(tarinter_state)
        pmu_inter, pvar_inter = self.passive_model(tar_state)
        return (pytorch_model.unwrap(self.interaction_model(tarinter_state)),
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
            bat.tarinter_state = np.concatenate([bat.target, bat.inter_state], axis=-1)
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
            val = bat.tarinter_state
            return self.interaction_model(pytorch_model.wrap(val, cuda=self.iscuda))

    def _target_dists(self, batch, params):
        target = batch.target_diff if self.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=self.iscuda)
        dist = self.dist(*params)
        log_probs = dist.log_prob(target)
        return target, dist, log_probs

    def normalize_batch(self, batch): # normalizes the components in the batch to be used for likelihoods, assumes the batch is an object batch
        batch.inter_state = self.norm(batch.inter_state, form="inter")
        batch.obs = self.norm(batch.obs, name=self.name)
        batch.tarinter_state = np.concatenate([batch.obs, batch.inter_state], axis=-1)
        batch.obs_next = self.norm(batch.obs_next, name=self.name)
        batch.target_diff = self.norm(batch.target_diff, form="dyn", name=self.name)
        return batch

    # likelihood functions (below) get the gaussian distributions output by the active and passive models
    def likelihoods(self, batch, normalize=False):
        if normalize: batch = self.normalize_batch(batch)
        inter = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda))
        print(batch.tarinter_state.shape, inter.shape)
        active_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), inter)
        passive_params = self.passive_model(pytorch_model.wrap(batch.obs, cuda=self.iscuda))
        # print(np.concatenate([batch.inter_state, batch.target_diff, pytorch_model.unwrap(active_params[0])], axis=-1))
        # print(np.concatenate([self.norm.reverse(batch.inter_state, form="inter"), self.norm.reverse(batch.target_diff, form="dyn"), self.norm.reverse(pytorch_model.unwrap(active_params[0]), form = 'dyn')], axis=-1))
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        return active_params, passive_params, inter, target, active_dist, passive_dist, active_log_probs, passive_log_probs        

    # gets the active likelihood without the interaction mask blocking any inputs
    def active_open_likelihood(self, batch, normalize=False):
        if normalize: batch = self.normalize_batch(batch)
        inter = pytorch_model.wrap(torch.ones(len(self.active_model.total_object_sizes)), cuda=self.iscuda)
        active_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), inter)
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        return active_params, active_dist, active_log_probs

    def passive_likelihoods(self, batch):
        passive_params = self.passive_model(pytorch_model.wrap(batch.obs, cuda=self.iscuda))
        target, dist, log_probs = self._target_dists(batch, passive_params)
        return passive_params, dist, log_probs

    def active_likelihoods(self, batch):
        inter = self.interaction_model(pytorch_model.wrap(batch.batch.tarinter_state, cuda=self.iscuda))
        active_params = self.active_model(pytorch_model.wrap(batch.batch.tarinter_state, cuda=self.iscuda), inter)
        target, dist, log_probs = self._target_dists(batch, active_params)
        return active_params, dist, log_probs

interaction_models = {'neural': FullNeuralInteractionForwardModel}