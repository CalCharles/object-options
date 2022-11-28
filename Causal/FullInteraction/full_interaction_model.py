import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
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

def assign_distribution(assign_dist):
        if assign_dist == "Gaussian": return torch.distributions.normal.Normal
        if assign_dist == "Identity": return None
        if assign_dist == "RelaxedBernoulli": return dist.relaxed_bernoulli.RelaxedBernoulli
        if assign_dist == "Bernoulli": return dist.bernoulli.Bernoulli
        # elif assign_dist == "Discrete": return Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        # elif assign_dist == "MultiBinary": return Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
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
    pad_size = extractor.pad_dim
    append_size = extractor.append_dim
    norm = PadNormalizationModule(environment.object_range, environment.object_dynamics, environment.object_instanced, environment.object_names, pad_size, append_size)
    return extractor, norm

MASKING_FORMS = {
    "weighting": 0,
    "relaxed": 1,
    "mixed": 2,
    "hard": 3,
}


def get_params(model, full_args, is_pair, multi_instanced, total_inter_size, total_target_size):
    full_args.interaction_net.object_names = model.names
    full_args.mask_dim = model.num_inter
    active_model_args = copy.deepcopy(full_args.interaction_net)
    active_model_args.num_inputs = total_inter_size
    active_model_args.num_outputs = total_target_size

    passive_model_args = copy.deepcopy(full_args.interaction_net)
    passive_model_args.num_inputs = total_target_size
    passive_model_args.num_outputs = total_target_size

    interaction_model_args = copy.deepcopy(full_args.interaction_net)
    interaction_model_args.num_inputs = total_inter_size
    interaction_model_args.num_outputs = 1
    
    if is_pair:
        pair = copy.deepcopy(full_args.interaction_net.pair)
        pair.object_dim = model.obj_dim
        pair.first_obj_dim = model.first_obj_dim
        pair.single_obj_dim = model.single_obj_dim
        pair.post_dim = -1
        # parameters specific to key-pair/transformer networks 
        pair.total_obj_dim = np.sum(model.extractor.complete_object_sizes)
        pair.expand_dim = model.extractor.expand_dim
        pair.total_instances = np.sum(model.extractor.complete_instances)
        pair.total_targets = model.multi_instanced
        pair.query_pair = False
        
        if model.multi_instanced:
            pair.aggregate_final = False # we are multi-instanced in outputs
        else:
            pair.aggregate_final = True # we are multi-instanced, but outputting a single value
        active_model_args.pair, passive_model_args.pair, interaction_model_args.pair = copy.deepcopy(pair), copy.deepcopy(pair), copy.deepcopy(pair)
        full_args.interaction_net.query_pair = True if full_args.interaction_net.net_type in ["keypair"] else False # a query pair means that the query network outputs pairwise
        if not model.multi_instanced: # passive model won't be a pairnet TODO: add additional to passive model
            passive_model_args.net_type = "mlp" # TODO: defaults to MLP
            print("passive", passive_model_args)
        if model.multi_instanced and full_args.interaction_net.net_type in ["keypair", "maskattn"]: # in keypair situations, the passive model is a pairnet
            passive_model_args.net_type = "pair"
            passive_model_args.pair.object_dim = model.single_obj_dim
        passive_model_args.pair.first_obj_dim = 0
        passive_model_args.pair.difference_first = False # difference first cannot be true since there is no first, at least for now
    print("active_model_args", active_model_args)
    return active_model_args, passive_model_args, interaction_model_args


class FullNeuralInteractionForwardModel(nn.Module):
    def __init__(self, args, target, environment, causal_extractor, normalization):
        super().__init__()
        # set input and output
        self.is_full = True
        self.name = target
        self.names = environment.object_names

        self.regenerate(causal_extractor, normalization, environment)
        # self.controllable = args.controllable

        # if we are predicting the dynamics
        self.predict_dynamics = True
        
        # construct the active model
        args.interaction_net.object_dim = self.obj_dim
        self.multi_instanced = environment.object_instanced[self.name] > 1 # an object CANNOT go from instanced to multi instanced
        self.active_model_args, self.passive_model_args, self.interaction_model_args = get_params(self, args, args.interaction_net.net_type in ["pair", "keypair", "maskattn"], environment.object_instanced[self.name], self.extractor.total_inter_size, self.extractor.single_object_size)
        self.active_model_args.mask_attn.passive_mask = self.check_passive_mask(np.zeros((1,)))

        # set the distributions
        self.dist = assign_distribution("Gaussian") # TODO: only one kind of dist at the moment
        self.relaxed_inter_dist = assign_distribution(args.full_inter.soft_distribution)
        self.dist_temperature = args.full_inter.dist_temperature
        self.inter_dist = assign_distribution("Bernoulli")
        self.mixing = args.full_inter.mixed_interaction# mostly only used for training

        # set the forward model
        self.active_model = DiagGaussianForwardPadMaskNetwork(self.active_model_args)

        # set the passive model
        self.use_active_as_passive = args.full_inter.use_active_as_passive # uses the active model with the one hot as the passive model
        self.passive_model = DiagGaussianForwardNetwork(self.passive_model_args) if not args.full_inter.use_active_as_passive else None

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

    def regenerate(self, extractor, norm, environment):
        self.norm, self.extractor = norm, extractor
        self.target_num = extractor.num_instances[extractor.names.index(self.name)]

        self.all_names = [n for n in environment.all_names if n not in ["Reward", "Done"]]
        self.num_inter = len(self.all_names)# number of instances to interact with
        self.name_index = self.all_names.index(self.name) if self.name in self.all_names else -1 # be careful using name index
        self.passive_mask = np.zeros(self.num_inter)
        self.target_selectors, self.full_select = self.extractor.get_selectors()
        self.target_select = self.target_selectors[self.name]
        if self.name not in ["Reward", "Done"]: self.passive_mask[self.name_index] = 1
        if hasattr(self, "active_model"): self.active_model.reset_passive_mask(self.passive_mask)
        self.target_num = environment.object_instanced[self.name]
        self.obj_dim, self.single_obj_dim, self.first_obj_dim = self.extractor._get_dims(self.name)
        if hasattr(self, "mask") and self.mask is not None: self.mask.regenerate_norm(norm)

    def reset_network(self, net = None):
        if net == "interaction":
            self.interaction_model.reset_parameters()
            net = self.interaction_model
        elif net == "active": 
            self.active_model.reset_parameters()
            net = self.active_model
        elif net == "passive" and not self.use_active_as_passive:
            self.passive_model.reset_parameters()
            net = self.passive_model
        return net

    def save(self, pth):
        torch.save(self.cpu(), os.path.join(create_directory(pth), self.name + "_inter_model.pt"))

    def cpu(self):
        super().cpu()
        self.active_model.cpu()
        self.interaction_model.cpu()
        if self.passive_model is not None: self.passive_model.cpu()
        self.iscuda = False
        return self

    def cuda(self, device=-1):
        device = cuda_string(device)
        super().cuda()
        self.active_model.cuda().to(device)
        self.interaction_model.cuda().to(device)
        if self.passive_model is not None: self.passive_model.cuda().to(device)
        self.iscuda = True
        return self

    def reset_parameters(self):
        self.active_model.reset_parameters()
        self.interaction_model.reset_parameters()
        if self.passive_model is not None: self.passive_model.reset_parameters()

    def _wrap_state(self, state):
        # takes in a state, either a full state (factored state dict (name to ndarray)), or tuple of (inter_state, target_state) 
        if type(state) == tuple:
            inter_state, tar_state = state
            inp_state = pytorch_model.wrap(np.concatenate([tar_state, inter_state], axis=-1), cuda=self.iscuda) # concatenates the tarinter state
            tar_state = pytorch_model.wrap(tar_state, cuda=self.iscuda)
        else: # assumes that state is a batch or dict
            if (type(state) == Batch or type(state) == dict) and ('factored_state' in state): state = state['factored_state'] # use gamma on the factored state
            tar_state = pytorch_model.wrap(self.norm(self.target_select(state)), cuda=self.iscuda)
            inp_state = torch.cat([tar_state, pytorch_model.wrap(self.norm(self.inter_select(state), form="inter"), cuda=self.iscuda)], dim=-1)
        return inp_state, tar_state

    def inter_passive(self, inter_mask):
        inter_mask = pytorch_model.unwrap(inter_mask)
        if self.name in ["Reward", "Done"]:
            return np.zeros(inter_mask.shape[:-1])
        passive_mask = self.generate_passive_mask(inter_mask.shape)
        return np.abs(np.sum(inter_mask - passive_mask, axis=-1)) == 0

    def check_passive_mask(self, state): # creates as many passive masks as there are states
        passive_mask = np.zeros((*state.shape[:-1], *(len(self.all_names), ))).astype(float)
        if type(state) == torch.Tensor: passive_mask = pytorch_model.wrap(passive_mask, cuda=self.iscuda)
        if self.name in ["Reward", "Done"]: return passive_mask
        passive_mask[...,self.name_index] = 1.0
        # TODO: get a passive mask
        return passive_mask

    def apply_passive(self, state):
        # assumes that the state is (inp_state, target_state)
        if type(state) == torch.Tensor:
            inp_state, tar_state = state, state
            return self.active_model(inp_state, self.check_passive_mask(tar_state))[0] if self.use_active_as_passive else self.passive_model(tar_state) 
        elif type(state) == tuple:
            if type(state[0]) == torch.Tensor:
                inp_state, tar_state = state
                return self.active_model(inp_state, self.check_passive_mask(tar_state))[0] if self.use_active_as_passive else self.passive_model(tar_state) 
            else:
                inp_state, tar_state = self._wrap_state(state)
                return pytorch_model.unwrap(self.active_model(inp_state, self.check_passive_mask(tar_state))[0]) if self.use_active_as_passive else pytorch_model.unwrap(self.passive_model(tar_state))
        else:
            inp_state, tar_state = self._wrap_state(state)
            return self.active_model(inp_state, self.check_passive_mask(tar_state))[0] if self.use_active_as_passive else self.passive_model(tar_state)

    def predict_next_state(self, state, normalized=False, difference=False):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        # @param difference returns the dynamics prediction instead of the active prediction, not used if the full model is not a dynamics predictor
        inp_state, tar_state = self._wrap_state(state)

        rv = self.norm.reverse
        inter = pytorch_model.unwrap(self.interaction_model(inp_state))
        inter_mask = self.apply_mask(inter, flat=True)

        # if predicting dynamics, add the mean of the model to the target state
        if self.predict_dynamics:
            if difference:
                fpred, ppred = rv(self.active_model(inp_state, inter_mask)[0], form="dyn"), rv(self.apply_passive((inp_state, tar_state))[0], form="dyn")
            else:
                fpred, ppred = rv(tar_state) + rv(self.active_model(inp_state, inter_mask)[0], form="dyn"), rv(tar_state) + rv(self.apply_passive((inp_state, tar_state))[0], form="dyn")
        else:
            fpred, ppred = rv(self.active_model(inp_state, inter_mask)[0]), rv(self.apply_passive((inp_state, tar_state))[0])
        
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
        inter = self.interaction_model(tarinter_state)
        inter_mask = self.apply_mask(inter, flat = True)
        mu_inter, var_inter = self.active_model(tarinter_state, inter_mask)
        pmu_inter, pvar_inter = self.apply_passive((tarinter_state, tar_state))
        return (pytorch_model.unwrap(inter),
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
            _, _, inter, _, _, _, active_log_probs, passive_log_probs = self.reduced_likelihoods(bat, masking = "full")
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

    def apply_mask(self, inter_mask, soft=True, flat=False):
        # TODO: generate the interaction mask out of the outputs of the interaction model
        revert_mask = False
        if type(inter_mask) != torch.Tensor: revert_mask, inter_mask = True, pytorch_model.wrap(inter_mask, cuda=self.iscuda)
        if flat: return pytorch_model.unwrap(self.test(inter_mask)) if revert_mask else self.test(inter_mask)
        if soft:
            if self.relaxed_inter_dist is None: return pytorch_model.unwrap(inter_mask) if revert_mask else inter_mask
            else: return pytorch_model.unwrap(self.relaxed_inter_dist(self.dist_temperature, probs=inter_mask).rsample()) if revert_mask else self.relaxed_inter_dist(self.dist_temperature, probs=inter_mask).rsample()
        return pytorch_model.unwrap(self.inter_dist(inter_mask).sample()) if revert_mask else self.inter_dist(inter_mask).sample() # hard masks don't need gradient

    def combine_mask(self, inter_mask, hard_mask, mixed=""):
        if len(mixed) == 0: mixed = self.mixing
        if MASKING_FORMS[mixed] == 0:
            return inter_mask
        elif MASKING_FORMS[mixed] == 1:
            return inter_mask
        elif MASKING_FORMS[mixed] == 2:
            return inter_mask * hard_mask
        elif MASKING_FORMS[mixed] == 3:
            return hard_mask
        else:
            raise ValueError("Unsupported masking form")

    # uses the flat mask
    def reduced_likelihoods(self, batch, normalize=False, masking=""):
        if normalize: batch = self.normalize_batch(batch)
        inter = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda))
        if masking == "flat": inter_mask = self.apply_mask(inter, soft=False, flat = True)
        elif masking == "full": inter_mask = pytorch_model.wrap(torch.ones(len(self.all_names) * self.target_num), cuda = self.iscuda)
        elif masking == "soft": inter_mask = self.apply_mask(inter, soft=True, flat = False)
        elif masking == "hard": inter_mask = self.apply_mask(inter, soft=False, flat = False)
        else: raise ValueError("Invalid Masking: " + masking) 
        active_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), inter_mask)
        passive_params = self.apply_passive((pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), pytorch_model.wrap(batch.obs, cuda=self.iscuda)))
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        return active_params, passive_params, inter, target, active_dist, passive_dist, active_log_probs, passive_log_probs

    # likelihood functions (below) get the gaussian distributions output by the active and passive models for all mask forms
    def likelihoods(self, batch, normalize=False, mixed=""):
        if normalize: batch = self.normalize_batch(batch)
        inter = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda))
        soft_inter_mask = self.apply_mask(inter, soft=True)
        hard_inter_mask = self.apply_mask(inter, soft=False)
        full_mask = pytorch_model.wrap(torch.ones(len(self.all_names) * self.target_num), cuda = self.iscuda)
        active_hard_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), hard_inter_mask)
        mixed_mask = self.combine_mask(soft_inter_mask, hard_inter_mask, mixed=mixed)
        active_soft_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), mixed_mask)
        active_full_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), full_mask)
        passive_params = self.apply_passive((pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), pytorch_model.wrap(batch.obs, cuda=self.iscuda)))
        # print(np.concatenate([batch.inter_state, batch.target_diff, pytorch_model.unwrap(active_params[0])], axis=-1))
        # print(np.concatenate([self.norm.reverse(batch.inter_state, form="inter"), self.norm.reverse(batch.target_diff, form="dyn"), self.norm.reverse(pytorch_model.unwrap(active_params[0]), form = 'dyn')], axis=-1))
        target, active_hard_dist, active_hard_log_probs = self._target_dists(batch, active_hard_params)
        target, active_soft_dist, active_soft_log_probs = self._target_dists(batch, active_soft_params)
        target, active_full_dist, active_full_log_probs = self._target_dists(batch, active_full_params)
        # print("full params", active_full_params[0], active_full_params[1], target,batch.tarinter_state, active_full_log_probs[0])
        target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        return active_hard_params, active_soft_params, active_full_params, passive_params,\
                 inter, soft_inter_mask, hard_inter_mask, \
                 target, \
                 active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
                 active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs        

    # gets the active likelihood without the interaction mask blocking any inputs
    def active_open_likelihood(self, batch, normalize=False):
        if normalize: batch = self.normalize_batch(batch)
        inter = pytorch_model.wrap(torch.ones(len(self.all_names) * self.target_num), cuda=self.iscuda)
        inter_mask = self.apply_mask(inter)
        active_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), inter_mask)
        target, active_dist, active_log_probs = self._target_dists(batch, active_params)
        return active_params, active_dist, active_log_probs

    def passive_likelihoods(self, batch):
        passive_params = self.apply_passive((pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), pytorch_model.wrap(batch.obs, cuda=self.iscuda)))
        target, dist, log_probs = self._target_dists(batch, passive_params)
        return passive_params, dist, log_probs

    def active_likelihoods(self, batch, normalize=False, soft=False, flat=True):
        if normalize: batch = self.normalize_batch(batch)
        inter = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda))
        inter_mask = self.apply_mask(inter, soft=soft, flat=flat)
        active_params = self.active_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), inter_mask)
        target, dist, log_probs = self._target_dists(batch, active_params)
        return active_params, dist, log_probs

interaction_models = {'neural': FullNeuralInteractionForwardModel}