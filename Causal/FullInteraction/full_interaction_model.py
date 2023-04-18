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
from Network.distributional_network import DiagGaussianForwardMaskNetwork, DiagGaussianForwardPadMaskNetwork, DiagGaussianForwardPadHotNetwork, InteractionMaskNetwork, InteractionSelectionMaskNetwork, DiagGaussianForwardNetwork, apply_probabilistic_mask
from Network.network_utils import pytorch_model, cuda_string, assign_distribution
from Network.Dists.mask_utils import count_keys_queries
from Causal.interaction_test import InteractionTesting
from Causal.Utils.instance_handling import compute_likelihood
from Causal.Utils.interaction_selectors import CausalExtractor
from Causal.active_mask import ActiveMasking
from Causal.FullInteraction.interaction_full_extractor import CausalPadExtractor, CausalFullExtractor
from Causal.interaction_model import get_params
from Causal.FullInteraction.full_interaction_testing import InteractionMaskTesting
from Causal.interaction_base_model import NeuralInteractionForwardModel, regenerate, KEYNETS, PAIR, MASKING_FORMS
from Environment.Normalization.norm import NormalizationModule
from Environment.Normalization.full_norm import FullNormalizationModule
from Environment.Normalization.pad_norm import PadNormalizationModule

def make_name(object_names):
    # return object_names.primary_parent + "->" + object_names.target
    return object_names.target # naming is target centric

def get_params(model, full_args, is_pair, multi_instanced, total_inter_size, total_target_size):
    full_args.interaction_net.object_names = model.names
    full_args.mask_dim = model.num_inter
    full_args.interaction_net.mask_attn.return_mask = False
    full_args.interaction_net.mask_attn.gumbel_temperature = full_args.full_inter.dist_temperature
    full_args.interaction_net.attention_mode = full_args.interaction_net.net_type == "rawattn"
    full_args.interaction_net.pair.total_instances = np.sum(model.extractor.complete_instances)
    full_args.interaction_net.selection_temperature = full_args.full_inter.selection_temperature
    full_args.interaction_net.symmetric_key_query = False # this will be false since we have a separate model for every class


    active_model_args = copy.deepcopy(full_args.interaction_net)
    active_model_args.num_inputs = total_inter_size
    active_model_args.num_outputs = total_target_size

    passive_model_args = copy.deepcopy(full_args.interaction_net)
    passive_model_args.num_inputs = total_target_size
    if passive_model_args.net_type in KEYNETS: passive_model_args.hidden_sizes = passive_model_args.hidden_sizes + passive_model_args.pair.final_layers
    passive_model_args.num_outputs = total_target_size

    interaction_model_args = copy.deepcopy(full_args.interaction_net)
    interaction_model_args.num_inputs = total_inter_size
    interaction_model_args.num_outputs = 1
    interaction_model_args.mask_attn.return_mask = False
    interaction_model_args.softmax_output = False

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
        pair.query_pair = False
        
        if model.multi_instanced:
            pair.aggregate_final = False # we are multi-instanced in outputs
        else:
            pair.aggregate_final = False # we are multi-instanced, but outputting a single value
        active_model_args.pair, passive_model_args.pair, interaction_model_args.pair = copy.deepcopy(pair), copy.deepcopy(pair), copy.deepcopy(pair)
        if interaction_model_args.cluster.cluster_mode:
            interaction_model_args.num_outputs = interaction_model_args.cluster.num_clusters
            interaction_model_args.cluster.cluster_mode = False # the interaction model does not take cluster mode
            # interaction_model_args.pair.aggregate_final = True
            interaction_model_args.softmax_output = True
            interaction_model_args.pair.total_instances = interaction_model_args.cluster.num_clusters # interaction models use this as a replacement for num_outputs
        interaction_model_args.query_pair = True if interaction_model_args.net_type in ["keypair"] else False # a query pair means that the query network outputs pairwise
        interaction_model_args.embedpair.query_aggregate = False # variable output based on the number of queries

        if full_args.full_inter.lightweight_passive:
            if not (model.multi_instanced): # passive model won't be a pairnet TODO: add additional to passive model
                passive_model_args.net_type = "mlp" # TODO: defaults to MLP
                print("passive", passive_model_args)
            if (model.multi_instanced) and interaction_model_args.net_type in KEYNETS: # in keypair situations, the passive model is a pairnet
                passive_model_args.net_type = "pair"
                passive_model_args.pair.object_dim = model.single_obj_dim
            passive_model_args.pair.first_obj_dim = 0
            passive_model_args.pair.difference_first = False # difference first cannot be true since there is no first, at least for now
    print("active_model_args", active_model_args)
    return active_model_args, passive_model_args, interaction_model_args


class FullNeuralInteractionForwardModel(NeuralInteractionForwardModel):
    def __init__(self, args, target, environment, causal_extractor, normalization):
        super().__init__(args, target, environment, causal_extractor, normalization, get_params, "full")

    def regenerate(self, extractor, norm, environment):
        super().regenerate(extractor, norm, environment)
        self.target_num = extractor.num_instances[extractor.names.index(self.name)]
        self.name_index = self.all_names.index(self.name) if self.name in self.all_names else -1 # be careful using name index
        self.target_select = self.target_selectors[self.name]
        self.obj_dim, self.single_obj_dim, self.first_obj_dim = self.extractor._get_dims(self.name)
        self.target_num = environment.object_instanced[self.name]
        if hasattr(self, "passive_model") and hasattr(self.passive_model, "reset_environment"):
            self.passive_model.reset_environment(self.name_index, self.num_inter, self.first_obj_dim)
        if hasattr(self, "active_model") and hasattr(self.active_model, "reset_environment"):
            self.active_model.reset_environment(self.name_index, self.num_inter, self.first_obj_dim)
        if hasattr(self, "interaction_model") and hasattr(self.interaction_model, "reset_environment"):
            print("resetting")
            self.interaction_model.reset_environment(self.name_index, self.num_inter, self.first_obj_dim)

    def _wrap_state(self, state, tensor=False, use_next = False):
        # takes in a state, either a full state (factored state dict (name to ndarray)), or tuple of (inter_state, target_state) 
        if state is None:
            return None, None
        if type(state) == tuple:
            inter_state, tar_state = state
            inp_state = pytorch_model.wrap(np.concatenate([tar_state, inter_state], axis=-1), cuda=self.iscuda) # concatenates the tarinter state
            tar_state = pytorch_model.wrap(tar_state, cuda=self.iscuda)
        else: # assumes that state is a batch or dict
            if (type(state) == Batch or type(state) == dict) and ('factored_state' in state): state = state['factored_state'] # use gamma on the factored state
            tar_state = pytorch_model.wrap(self.norm(self.target_select(state)), cuda=self.iscuda)
            inp_state = torch.cat([tar_state, pytorch_model.wrap(self.norm(self.inter_select(state), form="inter"), cuda=self.iscuda)], dim=-1)
        return inp_state, tar_state

    def get_interaction_state(self, state, next_state=None, factored=True):
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        if factored:
            inter_state, tar_state = self._wrap_inter(state, tensor=False) 
            next_inter_state, next_tar_state = self._wrap_inter(next_state, tensor=False)
        else:
            inp_state, tar_state = state.inter_state, state.obs
            next_inp_state, next_tar_state = state.next_inter_state, state.obs_next
        if self.nextstate_interaction:
            tar_dict = self.extractor.reverse_extract(tar_state, target=True)
            next_tar_dict = self.extractor.reverse_extract(next_tar_state, target=True)
            tar_dict = {n: np.concatenate([tar_dict[n], next_tar_dict[n]], axis=-1) for n in tar_dict.keys()}
            inp_state = np.concatenate([self.target_select(tar_dict), inter_state], axis=-1)
            return pytorch_model.wrap(inp_state, cuda=self.iscuda)
        else:
            inp_state, tar_state = self._wrap_state((inp_state, tar_state))
            return inp_state

    def inter_passive(self, inter_mask):
        inter_mask = pytorch_model.unwrap(inter_mask)
        if self.name in ["Reward", "Done"]:
            return np.zeros(inter_mask.shape[:-1])
        passive_mask = self.check_passive_mask(inter_mask)
        return np.abs(np.sum(inter_mask - passive_mask, axis=-1)) == 0

    def check_passive_mask(self, state): # creates as many passive masks as there are states
        # Does NOT handle non-batched state, expects state -> [batch_size, state_dim]
        n_keys, n_queries = self.count_keys_queries(state)
        passive_mask = np.zeros((state.shape[0], n_keys, self.num_inter, )).astype(float)
        if type(state) == torch.Tensor: passive_mask = pytorch_model.wrap(passive_mask, cuda=self.iscuda)
        if self.name in ["Reward", "Done"]: return passive_mask
        for i in range(n_keys):
            passive_mask[:,i, self.name_index + i] = 1.0
        passive_mask = passive_mask.reshape(state.shape[0], -1) # [batch_size, num_keys x num_instances]
        # print("passive_mask", passive_mask)
        return passive_mask
    
    def count_keys_queries(self, x):
        return count_keys_queries(self.first_obj_dim, self.single_obj_dim, self.obj_dim, x)

interaction_models = {'neural': FullNeuralInteractionForwardModel}