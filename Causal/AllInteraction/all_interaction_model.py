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
from Causal.interaction_model import get_params
from Causal.FullInteraction.full_interaction_testing import InteractionMaskTesting
from Causal.interaction_base_model import NeuralInteractionForwardModel, regenerate, KEYNETS, PAIR, MASKING_FORMS
from Environment.Normalization.norm import NormalizationModule
from Environment.Normalization.full_norm import FullNormalizationModule
from Environment.Normalization.pad_norm import PadNormalizationModule

def get_params_all(model, full_args, is_pair, multi_instanced, total_inter_size, single_target_size):
    full_args.interaction_net.object_names = model.names
    full_args.mask_dim = model.num_inter
    full_args.interaction_net.mask_attn.return_mask = False
    full_args.interaction_net.mask_attn.gumbel_temperature = full_args.full_inter.dist_temperature
    full_args.interaction_net.attention_mode = full_args.interaction_net.net_type == "rawattn"
    full_args.interaction_net.pair.total_instances = np.sum(model.extractor.num_instances)
    full_args.interaction_net.selection_temperature = full_args.full_inter.selection_temperature
    full_args.interaction_net.symmetric_key_query = True # this will be true since we need class-ided keys and queries
    full_args.interaction_net.multi.num_masks = full_args.EMFAC.num_masks
    
    active_model_args = copy.deepcopy(full_args.interaction_net)
    active_model_args.num_inputs = total_inter_size
    active_model_args.num_outputs = single_target_size

    passive_model_args = copy.deepcopy(full_args.interaction_net)
    passive_model_args.num_inputs = total_inter_size
    if passive_model_args.net_type in KEYNETS: passive_model_args.hidden_sizes = passive_model_args.hidden_sizes + passive_model_args.pair.final_layers
    passive_model_args.num_outputs = single_target_size

    interaction_model_args = copy.deepcopy(full_args.interaction_net)
    interaction_model_args.num_inputs = total_inter_size
    interaction_model_args.num_outputs = 1
    interaction_model_args.mask_attn.return_mask = False
    interaction_model_args.softmax_output = False
    interaction_model_args.cluster.use_cluster = interaction_model_args.cluster.cluster_mode # cluster information passed here 

    if is_pair:
        pair = copy.deepcopy(full_args.interaction_net.pair)
        pair.object_dim = model.obj_dim
        pair.first_obj_dim = model.first_obj_dim
        pair.single_obj_dim = model.obj_dim # the "single objects`" are symmetric with the objcts
        pair.post_dim = -1
        # parameters specific to key-pair/transformer networks 
        pair.total_obj_dim = np.sum(model.extractor.full_object_sizes)
        pair.expand_dim = model.extractor.expand_dim
        pair.total_instances = np.sum(model.extractor.num_instances)
        pair.query_pair = False
        pair.expand_dim = model.extractor.expand_dim
        
        pair.aggregate_final = False # we are multi-instanced in outputs
        active_model_args.pair, passive_model_args.pair, interaction_model_args.pair = copy.deepcopy(pair), copy.deepcopy(pair), copy.deepcopy(pair)
        interaction_model_args.query_pair = True
        interaction_model_args.embedpair.query_aggregate = False # variable output based on the number of queries
        if interaction_model_args.cluster.cluster_mode:
            interaction_model_args.num_outputs = interaction_model_args.cluster.num_clusters 
            interaction_model_args.cluster.cluster_mode = False # the interaction model does not take cluster mode
            # interaction_model_args.pair.aggregate_final = True
            interaction_model_args.softmax_output = True
            interaction_model_args.pair.total_instances = interaction_model_args.cluster.num_clusters # interaction models use this as a replacement for num_outputs
        if full_args.full_inter.lightweight_passive:
            passive_model_args.net_type = "conv"
            passive_model_args.pair.difference_first = False # difference first cannot be true since there is no first, at least for now
    return active_model_args, passive_model_args, interaction_model_args


class AllNeuralInteractionForwardModel(NeuralInteractionForwardModel):
    def __init__(self, args, target, environment, causal_extractor, normalization):
        super().__init__(args, "all", environment, causal_extractor, normalization, get_params_all, "all")

    def regenerate(self, extractor, norm, environment):
        super().regenerate(extractor, norm, environment)
        self.all_names = [n for n in environment.all_names]
        self.num_inter = len(self.all_names)# number of instances to interact with
        self.target_num = self.num_inter
        self.obj_dim, self.single_obj_dim, self.first_obj_dim = self.extractor._get_dims(None)
        self.target_select = extractor.target_select # target prediction without identity components
        if hasattr(self, "passive_model") and self.passive_model is not None and hasattr(self.passive_model, "reset_environment"):
            self.passive_model.reset_environment(0, self.num_inter, self.first_obj_dim)
        if hasattr(self, "active_model") and hasattr(self.active_model, "reset_environment"):
            self.active_model.reset_environment(0, self.num_inter, self.first_obj_dim)
        if hasattr(self, "interaction_model") and hasattr(self.interaction_model, "reset_environment"):
            self.interaction_model.reset_environment(0, self.num_inter, self.first_obj_dim)

    def _wrap_state(self, state, tensor=True, use_next=False):
        return self._wrap_inter(state, tensor=tensor, use_next=use_next)

    def get_interaction_state(self, state, next_state=None, factored=True):
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        if factored:
            inp_state, tar_state = self._wrap_state(state, tensor=False) 
            next_inp_state, next_tar_state = self._wrap_state(next_state, tensor=False)
        else:
            inp_state, tar_state = state.obs, state.target
            next_inp_state, next_tar_state = state.obs_next, state.next_target
        if self.nextstate_interaction:
            obs_dict = self.extractor.reverse_extract(inp_state, target=False)
            next_obs_dict = self.extractor.reverse_extract(next_inp_state, target=False)
            obs_dict = {n: np.concatenate([obs_dict[n], next_obs_dict[n]], axis=-1) for n in obs_dict.keys()}
            inp_state = self.inter_select(obs_dict)

        return pytorch_model.wrap(inp_state, cuda=self.iscuda)

    def inter_passive(self, inter_mask):
        inter_mask = pytorch_model.unwrap(inter_mask)
        passive_mask = self.check_passive_mask(inter_mask)
        return np.abs(np.sum(inter_mask - passive_mask, axis=-1)) == 0

    def check_passive_mask(self, state): # creates as many passive masks as there are states
        # Does NOT handle non-batched state, expects state -> [batch_size, state_dim]
        passive_mask = np.broadcast_to(np.identity(self.num_inter), (state.shape[0], self.num_inter, self.num_inter))
        if type(state) == torch.Tensor: passive_mask = pytorch_model.wrap(passive_mask, cuda=self.iscuda)
        passive_mask = passive_mask.reshape(state.shape[0], -1) # [batch_size, num_keys x num_instances]
        return passive_mask
    
    def count_keys_queries(self, x):
        return count_keys_queries(0, 1, self.obj_dim, x)

    def _compute_passive(self, inp_state, tar_state):
        # print(inp_state.shape, tar_state.shape)
        if self.cluster_mode:
            batch_size = 0 if len(inp_state.shape) < 2 else inp_state.shape[0]
            passive_inter = self.active_model.get_hot_passive_mask(batch_size, self.num_inter)
            return self.active_model(inp_state, passive_inter)[0]
        else:
            return super()._compute_passive(inp_state, tar_state)

    def normalize_batch(self, batch): # normalizes the components in the batch to be used for likelihoods, assumes the batch is an object batch
        batch = super().normalize_batch(batch)
        batch.tarinter_state = batch.inter_state
        return batch

    def get_cluster_full_mask(self, x, all=False):
        # only works if cluster_mode is true
        # All returns the average value, which should only be used for evaluation
        n_keys, n_queries = self.count_keys_queries(x)
        n_keys = n_queries
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        if all: return self.active_model.get_all_mask(batch_size, n_keys, n_queries)
        return self.active_model.get_hot_full_mask(batch_size, n_keys, n_queries)