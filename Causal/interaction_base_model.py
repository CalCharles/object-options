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
from Causal.FullInteraction.full_interaction_testing import InteractionMaskTesting
from Environment.Normalization.norm import NormalizationModule
from Environment.Normalization.full_norm import FullNormalizationModule
from Environment.Normalization.pad_norm import PadNormalizationModule

KEYNETS = ["keypair", "keyembed", "maskattn", "rawattn"]
PAIR=  ["pair", "keypair", "keyembed", "maskattn", "rawattn"]

MASKING_FORMS = {
    "weighting": 0,
    "relaxed": 1,
    "mixed": 2,
    "hard": 3,
}

def regenerate(append_id, environment):
    extractor = CausalPadExtractor(environment, append_id)
    # norm = FullNormalizationModule(environment.object_range, environment.object_dynamics, name, environment.object_instanced, environment.object_names)
    pad_size = extractor.pad_dim
    append_size = extractor.append_dim
    environment.object_range_true = environment.object_range
    norm = PadNormalizationModule(environment.object_range, environment.object_range_true, environment.object_dynamics, environment.object_instanced, environment.object_names, pad_size, append_size)
    return extractor, norm

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


class NeuralInteractionForwardModel(nn.Module):
    # base class for full interaction model and all interaction model
    def __init__(self, args, target, environment, causal_extractor, normalization, get_params):
        super().__init__()
        self.names = environment.object_names

        self.regenerate(causal_extractor, normalization, environment)
        # self.controllable = args.controllable

        # if we are predicting the dynamics
        self.predict_dynamics = True
        
        # construct the active model
        args.interaction_net.object_dim = self.obj_dim
        self.multi_instanced = environment.object_instanced[self.name] > 1 # an object CANNOT go from instanced to multi instanced
        self.cluster_mode = self.active_model_args.cluster.cluster_mode # uses a mixture of experts implementation, which shoudl return different interaction masks
        self.attention_mode = self.active_model_args.attention_mode # gets the interaction mask from the active model
        self.num_clusters = self.active_model_args.cluster.num_clusters # uses a mixture of experts implementation, which shoudl return different interaction masks
        self.selection_mask = args.full_inter.selection_mask
        self.active_model_args, self.passive_model_args, self.interaction_model_args = get_params(self, args, args.interaction_net.net_type in PAIR, environment.object_instanced[self.name], self.extractor.total_inter_size, self.extractor.single_object_size)

        # set the distributions
        self.dist = assign_distribution("Gaussian") # TODO: only one kind of dist at the moment
        self.relaxed_inter_dist = assign_distribution(args.full_inter.soft_distribution)
        self.dist_temperature = args.full_inter.dist_temperature
        self.inter_dist = assign_distribution("Bernoulli")
        self.mixing = args.full_inter.mixed_interaction# mostly only used for training
        
        # set the testing module
        self.test = InteractionMaskTesting(args.inter.interaction_testing)

        # set the forward model
        self.active_model_args.mask_attn.inter_dist, self.active_model_args.mask_attn.relaxed_inter_dist, self.active_model_args.mask_attn.dist_temperature, self.active_model_args.mask_attn.test = self.inter_dist, self.relaxed_inter_dist, self.dist_temperature, self.test
        self.active_model = DiagGaussianForwardPadHotNetwork(self.active_model_args) if self.cluster_mode else DiagGaussianForwardPadMaskNetwork(self.active_model_args)

        # set the passive model
        self.use_active_as_passive = args.full_inter.use_active_as_passive or self.cluster_mode # uses the active model with the one hot as the passive model
        self.lightweight_passive = args.full_inter.lightweight_passive
        # self.passive_model = DiagGaussianForwardPadMaskNetwork(self.passive_model_args) # TODO: comment this out
        self.passive_model = (DiagGaussianForwardNetwork(self.passive_model_args) 
                                if self.lightweight_passive else DiagGaussianForwardPadMaskNetwork(self.passive_model_args) 
                                ) if not self.use_active_as_passive else None
        

        # construct the interaction model
        self.soft_inter_dist = assign_distribution("RelaxedHot") if self.cluster_mode else assign_distribution("Identity")
        self.hard_inter_dist = assign_distribution("CategoricalHot") if self.cluster_mode else assign_distribution("Identity")
        self.interaction_model = (InteractionSelectionMaskNetwork(self.interaction_model_args) if self.selection_mask else InteractionMaskNetwork(self.interaction_model_args)) if not self.attention_mode else None

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
        self.cuda(device = args.torch.gpu) if args.torch.cuda else self.cpu()
        self.gpu = args.torch.gpu
        self.regenerate(causal_extractor, normalization, environment)
        print(self)


    def toggle_active_as_passive(self, use_active_as_passive):
        # dangerous to toggle if no passive model implemented
        self.use_active_as_passive = use_active_as_passive
    
    def regenerate(self, extractor, norm, environment):
        self.norm, self.extractor = norm, extractor
        self.all_names = [n for n in environment.all_names if n not in ["Reward", "Done"]]
        self.num_inter = len(self.all_names)# number of instances to interact with
        self.target_selectors, self.full_select = self.extractor.get_selectors()
        self.obj_dim = self.extractor.get_base_dim()
        if hasattr(self, "mask") and self.mask is not None: self.mask.regenerate_norm(norm)
    
    def load_forward_only(self, new_full_model):
        self.passive_model = new_full_model.passive_model
        self.active_model = new_full_model.active_model

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
        if self.interaction_model is not None: self.interaction_model.cpu()
        if self.passive_model is not None: self.passive_model.cpu()
        self.iscuda = False
        return self

    def cuda(self, device=-1):
        gpu = device if device >=0 else self.gpu
        device = cuda_string(gpu)
        super().cuda()
        self.active_model.cuda(gpu=gpu).to(device)
        if self.interaction_model is not None: self.interaction_model.cuda(gpu=gpu).to(device)
        if self.passive_model is not None: self.passive_model.cuda(gpu=gpu).to(device)
        self.iscuda = True
        return self

    def reset_parameters(self):
        self.active_model.reset_parameters()
        if self.interaction_model is not None: self.interaction_model.reset_parameters()
        if self.passive_model is not None: self.passive_model.reset_parameters()

    def _compute_passive(self, inp_state, tar_state):
        # print(inp_state.shape, tar_state.shape)
        if self.cluster_mode:
            batch_size = 0 if len(inp_state.shape) < 2 else inp_state.shape[0]
            num_keys, num_queries = self.count_keys_queries(inp_state)
            passive_inter = self.active_model.get_hot_passive_mask(batch_size, num_keys, num_queries)
            # print("passive", passive_inter, inp_state, self.active_model(inp_state, passive_inter)[1])
            return self.active_model(inp_state, passive_inter)[0]
        else:
            if self.use_active_as_passive: return self.active_model(inp_state, self.check_passive_mask(tar_state))[0]
            elif self.lightweight_passive: return self.passive_model(tar_state)
            else: return self.passive_model(inp_state, self.check_passive_mask(tar_state))[0]

    def apply_passive(self, state):
        # assumes that the state is (inp_state, target_state) if tuple
        # if a single tensor, the passive model should NOT require a target state
        # if a batch, it will look in target_select, inter_select
        if type(state) == torch.Tensor:
            inp_state, tar_state = state, state
            return self._compute_passive(inp_state, tar_state)
        elif type(state) == tuple:
            if type(state[0]) == torch.Tensor:
                inp_state, tar_state = state
                return self._compute_passive(inp_state, tar_state)
            else:
                inp_state, tar_state = self._wrap_state(state)
                return pytorch_model.unwrap(self._compute_passive(inp_state, tar_state))
        else:
            inp_state, tar_state = self._wrap_state(state)
            return self._compute_passive(inp_state, tar_state)


    def predict_next_state(self, state, normalized=False, difference=False):
        # returns the interaction value and the predicted next state (if interaction is low there is more error risk)
        # state is either a single flattened state, or batch x state size, or factored_state with sufficient keys
        # @param difference returns the dynamics prediction instead of the active prediction, not used if the full model is not a dynamics predictor
        inp_state, tar_state = self._wrap_state(state)

        rv = self.norm.reverse
        if self.attention_mode:
            inter = pytorch_model.unwrap(self.interaction_model(inp_state))
            inter_mask = self.apply_mask(inter, flat=True, x = inp_state)
        else: inter, inter_mask = None, None
        # if predicting dynamics, add the mean of the model to the target state
        active_params, mask = self.active_model(inp_state, inter_mask)
        inter = mask if self.cluster_mode or self.attention_mode else inter
        if self.predict_dynamics:
            if difference:
                fpred, ppred = rv(active_params[0], form="dyn"), rv(self.apply_passive((inp_state, tar_state))[0], form="dyn")
            else:
                fpred, ppred = rv(tar_state) + rv(active_params[0], form="dyn"), rv(tar_state) + rv(self.apply_passive((inp_state, tar_state))[0], form="dyn")
        else:
            fpred, ppred = rv(active_params[0]), rv(self.apply_passive((inp_state, tar_state))[0])
        
        if normalized: fpred, ppred = self.norm(fpred, form="dyn" if difference else "target"), self.norm(ppred, form="dyn" if difference else "target")

        # TODO: remove this conditional with appropriate slicing
        # select active or passive based on inter value
        # if len(fpred.shape) == 1:
        #     return (inter, fpred)
        # else:
        #     pred = np.stack((ppred, fpred), axis=1)
        #     intera = inter_bin.squeeze().astype(int)
        #     pred = pred[np.arange(pred.shape[0]).astype(int), intera]
        return inter, fpred


    def hypothesize(self, state):
        # takes in a full state (dict with raw_state, factored_state) or tuple of ndarray of input_state, target_state 
        # computes the interaction value, the mean, var of forward model, the mean, var of the passive model
        inter_state, tar_state = self._wrap_state(state)
        rv = self.norm.reverse
        if self.attention_mode: inter, inter_mask = None, None
        else:
            inter = self.interaction_model(inter_state)
            inter_mask = self.apply_mask(inter, flat = True, x=inter_state)
        (mu_inter, var_inter), m = self.active_model(inter_state, inter_mask)
        inter = m if self.cluster_mode or self.attention_mode else inter
        pmu_inter, pvar_inter = self.apply_passive((inter_state, tar_state))
        return (pytorch_model.unwrap(inter),
            (rv(pytorch_model.unwrap(mu_inter)), rv(pytorch_model.unwrap(var_inter))), 
            (rv(pytorch_model.unwrap(pmu_inter)), rv(pytorch_model.unwrap(pvar_inter))))

    def check_interaction(self, inter):
        return self.test(inter)

    def get_active_mask(self):
        return self.test.selection_binary

    def interaction(self, val, target=None, next_target=None, target_diff=None, prenormalize=False, use_binary=False, return_hot=False, all=False): # val is either a batch, or a ndarray of inter_state. Does NOT unwrap, Does NOT normalize
        if type(val) != Batch:
            bat = Batch()
            bat.inter_state = val # state from the full_rollout
            bat.target = target # state from the object rollout
            bat.tarinter_state = bat.inter_state if all else np.concatenate([bat.target, bat.inter_state], axis=-1)
            bat.next_target = next_target
            bat.target_diff = target_diff
        else:
            bat = val
        if type(val) != np.ndarray: val = val.inter_state # if not an array, assume it is a Batch
        if prenormalize: 
            bat = self.normalize_batch(bat)

        # print("interactions", use_binary, self.cluster_mode, self.attention_mode)
        if use_binary:
            _, _, inter, inter_mask, _, _, _, active_log_probs, passive_log_probs = self.reduced_likelihoods(bat, masking = "full")
            binary = self.test.compute_binary(- active_log_probs.sum(dim=-1),
                                                - passive_log_probs.sum(dim=-1)).unsqueeze(-1)
            return binary
        elif self.cluster_mode:
            val = bat.tarinter_state
            inter_hot = self.interaction_model(pytorch_model.wrap(val, cuda=self.iscuda))
            (mu_inter, var_inter), m = self.active_model(pytorch_model.wrap(val, cuda=self.iscuda), inter_hot)
            if return_hot: return inter_hot, m
            return m
        elif self.attention_mode:
            val = bat.tarinter_state
            (mu_inter, var_inter), m = self.active_model(pytorch_model.wrap(val, cuda=self.iscuda), None)
            if return_hot: return None, m
            return m
        else:
            val = bat.tarinter_state
            if return_hot: return None, self.interaction_model(pytorch_model.wrap(val, cuda=self.iscuda))
            return self.interaction_model(pytorch_model.wrap(val, cuda=self.iscuda))
    
    def _target_dists(self, batch, params, skip=None):
        # start = time.time()
        target = batch.target_diff if self.predict_dynamics else batch.next_target
        target = pytorch_model.wrap(target, cuda=self.iscuda)
        # print(target.shape, target[:6])
        # print("wrap", time.time() - start)
        num_param_sets = int(params[0].shape[-1] // target.shape[-1]) if params[0].shape[-1] > target.shape[-1] else 1
        log_probs = list()
        # print(params[0].shape[-1], target.shape)
        for i in range(num_param_sets):
            if skip is not None and skip[i] == 0: continue # only add the non-skipped params
            new_params = [p[..., target.shape[-1]*i:target.shape[-1] * (i+1)] for p in params]
            dist = self.dist(*new_params)
            # print(target.shape, dist)
            # print("dist", time.time() - start)
            log_probs.append(dist.log_prob(target))
        log_probs = torch.cat(log_probs, dim=-1)
        # print("log_probs", time.time() - start)
        return target, dist, log_probs

    def normalize_batch(self, batch): # normalizes the components in the batch to be used for likelihoods, assumes the batch is an object batch
        batch.inter_state = self.norm(batch.inter_state, form="inter")
        batch.obs = self.norm(batch.obs, name=self.name)
        batch.tarinter_state = np.concatenate([batch.obs, batch.inter_state], axis=-1)
        batch.obs_next = self.norm(batch.obs_next, name=self.name)
        batch.target_diff = self.norm(batch.target_diff, form="dyn", name=self.name)
        return batch

    def apply_cluster_mask(self, inter_mask, cluster_hard=False):
        # only applicable in cluster mode, turns a softmax selection over clusters into the mask corresponding to that selection
        inter_mask = inter_mask.reshape(inter_mask.shape[0], -1, self.num_clusters)
        if cluster_hard: return self.hard_inter_dist(inter_mask).sample()
        else: return self.soft_inter_dist(self.dist_temperature, probs=inter_mask).rsample()

    def apply_mask(self, inter_mask, soft=True, flat=False, cluster_hard=False, x=None):
        # generate the interaction mask out of the outputs of the interaction model
        # if the interaction model is in cluster mode, extracts the cluster interaction mask first
        # inter_mask in this case is the selection over cluster modes
        # x is required in this mode
        # does not apply when in attention_mode
        revert_mask = type(inter_mask) != torch.Tensor
        if self.cluster_mode:
            if len(inter_mask.shape) < 2: total_len = 0
            else: total_len = x.shape[0]
            true_inter_mask = list()
            for i in range(int(np.ceil(total_len / 512))): # break it up to avoid overloading the GPU
                batch = x[i*512:(i+1) * 512]
                batch_len = batch.shape[0]

                # in cluster mode, m has the form num_batch, num_keys, num_cluster_heads
                num_keys, num_queries = self.count_keys_queries(batch)
                if self.form == "all": num_keys = num_queries
                # print(type(batch), batch.shape)
                all_masks, inter_m = self.active_model.compute_cluster_masks(pytorch_model.wrap(batch, cuda = self.iscuda), pytorch_model.wrap(inter_mask[i*512:(i+1) * 512], cuda=self.iscuda), num_keys, num_queries)
                true_inter_mask.append(inter_m)
            inter_mask = torch.cat(true_inter_mask, dim=0)
            # print(inter_mask.shape, int(np.ceil(total_len / 512)), total_len)
        if revert_mask: inter_mask = pytorch_model.wrap(inter_mask, cuda=self.iscuda)
        mixed = MASKING_FORMS[self.mixing] == 2
        inter_mask = inter_mask - 1e-5
        inter_mask[inter_mask < 0] = 0
        return apply_probabilistic_mask(inter_mask, inter_dist=self.inter_dist if ((not soft) or (soft and mixed)) else None, relaxed_inter_dist=self.relaxed_inter_dist if (soft and not mixed) else None, mixed=mixed, test=self.test if flat else None, dist_temperature=self.dist_temperature, revert_mask=revert_mask)

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

    # likelihood functions (below) get the gaussian distributions output by the active and passive models for all mask forms
    def _likelihoods(self, batch, normalize=False, 
                        mixed="", input_grad=False, 
                        soft_select=False, soft_eval=False, skip_dists=0, # cluster specific parameters
                        return_selection=False, # interaction selection specific parameters
                        compute_values = "all", # a list of some subset of ["hard", "soft", "full", "passive"] or "all" to denote all 
                    ):
        if normalize: batch = self.normalize_batch(batch)

        # flags for which values to compute
        use_hard = (compute_values == "all") or ("hard" in compute_values)
        use_soft = (compute_values == "all") or ("soft" in compute_values)
        use_passive = (compute_values == "all") or ("passive" in compute_values)
        use_full = (compute_values == "all") or ("full" in compute_values)
        use_flat = ("flat" in compute_values) # all does NOT include flat
        use_active = use_hard or use_soft or use_full

        # logic for handling input gradients, if needed
        iv = pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda)
        active_hard_input = pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda) if use_hard and input_grad else iv
        active_soft_input = pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda) if use_soft and input_grad else iv
        active_full_input = pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda) if use_full in compute_values and input_grad else iv
        passive_full_input = pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda) if use_passive in compute_values and input_grad else iv
        active_hard_input.requires_grad, active_soft_input.requires_grad, active_full_input.requires_grad, passive_full_input.requires_grad = input_grad, input_grad, input_grad, input_grad

        if use_active: # otherwise, the active model is not needed
            if self.cluster_mode:
                # the "inter" here is the hot mask 
                inter = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda))
                hot_mask = inter
                if use_hard or use_soft:
                    # get the one-hot mask applies categorical or gumbel softmax
                    hot_inter_soft, hot_inter_hard = self.apply_cluster_mask(inter, cluster_hard=False), self.apply_cluster_mask(inter, cluster_hard=True) 
                    use_hot_mask = hot_inter_soft if soft_select else hot_inter_hard # in general, the selection mask (soft_select) should be hard
                    mixed = MASKING_FORMS[self.mixing] == 2
                    if use_hard: active_hard_params, hard_inter_mask = self.active_model(active_hard_input, hot_inter_hard, soft=False, mixed=False, flat=False)
                    if use_flat: active_flat_params, flat_inter_mask = self.active_model(active_hard_input, hot_inter_hard, soft=False, mixed=False, flat=True)
                    if use_soft: active_soft_params, soft_inter_mask = self.active_model(active_soft_input, use_hot_mask, soft=True, mixed=mixed, flat=False)
                    inter = soft_inter_mask
                # the full output either outputs all of the inputs
                if use_full:
                    full_mask = self.get_cluster_full_mask(batch.tarinter_state) # should be unused
                    active_full_params, m = self.active_model(active_full_input, full_mask, soft=True, mixed=mixed, full=True)
            if self.attention_mode:
                mixed = MASKING_FORMS[self.mixing] == 2
                if use_hard: active_hard_params, hard_inter_mask = self.active_model(active_hard_input, None, soft=False, mixed=False, flat=False)
                if use_soft: active_soft_params, soft_inter_mask = self.active_model(active_soft_input, None, soft=True, mixed=mixed, flat=False)
                if use_flat: active_flat_params, flat_inter_mask = self.active_model(active_flat_input, None, soft=False, mixed=False, flat=True)
                if use_full:
                    full_mask = pytorch_model.wrap(torch.ones(len(self.all_names) * self.target_num), cuda = self.iscuda)
                    active_full_params, m = self.active_model(active_full_input, None, soft=False, mixed=mixed) # this one is not really a good idea to train on
                hot_mask = hard_inter_mask
                inter = soft_inter_mask
            else:
                if self.selection_mask: inter, hot_mask = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), return_selection=True)
                else:
                    inter = self.interaction_model(pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda))
                    hot_mask = inter # hot mask not used
                if use_full:
                    full_mask = pytorch_model.wrap(torch.ones(len(self.all_names) * self.target_num), cuda = self.iscuda)
                    active_full_params, m = self.active_model(active_full_input, full_mask)
                if use_hard: 
                    hard_inter_mask = self.apply_mask(inter, soft=False)
                    active_hard_params, m = self.active_model(active_hard_input, hard_inter_mask)
                if use_flat:
                    flat_inter_mask = self.apply_mask(inter, soft=False, flat = True)
                    active_flat_params, m = self.active_model(active_flat_input, flat_inter_mask)
                if use_soft:
                    soft_inter_mask = self.apply_mask(inter, soft=True)
                    mixed_mask = self.combine_mask(soft_inter_mask, hard_inter_mask, mixed=mixed)
                    soft_inter_mask = mixed_mask
                    active_soft_params, m = self.active_model(active_soft_input, mixed_mask)
                # print(full_mask, mixed_mask)
        if self.use_passive: passive_params = self.apply_passive((pytorch_model.wrap(batch.tarinter_state, cuda=self.iscuda), pytorch_model.wrap(batch.obs, cuda=self.iscuda)))

        if use_hard: target, active_hard_dist, active_hard_log_probs = self._target_dists(batch, active_hard_params)
        if use_soft: target, active_soft_dist, active_soft_log_probs = self._target_dists(batch, active_soft_params)
        if use_flat: target, active_flat_dist, active_flat_log_probs = self._target_dists(batch, active_flat_params)
        if self.cluster_mode:
            skip_dist_indexes = np.ones(self.num_clusters)
            if skip_dists > 0: #otherwise, don't skip anything
                skip_dist_indexes[:skip_dists] = 0
        else: skip_dist_indexes = None
        if use_full:
            target, active_full_dist, active_full_log_probs = self._target_dists(batch, active_full_params, skip=skip_dist_indexes)
            if self.cluster_mode: # scale the full params by the selection output
                # print(hot_mask[:6], active_full_log_probs[:6], active_full_params[0][:6], target[:6])
                if not soft_eval:
                    # print(hot_mask.shape, len(batch), active_full_log_probs.reshape(len(batch), self.num_clusters, -1).shape)
                    active_full_log_probs = (active_full_log_probs.reshape(len(batch), self.num_clusters, -1) * hot_mask[skip_dists:].unsqueeze(-1)).sum(dim=1)
                else:
                    active_full_log_probs = active_full_log_probs.reshape(len(batch), self.num_clusters - skip_dists, -1).mean(dim=1)
        # print("full params", active_full_params[0], active_full_params[1], target,batch.tarinter_state, active_full_log_probs[0])
        if use_passive: target, passive_dist, passive_log_probs = self._target_dists(batch, passive_params)
        
        params = list()
        masks = list()
        dists = list()
        log_probs = list()
        inps = list()
        if use_active:
            masks += [inter, hot_mask]
        if use_hard:
            params += [active_hard_params]
            masks += [hard_inter_mask]
            dists += [active_hard_dist]
            log_probs += [active_hard_log_probs]
            inps += [active_hard_input]
        if use_soft:
            params += [active_soft_params]
            masks += [soft_inter_mask]
            dists += [active_soft_dist]
            log_probs += [active_soft_log_probs]
            inps += [active_soft_input]
        if use_full:
            params += [active_full_params]
            masks += [full_inter_mask]
            dists += [active_full_dist]
            log_probs += [active_full_log_probs]
            inps += [active_full_input]
        if use_flat:
            params += [active_flat_params]
            masks += [flat_inter_mask]
            dists += [active_flat_dist]
            log_probs += [active_flat_log_probs]
            inps += [active_flat_input]
        if use_passive:
            params += [passive_params]
            dists += [passive_dist]
            log_probs += [passive_log_probs]
            inps += [passive_input]
        
        return (*params, target, *dists, *log_probs, *inps)

        # TODO: adjusted the order of parameters: inter, hot_mask, soft_inter_mask, hard_inter mask, added passive_input, added flat
        # return active_hard_params, active_soft_params, active_full_params, passive_params,\
        #          inter, soft_inter_mask, hard_inter_mask, hot_mask,\
        #          target, \
        #          active_hard_dist, active_soft_dist, active_full_dist, passive_dist, \
        #          active_hard_log_probs, active_soft_log_probs, active_full_log_probs, passive_log_probs,\
        #          active_hard_input, active_soft_input, active_full_input


    # batch, normalize=False, mixed="", input_grad=False, soft_select=False, soft_eval=False, skip_dists=0, return_selection=False
                        # compute_values = "all"
    # uses the flat mask
    def likelihoods(self, batch, normalize=False, mixed="", input_grad=False, soft_select=False, soft_eval=False, skip_dists=0, return_selection=False):
        return self._likelihoods(batch, normalize=normalize, mixed=mixed, input_grad=input_grad, soft_select=soft_select, soft_eval=soft_eval, skip_dists=skip_dists, return_selection=return_selection, compute_values = "all")

    def reduced_likelihoods(self, batch, normalize=False, masking="", return_selection):
        return self._likelihoods(batch, normalize=normalize, mixed="", return_selection = False, compute_values = masking)

    # gets the active likelihood without the interaction mask blocking any inputs
    def active_open_likelihoods(self, batch, normalize=False, all=False):
        return self._likelihoods(batch, normalize=normalize, mixed="", return_selection = False, compute_values = ["full"])

    def passive_likelihoods(self, batch, normalize=False):
        return self._likelihoods(batch, normalize=normalize, mixed="", return_selection = False, compute_values = ["passive"])

    def active_likelihoods(self, batch, normalize=False, soft=False, flat=False, cluster_choice=-1): # TODO: flat not implemented
        return self._likelihoods(batch, normalize=normalize, compute_values = ["soft"] if soft else ["hard"])
