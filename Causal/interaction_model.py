import numpy as np
import os, cv2, time, copy, psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter
from EnvironmentModels.environment_model import get_selection_list, FeatureSelector, ControllableFeature, sample_multiple
from EnvironmentModels.environment_normalization import hardcode_norm
from Counterfactual.counterfactual_dataset import counterfactual_mask
from DistributionalModels.distributional_model import DistributionalModel
from DistributionalModels.InteractionModels.dummy_models import DummyModel
from DistributionalModels.InteractionModels.state_management import StateSet
from file_management import save_to_pickle, load_from_pickle
from Networks.distributions import Bernoulli, Categorical, DiagGaussian
from Networks.DistributionalNetworks.forward_network import forward_nets
from Networks.DistributionalNetworks.interaction_network import interaction_nets
from Networks.network import ConstantNorm, pytorch_model
from Networks.input_norm import InterInputNorm, PointwiseNorm
from Rollouts.rollouts import ObjDict, merge_rollouts
from tianshou.data import Collector, Batch, ReplayBuffer

def assign_distribution(dist):
        if kwargs['dist'] == "Discrete": return Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        elif kwargs['dist'] == "Gaussian": return torch.distributions.normal.Normal
        elif kwargs['dist'] == "MultiBinary": return Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
        else: raise NotImplementedError

def load_interaction(pth):
    for name in os.listdir(pth):
        if "inter_model.pt" in name:
            break
    return torch.load(os.path.join(pth, name))

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # set input and output
        self.names = kwargs["object_names"]
        self.inter_select = kwargs['inter_select']
        self.target_select = kwargs['target_select']
        self.parent_selectors = kwargs['parent_selectors']
        self.parent_select = kwargs['parent_select']
        self.controllable = kwargs['controllable']

        # if we are predicting the dynamics
        self.predict_dynamics = kwargs["predict_dynamics"]
        
        # construct the active model
        self.first_obj_dim = [ps.output_size() for ps in self.parent_selectors] # the first object dim is the combined length of the parents
        self.obj_dim = self.target_select.output_size() # the selector gets the size of a single instance
        self.forward_model = forward_nets[kwargs['active_class']](**kwargs['active_model_args'])

        # set the passive model
        self.passive_model = forward_nets[kwargs['passive_class']](**kwargs['passive_model_args'])

        # construct the interaction model        
        self.interaction_model = interaction_nets[kwargs['interaction_class']](**kwargs['interaction_model_args'])

        # set the testing module
        self.test = InteractionTesting(kwargs)

        # set the normalization function
        self.norm = NormalizationModule(kwargs)

        # set the masking module to None as a placeholder
        self.mask = ActiveMasking(kwargs)

    def save(self, pth):
        try:
            os.mkdir(pth)
        except OSError as e:
            pass
        torch.save(self, os.path.join(pth, self.name + "inter_model.pt"))

    def cpu(self):
        super().cpu()
        self.forward_model.cpu()
        self.interaction_model.cpu()
        self.passive_model.cpu()
        self.norm.cpu()
        self.test.cpu()
        self.iscuda = False
        return self

    def cuda(self):
        super().cuda()
        self.forward_model.cuda()
        self.interaction_model.cuda()
        self.passive_model.cuda()
        self.norm.cuda()
        self.test.cuda()
        self.iscuda = True
        return self

    def reset_parameters(self):
        self.forward_model.reset_parameters()
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
            fpred, ppred = tar_state + rv(self.forward_model(inp_state)[0]), tar_state + rv(self.passive_model(tar_state)[0])
        else:
            fpred, ppred = rv(self.forward_model(inp_state)[0]), rv(self.passive_model(tar_state)[0])
        
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
        mu_inter, var_inter = self.forward_model(inter_state)
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

interaction_models = {'neural': NeuralInteractionForwardModel, 'dummy': DummyModel}