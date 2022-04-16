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

# def nflen(x):
#     return ConstantNorm(mean= pytorch_model.wrap(sum([[84//2,84//2,0,0,0] for i in range(x // 5)], list())), variance = pytorch_model.wrap(sum([[84,84, 5, 5, 1] for i in range(x // 5)], list())), invvariance = pytorch_model.wrap(sum([[1/84,1/84, 1/5, 1/5, 1] for i in range(x // 5)], list())))

# nf5 = ConstantNorm(mean=0, variance=5, invvariance=.2)

# def default_model_args(predict_dynamics, model_class, norm_fn=None, delta_norm_fn=None):    
#     model_args = ObjDict({ 'model_type': 'neural',
#      'dist': "Gaussian",
#      'passive_class': model_class,
#      "forward_class": model_class,
#      'interaction_class': model_class,
#      'init_form': 'xnorm',
#      'activation': 'relu',
#      'factor': 8,
#      'num_layers': 2,
#      'use_layer_norm': False,
#      'normalization_function': norm_fn,
#      'delta_normalization_function': delta_norm_fn,
#      'interaction_binary': [],
#      'active_epsilon': .5,
#      'base_variance': .0001 # TODO: this parameter is extremely sensitive, and that is a problem
#      })
#     return model_args

# def load_hypothesis_model(pth):
#     for root, dirs, files in os.walk(pth):
#         for file in files:
#             if file.find(".pt") != -1: # return the first pytorch file
#                 return torch.load(os.path.join(pth, file))

def assign_distribution(dist):
        if kwargs['dist'] == "Discrete": return Categorical(kwargs['num_outputs'], kwargs['num_outputs'])
        elif kwargs['dist'] == "Gaussian": return torch.distributions.normal.Normal
        elif kwargs['dist'] == "MultiBinary": return Bernoulli(kwargs['num_outputs'], kwargs['num_outputs'])
        else: raise NotImplementedError

class NeuralInteractionForwardModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # set input and output
        self.inter_state = kwargs['inter_state']
        self.target_state = kwargs['target_state']
        self.parent_states = kwargs['parent_states']
        self.controllable = kwargs['controllable']
        
        # construct the active model
        self.first_obj_dim = kwargs['first_obj_dim']
        forward_args = kwargs['forward_model_args'] 
        self.forward_model = forward_nets[kwargs['forward_class']](**kwargs['passive_model_args'])

        # set the passive model
        self.passive_model = forward_nets[kwargs['passive_class']](**kwargs['forward_model_args'])

        # construct the interaction model        
        self.interaction_model = interaction_nets[kwargs['interaction_class']](**kwargs['interaction_model_args'])

        # set the testing module
        self.testing_module = TestingModule(kwargs)

        # set the normalization function
        self.norm = NormalizationModule(kwargs)

    def save(self, pth):
        try:
            os.mkdir(pth)
        except OSError as e:
            pass
        torch.save(self, os.path.join(pth, self.name + "_model.pt"))

    def cpu(self):
        super().cpu()
        self.forward_model.cpu()
        self.interaction_model.cpu()
        self.passive_model.cpu()
        self.norm.cpu()
        self.testing_module.cpu()
        self.iscuda = False
        return self

    def cuda(self):
        super().cuda()
        self.forward_model.cuda()
        self.interaction_model.cuda()
        self.passive_model.cuda()
        self.norm.cuda()
        self.testing_module.cuda()
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

        rv = self.output_normalization_function.reverse
        inter = pytorch_model.unwrap(self.interaction_model(inp_state))
        inter_bin = self.testing_module(inter)

        # if predicting dynamics, add the mean of the model to the target state
        if self.predict_dynamics:
            fpred, ppred = tar_state + rv(self.forward_model(inp_state)[0]), tar_state + rv(self.passive_model(tar_state)[0])
        else:
            fpred, ppred = rv(self.forward_model(inp_state)[0]), rv(self.passive_model(tar_state)[0])
        
        # TODO: remove this conditional with appropriate slicing
        if len(state.shape) == 1:
            return (inter, fpred) if pytorch_model.unwrap(inter) > self.interaction_prediction else (inter, ppred)
        else:
            pred = torch.stack((ppred, fpred), dim=1)
            intera = pytorch_model.wrap(intera.squeeze().long(), cuda=self.iscuda)
            pred = pred[torch.arange(pred.shape[0]).long(), intera]
        return inter, pred

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
        return self.testing_module(inter)

    def get_active_mask(self):
        return self.testing_module.selection_binary

interaction_models = {'neural': NeuralInteractionForwardModel, 'dummy': DummyModel}