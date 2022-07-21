import numpy as np
import torch
from Network.network_utils import pytorch_model

def split_instances(state, obj_dim):
    # split up a state or batch of states into instances
    nobj = state.shape[-1] // obj_dim
    if len(state.shape) == 1:
        state = state.reshape(nobj, obj_dim)
    elif len(state.shape) == 2:
        state = state.reshape(-1, nobj, obj_dim)
    return state

def flat_instances(state, obj_dim):
    # change an instanced state into a flat state
    if len(state.shape) == 2:
        state = state.flatten()
    elif len(state.shape) == 3:
        batch_size = state.shape[0]
        state = state.reshape(batch_size, state.shape[1] * state.shape[2])
    return state

def compute_likelihood(full_model, batch_size, likelihood_full, done_flags=None, reduced=True):
    # computes either the multi-instanced likelihood, or the normal summed likelihood, regulated by the done flags
    if done_flags is None: done_flags = np.ones((batch_size, 1))
    if type(likelihood_full) == torch.Tensor:
        if full_model.multi_instanced:
            loss = likelihood_full.reshape(batch_size, -1, full_model.obj_dim).sum(dim=-1)
            if reduced:
                loss = likelihood_full.reshape(batch_size, -1, full_model.obj_dim).sum(dim=-1).max(dim=-1)[0].unsqueeze(1)  * pytorch_model.wrap(done_flags, cuda=full_model.iscuda)# only take the worst object performance
            return loss
        else: loss = likelihood_full.sum(dim=-1).unsqueeze(1) * pytorch_model.wrap(done_flags, cuda=full_model.iscuda)
    else: # assumes it's a numpy array and perform the same operation
        if full_model.multi_instanced: loss = np.expand_dims(np.max(np.sum(likelihood_full.reshape(batch_size, -1, full_model.obj_dim), axis=-1), axis=-1), axis=-1)  * done_flags
        else: loss = np.expand_dims(np.sum(likelihood_full, axis=-1), axis=-1) * done_flags
    return loss

def compute_l1(full_model, batch_size, params, targets):
    # computes either the multi-instanced likelihood, or the normal summed likelihood, regulated by the done flags
    if full_model.multi_instanced:
        l1_error_element =np.abs(pytorch_model.unwrap(params[0].reshape(batch_size, -1, full_model.obj_dim) - targets.reshape(batch_size, -1, full_model.obj_dim))) 
        l1_error = np.max(l1_error_element, axis=1)
        l1_error_element =np.sum(l1_error_element, axis=-1)
    else: 
        l1_error = np.abs(pytorch_model.unwrap(params[0] - targets))
        l1_error_element = np.abs(pytorch_model.unwrap(params[0] - targets))
    return l1_error, l1_error_element

def decide_multioption():
    '''
    Multioption: first_obj_dim is always going to be any of the single objects in additional, parent, target
    TODO: make he additional elements contained in the passive model?

    '''
    self.first_obj_dim = np.sum([self.parent_selectors[p].output_size() for p in self.names.parents]) # the first object dim is the combined length of the parents
    self.obj_dim = self.target_select.output_size() # the selector gets the size of a single instance
    self.additional_dim = environment.object_sizes[self.names.additional[0]] if len(self.names.additional) > 0 else 0# all additional objects must have the same dimension
