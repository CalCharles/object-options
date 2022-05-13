import numpy as np
import os, cv2, time, copy, itertools
import torch
from collections import OrderedDict
from tianshou.data import Batch
from Network.network_utils import pytorch_model

def tensor_state(factored_state, cuda=False):
    fs = copy.deepcopy(factored_state)
    for k in factored_state.keys():
        fs[k] = pytorch_model.wrap(fs[k], cuda=cuda)
    return fs

def broadcast(arr, size, cat=True):
    if cat: return np.concatenate([arr.copy() for i in range(size)])
    return np.stack([arr.copy() for i in range(size)], axis=-1)

def sample_feature(feature_range, step, idx, states):
    all_states = [] # states to return
    num = int((feature_range[1] - feature_range[0] ) / step)

    # steps through the range at step
    f = feature_range[0]
    while f <= feature_range[1]:
        assigned_states = states.copy()
        assigned_states[..., idx] = f
        all_states.append(assigned_states)
        f += step

    # TODO: remove this if statement
    if len(states.shape) == 1: # a single flattened state
        return np.concatenate(all_states, axis=0) # if there are no batches, then this is the 0th dim
    return np.concatenate(all_states, axis=1) # if we have a batch of states, then this is the 1st dim

def construct_object_selector(names, environment):
    '''
    constructs a selector to select the elements of all the objects in names
    '''
    factored = dict()
    for name in names:
        sze = environment.object_sizes[name]
        factored[name] = np.arange(sze)
    return FeatureSelector(factored, names)

def numpy_factored(factored_state):
    for n in factored_state.keys():
        factored_state[n] = np.array(factored_state[n])
    return factored_state

class FeatureSelector():
    def __init__(self, factored_features, names):
        '''
        factored_features: a dict of the entity name and the feature indices as a numpy array, with only one index per tuple
        names: the order of names for hashes
        '''
        self.factored_features = factored_features
        self.names=names

    def __hash__(self):
        # hash is tuple of string names followed by corresponding features
        total = list()
        for n in self.names:
            total.append(str(hash(n)))
            total.append(self.factored_features[n])
        return tuple(total)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_entity(self):
        return self.names

    def output_size(self):
        return sum([len(self.factored_features[n]) for n in self.names])

    def __call__(self, states):
        '''
        states are dict[name] -> nparray
        target shape can be [batchlen, factored feature shape], or [factored feature shape] 
        '''
        cut_state = [states[name][...,self.factored_features[name]] for name in self.names]
        state = np.concatenate(cut_state, axis=0)
        return state

    def reverse(self, delta_state, insert_state, names=None):
        '''
        assigns the relavant values of insert_state to delta_state
        if names is not None, then only assigns the components of names in names, in the order of names
        '''
        drng = 0
        if names is None: names = self.names
        for name in names:
            idxes = self.factored_features[name]
            delta_state[name][idxes] = insert_state[drng:drng+len(idxes)]
            drng += len(idxes)
        return delta_state

def assign_feature(self, states, assignment, edit=False, clipped=None):
    # assigns the values of states to assignment
    # assignment is a tuple assignment keys (tuples of (name, indexes)), and assignment values
    # edit means that the assignment is added
    # clipped is a tuple of the clipping range
    if type(states) is dict or type(states) == OrderedDict or type(states) == Batch: # factored features, assumes that shapes for assignment[0][1] and assignment[1] match
        states[assignment[0][0]][...,assignment[0][1]] = assignment[1] if not edit else states[assignment[0][0]][...,assignment[0][1]] + assignment[1]
        if clipped is not None: states[assignment[0][0]][...,assignment[0][1]] = states[assignment[0][0]][...,assignment[0][1]].clip(clipped[0], clipped[1])