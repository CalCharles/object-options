import numpy as np
import os, cv2, time, copy, itertools
import torch
from collections import OrderedDict
from tianshou.data import Batch
from Network.network_utils import pytorch_model
from State.pad_selector import PadSelector

def tensor_state(factored_state, cuda=False):
    fs = copy.deepcopy(factored_state)
    for k in factored_state.keys():
        fs[k] = pytorch_model.wrap(fs[k], cuda=cuda)
    return fs

def broadcast(arr, size, cat=True, axis=0): # always broadcasts on axis 0
    if cat: return np.concatenate([arr.copy() for i in range(size)], axis= axis)
    return np.stack([arr.copy() for i in range(size)], axis=axis)

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

def construct_object_selector(names, environment, masks=None, pad=False, append_id=False):
    '''
    constructs a selector to select the elements of all the objects in names\
    masks will select particular features from that object, one mask for each object
    This should be the only entry point for constructing feature selectors
    '''
    factored = dict()
    if masks is None: masks = [np.ones(environment.object_sizes[name]) for name in names] # if no masks select all features
    for mask, name in zip(masks, names):
        sze = environment.object_sizes[name]
        factored[name] = np.arange(sze)[np.array(mask).astype(bool)]
    if pad: return PadSelector(environment.object_sizes, environment.object_instanced, names, factored, append_id)
    return FeatureSelector(factored, names, multiinstanced={name: environment.object_instanced[name] != 1 for name in names})

def construct_object_selector_dict(names, object_sizes, object_instanced, masks=None):
    '''
    constructs a selector to select the elements of all the objects in names\
    masks will select particular features from that object, one mask for each object
    This should be the only entry point for constructing feature selectors
    '''
    factored = dict()
    if masks is None: masks = [np.ones(object_sizes[name]) for name in names] # if no masks select all features
    for mask, name in zip(masks, names):
        sze = object_sizes[name]
        factored[name] = np.arange(sze)[np.array(mask).astype(bool)]
    return FeatureSelector(factored, names, multiinstanced={name: object_instanced[name] != 1 for name in names})


def numpy_factored(factored_state):
    for n in factored_state.keys():
        factored_state[n] = np.array(factored_state[n])
    return factored_state

class FeatureSelector():
    def __init__(self, factored_features, names, multiinstanced=None):
        '''
        factored_features: a dict of the entity name and the feature indices as a numpy array, with only one index per tuple
        names: the order of names for hashes
        '''
        self.factored_features = factored_features
        self.names=names
        self.multiinstanced = multiinstanced

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
        if len(self.names) == 0: return np.zeros(states["Action"].shape) # return a dummy value
        # print(self.factored_features, self.names, [states[name].shape for name in self.names])
        if self.multiinstanced and np.any([mi for mi in self.multiinstanced.values()]):
            cut_state = list()
            for name in self.names:
                if self.multiinstanced[name]: # iterates through all of the instances
                    i=0
                    itr_name = name + str(i)
                    while itr_name in states:
                        cut_state.append(states[itr_name][...,self.factored_features[name]])
                        i += 1
                        itr_name = name + str(i)
                else: cut_state += [states[name][...,self.factored_features[name]]]
        else: cut_state = [states[name][...,self.factored_features[name]] for name in self.names]
        state = np.concatenate(cut_state, axis=-1)
        return state

    def reverse(self, delta_state, insert_state, names=None, mask=None):
        '''
        assigns the relavant values of insert_state to delta_state
        if names is not None, then only assigns the components of names in names, in the order of names
        '''
        drng, frng = 0, 0
        if names is None: names = self.names
        if type(delta_state) == list: delta_state = np.array(delta_state) # needs advanced slicing
        for name in names:
            # only use the component of the mask corresponding to this name
            num_features_name = len(self.factored_features[name])
            mask_comp = np.ones(num_features_name).astype(bool) if mask is None else mask[frng:frng + num_features_name].astype(bool)
            # get the masked components
            idxes = self.factored_features[name][mask_comp]
            idxes += frng
            # assign
            delta_state[...,idxes] = insert_state[...,drng:drng+len(idxes)]
            # increment index indicators in insert state and delta state
            drng += len(idxes)
            frng += len(self.factored_features[name])
        return delta_state

def assign_feature(self, states, assignment, edit=False, clipped=None):
    # assigns the values of states to assignment
    # assignment is a tuple assignment keys (tuples of (name, indexes)), and assignment values
    # edit means that the assignment is added
    # clipped is a tuple of the clipping range
    if type(states) is dict or type(states) == OrderedDict or type(states) == Batch: # factored features, assumes that shapes for assignment[0][1] and assignment[1] match
        states[assignment[0][0]][...,assignment[0][1]] = assignment[1] if not edit else states[assignment[0][0]][...,assignment[0][1]] + assignment[1]
        if clipped is not None: states[assignment[0][0]][...,assignment[0][1]] = states[assignment[0][0]][...,assignment[0][1]].clip(clipped[0], clipped[1])