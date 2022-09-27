import numpy as np
import os, cv2, time, copy, itertools
import torch
from collections import OrderedDict
from tianshou.data import Batch
from Network.network_utils import pytorch_model

def flatten(factored_state, names):
    flat = list()
    for name in names:
        flat.append(factored_state[name])
    return np.concatenate(flat, axis=-1)

class FullSelector():
    def __init__(self, sizes, instanced, names):
        self.instanced = instanced
        self.names = names
        self.sizes = sizes 

    def __call__(self, states):
        '''
        states are dict[name] -> ndarray: [batchlen, object state]
        returns [batchlen, flattened state]
        '''
        flattened = list()
        for name in self.names:
            if self.instanced[name]:
                for i in range(self.instanced[name]):
                    flattened.append(states[name + str(i)])
            else:
                flattened.append(states[name])
        return np.concatenate(flattened, axis=-1)

    def get_entity(self):
        return self.names

    def output_size(self):
        return sum([self.sizes[n] * self.instanced[n] for n in self.names])

    def reverse(self, flat_state):
        '''
        unflattens a flat state [batch, output_size]
        '''
        factored = dict()
        at = 0
        for name in self.names:
            if self.instanced[name]:
                for i in range(self.instanced[name]):
                    factored[name + str(i)] = flat_state[...,at:at + self.sizes[name]]
                    at = at + self.sizes[name]
            else:
                factored_state[name] = flat_state[...,at: at + self.sizes[name]]
                at = at + self.sizes[name]
        return factored

    def get_idxes(self, names):
        at = 0
        idxes = list()
        name_check = set(names)
        for name in self.names:
            if self.instanced[name]:
                for i in range(self.instanced[name]):
                    full_name = name + str(i)
                    if full_name in name_check:
                        idxes += list(range(at, at + self.sizes[name]))
                    at += self.sizes[name]
            else:
                if name in name_check:
                    idxes += list(range(at, at + self.sizes[name]))
                at += self.sizes[name]                
        return np.array(idxes)

    def assign(self, state, insert_state, names):
        if type(insert_state) == np.ndarray:
            if type(state) == np.ndarray:
                idxes = self.get_idxes(names)
                state[...,idxes] = insert_state
            else:
                at = 0
                for name in names:
                    size = self.sizes[name.strip("0123456789")]
                    state[name] = insert_state[at:at + size]
                    at += size
        else: # assume that insert state is a dict
            if type(state) == np.ndarray:
                idxes = self.get_idxes(names)
                state[...,idxes] = flatten(insert_state, names)
            else:
                for name in insert_state.keys():
                    state[name] = insert_state[name]
        return state