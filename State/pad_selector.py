import numpy as np
import os, cv2, time, copy, itertools
import torch
from collections import OrderedDict
from tianshou.data import Batch
from Network.network_utils import pytorch_model
from State.full_selector import flatten

def add_pad(self, states, name, pad_size):
    pad = pad_size - states[name].shape[-1]
    add_state = np.concatenate((states[name], np.zeros(states.shape[:-1] + (pad, ))), axis=-1) if pad > 0 else states[name]


class PadSelector():
    def __init__(self, sizes, instanced, names):
        self.instanced = instanced
        self.names = names
        self.sizes = sizes 
        self.pad_size = np.max(list(sizes.values()))

    def __call__(self, states):
        '''
        states are dict[name] -> ndarray: [batchlen, object state + zero padding]
        returns [batchlen, flattened state]
        '''
        flattened = list()
        for name in self.names:
            if self.instanced[name]:
                for i in range(self.instanced[name]):
                    flattened.append(add_pad(states, name + str(i), self.pad_size))
            else:
                flattened.append(add_pad(states, name, self.pad_size))
        return np.concatenate(flattened, axis=-1)

    def get_entity(self):
        return self.names

    def output_size(self):
        return sum([self.pad_size * self.instanced[n] for n in self.names])

    def reverse(self, flat_state):
        '''
        unflattens a flat state [batch, output_size]
        pretty much the same logic as the full selector, but goes by pad size
        '''
        factored = dict()
        at = 0
        for name in self.names:
            if self.instanced[name]:
                for i in range(self.instanced[name]):
                    factored[name + str(i)] = flat_state[...,at:at + self.sizes[name]]
                    at = at + self.pad_size # skip any padding
            else:
                factored_state[name] = flat_state[...,at: at + self.sizes[name]]
                at = at + self.pad_size
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
                    at += self.pad_size
            else:
                if name in name_check:
                    idxes += list(range(at, at + self.sizes[name]))
                at += self.pad_size              
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
                    at += pad_size
        else: # assume that insert state is a dict
            if type(state) == np.ndarray:
                idxes = self.get_idxes(names)
                state[...,idxes] = flatten(insert_state, names)
            else:
                for name in insert_state.keys():
                    state[name] = insert_state[name]
        return state