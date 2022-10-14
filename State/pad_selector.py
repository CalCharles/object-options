import numpy as np
import os, cv2, time, copy, itertools
import torch
from collections import OrderedDict
from tianshou.data import Batch
from Network.network_utils import pytorch_model
from State.full_selector import flatten

def add_pad(states, name, pad_size, append_id, num_objects):
    pad = pad_size - states[name].shape[-1]
    if append_id >= 0:
        id_append_hot = np.zeros(states[name].shape[:-1] + (num_objects, ))
        id_append_hot[append_id] = 1
        # print(append_id, id_append_hot.shape, pad, np.zeros(states[name].shape[:-1] + (pad, )).shape)
        add_state = np.concatenate((states[name], np.zeros(states[name].shape[:-1] + (pad, ))), axis=-1) if pad > 0 else states[name]
        add_state = np.concatenate((add_state, id_append_hot), axis=-1)
    else: add_state = np.concatenate((states[name], np.zeros(states[name].shape[:-1] + (pad, ))), axis=-1) if pad > 0 else states[name]
    return add_state

class PadSelector():
    def __init__(self, sizes, instanced, names, factored, append_id=False):
        # if factored does not contain the full state, this extractor is just for extracting from padded states 
        self.instanced = instanced
        self.names = names
        self.name_id = {n: self.names.index(n) for n in self.names}
        self.factored = factored
        self.sizes = sizes 
        self.pad_size = np.max(list(sizes.values()))
        self.num_objects = len(list(sizes.values()))
        self.append_id = append_id
        self.append_pad_size = self.pad_size + int(self.append_id) * self.num_objects

    def __call__(self, states):
        '''
        states are dict[name] -> ndarray: [batchlen, object state + zero padding]
        returns [batchlen, flattened state], where the flattened state selects objects in names
        it does not select only the masked values 
        '''
        flattened = list()
        for name in self.names:
            id_append = self.name_id[name] if self.append_id else -1
            if self.instanced[name] > 1:
                for i in range(self.instanced[name]):
                    flattened.append(add_pad(states, name + str(i), self.pad_size, id_append, self.num_objects))
            else:
                flattened.append(add_pad(states, name, self.pad_size, id_append, self.num_objects))
        return np.concatenate(flattened, axis=-1)

    def get_entity(self):
        return self.names

    def output_size(self):
        return sum([self.pad_size * self.instanced[n] for n in self.names])

    def reverse(self, flat_state, prev_factored=None):
        '''
        unflattens a flat state [batch, output_size]
        sets the values of prev_factored if possible, otherwise assumes that all the features are being selected
        pretty much the same logic as the full selector, but goes by pad size
        '''
        factored = dict() if prev_factored is None else prev_factored
        at = 0
        for name in self.names:
            if self.instanced[name]:
                for i in range(self.instanced[name]):
                    if name + str(i) in factored: factored[name + str(i)][...,self.factored] = flat_state[...,at +self.factored]
                    else: factored[name + str(i)] = flat_state[...,at +self.factored]
                    at = at + self.pad_size # skip any padding
            else:
                if name in factored: factored[name][...,self.factored] = flat_state[...,at + self.factored]
                else: factored[name] = flat_state[...,at + self.factored]
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
                        idxes += (at + self.factored[name]).tolist()
                    at += self.pad_size
            else:
                if name in name_check:
                    idxes += (at + self.factored[name]).tolist()
                at += self.pad_size
        return np.array(idxes)

    def assign(self, state, insert_state, names = None):
        # assigns only the factored indices, names should overlap with self.names
        if names is None: names = self.names 
        if type(insert_state) == np.ndarray:
            if type(state) == np.ndarray:
                idxes = self.get_idxes(names)
                state[...,idxes] = insert_state
            else:
                at = 0
                for name in names:
                    o_name = name.strip("0123456789")
                    size = len(self.factored[o_name])#self.sizes[o_name]
                    state[name][self.factored[o_name]] = insert_state[at:at + size]
                    at += self.append_pad_size
        else: # assume that insert state is a dict
            if type(state) == np.ndarray:
                idxes = self.get_idxes(names)
                state[...,idxes] = flatten(insert_state, names)
            else:
                for name in insert_state.keys():
                    state[name] = insert_state[name]
        return state