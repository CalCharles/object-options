import pickle, os
import numpy as np
import imageio as imio
import sys, cv2 

def load_from_pickle(pth):
    fid = open(pth, 'rb')
    save_dict = pickle.load(fid)
    fid.close()
    return save_dict

def save_to_pickle(pth, val):
    try:
        splt_path = pth.split("/")[:-1]
        target = os.path.join(*splt_path)
        if splt_path[0] == "":
            target = "/" + target
        
        os.makedirs(target)
    except OSError:
        pass
    fid = open(pth, 'wb')
    pickle.dump(val, fid)
    fid.close()

class ObjDict(dict):
    def __init__(self, ins_dict=None):
        super().__init__()
        if ins_dict is not None:
            for n in ins_dict.keys(): 
                self[n] = ins_dict[n]

    def insert_dict(self, ins_dict):
        for n in ins_dict.keys(): 
            self[n] = ins_dict[n]

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

def _dump_from_line(line, time_dict):
    '''
    factored state saved as strings, where each line is at one timestep
    reserved strings: ":" separates name from state
    " " separated sequence of floats encodes state
    '''
    for obj in line.split('\t'):
        if obj == "\n":
            continue
        else:
            split = obj.split(":")
            name = split[0]
            vals = split[1].split(" ")
            state = [float(i) for i in vals]
            time_dict[name] = np.array(state)
    return time_dict

def state_to_dump(full_state, name_order):
    '''
    a factored state: dictionary of factored_state, full_state, with factored state as str->nparray
    name order: the order to save the state 
    '''
    dump = ""
    for name in name_order:
        dump += name
        dump += ":" + " ".join([str(v) for v in full_state['factored_state'][name]])
    return dump

def read_obj_dumps(pth, i= 0, rng=-1, filename='object_dumps.txt'):
    '''
    pth: a string to the directory where the object dumps are
    i = -1 means count rng from the back
    rng = -1 means take all after i
    filename: the name of the file containing the strings
    i is start position, rng is number of values
    '''
    obj_dumps = []
    i, total_len = get_start(pth, filename, i, rng)
    current_len = 0
    for line in open(os.path.join(pth, filename), 'r'):
        current_len += 1
        if current_len < i:
            continue
        if rng != -1 and current_len > i + rng:
            break
        time_dict = _dump_from_line(line, dict())
        obj_dumps.append(time_dict)
    return obj_dumps

def get_start(pth, filename, i, rng, tab_count=False):
    total_len = 0
    if i <= 0:
        if tab_count:
            for line in open(os.path.join(pth, filename), 'r'):
                for action_str in line.split("\t"):
                    total_len += len(action_str) > 0
        else:
            for line in open(os.path.join(pth, filename), 'r'):
                total_len += 1
        if rng == -1:
            i = 0
        else:
            i = max(total_len - rng, 0)
    return i, total_len

def numpy_factored(factored_state):
    for n in factored_state.keys():
        factored_state[n] = np.array(factored_state[n])
    return factored_state

def action_toString(action):
    if type(action) == list: 
        action = np.array(action)
    if type(action) == np.ndarray:
        action = action.squeeze()
    else:
        return str(action)
    if len(action.shape) == 0:
        return str(action)
    return ",".join(map(str, action))

def read_action_dumps(pth, i=0, rng=-1, filename='action_dumps.txt', indexed=False):
    action_dumps = list()
    i, total_len = get_start(pth, filename, i, rng, tab_count = True)
    current_len = 0
    idxes = list()
    additional = list()
    for line in open(os.path.join(pth, filename), 'r'): # there should only be one line since actions are tab separated
        for action_str in line.split("\t"):
            current_len += 1
            if current_len < i:
                continue
            if rng != -1 and current_len > i + rng:
                break
            if indexed and len(action_str) > 0:
                action_str = action_str.split(":")
                idx_str, action_str = action_str[0], action_str[1] 
                idx = int(idx_str)
                idxes.append(idx)
            extra_splt = action_str.split('|')
            if len(extra_splt) > 1:
                additional.append(list(map(int, extra_splt[1].split(','))))
            splt = extra_splt[0]
            splt = splt.split(',')
            if len(splt) > 1:
                action_dumps.append([float(s.strip("\t\n")) for s in splt])
            elif len(action_str) > 0:
                action_dumps.append(float(splt[0].strip("\t\n")))
    return action_dumps, idxes, additional

def get_raw_data(pth, i=0, rng=-1):
    '''
    loads raw frames, i denotes starting position, rng denotes range of values. 
    '''
    frames = []
    if rng == -1:
        try:
            f = i
            while True:
                frames.append(imio.load(os.path.join(pth, "state" + str(f) + ".png")))
                f += 1
        except OSError as e:
            return frames
    else:
        for f in range(i, i + rng[1]):
            frames.append(imio.load(os.path.join(pth, "state" + str(f) + ".png")))
    return frames