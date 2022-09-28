import pickle, os
import numpy as np
import imageio as imio
import sys, cv2, copy

def load_from_pickle(pth):
    fid = open(pth, 'rb')
    save_dict = pickle.load(fid)
    fid.close()
    return save_dict

def save_to_pickle(pth, val):
    try:
        splt_path = os.path.split(pth)[0]#.split("/")[:-1]
        target = os.path.join(*splt_path)
        if splt_path[0] == "":
            target = "/" + target
        
        os.makedirs(target)
    except OSError:
        pass
    fid = open(pth, 'wb')
    pickle.dump(val, fid)
    fid.close()

def create_directory(pth, drop_last = False):
    new_pth = pth
    if drop_last:
        new_pth = os.path.split(pth)[0]#os.path.join(*os.path.split(pth)[:-1])
        if len(new_pth) == 0:
            return pth
    try:
        os.makedirs(new_pth)
    except OSError as e:
        pass
    return pth

def append_string(pth, strv):
    filev = open(pth, 'a')
    filev.write(strv)
    filev.close()

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

def write_string(file_str, wstring, form="a"):
    option_dumps = open(file_str, form)
    option_dumps.write(wstring)
    option_dumps.close()


def action_chain_string(action):
    # expects a list of lists or individual values, returns tab separated actions, and comma separated values
    action_str = ""
    for a in action:
        if type(a) == list: 
            a = np.array(action)
        if type(a) == np.ndarray:
            a = a.squeeze()
            if len(a.shape) == 0:
                action_str += str(a) + '\t' # a single value string
            else: 
                action_str += ",".join(map(str, a)) + '\t'
        else:
            action_str += str(a) + '\t'
    return action_str[:-1]

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

def display_frame(frame, waitkey=10, rescale=-1):
    if rescale > 0: frame = cv2.resize(frame, (frame.shape[0] * rescale, frame.shape[1] * rescale), interpolation = cv2.INTER_NEAREST)
    cv2.imshow('image',frame)
    cv2.waitKey(waitkey) # waits until a key is pressed

def display_param(frame, param, waitkey=10, rescale=-1, dot=True, transpose = True):
    param = copy.deepcopy(param)
    if param is not None:
        loc = param.squeeze()[:2]
        if transpose: loc[0], loc[1] = loc[1], loc[0]
        angle = None
        if len(param.squeeze()) >= 4:
            if transpose: param[...,2], param[...,3] =  param[...,3], param[...,2]
            angle = param.squeeze()[2:4]
            angle[1] = - angle[1]
        color = (0,128,0)
        if len(param.squeeze()) == 3 or len(param.squeeze()) == 5:
            if param.squeeze()[len(param.squeeze()) - 1:] < 0.5:
                color = (0,0,128)
        frame = np.stack([frame.copy() for i in range(3)], axis = -1)
        print(loc, angle)
        if angle is not None:
            cv2.line(frame, loc.astype(int), (loc + 2 * angle).astype(int), color,2)
        else:
            if dot:
                frame[np.round(loc).astype(int)[0], np.round(loc).astype(int)[1]] += np.array(color).astype(np.int8)
            else:
                loc[0], loc[1] = loc[1], loc[0]
                cv2.circle(frame, loc.astype(int), 3, color, 1)

    if rescale > 0: frame = cv2.resize(frame, (frame.shape[0] * rescale, frame.shape[1] * rescale), interpolation = cv2.INTER_NEAREST)
    cv2.imshow('param_image',frame)
    cv2.waitKey(int(waitkey)) # waits until a key is pressed
    return frame
