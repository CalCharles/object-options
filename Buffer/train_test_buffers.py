import numpy as np
import os
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from Buffer.fill_buffer import fill_buffer
from Buffer.buffer import InterWeightedReplayBuffer

def set_batch(buffer, batch):
    # sets a batch and also sets internal variables of the buffer (TODO: breaks TS abstraction boundary, but that's because of TS improper design)
    buffer.set_batch(batch)
    buffer._index = 0
    buffer._size = len(batch)

def train_test_indices(args, buffer):
    if args.train.train_test_order == "random":
        train_indices = np.random.choice(list(range(len(buffer))), size=int(len(buffer) * args.train.train_test_ratio), replace=False)
        test_indices = [i for i in range(len((buffer))) if i not in train_indices]
    elif args.train.train_test_order == "time":
        train_indices, test_indices = list(range(len(buffer)))[:int(len(buffer) * args.train.train_test_ratio)], list(range(len(buffer)))[int(len(buffer) * args.train.train_test_ratio):]
    else:
        raise ValueError("invalid ordering setting")
    return train_indices, test_indices

def generate_buffers(environment, args, object_names, full_model, train=True):
    # load data
    data = read_obj_dumps(args.train.load_rollouts, i=-1, rng = args.train.num_frames, filename='object_dumps.txt')

    # get the buffers
    buffer = fill_buffer(environment, data, args, object_names, full_model.norm, full_model.predict_dynamics)
    if not train: return buffer

    # get indices for train/test, there are various settings for this
    train_indices, test_indices = train_test_indices(args, buffer)

    # fill the train/test buffer
    train_buffer = InterWeightedReplayBuffer(len(train_indices), stack_num=1)
    set_batch(train_buffer, buffer[train_indices])
    test_buffer = InterWeightedReplayBuffer(len(test_indices), stack_num=1)
    set_batch(test_buffer, buffer[test_indices])
    if args.inter.save_intermediate: save_to_pickle(os.path.join(args.inter.save_intermediate, environment.name + "_" + full_model.name + "_full_rollouts.pkl"), buffer)
    del buffer
    return train_buffer, test_buffer
