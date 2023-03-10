import numpy as np
import os
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle, create_directory
from Buffer.fill_buffer import fill_buffer
from Buffer.buffer import InterWeightedReplayBuffer
from Buffer.FullBuffer.full_buffer import FullReplayBuffer, ObjectReplayBuffer
from Buffer.FullBuffer.fill_full_buffer import fill_full_buffer
from Buffer.FullBuffer.fill_all_buffer import fill_all_buffer

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

def generate_buffers(environment, args, object_names, full_model, train=True, full=0):
    # load data
    data = read_obj_dumps(args.train.load_rollouts, i=-1, rng = args.train.num_frames, filename='object_dumps.txt')
    # get the buffers
    if full == 2:
        buffer = fill_all_buffer(full_model, environment, data, args, object_names, full_model.norm, full_model.predict_dynamics)
    elif full == 1:
        buffer, object_buffers = fill_full_buffer(full_model, environment, data, args, object_names, full_model.norm, full_model.predict_dynamics)
    else: buffer = fill_buffer(environment, data, args, object_names, full_model.norm, full_model.predict_dynamics)
    if not train: return buffer

    # get indices for train/test, there are various settings for this
    train_indices, test_indices = train_test_indices(args, buffer)

    # fill the train/test buffer
    if full == 2:
        if args.inter.save_intermediate: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_mask_rollouts.pkl"), buffer)
        # fill buffers for the train set
        alpha, beta = (1e-10, 0) if len(args.collect.prioritized_replay) == 0 else args.collect.prioritized_replay
        train_buffer = FullReplayBuffer(len(train_indices), stack_num=1)
        set_batch(train_buffer, buffer[train_indices])
        # fill buffers for the test set
        test_buffer = FullReplayBuffer(len(test_indices), stack_num=1)
        set_batch(test_buffer, buffer[test_indices])
        del buffer
        return train_buffer, test_buffer
    elif full == 1:
        if args.inter.save_intermediate: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_mask_rollouts.pkl"), buffer)
        # fill buffers for the train set
        alpha, beta = (1e-10, 0) if len(args.collect.prioritized_replay) == 0 else args.collect.prioritized_replay
        train_buffer = FullReplayBuffer(len(train_indices), stack_num=1)
        set_batch(train_buffer, buffer[train_indices])
        train_object_buffers = dict()
        for n in environment.object_names:
            train_object_buffers[n] = ObjectReplayBuffer(len(train_indices), stack_num=1, alpha=alpha, beta=beta)
            set_batch(train_object_buffers[n], object_buffers[n][train_indices])
        # fill buffers for the test set
        test_buffer = FullReplayBuffer(len(test_indices), stack_num=1)
        set_batch(test_buffer, buffer[test_indices])
        test_object_buffers = dict()
        for n in environment.object_names:
            test_object_buffers[n] = ObjectReplayBuffer(len(test_indices), stack_num=1, alpha=alpha, beta=beta)
            set_batch(test_object_buffers[n], object_buffers[n][test_indices])
        del buffer
        return train_buffer, train_object_buffers, test_buffer, test_object_buffers
    else:
        if args.inter.save_intermediate: save_to_pickle(os.path.join(create_directory(args.inter.save_intermediate), environment.name + "_" + full_model.name + "_full_rollouts.pkl"), buffer)
        train_buffer = InterWeightedReplayBuffer(len(train_indices), stack_num=1)
        print(len(buffer), buffer._size, len(buffer.obs), len(buffer[train_indices]), len(train_buffer), train_buffer.maxsize == len(train_indices))
        set_batch(train_buffer, buffer[train_indices])
        test_buffer = InterWeightedReplayBuffer(len(test_indices), stack_num=1)
        set_batch(test_buffer, buffer[test_indices])
        del buffer
        return train_buffer, test_buffer
