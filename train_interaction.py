import os, torch
from arguments import get_args
from Record.file_management import read_obj_dumps, load_from_pickle, save_to_pickle
from State.object_dict import ObjDict
from State.feature_selector import construct_object_selector
from Buffer.fill_buffer import fill_buffer
from Buffer.buffer import InterWeightedReplayBuffer
from Causal.interaction_model import NeuralInteractionForwardModel
from Causal.Training.train_full import train_full

from Environment.Environments.initialize_environment import initialize_environment

from Network.network_utils import pytorch_model
import numpy as np
import sys
import psutil

def set_batch(buffer, batch):
    # sets a batch and also sets internal variables of the buffer (TODO: breaks TS abstraction boundary, but that's because of TS improper design)
    buffer.set_batch(batch)
    buffer._index = 0
    buffer._size = len(batch)


def generate_buffers(environment, args, object_names, full_model, train=True):
    # load data
    data = read_obj_dumps(args.train.load_rollouts, i=-1, rng = args.train.num_frames, filename='object_dumps.txt')

    # get the buffers
    buffer = fill_buffer(environment, data, args, object_names, full_model.norm)

    if not train: return buffer

    # fill the train/test buffer
    train_indices = np.random.choice(list(range(len(buffer))), size=int(len(buffer) * args.train.train_test_ratio), replace=False)
    test_indices = [i for i in range(len((buffer))) if i not in train_indices]
    train_buffer = InterWeightedReplayBuffer(len(train_indices), stack_num=1)
    set_batch(train_buffer, buffer[train_indices])
    test_buffer = InterWeightedReplayBuffer(len(test_indices), stack_num=1)
    set_batch(test_buffer, buffer[test_indices])
    if args.inter.save_intermediate: save_to_pickle("/hdd/datasets/object_data/temp/full_rollouts.pkl", buffer)
    del buffer
    return train_buffer, test_buffer


if __name__ == '__main__':
    args = get_args()
    print(args) # print out args for records
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)

    environment, record = initialize_environment(args.environment, args.record)

    # create a dictionary for the names
    object_names = ObjDict()
    object_names.target = args.train.train_edge[-1]
    object_names.primary_parent = args.train.train_edge[0]
    object_names.parents = args.train.train_edge[:-1]
    args.object_names = object_names

    # build the selectors for the passive (target), interaction or active (parent + target), parent (just parent) states
    args.target_select = construct_object_selector([object_names.target], environment)
    args.parent_selectors = {p: construct_object_selector([p], environment) for p in object_names.parents}
    args.parent_select = construct_object_selector(object_names.parents, environment)
    args.inter_select = construct_object_selector(object_names.parents + [object_names.target], environment)
    args.controllable = None # this is filled in with controllable features of the target

    # initialize the full model
    full_model = NeuralInteractionForwardModel(args, environment)
    
    # get the train and test buffers
    if args.inter.load_intermediate: train_buffer, test_buffer = load_from_pickle("/hdd/datasets/object_data/temp/rollouts.pkl")
    else: train_buffer, test_buffer = generate_buffers(environment, args, object_names, full_model)
    if args.inter.save_intermediate: save_to_pickle("/hdd/datasets/object_data/temp/rollouts.pkl", (train_buffer, test_buffer))

    if args.train.train: train_full(full_model, train_buffer, test_buffer, args, object_names, environment)
    elif len(args.record.load_dir) > 0: full_model = torch.load(os.path.join(args.record.load_dir, full_model.name + "_inter_model.pt"))
    test_full(full_model, train_buffer, test_buffer, args, object_names, environment)