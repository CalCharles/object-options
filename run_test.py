from Graph.graph import Graph, load_graph
from arguments import get_args
import torch
import os
import numpy as np
from train_interaction import init_names
from Testing.small_test import small_test


NUM_ITERS = 100
LOG_INTERVAL = 10
BATCH_SIZE = 64

if __name__ == '__main__':
    args = get_args()
    print(args)
    torch.manual_seed(args.torch.torch_seed)
    torch.cuda.set_device(args.torch.gpu)
    np.set_printoptions(threshold=3000, linewidth=120, precision=4, suppress=True)
    torch.set_printoptions(precision=4, sci_mode=False)
    graph = load_graph(args.record.load_dir, args.torch.gpu)
    object_names = init_names(args)
    print(graph.nodes[object_names.target].option, object_names.target)
    target_option = graph.nodes[object_names.target].option
    discrete_actions = target_option.action_map.discrete_actions
    total_size = target_option.state_extractor.total_size
    policy_action_space = target_option.action_map.policy_action_space
    file_path = os.path.join(args.record.checkpoint_dir, "pretrain_buffers.bf")

    small_test(args, file_path, discrete_actions, total_size, policy_action_space, 100, 10, 32)
