# small buffer test
# arguments list: file path
from ReinforcementLearning.utils.RL_logger import RLLogger
from ReinforcementLearning.policy import Policy
from Record.file_management import load_from_pickle

def small_test(args):
    graph = load_graph(args.record.load_dir, args.torch.gpu)
    object_names = init_names(args)
    print(graph.nodes[object_names.target].option, object_names.target)
    target_option = graph.nodes[object_names.target].option
    discrete_actions = target_option.action_map.discrete_actions
    total_size = target_option.state_extractor.total_size
    policy_action_space = target_option.action_map.policy_action_space
    file_path = os.path.join(args.record.checkpoint_dir, "pretrain_buffers.bf")
    rollouts = load_from_pickle(file_path)
    train_logger = RLLogger("train", "", log_interval, 1000, "")

    policy = Policy(discrete_actions, total_size, policy_action_space, args)
    for i in range(num_iters):
        losses = policy.update(args.train.batch_size, train_collector.buffer, train_collector.her_buffer)
        train_logger.log_losses(losses)
        train_logger.print_losses(i)
