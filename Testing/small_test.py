# small buffer test
# arguments list: file path
from ReinforcementLearning.utils.RL_logger import RLLogger
from ReinforcementLearning.policy import Policy
from Record.file_management import load_from_pickle

def small_test(args, file_path, discrete_actions, total_size, policy_action_space, num_iters, log_interval, batch_size):
    rollouts = load_from_pickle(file_path)
    train_logger = RLLogger("train", "", log_interval, 1000, "")

    policy = Policy(discrete_actions, total_size, policy_action_space, args)
    for i in range(num_iters):
        losses = policy.update(args.train.batch_size, train_collector.buffer, train_collector.her_buffer)
        train_logger.log_losses(losses)
        train_logger.print_losses(i)
