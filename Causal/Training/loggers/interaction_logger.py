import logging, os
from Network.network_utils import pytorch_model
from Record.logging import Logger
from Record.file_management import create_directory
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class interaction_logger(Logger):
    def __init__(self, name, record_graphs, log_interval, full_model, filename=""):
        super().__init__(filename)
        self.name = name
        self.record_graphs = record_graphs
        if len(record_graphs) != 0:
            full_logdir = os.path.join(create_directory(record_graphs+ "/logs"))
            self.tensorboard_logger = SummaryWriter(log_dir=full_logdir)
        self.log_interval = log_interval
        self.interaction_prediction = full_model.test.interaction_prediction
        self.reset()

    def log_gradient(self, gradient_dict):
        # losses may differ for values
        for k in test_dict.keys():
            if len(test_dict[k].squeeze().shape) == 0:
                if k not in self.testing_log:
                    self.testing_log[k] = deque(maxlen=self.maxlen)
                self.testing_log[k].append(test_dict[k])
        print(self.testing_log)


    def reset(self):
        self.loss = list()
        self.total_seen = 0
        self.binary_false_positive = 0
        self.binary_false_negative = 0
        self.binary_true_false_positive = 0
        self.binary_true_false_negative = 0
        self.trace_false_positive = 0
        self.trace_false_negative = 0
        self.weight_count = 0
        self.total_inter = 0
        self.total_true = 0
        self.total_bin = 0

    def log(self, i, loss, interactions, binaries, done_flags, weights, trace=None, no_print=False):
        self.loss.append(pytorch_model.unwrap(loss))

        # 0-1 vector when the interactions are greater than interaction prediction
        bin_interactions = pytorch_model.unwrap(interactions).copy()
        bin_interactions[bin_interactions < self.interaction_prediction] = 0
        bin_interactions[bin_interactions >= self.interaction_prediction] = 1

        # total_inter, weights
        self.total_inter += np.sum(bin_interactions * done_flags.astype(int))
        if trace is not None: self.total_true += np.sum(trace * done_flags.astype(int))
        self.weight_count += weights

        # -1 is a false negative (compared to the binaries), +1 is a false positive
        bin_diffs = bin_interactions - pytorch_model.unwrap(binaries)
        self.total_bin += np.sum(pytorch_model.unwrap(binaries).astype(int) * done_flags.astype(int)) 
        self.binary_false_negative += np.sum((bin_diffs < 0).astype(int) * done_flags.astype(int))
        self.binary_false_positive += np.sum((bin_diffs > 0).astype(int) * done_flags.astype(int))
        # comparing to the trace, if trace is not used the trace vector is expected to be all 1
        if trace is not None:
            bin_trace_diffs = pytorch_model.unwrap(binaries) - trace
            self.binary_true_false_negative += np.sum((bin_trace_diffs < 0).astype(int) * done_flags.astype(int))
            self.binary_true_false_positive += np.sum((bin_trace_diffs > 0).astype(int) * done_flags.astype(int))
            trace_diffs = bin_interactions - trace
            self.trace_false_negative += np.sum((trace_diffs < 0).astype(int) * done_flags.astype(int))
            self.trace_false_positive += np.sum((trace_diffs > 0).astype(int) * done_flags.astype(int))
        self.total_seen += len(binaries) - np.sum((done_flags == 0).astype(float))
        
        if i % self.log_interval == 0 and not no_print:
            log_str = self.name + f' interaction at {i}, mean loss: {np.mean(self.loss)}, total_inter: {self.total_inter/self.total_seen},  total_bin: {self.total_bin/self.total_seen}'
            if self.total_true > 0: log_str += f'\ntotal true: {self.total_true/self.total_seen}, weight rate: {self.weight_count/self.total_seen}' # assumes that there would not be no true interactions unless unused
            log_str  += f'\nbinary FP: {self.binary_false_positive/max(1, self.binary_false_positive + self.total_seen - self.total_bin)}, binary FN: {self.binary_false_negative/max(1, self.binary_false_negative + self.total_bin)}'
            # if self.total_true > 0: log_str  += f'\nbinary true FP: {self.binary_true_false_positive/max(1, self.binary_true_false_positive + self.total_seen - self.total_true)}, binary FN: {self.binary_true_false_negative/max(1, self.binary_true_false_negative + self.total_true)}'
            if self.total_true > 0: log_str  += f'\nbinary true FP: {self.binary_true_false_positive/max(1, self.total_true)}, binary FN: {self.binary_true_false_negative/max(1, self.total_true)}'
            # if self.total_true > 0: log_str += f'\ntrace FP: {self.trace_false_positive/max(1, self.trace_false_positive + self.total_seen - self.total_true)}, trace FN: {self.trace_false_negative/max(1, self.total_true + self.trace_false_negative)}'
            if self.total_true > 0: log_str += f'\ntrace FP: {self.trace_false_positive/max(1, self.total_true)}, trace FN: {self.trace_false_negative/max(1, self.total_true)}'
            logging.info(log_str)
            print(log_str)
            if len(self.record_graphs) != 0:
                # adds to the tensorboard logger for graphing
                # self.tensorboard_logger.add_scalar("Return/"+self.name, np.sum(self.reward)/np.sum(self.current_episodes), i)
                # self.tensorboard_logger.add_scalar("Success/"+self.name, np.sum(self.success)/np.sum(self.current_term), i)
                # self.tensorboard_logger.add_scalar("Success/"+self.name + "_h/m", np.sum(self.success)/miss_hit, i)
                self.tensorboard_logger.add_scalar("interaction_loss" +"/" + self.name, np.mean(self.loss), i) # TODO add other losses
                # log the loss values
                # print(self.testing_log)
                self.tensorboard_logger.flush()
            if i % (self.log_interval * 10) == 0: self.reset()