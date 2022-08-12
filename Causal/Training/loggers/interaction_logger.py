import logging
from Network.network_utils import pytorch_model
from Record.logging import Logger
import numpy as np

class interaction_logger(Logger):
    def __init__(self, name, log_interval, full_model, filename=""):
        super().__init__(filename)
        self.name = name
        self.log_interval = log_interval
        self.interaction_prediction = full_model.test.interaction_prediction
        self.reset()

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
            log_str = f'interaction at {i}, mean loss: {np.mean(self.loss)}, total_inter: {self.total_inter/self.total_seen},  total_bin: {self.total_bin/self.total_seen}'
            if self.total_true > 0: log_str += f'\ntotal true: {self.total_true/self.total_seen}, weight rate: {self.weight_count/self.total_seen}' # assumes that there would not be no true interactions unless unused
            log_str  += f'\nbinary FP: {self.binary_false_positive/max(1, self.binary_false_positive + self.total_seen - self.total_bin)}, binary FN: {self.binary_false_negative/max(1, self.binary_false_negative + self.total_bin)}'
            if self.total_true > 0: log_str  += f'\nbinary true FP: {self.binary_true_false_positive/max(1, self.binary_true_false_positive + self.total_seen - self.total_true)}, binary FN: {self.binary_true_false_negative/max(1, self.binary_true_false_negative + self.total_true)}'
            if self.total_true > 0: log_str += f'\ntrace FP: {self.trace_false_positive/max(1, self.trace_false_positive + self.total_seen - self.total_true)}, trace FN: {self.trace_false_negative/max(1, self.total_true + self.trace_false_negative)}'
            logging.info(log_str)
            print(log_str)
            self.reset()