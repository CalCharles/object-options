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
        self.trace_false_positive = 0
        self.trace_false_negative = 0
        self.weight_count = 0
        self.total_inter = 0
        self.total_true = 0

    def log(self, i, loss, interactions, binaries, dones, weights, trace=None, no_print=False):
        self.loss.append(pytorch_model.unwrap(loss))

        # 0-1 vector when the interactions are greater than interaction prediction
        bin_interactions = pytorch_model.unwrap(interactions).copy()
        bin_interactions[bin_interactions < self.interaction_prediction] = 0
        bin_interactions[bin_interactions >= self.interaction_prediction] = 1

        # total_inter, weights
        self.total_inter += np.sum(bin_interactions)
        if trace is not None: self.total_true += np.sum(trace)
        self.weight_count += weights

        # -1 is a false negative (compared to the binaries), +1 is a false positive
        bin_diffs = bin_interactions - pytorch_model.unwrap(binaries)
        self.binary_false_negative += np.sum((bin_diffs < 0).astype(int) * dones)
        self.binary_false_positive += np.sum((bin_diffs > 0).astype(int) * dones)
        # comparing to the trace, if trace is not used the trace vector is expected to be all 1
        if trace is not None:
            trace_diffs = bin_interactions - trace
            self.trace_false_negative += np.sum((trace_diffs < 0).astype(int) * dones)
            self.trace_false_positive += np.sum((trace_diffs > 0).astype(int) * dones)
        self.total_seen += len(binaries) - np.sum(dones)
        
        if i % self.log_interval == 0 and not no_print:
            log_str = f'interaction at {i}, mean loss: {np.mean(self.loss)}, total_inter: {self.total_inter/self.total_seen}'
            if self.total_true > 0: log_str += f', total true: {self.total_true/self.total_seen}, weight rate: {self.weight_count/self.total_seen}' # assumes that there would not be no true interactions unless unused
            log_str  += f'\nbinary FP: {self.binary_false_positive/self.total_seen}, binary FN: {self.binary_false_negative/self.total_seen}'
            if self.total_true > 0: log_str += f'\ntrace FP: {self.trace_false_positive/self.total_seen}, trace FN: {self.trace_false_negative/self.total_seen}'
            logging.info(log_str)
            print(log_str)
            self.reset()