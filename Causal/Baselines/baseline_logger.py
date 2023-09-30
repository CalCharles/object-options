import logging, os
from Network.network_utils import pytorch_model
from Record.logging import Logger
from Record.file_management import create_directory
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque

class baseline_interaction_logger(Logger):
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

    def log_gradient(self, test_dict):
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
        self.element_loss = None
        self.sum_soft_over = None
        self.sum_soft_under = None
        self.total_inter = 0
        self.weight_count = 0
        self.active_loss = list()
        self.total_true = 0

    def log(self, i, soft_bins, bins, trace, active_loss, done, active_prediction_params, target, full_model, valid=None, no_print=False):
        done_flags = 1-done
        loss = np.mean(np.abs(bins - trace) * done_flags)
        self.loss.append(pytorch_model.unwrap(loss))
        self.active_loss.append(pytorch_model.unwrap(active_loss))
        self.total_inter += np.sum(bins * done_flags.astype(int))
        self.total_true += np.sum(trace * done_flags.astype(int))

        # the per-element binary error
        self.element_loss = np.sum(np.abs(bins - trace) * done_flags, axis=0) if self.element_loss is None else self.element_loss +  np.sum(np.abs(bins - trace) * done_flags, axis=0)
        # the sum of soft values when the trace is 1
        over_dones = done_flags[np.nonzero(np.expand_dims(trace.sum(axis=-1), axis=-1))]
        if self.sum_soft_over is None: self.sum_soft_over = np.zeros(trace.shape[-1])
        for k in range(trace.shape[-1]):
            self.sum_soft_over[k] = self.sum_soft_over[k] + np.sum((soft_bins[:,k] * over_dones)[np.nonzero(trace[:,k])], axis=0)
        # the sum of soft values when the trace is 
        under_trace = 1-trace
        under_dones = done_flags[np.nonzero(np.expand_dims(under_trace.sum(axis=-1), axis=-1))]
        if self.sum_soft_under is None: self.sum_soft_under = np.zeros(trace.shape[-1])
        for k in range(trace.shape[-1]):
            self.sum_soft_under[k] = self.sum_soft_under[k] + np.sum((soft_bins[:,k] * under_dones)[np.nonzero(under_trace[:,k])], axis=0)
        self.total_seen += len(soft_bins) - np.sum((done_flags == 0).astype(float))
        
        if i % self.log_interval == 0 and not no_print:
            log_str = self.name + f' interaction at {i}, mean loss: {np.mean(self.loss)}, active loss: {np.mean(self.active_loss)}'
            log_str += f'\ntotal_inter: {self.total_inter/self.total_seen} total_seen: {self.total_seen}'
            if self.total_true > 0: log_str += f'\ntotal true: {self.total_true/self.total_seen}' # assumes that there would not be no true interactions unless unused
            if self.element_loss is not None: log_str  += '\nflat error: ' + str(self.element_loss / self.total_seen)
            if self.sum_soft_over is not None: log_str  += '\nsoft Positive: ' + str(self.sum_soft_over / self.total_seen)
            if self.sum_soft_under is not None: log_str  += '\nsoft Negative: ' + str(self.sum_soft_under / self.total_seen)
            log_str += '\ntrace '  + str(trace[0])
            log_str += '\nsoft '  + str(soft_bins[0])
            log_str += '\nhard '  + str(bins[0])
            logging.info(log_str)
            print(log_str)
            if len(self.record_graphs) != 0:
                # adds to the tensorboard logger for graphing
                self.tensorboard_logger.add_scalar("interaction_loss" +"/" + self.name, np.mean(self.loss), i) # TODO add other losses
                if self.sum_soft_over is not None:
                    for ii in range(len(self.sum_soft_over)):
                        self.tensorboard_logger.add_scalar("flat error" +"/" + self.name + str(ii), self.element_loss[ii] / self.total_seen, i) # TODO add other losses
                        self.tensorboard_logger.add_scalar("soft Positive" +"/" + self.name + str(ii), self.sum_soft_over[ii] / self.total_seen, i) # TODO add other losses
                        self.tensorboard_logger.add_scalar("soft Negative" +"/" + self.name + str(ii), self.sum_soft_under[ii] / self.total_seen, i) # TODO add other losses

                # log the loss values
                # print(self.testing_log)
                self.tensorboard_logger.flush()
            if i % (self.log_interval * 10) == 0: self.reset()