import logging, os
from Network.network_utils import pytorch_model
from Record.logging import Logger
from Record.file_management import create_directory
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque

class full_interaction_logger(Logger):
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
        self.sum_trace = None
        self.sum_soft_error = None
        self.sum_soft_over = None
        self.sum_soft_under = None
        self.sum_hard_error = None
        self.sum_flat_error = None
        self.sum_soft_entropy = 0
        self.total_inter = 0
        self.weight_count = 0
        self.active_loss = list()
        self.total_true = 0
        self.num_iters = 0
        self.lambdas = None

    def log(self, i, loss, active_loss, soft_interactions, hard_interactions, done_flags, weights, trace=None, no_print=False, lambdas=None):
        self.loss.append(pytorch_model.unwrap(loss))
        self.active_loss.append(pytorch_model.unwrap(active_loss))

        # 0-1 vector when the interactions are greater than interaction prediction
        flat_interactions = pytorch_model.unwrap(soft_interactions).copy()
        flat_interactions[flat_interactions < self.interaction_prediction] = 0
        flat_interactions[flat_interactions >= self.interaction_prediction] = 1

        # total_inter, weights
        self.total_inter += np.sum(soft_interactions * done_flags.astype(int))
        if trace is not None: self.total_true += np.sum(trace * done_flags.astype(int))
        self.weight_count += weights
        self.sum_soft_entropy += np.sum(-np.sum(soft_interactions * np.log(soft_interactions), axis=-1))

        if lambdas is not None:
            self.lambdas = np.array(lambdas) if self.lambdas is None else self.lambdas + np.array(lambdas)
        if trace is not None:
            self.sum_trace = np.sum(trace * done_flags, axis=0) if self.sum_trace is None else self.sum_trace + np.sum(trace * done_flags, axis=0)
            self.sum_flat_error = np.sum(np.abs(flat_interactions - trace) * done_flags, axis=0) if self.sum_flat_error is None else self.sum_soft_error + np.sum(np.abs(flat_interactions - trace) * done_flags, axis=0)
            self.sum_soft_error = np.sum(np.abs(soft_interactions - trace) * done_flags, axis=0) if self.sum_soft_error is None else self.sum_soft_error + np.sum(np.abs(soft_interactions - trace) * done_flags, axis=0)
            self.sum_hard_error = np.sum(np.abs(hard_interactions - trace) * done_flags, axis=0) if self.sum_hard_error is None else self.sum_soft_error + np.sum(np.abs(hard_interactions - trace) * done_flags, axis=0)
            sum_soft_over = (soft_interactions - trace) * done_flags
            sum_soft_over[sum_soft_over<0] = 0
            self.sum_soft_over = np.sum(np.abs(sum_soft_over), axis=0) if self.sum_soft_over is None else self.sum_soft_over + np.sum(np.abs(sum_soft_over), axis=0)
            sum_soft_under = (soft_interactions - trace) * done_flags
            sum_soft_under[sum_soft_under>0] = 0
            self.sum_soft_under = np.sum(np.abs(sum_soft_under), axis=0) if self.sum_soft_under is None else self.sum_soft_under + np.sum(np.abs(sum_soft_under), axis=0)
        self.total_seen += len(soft_interactions) - np.sum((done_flags == 0).astype(float))
        self.num_iters += 1
        
        if i % self.log_interval == 0 and not no_print:
            log_str = self.name + f' interaction at {i}, mean loss: {np.mean(self.loss)}, active loss: {np.mean(self.active_loss)}'
            log_str += f'\ntotal_inter: {self.total_inter/self.total_seen}, entropy: {self.sum_soft_entropy/self.total_seen}'
            log_str += f'\nlambda values: {self.lambdas / self.num_iters}'
            if self.total_true > 0: log_str += f'\ntotal true: {self.total_true/self.total_seen}, weight rate: {self.weight_count/self.total_seen}' # assumes that there would not be no true interactions unless unused
            if self.sum_trace is not None: log_str  += '\ntrace rate: ' + str(self.sum_trace / self.total_seen)
            if self.sum_flat_error is not None: log_str  += '\nflat error: ' + str(self.sum_flat_error / self.total_seen)
            if self.sum_soft_error is not None: log_str  += '\nsoft error: ' + str(self.sum_soft_error / self.total_seen)
            if self.sum_hard_error is not None: log_str  += '\nhard error: ' + str(self.sum_hard_error / self.total_seen)
            if self.sum_soft_over is not None: log_str  += '\nsoft FP: ' + str(self.sum_soft_over / self.total_seen)
            if self.sum_soft_under is not None: log_str  += '\nsoft FN: ' + str(self.sum_soft_under / self.total_seen)
            if trace is not None: log_str += '\ntrace '  + str(trace[0])
            log_str += '\nsoft '  + str(soft_interactions[0])
            log_str += '\nhard '  + str(hard_interactions[0])
            logging.info(log_str)
            print(log_str)
            if len(self.record_graphs) != 0:
                # adds to the tensorboard logger for graphing
                self.tensorboard_logger.add_scalar("interaction_loss" +"/" + self.name, np.mean(self.loss), i) # TODO add other losses
                if self.sum_flat_error is not None:
                    for ii in range(len(self.sum_flat_error)):
                        self.tensorboard_logger.add_scalar("flat error" +"/" + self.name + str(ii), self.sum_flat_error[ii] / self.total_seen, i) # TODO add other losses
                        self.tensorboard_logger.add_scalar("soft error" +"/" + self.name + str(ii), self.sum_soft_error[ii] / self.total_seen, i) # TODO add other losses
                        self.tensorboard_logger.add_scalar("hard error" +"/" + self.name + str(ii), self.sum_hard_error[ii] / self.total_seen, i) # TODO add other losses
                        self.tensorboard_logger.add_scalar("soft FP" +"/" + self.name + str(ii), self.sum_soft_over[ii] / self.total_seen, i) # TODO add other losses
                        self.tensorboard_logger.add_scalar("soft FN" +"/" + self.name + str(ii), self.sum_soft_under[ii] / self.total_seen, i) # TODO add other losses
                    self.tensorboard_logger.add_scalar("entropy" +"/" + self.name + str(ii), self.sum_soft_entropy/self.total_seen, i) # TODO add other losses

                # log the loss values
                # print(self.testing_log)
                self.tensorboard_logger.flush()
            if i % (self.log_interval * 10) == 0: self.reset()