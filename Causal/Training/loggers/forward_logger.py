import logging
from Network.network_utils import pytorch_model
import numpy as np
from Record.logging import Logger
from Causal.Utils.instance_handling import compute_l1

class forward_logger(Logger):
# Logging forward model:
#     iteraction number
#     raw likelihood
#     weighted likelihood
#     likelihood weighted with true values
#     l1 average error per element

    def __init__(self, net_type, log_interval, full_model, filename=""):
        super().__init__(filename)
        self.log_interval = log_interval
        self.type = net_type
        self.reset()

    def reset(self):
        self.weight_rates = list()
        self.trace_rates = list()
        self.loss = list()
        self.raw_likelihood = list()
        self.weighted_likelihood = list()
        self.raw_likelihood_expanded = list()
        self.true_weighted_likelihood = list()
        self.l1_average_error = list()
        self.l1_average_weighted_error = list()
        self.l1_average_true_error = list()

    def log(self, i, loss, raw_likelihood, weighted_likelihood, 
                    raw_likelihood_expanded, trace, weight_rate, dones,
                    params, targets, interaction_likelihoods, full_model):
        self.loss.append(pytorch_model.unwrap(loss))
        if raw_likelihood is not None: self.raw_likelihood.append(pytorch_model.unwrap(raw_likelihood))
        if weighted_likelihood is not None: self.weighted_likelihood.append(pytorch_model.unwrap(weighted_likelihood))
        if raw_likelihood_expanded is not None: self.raw_likelihood_expanded += pytorch_model.unwrap(raw_likelihood_expanded).tolist()
        # print(self.type, raw_likelihood_expanded[...,1:2], np.mean(self.raw_likelihood_expanded, axis=0)[...,1:2])

        # l1 errors using the means
        if params is not None:
            l1_error, l1_error_element = compute_l1(full_model, len(targets), params, targets)
            self.l1_average_error.append(np.mean(l1_error, axis=0))

        # the weighted with the trace instead of interaction values. TODO: use only raw[trace==1] to compute mean
        if trace is not None:
            true_weighted_likelihood = pytorch_model.unwrap(raw_likelihood) * trace
            self.true_weighted_likelihood.append(true_weighted_likelihood)
            self.l1_average_true_error.append(np.sum(l1_error * trace, axis=0) / np.sum(trace) )
        if weight_rate is not None: self.weight_rates.append(weight_rate)
        if trace is not None: self.trace_rates.append(np.sum(trace) / max(1,len(trace)))
        
        # weighted error according to trace, or according to interaction
        if interaction_likelihoods is not None:
            unwrapped_likelihoods = pytorch_model.unwrap(interaction_likelihoods)
            self.l1_average_weighted_error.append(np.sum(l1_error_element * unwrapped_likelihoods, axis=0) / np.sum(unwrapped_likelihoods) )

        if i % self.log_interval == 0:
            logging_str = "================\n"
            logging_str = self.type + f' at {i}, mean loss: {np.mean(self.loss)}, '
            if len(self.raw_likelihood) > 0: logging_str += f'raw: {np.mean(self.raw_likelihood)}, '
            if len(self.weighted_likelihood) > 0: logging_str += f'weighted: {np.mean(self.weighted_likelihood)}, '
            if len(self.true_weighted_likelihood) > 0: logging_str += f'true: {np.mean(true_weighted_likelihood)}'
            if len(self.raw_likelihood_expanded) > 0: logging_str += f'\nexpanded: {np.mean(self.raw_likelihood_expanded, axis=0)}'
            if len(self.l1_average_error) > 0: logging_str += f'\nl1 error: {np.mean(self.l1_average_error, axis=0)}'
            if len(self.l1_average_weighted_error) > 0: logging_str += f'\nl1 weighted: {np.mean(self.l1_average_weighted_error, axis=0)}'
            if len(self.l1_average_true_error) > 0: logging_str += f'\nl1 true: {np.mean(self.l1_average_true_error, axis=0)}'
            if len(self.weight_rates) > 0: logging_str += f'\npercent (weight, trace): {np.mean(self.weight_rates)}, {np.mean(self.trace_rates)}'
            logging_str += f"\ntarget: {full_model.norm.reverse(pytorch_model.unwrap(targets[0]), form='dyn' if full_model.predict_dynamics else 'target')}, {dones[0]}\n"
            logging_str += f"mean: {full_model.norm.reverse(pytorch_model.unwrap(params[0][0]), form='dyn' if full_model.predict_dynamics else 'target')}\n"
            logging_str += f"variance: {full_model.norm.reverse(pytorch_model.unwrap(params[1][0]), form='dyn' if full_model.predict_dynamics else 'diff')}"
            logging.info(logging_str)
            print(logging_str)
            self.reset()