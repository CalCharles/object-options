import logging
from Network.network_utils import pytorch_model
import numpy as np

class forward_logger():
# Logging forward model:
#     iteraction number
#     raw likelihood
#     weighted likelihood
#     likelihood weighted with true values
#     l1 average error per element

    def __init__(self, net_type, log_interval, full_model):
        self.log_interval = log_interval
        self.type = net_type
        self.reset()

    def reset(self):
        self.loss = list()
        self.raw_likelihood = list()
        self.weighted_likelihood = list()
        self.raw_likelihood_expanded = list()
        self.true_weighted_likelihood = list()
        self.l1_average_error = list()
        self.l1_average_weighted_error = list()
        self.l1_average_true_error = list()

    def log(self, i, loss, raw_likelihood, weighted_likelihood, 
                    raw_likelihood_expanded, trace,
                    params, targets, interaction_likelihoods, full_model):
        self.loss.append(pytorch_model.unwrap(loss))
        if raw_likelihood is not None: self.raw_likelihood.append(pytorch_model.unwrap(raw_likelihood))
        if weighted_likelihood is not None: self.weighted_likelihood.append(pytorch_model.unwrap(weighted_likelihood))
        if raw_likelihood_expanded is not None: self.raw_likelihood_expanded += pytorch_model.unwrap(raw_likelihood_expanded).tolist()

        # l1 errors using the means
        if params is not None:
            l1_error = np.abs(pytorch_model.unwrap(params[0] - targets))
            self.l1_average_error.append(np.mean(l1_error, axis=0))

        # the weighted with the trace instead of interaction values. TODO: use only raw[trace==1] to compute mean
        if trace is not None:
            true_weighted_likelihood = pytorch_model.unwrap(raw_likelihood) * trace
            self.true_weighted_likelihood.append(true_weighted_likelihood)
            self.l1_average_true_error.append(np.sum(l1_error * trace, axis=0) / np.sum(trace) )

        # weighted error according to trace, or according to interaction
        if interaction_likelihoods is not None:
            unwrapped_likelihoods = pytorch_model.unwrap(interaction_likelihoods)
            self.l1_average_weighted_error.append(np.sum(l1_error * unwrapped_likelihoods, axis=0) / np.sum(unwrapped_likelihoods) )

        if i % self.log_interval == 0:
            logging_str = self.type + f' at {i}, mean loss: {np.mean(self.loss)}, '
            if len(self.raw_likelihood) > 0: logging_str += f'raw: {np.mean(self.raw_likelihood)}, '
            if len(self.weighted_likelihood) > 0: logging_str += f'weighted: {np.mean(self.weighted_likelihood)}, '
            if len(self.true_weighted_likelihood) > 0: logging_str += f'true: {np.mean(true_weighted_likelihood)}'
            if len(self.raw_likelihood_expanded) > 0: logging_str += f'\nexpanded: {np.mean(self.raw_likelihood_expanded, axis=0)}'
            if len(self.l1_average_error) > 0: logging_str += f'\nl1 error: {np.mean(self.l1_average_error, axis=0)}'
            if len(self.l1_average_weighted_error) > 0: logging_str += f'\nl1 weighted: {np.mean(self.l1_average_weighted_error, axis=0)}'
            if len(self.l1_average_true_error) > 0: logging_str += f'\nl1 true: {np.mean(self.l1_average_true_error, axis=0)}'
            logging_str += f"\ntarget: {full_model.norm.reverse(pytorch_model.unwrap(targets[0]), form='dyn' if full_model.predict_dynamics else 'target')}\n"
            logging_str += f"params: {full_model.norm.reverse(pytorch_model.unwrap(params[0][0]), form='dyn' if full_model.predict_dynamics else 'target')}\n"
            logging_str += f"params: {full_model.norm.reverse(pytorch_model.unwrap(params[1][0]), form='dyn' if full_model.predict_dynamics else 'target')}"
            logging.info(logging_str)
            print(logging_str)
            self.reset()