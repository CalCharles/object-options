import logging

class interaction_logger():
Logging forward model:
    iteraction number
    raw likelihood
    weighted likelihood
    likelihood weighted with true values
    l1 average error per element

    def __init__(self, net_type, train_args, full_model):
        self.forward_logging_schedule = train_args.forward_logging_schedule
        self.type = net_type
        self.reset()

    def reset(self):
        self.loss = list()
        self.raw_likelihood = list()
        self.weighted_likelihood = list()
        self.true_weighted_likelihood = list()
        self.l1_average_error = list()
        self.l1_average_weighted_error = list()
        self.l1_average_true_error = list()

    def log(i, loss, raw_likelihood, weighted_likelihood, 
                    raw_likelihood_expanded, trace,
                    params, targets, interaction_likelihoods):
        self.loss.append(pytorch_model.unwrap(loss))
        self.raw_likelihood.append(pytorch_model.unwrap(raw_likelihood))
        self.weighted_likelihood.append(pytorch_model.unwrap(weighted_likelihood))

        # the weighted with the trace instead of interaction values. TODO: use only raw[trace==1] to compute mean
        true_weighted_likelihood = pytorch_model.unwrap(raw_likelihood_expanded) * trace
        self.true_weighted_likelihood.append(true_weighted_likelihood)

        # l1 errors using the means
        l1_error = np.abs(pytorch_model.unwrap(params[0]) - targets)
        self.l1_average_error.append(np.mean(l1_error, axis=0))

        # weighted error according to trace, or according to interaction
        unwrapped_likelihoods = pytorch_model.unwrap(interaction_likelihoods)
        self.l1_average_weighted_error.append(np.sum(l1_error * unwrapped_likelihoods, axis=0) / np.sum(unwrapped_likelihoods) )
        self.l1_average_true_error.append(np.sum(l1_error * trace, axis=0) / np.sum(trace) )

        if i % self.forward_logging_schedule:
            logging.info(net_type + f' at {i}, mean loss: {np.mean(self.loss)}, raw likelihood: {np.mean(self.raw_likelihood)}, weighted likelihood: {np.mean(self.weighted_likelihood)}' +
                f'true likelihood: {np.mean(true_weighted_likelihood)}\n' +
                f'l1 error: {np.mean(self.l1_average_error, axis=0)}\nl1 weighted: {np.mean(self.l1_average_weighted_error, axis=0)}\nl1 true: {np.mean(self.l1_average_true_error, axis=0)}')
            self.reset()