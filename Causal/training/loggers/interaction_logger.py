import logging

class interaction_logger():
    def __init__(self, train_args, full_model):
        self.interaction_logging_schedule = train_args.interaction_logging_schedule
        self.interaction_prediction = full_model.test.interaction_prediction
        self.reset()

    def reset(self):
        self.loss = list()
        self.total_seen = 0
        self.binary_false_positive = 0
        self.binary_false_negative = 0
        self.trace_false_positive = 0
        self.trace_false_negative = 0
        self.total_inter = 0
        self.total_true = 0

    def log(i, loss, interactions, binaries, trace):
        self.loss.append(pytorch_model.unwrap(loss))

        # 0-1 vector when the interactions are greater than interaction prediction
        bin_interactons = interactions.copy()
        bin_interactons[bin_interactions < self.interaction_prediction] = 0
        bin_interactons[bin_interactions >= self.interaction_prediction] = 1

        # total_inter
        self.total_inter += np.sum(bin_interactions)
        self.total_true += np.sum(trace)

        # -1 is a false negative (compared to the binaries), +1 is a false positive
        bin_diffs = bin_interactions - binaries
        self.binary_false_negative += np.sum((bin_diffs < 0).astype(int))
        self.binary_false_positive += np.sum((bin_diffs > 0).astype(int))
        # comparing to the trace, if trace is not used the trace vector is expected to be all 1
        trace_diffs = bin_interactions - trace
        self.trace_false_negative += np.sum((bin_diffs < 0).astype(int))
        self.trace_false_positive += np.sum((bin_diffs > 0).astype(int))

        self.total_seen += len(binaries)
        
        if i % self.interaction_logging_schedule:
            logging.info(f'interaction at {i}, mean loss: {np.mean(self.loss)}, total_inter: {self.total_inter}, total true: {self.total_true}' +
                f'binary FP: {self.binary_false_positive/self.total_seen}, binary FN: {self.binary_false_negative/self.total_seen}' +
                f'trace FP: {self.trace_false_positive/self.total_seen}, trace FN: {self.trace_false_negative/self.total_seen}')
            self.reset()