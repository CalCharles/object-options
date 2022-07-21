import numpy as np
from Option.rew_term_done import RTD

class BinaryParameterizedOptionControl(RTD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_close = kwargs['epsilon_close']
        self.norm_p = kwargs['param_norm']
        self.constant_lambda = kwargs['constant_lambda']

    def compute_rew_term_done(self, inter_state, target, next_target, param, mask, true_done, true_reward):
        if self.norm_p > 2: inside = np.max(np.abs((next_target - param) * mask), axis=-1) <= self.epsilon_close 
        else: inside = np.linalg.norm((next_target - param) * mask, ord = self.norm_p, axis=-1) <= self.epsilon_close
        term, rew = inside.copy(), inside.copy().astype(np.float64)
        return term, rew + self.constant_lambda, np.array(True)