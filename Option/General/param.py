import numpy as np
from Option.rew_term_done import RTD

def check_close(epsilon_close, norm_p, next_target, param, mask):
    # print(epsilon_close.shape)
    if len(epsilon_close.shape) > 0:
        inside = np.min(np.abs((next_target.squeeze() - param.squeeze()) * mask.squeeze()) <= epsilon_close, axis=-1)
    else:
        if norm_p > 2: inside = np.max(np.abs((next_target.squeeze() - param.squeeze()) * mask.squeeze()), axis=-1) <= epsilon_close
        else: inside = np.linalg.norm((next_target.squeeze() - param.squeeze()) * mask.squeeze(), ord = norm_p, axis=-1) <= epsilon_close
    return inside.squeeze()

class BinaryParameterizedOptionControl(RTD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_close = np.array(kwargs['epsilon_close']).squeeze()
        self.norm_p = kwargs['param_norm']
        self.constant_lambda = kwargs['constant_lambda']

    def compute_rew_term_done(self, inter_state, target, next_target, target_diff, param, mask, true_done, true_reward):
        inside = check_close(self.epsilon_close, self.norm_p, next_target, param, mask)
        term, rew = inside.copy(), inside.copy().astype(np.float64)
        return term, rew + self.constant_lambda, np.array(True)