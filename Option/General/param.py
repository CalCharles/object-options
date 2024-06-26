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
        self.true_lambda = kwargs["true_lambda"]
        self.negative_true = kwargs["negative_true"]

    def compute_rew_term_done(self, inter_state, target, next_target, target_diff, param, mask, true_done, true_reward):
        inside = check_close(self.epsilon_close, self.norm_p, next_target, param, mask)
        term, rew = inside.copy(), inside.copy().astype(np.float64)
        if hasattr(self, "true_lambda"):
            if hasattr(self, "negative_true") and self.negative_true:
                negative_true = true_reward.copy()
                if type(negative_true) == np.ndarray: negative_true[negative_true > 0] = 0.0
                else: negative_true = np.array(negative_true) if negative_true < 0 else np.array(0.0)
                true_reward_component = negative_true.squeeze() * self.true_lambda
            else: true_reward_component = true_reward.squeeze() * self.true_lambda
            # print(term, param, self.epsilon_close, self.norm_p, mask, next_target)
        else:
            true_reward_component = 0.0
        return term, rew + self.constant_lambda + true_reward_component, np.array(True)