from Option.rew_term_done import RTD
import numpy as np
from Network.network_utils import pytorch_model

class BinaryInteractionParameterizedOptionControl(RTD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.interaction_model = kwargs['interaction_model']
        self.epsilon_close = kwargs['epsilon_close']
        self.norm_p = kwargs['param_norm']
        self.target_select = kwargs['target_select']
        self.inter_select = kwargs['inter_select']
        self.param_lambda = kwargs["param_lambda"]
        self.inter_lambda = kwargs["inter_lambda"]
        self.constant_lambda = kwargs['constant_lambda']

    def compute_rew_term_done(self, full_state, next_full_state, param, mask, true_done, true_reward):
        state = self.target_select(next_full_state['factored_state'])
        inside = np.linalg.norm((state - param) * mask, ord = self.norm_p, axis=-1) <= self.epsilon_close
        inter = self.interaction_model.test(pytorch_model.unwrap(self.interaction_model.interaction(self.inter_select(full_state['factored_state']))))
        term = inside * inter
        rew = (inside * inter).astype(np.float64) * self.param_lambda + inter * self.inter_lambda + self.constant_lambda
        return term, rew, inter