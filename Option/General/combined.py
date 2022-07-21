from Option.rew_term_done import RTD
import numpy as np
from Network.network_utils import pytorch_model

class BinaryInteractionParameterizedOptionControl(RTD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_close = kwargs['epsilon_close']
        self.norm_p = kwargs['param_norm']
        self.param_lambda = kwargs["param_lambda"]
        self.inter_lambda = kwargs["inter_lambda"]
        self.constant_lambda = kwargs['constant_lambda']
        self.inter_term = kwargs['interaction_as_termination']

    def compute_rew_term_done(self, inter_state, target, next_target, param, mask, true_done, true_reward):
        if self.norm_p > 2: inside = (np.max(np.abs((next_target.squeeze() - param.squeeze()) * mask.squeeze()), axis=-1) <= self.epsilon_close ).squeeze()
        else: inside = (np.linalg.norm((next_target.squeeze() - param.squeeze()) * mask.squeeze(), ord = self.norm_p, axis=-1) <= self.epsilon_close).squeeze()
        interv = pytorch_model.unwrap(self.interaction_model.interaction(inter_state))
        inter = self.interaction_model.test(interv).squeeze()
        term = inside * inter + inter * self.inter_term
        rew = (inside * inter).astype(np.float64) * self.param_lambda + inter.astype(np.float64) * self.inter_lambda + self.constant_lambda
        return term, rew, inter