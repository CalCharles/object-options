from Option.rew_term_done import RTD
import numpy as np
from Network.network_utils import pytorch_model
from tianshou.data import Batch

class RewardOptionControl(RTD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constant_lambda = kwargs['constant_lambda']
        self.true_lambda = kwargs['true_lambda']
        self.inter_term = kwargs['interaction_as_termination']
        self.epsilon_close = kwargs['epsilon_close']

    def compute_rew_term_done(self, inter_state, target, next_target, target_diff, param, mask, true_done, true_reward):
        inter = true_done.squeeze()
        term = self.interaction_model.interaction(Batch(inter_state = inter_state, target=target, next_target=next_target)) if self.inter_term else inter
        rew = true_reward + self.constant_lambda
        return term, rew, inter