from Option.rew_term_done import RTD
import numpy as np
from Network.network_utils import pytorch_model
from Option.General.param import check_close

class BinaryInteractionParameterizedOptionControl(RTD):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_close = np.array(kwargs['epsilon_close']).squeeze()
        self.norm_p = kwargs['param_norm']
        self.param_lambda = kwargs["param_lambda"]
        self.inter_lambda = kwargs["inter_lambda"]
        self.true_lambda = kwargs["true_lambda"]
        self.negative_true = kwargs["negative_true"]
        self.constant_lambda = kwargs['constant_lambda']
        self.inter_term = kwargs['interaction_as_termination']
        self.use_binary = kwargs['use_binary']

    def compute_rew_term_done(self, inter_state, target, next_target, target_diff, param, mask, true_done, true_reward):
        self.epsilon_close = np.array(self.epsilon_close).squeeze()
        self.true_lambda = 0
        inside = check_close(self.epsilon_close, self.norm_p, next_target, param, mask)
        interv = pytorch_model.unwrap(self.interaction_model.interaction(inter_state, target, next_target, target_diff, prenormalize = True, use_binary=self.use_binary if hasattr(self, "use_binary") else False))
        # print("interv", interv, inter_state)
        inter = self.interaction_model.test(interv).squeeze()
        term = inside * inter + inter * self.inter_term

        # use the negative extrinsic reward if this is active
        if hasattr(self, "negative_true") and self.negative_true:
            negative_true = true_reward.copy()
            if type(negative_true) == np.ndarray: negative_true[negative_true > 0] = 0.0
            else: negative_true = np.array(negative_true) if negative_true < 0 else np.array(0.0)
            true_reward_component = negative_true.squeeze() * self.true_lambda
        else: true_reward_component = true_reward.squeeze() * self.true_lambda
        rew = ((inside * inter).astype(np.float64) * self.param_lambda 
                + inter.astype(np.float64) * self.inter_lambda + self.constant_lambda
                + true_reward_component)
        return term, rew, inter