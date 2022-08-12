import numpy as np
from ReinforcementLearning.GroundTruth.ground_truth import GroundTruthPolicy
from Environment.Environments.Breakout.breakout_objects import *
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy

class GroundTruthPaddlePolicy(GroundTruthPolicy):
    '''
    for dummy policies that generate perfect behavior
    '''
    def __init__(self, discrete_actions, input_shape, policy_action_space, args, parent_algo_policy):
        super().__init__(discrete_actions, input_shape, policy_action_space, args, parent_algo_policy)

    def internal_policy(self, batch):
        self.internal_environment.set_from_factored_state(batch.full_state.factored_state[0])
        paddle_pos = self.internal_environment.paddle.getMidpoint()

        direction = np.sign(paddle_pos[1] - batch.param[1])
       	print(direction, paddle_pos, batch.param, np.abs(paddle_pos[1] - batch.param[1]))
        if np.abs(paddle_pos[1] - batch.param[1]) > 1:
            if direction > 0:
                return np.array(2)
            else:
                return np.array(3)
        return np.array(1) # doesn't use action 0
