import numpy as np
import copy
from ReinforcementLearning.policy import Policy
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Environment.Environments.initialize_environment import initialize_environment


class GroundTruthPolicy(Policy):
    '''
    for dummy policies that generate perfect behavior
    '''
    def __init__(self, discrete_actions, input_shape, policy_action_space, args, parent_policy):
        super().__init__(discrete_actions, input_shape, policy_action_space, args, parent_policy)
        self.internal_environment, _ = initialize_environment(args.environment, None)
        self.action_map = args.action_map_object # we need to reverse actions

    def internal_policy(self, batch): # where the magic happens, returns mapped action
        return 

    def forward(self, batch: Batch, state: Optional[Union[dict, Batch, np.ndarray]] = None, input: str = "obs", use_gt: bool = False, **kwargs: Any):
        '''
        Matches the call for the forward of another algorithm method 
        '''
        batch = copy.deepcopy(batch) # make sure input norm does no    def reverse_map_action(self, mapped_act, batch):t alter the input batch
        vals = self.algo_policy(batch, state = state, input=input, **kwargs)
        # technically, this makes forward invalid for non-action batched cases if use_gt is True
        mapped_action = self.internal_policy(batch)
        vals.act = np.expand_dims(self.action_map.reverse_map_action(mapped_action, batch), 0)
        return vals


