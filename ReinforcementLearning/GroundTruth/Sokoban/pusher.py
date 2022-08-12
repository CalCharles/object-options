from Environment.Environments.Sokoban.sokoban_objects import *
import numpy as np
from ReinforcementLearning.GroundTruth.ground_truth import GroundTruthPolicy
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
import copy

action_direction = {
    0: np.array([-1, 0]),
    1: np.array([1, 0]),
    2: np.array([0, -1]),
    3: np.array([0, 1])
}

def check_bounds(pos, num_rows, num_columns):
    return pos[0] < 0 or pos[1] < 0 or pos[0] >= num_rows or pos[1] >= num_columns


class GroundTruthPusherPolicy(GroundTruthPolicy):
    '''
    for dummy policies that generate perfect behavior
    '''
    def __init__(self, discrete_actions, input_shape, policy_action_space, args, parent_algo_policy):
        super().__init__(discrete_actions, input_shape, policy_action_space, args, parent_algo_policy)
    
    def check_occupancy(self, next_pos,backtrack_grid, next_next = None):
        if check_bounds(next_pos, self.internal_environment.num_rows, self.internal_environment.num_columns):
            return True
        elif backtrack_grid[next_pos[0]][next_pos[1]] is not None:
            return True
        elif type(self.internal_environment.occupancy_matrix[next_pos[0]][next_pos[1]]) == Obstacle:
            return True
        elif type(self.internal_environment.occupancy_matrix[next_pos[0]][next_pos[1]]) == tuple or type(self.internal_environment.occupancy_matrix[next_pos[0]][next_pos[1]]) == Block:
            if next_next is not None:
                if check_bounds(next_next, self.internal_environment.num_rows, self.internal_environment.num_columns):
                    return True
                if self.internal_environment.occupancy_matrix[next_next[0]][next_next[1]] is not None and type(self.internal_environment.occupancy_matrix[next_next[0]][next_next[1]]) != Target:
                    return True
        return False

    def internal_policy(self, batch):
        param = copy.deepcopy(batch.param)
        self.internal_environment.set_from_factored_state(batch.full_state.factored_state[0])
        pusher_pos = self.internal_environment.pusher.pos
        backtrack_grid = [[None for i in range(self.internal_environment.num_columns)] for j in range(self.internal_environment.num_rows)]
        
        queue = [(pusher_pos, None, -1)]
        param_loc = np.round(param).astype(int)
        if self.check_occupancy(param_loc, backtrack_grid):
            closest = None
            for a in range(4):
                closest = param_loc + action_direction[a]
                if not self.check_occupancy(closest, backtrack_grid):
                    param = param + action_direction[a]
                    break
        if np.max(np.abs(pusher_pos - param)) < 0.5: # if we are already at the location, use 0
            return np.array(0)
        found = False
        while len(queue) > 0:
            at_pos, last_pos, a = queue.pop(0)
            backtrack_grid[at_pos[0]][at_pos[1]] = (last_pos, a)
            if np.max(np.abs(at_pos - param)) < 0.5:
                # print("broke", at_pos, param, a)
                found = True
                break
            for a in range(4):
                next_pos = at_pos + action_direction[a]
                # print(next_pos, at_pos)
                if self.check_occupancy(next_pos, backtrack_grid, next_pos+action_direction[a]):
                    continue
                # print("added", next_pos, at_pos, a)
                queue.append((next_pos, at_pos, a))
        prior_action = a
        if not found:
            return np.array(0)
        while True:
            last_pos, a = backtrack_grid[at_pos[0]][at_pos[1]]
            if a == -1:
                break
            prior_action = a
            at_pos = last_pos
        return np.array(prior_action)

