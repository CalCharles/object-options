import numpy as np
from ReinforcementLearning.GroundTruth.ground_truth import GroundTruthPolicy
from Environment.Environments.Asteroids.asteroid_objects import *
from typing import Any, Dict, Tuple, Union, Optional, Callable
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from State.angle_calculator import sincos_to_angle, sincos_to_angle2

class GroundTruthShipPolicy(GroundTruthPolicy):
    '''
    for dummy policies that generate perfect behavior
    '''
    def __init__(self, discrete_actions, input_shape, policy_action_space, args, parent_algo_policy):
        super().__init__(discrete_actions, input_shape, policy_action_space, args, parent_algo_policy)

    def internal_policy(self, batch):
        self.internal_environment.set_from_factored_state(batch.full_state.factored_state[0])
        ship_location = self.internal_environment.ship.pos
        ship_angle = self.internal_environment.ship.angle
        ship_direction = np.dot(rotation_matrix(self.internal_environment.ship.angle), np.array([[1,0]]).T).squeeze()
        ship_to_target = batch.param[:2] - ship_location
        # cos_sim = np.sum(ship_direction * ship_to_target) / (np.linalg.norm(ship_direction) * np.linalg.norm(ship_to_target))
        # cos_ang = np.arccos(cos_sim)
        # cos_ang = cos_ang if cos_ang < np.pi/2 else np.pi - cos_ang

        to_ang = sincos_to_angle(ship_to_target[1], ship_to_target[0])
        to_ang = 2 * np.pi + to_ang if to_ang < 0 else to_ang

        angle_mag = np.min([np.abs(ship_angle - to_ang), np.abs(2 * np.pi - np.abs(ship_angle - to_ang))])

        orthogonal_direction = np.dot(rotation_matrix(self.internal_environment.ship.angle + np.pi/2), np.array([[1,0]]).T).squeeze()
        orthogonal_angle = sincos_to_angle(orthogonal_direction[1], orthogonal_direction[0])
        if np.max(np.abs(ship_to_target)) < 2:
            ship_sincos = batch.full_state.factored_state[0].Ship[2:4]
            if np.max(np.abs(batch.param[2:4] - ship_sincos)) < 0.3:
                if batch.param[4] > 0.5:
                    act = np.array(5)
                else:
                    act = np.array(0)
            else: # need to align angle to target
                angle = sincos_to_angle(batch.param[2], batch.param[3])
                looped = np.abs(ship_angle - angle) > np.abs(2 * np.pi - np.abs(ship_angle - angle))
                if looped:
                    if ship_angle - angle > 0:
                        act= np.array(3)
                    else:
                        act= np.array(4)
                else:
                    if ship_angle - angle > 0:
                        act= np.array(4)
                    else:
                        act= np.array(3)

                # angle_dir = np.dot(rotation_matrix(angle), np.array([[1,0]]).T).squeeze()
                # if np.sum(orthogonal_direction * angle_dir) > 0:
                #     act = np.array(3)
                # else:
                #     act = np.array(4)
        elif angle_mag > self.internal_environment.ship_speed[1] / 2:
            looped = np.abs(ship_angle - to_ang) > np.abs(2 * np.pi - np.abs(ship_angle - to_ang))
            if looped:
                if ship_angle - to_ang > 0:
                    act= np.array(3)
                else:
                    act= np.array(4)
            else:
                if ship_angle - to_ang > 0:
                    act= np.array(4)
                else:
                    act= np.array(3)
            # side = np.sign(np.sum(orthogonal_direction * ship_to_target)) # sign should not be zero or angle would be 0
            # if side > 0:
            #     act= np.array(3)
            # else:
            #     act = np.array(4)
        else:
            direction = np.sign(np.sum(ship_direction * ship_to_target))
            # if direction > 0:
            #     act= np.array(1)
            # else:
            act= np.array(2)
        return act