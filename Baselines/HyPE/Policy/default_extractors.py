import copy
import numpy as np

class BreakoutExtractor():
    def __init__(self, input_scaling, normalized, num_blocks, prox):
        self.input_scaling = input_scaling
        self.normalized = normalized
        self.num_blocks = num_blocks
        self.prox = prox

    def get_target(self, full_state):
        return np.array(copy.deepcopy(full_state['factored_state']["Reward"]))

    def get_parent(self, full_state):
        return np.array(copy.deepcopy(full_state['factored_state']["Ball"]))

    def get_diff(self, full_state, next_full_state):
        return np.array(full_state['factored_state']["Reward"]) - np.array(full_state['factored_state']["Reward"])

    def get_obs(self, full_state):
        # print(full_state["factored_state"])
        var = np.array([84,84, 2,1.0,1.0])
        paddle = np.array(copy.deepcopy(full_state['factored_state']["Paddle"]))
        ball = np.array(copy.deepcopy(full_state['factored_state']["Ball"]))
        blocks = [np.array(full_state['factored_state']["Block" + str(i)]) for i in range(self.num_blocks)] if self.num_blocks > 1 else [np.array(full_state['factored_state']["Block"])]
        rel = paddle - ball
        # print(parent.shape, target.shape, rel.shape)
        if self.prox:
            param = np.array(copy.deepcopy(full_state['factored_state']["Param"]))
            if self.normalized: components = [paddle / var - 0.5, ball / var - 0.5, rel / var, param / var - 0.5] + [block / var - 0.5 for block in blocks]
            else: components = [paddle, ball, rel, param] + blocks
        else:
            if self.normalized: components = [paddle / var - 0.5, ball / var - 0.5, rel / var] + [block / var - 0.5 for block in blocks]
            else: components = [paddle, ball, rel] + blocks
        # print(cat, self.obs_components)
        if hasattr(self, "input_scaling"):
            return np.concatenate(components, axis=-1) * self.input_scaling
        else: return np.concatenate(components, axis=-1)

    def pair_args(self):
        if self.prox:
            return 20, 5
        return 15, 5

class RoboPushingExtractor():
    def __init__(self, input_scaling, normalized, num_obstacles):
        self.input_scaling = input_scaling
        self.normalized = normalized
        self.num_obstacles = num_obstacles

    def get_target(self, full_state):
        return np.array(copy.deepcopy(full_state['factored_state']["Reward"]))

    def get_parent(self, full_state):
        return np.array(copy.deepcopy(full_state['factored_state']["Block"]))

    def get_diff(self, full_state, next_full_state):
        return np.array(full_state['factored_state']["Reward"]) - np.array(full_state['factored_state']["Reward"])

    def get_obs(self, full_state):
        # print(full_state["factored_state"])
        gvar = np.array([0.2,0.2, 0.2])
        gmean = np.array([-0.1, 0.0, 0.9])
        gripper = np.array(copy.deepcopy(full_state['factored_state']["Gripper"]))
        bvar = np.array([0.2,0.2, 0.05])
        bmean = np.array([-0.1, 0.0, 0.83])
        block = np.array(copy.deepcopy(full_state['factored_state']["Block"]))
        target = np.array(copy.deepcopy(full_state['factored_state']['Target']))
        obstacles = [np.array(full_state['factored_state']["Obstacle" + str(i)]) for i in range(self.num_obstacles)]
        rvar1 = np.array([0.4,0.4, 0.05])
        rel1 = block - target
        rvar2 = np.array([0.4,0.4, 0.2])
        rel2 = block - target
        # print(parent.shape, target.shape, rel.shape)
        if self.normalized: components = [(gripper - gmean) / gvar, (block - bmean) / bvar, rel1 / rvar1, rel2/rvar2, 
        										(target - bmean) / bvar] + [(obstacle - bmean) / bvar for obstacle in obstacles]
        else: components = [gripper, block, rel1, rel2, target] + obstacles
        # print(cat, self.obs_components)
        if hasattr(self, "input_scaling"):
            return np.concatenate(components, axis=-1) * self.input_scaling
        else: return np.concatenate(components, axis=-1)

    def pair_args(self):
        return 15, 3
