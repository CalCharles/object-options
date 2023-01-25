import numpy as np

class TrueReward():
    def __init__(self, object_names):
        self.name = "true reward"
        self.names = object_names
        self.true_num_modes = 1
        self.num_modes = 1
        self.parameter_minmax = [np.array([0]), np.array([84])] # TODO: where does this come from?
        self.desired_modes = [0]
        self.changepoint_reward = 0
        self.reward_base = 0
        self.param_reward = 0
        self.extractor = None
        self.norm = None
        self.one_mode = False

    def toggle_one_mode(self, one_mode):
        # one mode uses only a single mode for reward assignment---every nonnegative assignment of mode gets reward
        pass

    def set_extractor_norm(self, extractor, norm):
        self.extractor = extractor
        self.norm = norm

    def set_params(self, reward_base, param_reward, changepoint_reward, one_mode):
        self.reward_base = 0
        self.param_reward = 0
        self.changepoint_reward = 0

    def compute_reward(self, target_diff, target_states, parent_states, dones):
        return None, None