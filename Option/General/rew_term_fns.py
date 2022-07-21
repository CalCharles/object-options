from Option.General.combined import BinaryInteractionParameterizedOptionControl
from Option.General.param import BinaryParameterizedOptionControl
from Option.General.reward import RewardOptionControl
terminate_reward = {"combined": BinaryInteractionParameterizedOptionControl, "param": BinaryParameterizedOptionControl, "reward": RewardOptionControl}