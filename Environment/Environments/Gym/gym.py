import torch
import numpy as np
import gym
# from matplotlib.pyplot import imshow
# import matplotlib as plt
from copy import deepcopy as dc
from Environments.environment_specification import RawEnvironment

class Gym(RawEnvironment): # wraps openAI gym environment
    def __init__(self, frameskip=1, gym_name=""):
        super().__init__()
        # gym wrapper specific properties
        gymenv = gym.make(gym_name)
        self.env = gymenv if self.discrete_actions else NormalizedActions(gymenv)


        # environment properties
        self.name = "ABSTRACT_BASE" # required for an environment 
        self.discrete_actions = type(gymenv.action_space) == gym.spaces.Discrete
        self.num_actions = self.env.action_space.N if self.discrete_actions else -1
        self.frameskip = frameskip

        # spaces
        self.action_space = self.env.action_space # gym.spaces
        self.action_shape = (1,) if self.discrete_actions else self.action_space.shape
        self.observation_space = self.env.observation_space

        # state components
        self.frame = self.env.observation_space.sample() # the image generated by the environment
        self.reward = 0
        self.done = False
        self.action = np.zeros(self.action_shape)
        self.extracted_state = self.dict_state(self.frame, 0, 0, self.action)

        # saving component
        self.save_module = save_module

        # factorized state properties
        self.object_names = ["Action", "State", "Reward", "Done"] # must be initialized, a list of names that controls the ordering of things
        self.object_sizes = {"Action": self.action_shape[0], "State": self.observation_space.shape[0], "Reward": 1, "Done": 1} # must be initialized, a dictionary of name to length of the state
        self.object_range = {"Action": [self.action_space.low, self.action_space.high], "State": [self.observation_space.low, self.observation_space.high], "Reward": [-1,1], "Done": [0,1]} # the minimum and maximum values for a given feature of an object
        self.instance_length = 6
        return {"State": observation, "Frame": observation, "Object": observation, "Reward": np.array([reward]), "Done": np.array([int(done)]), "Action": action}

    def seed(self, seed):
        super().seed(seed)
        self.env.seed(seed)

    def reset(self):
        self.frame = self.env.reset()
        self.extracted_state = self.dict_state(self.frame, self.reward, self.done, self.action)
        # print("resetting")
        return {"raw_state": self.frame, "factored_state": self.extracted_state}

    def step(self, action):
        self.action = action
        observation, self.reward, self.done, info = self.env.step(action)
        if len(action.shape) == 0:
            action = np.array([action])
            self.action = action
        extracted_state = self.dict_state(observation, self.reward, self.done, action)
        frame = observation
        self.extracted_state, self.frame = extracted_state, frame
        return {"raw_state": frame, "factored_state": extracted_state}, self.reward, bool(self.done), info

    def extracted_state_dict(self):
        return dc(self.extracted_state)

    def dict_state(self, observation, reward, done, action):
        return {"State": observation, "Frame": observation, "Object": observation, "Reward": np.array([reward]), "Done": np.array([int(done)]), "Action": action}

    def toString(self, extracted_state):
        names = ["Action", "State", "Frame", "Object", "Reward", "Done"]
        es = ""
        for name in names:
            es += name + ":" + " ".join(map(str, extracted_state[name])) + "\t"
        return es

    def get_trace(self, factored_state, action, object_names):
        return [1]

    def get_full_trace(self, factored_state, action, target_name):
        return np.ones(len(self.all_names))

    def current_trace(self, object_names):
        return [1]

    def get_state(self):
        return {'raw_state': self.frame, 'factored_state': self.extracted_state_dict()}

    def set_from_factored_state(self, factored_state): # TODO: implement
        return super().set_from_factored_state()

    def get_trace(factored_state, action, object_names):  # TODO: implement
        return super().get_trace()