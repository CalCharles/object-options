import numpy as np
import gymnasium as gym

class GymnasiumWrapper(gym.Env):
    def __init__(self, gym_environment):
        ''' required attributes:
            num actions: int or None
            action_space: gym.Spaces
            action_shape: tuple of ints
            observation_space = gym.Spaces
            done: boolean
            reward: int
            seed_counter: int
            discrete_actions: boolean
            name: string
        All the below properties are set by the subclass
        '''
        # environment properties
        self.gym = gym_environment
        self.self_reset = self.gym.self_reset
        self.num_actions = self.gym.num_actions
        self.name = self.gym.name
        self.fixed_limits = self.gym.fixed_limits
        self.discrete_actions = self.gym.discrete_actions
        self.frameskip = self.gym.frameskip
        self.transpose = self.gym.transpose

        # spaces
        self.action_shape = self.gym.action_shape
        self.action_space = self.gym.action_space
        self.observation_space = self.gym.observation_space
        self.pos_size = self.gym.pos_size

        # state components
        self.frame = self.gym.frame
        self.reward = self.gym.reward
        self.done = self.gym.done
        self.action = self.gym.action
        self.extracted_state = self.gym.extracted_state

        # running values
        self.itr = self.gym.itr

        # factorized state properties
        self.all_names = self.gym.all_names
        self.object_names = self.gym.object_names
        self.object_sizes = self.gym.object_sizes
        self.object_range = self.gym.object_range
        self.object_dynamics = self.gym.object_dynamics
        self.object_range_true = self.gym.object_range_true
        self.object_dynamics_true = self.gym.object_dynamics_true
        self.object_instanced = self.gym.object_instanced
        self.object_proximal = self.gym.object_proximal
        self.object_name_dict = self.gym.object_name_dict
        self.instance_length = self.gym.instance_length

        # proximity state components
        self.position_masks = self.gym.position_masks

    def step(self, action):
        obs, rew, done, info= self.gym.step(action)
        return obs, rew, done, False if "Timelimit.truncated" not in info else info["Timelimit.truncated"], info

    def reset(self):
        return self.gym.reset(), self.get_info() # returns a dummy info on reset

    def render(self, mode='human'):
        return self.gym.render(mode=mode)

    def close(self):
        return self.gym.close()

    def seed(self, seed):
        return self.gym.seed(seed)


    def get_state(self):
        return self.gym.get_state()

    def get_info(self): # returns the info, the most important value is TimeLimit.truncated, can be overriden
        return self.gym.get_info()

    def get_itr(self):
        return self.gym.get_itr()

    def run(self, policy, iterations = 10000):
        self.gym.run(policy, iterations=iterations)

    def set_from_factored_state(self, factored_state):
        return self.gym.set_from_factored_state(factored_state)

    def current_trace(self, names):
        return self.gym.current_trace(names)

    def get_trace(self, factored_state, action, names):
        return self.gym.get_trace(factored_state, action, names)

    def get_full_trace(self, factored_state, action):
        return self.gym.get_full_trace(factored_state, action)

    def demonstrate(self):
        return self.gym.demonstrate()

    def toString(self, extracted_state):
        return self.gym.toString(extracted_state)
