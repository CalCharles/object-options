import numpy as np
import gymnasium as gym
from Environment.environment import Environment

class ACDomain(Environment):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False):
        '''
        Environmental future
        '''
        # environment properties
        self.self_reset = True
        self.num_actions = 2 # this must be defined, -1 for continuous. Only needed for primitive actions

        # spaces
        self.action_shape = (1,) # should be set in the environment, (1,) is for discrete action environments
        self.action_space = None # gym.spaces
        self.observation_space = None # raw space, gym.spaces
        self.pos_size = 2 # the dimensionality, should be set

        # running values
        self.itr = 0

        # factorized state properties
        # self.all_names = [] # set prior to calling super()
        # self.objects = {} # dict of name to value
        # self.binary_relations = [] # must get set prior to calling super (), the order follows the order of operations
        # self.outcome_variable = "" # the name of the variable treated as "outcome" (not in state, but used to evaluate outcomes)
        self.valid_names = self.all_names
        self.num_objects = len(self.all_names) 
        self.object_names = self.all_names 
        self.object_range = {n: (0, self.objects[n].num_values) for n in self.all_names} # set to be the number of discrete values
        self.object_sizes = {n: (1,) for n in self.all_names}
        self.object_dynamics = {self.object_range[n][1] - self.object_range[n][0] for n in self.all_names}
        self.object_range_true = self.object_range
        self.object_dynamics_true = self.object_dynamics
        self.object_instanced = {n: 1 for n in self.all_names}
        self.object_proximal = {n: False for n in self.all_names}
        self.object_name_dict = {n: n for n in self.all_names}
        self.instance_length = len(self.all_names)

        # proximity state components
        self.position_masks = {n: [1] for n in self.all_names}
        # self.all_states = np.array(np.meshgrid(*[np.arange(self.objects[n].num_values) for n in self.all_names])).T.reshape(-1,len(self.all_names))
        self.all_states, self.outcomes = self.exhaustive_evaluation()

    def return_state(self):
        return np.array([self.objects[n].attribute for n in self.all_names if n != self.outcome_variable])

    def step(self, action):
        '''
        Only takes a single step, returns Done regardless
        '''
        for relation in self.binary_relations:
            self.action.attribute = action
            relation(self.objects)
        self.done.attribute = True
        return self.get_state(), 0.0, True, dict()

    def reset(self, name_dict_assignment = None):
        for n in self.objects.keys():
            self.objects[n].attribute = np.random.randint(self.objects[n].num_values if name_dict_assignment is None else name_dict_assignment[n])
        self.done.attribute = False

    def exhaustive_evaluation(self):
        arrays = [np.arange(self.objects[n].num_values) for n in self.all_names]
        combinations = np.array(np.meshgrid(*arrays)).T.reshape(-1,len(arrays))
        states = set()
        outcomes = set()
        for combination in combinations:
            name_dict_assignment = {n: combination[i] for i,n in enumerate(self.all_names)}
            self.reset(name_dict_assignment=name_dict_assignment)
            state, reward, done, info = self.step(1)
            state = tuple(state.tolist())
            states.add(state)
            outcomes.add(self.objects[self.outcome_variable].attribute)
        return [np.array(s) for s in states], np.array(list(outcomes))



class ACObject():
    def __init__(self, name, num_values):
        self.name = name
        self.num_values = num_values
        self.attribute = np.random.randint(num_values)
        self.interaction_trace = list()
    
    def get_state(self):
        return self.attribute


class Action():
    def __init__(self, discrete, num_actions):
        self.name = "Action"
        self.discrete = discrete
        self.attribute = np.random.randint(num_actions) if discrete else (np.random.rand(num_actions) - .5) * 2
        self.num_actions = num_actions
        self.interaction_trace = list()

    def get_state(self):
        if self.discrete: return np.eye(self.num_actions)[np.array([[self.attribute]])]
        return self.attribute

    def step_state(self):
        return
