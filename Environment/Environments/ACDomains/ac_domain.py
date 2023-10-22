import numpy as np
import gymnasium as gym
from Environment.environment import Environment
import copy

class ACDomain(Environment):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False, cf_states=False):
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
        # self.relation_outcome = [] # the outcome variable of a binary relation (binary relations should only affect one variable)
        # self.outcome_variable = "" # the name of the variable treated as "outcome" (not in state, but used to evaluate outcomes)
        self.valid_names = self.all_names
        self.num_objects = len(self.all_names) - 1 # don't include the outcome object
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
        self.all_states, self.outcomes = self.exhaustive_evaluation(counterfactuals=cf_states)
        self.use_zero = True # if true, allows exhaustive EM to use the zero mask
        print(self.all_names, self.outcome_variable)

    def return_state(self):
        return np.array([self.objects[n].attribute for n in self.all_names if n != self.outcome_variable])

    def step(self, action, frozen_relations=None):
        '''
        Only takes a single step, returns Done regardless
        '''
        for out_var, relation in zip(self.relation_outcome, self.binary_relations):
            if frozen_relations is None or out_var not in frozen_relations:
                relation(self.objects)
        return self.return_state(), 0.0, True, dict()

    def reset(self, name_dict_assignment = None):
        
        if name_dict_assignment is None:
            for n in self.objects.keys():
                self.objects[n].attribute = np.random.randint(self.objects[n].num_values) 
        else:
            for n in name_dict_assignment.keys():
                self.objects[n].attribute = name_dict_assignment[n]

    def _get_counterfactuals(self, check_names, frozen_relations=None):
        if len(check_names) == 0:
            return dict(), 0
        arrays = [np.arange(self.objects[n].num_values) for n in check_names]
        combinations = np.array(np.meshgrid(*arrays)).T.reshape(-1,len(arrays))
        sos = dict()
        for combination in combinations:
            # assign state to the counterfactual combination
            name_dict_assignment = {n: combination[i] for i,n in enumerate(check_names)}
            self.reset(name_dict_assignment=name_dict_assignment)
            # step the evironment to reassign observational values, excepting frozen relations
            state, reward, done, info = self.step(1, frozen_relations=frozen_relations)
            state = tuple(state.tolist())
            # by construction, there should never be a double mapping
            sos[state] = self.objects[self.outcome_variable].attribute
            print(name_dict_assignment, sos[state], frozen_relations)
        return sos, len(combinations)

    def exhaustive_evaluation(self, counterfactuals=False):
        # gets all possible state-outcome pairs in the environment
        frozen_relations = None
        if counterfactuals:
            frozen_relations = copy.deepcopy(self.all_names)
            frozen_relations.pop(frozen_relations.index(self.outcome_variable))
        print(frozen_relations, counterfactuals)
        sos, cost = self._get_counterfactuals(self.all_names, frozen_relations = frozen_relations)
        states = [np.array(s) for s in sos.keys()] # using keys() means the order might change
        outcomes = [np.array(sos[s]) for s in sos.keys()]
        return states, outcomes

    def set_state(self, state):
        name_dict_assignment = {self.all_names[i]: s for i,s in enumerate(state)}
        name_dict_assignment[self.outcome_variable] = 0 # outcome variable not in state, must be assigned, will be assigned upon reset to 0
        self.reset(name_dict_assignment=name_dict_assignment)


    def evaluate_split_counterfactuals(self, binary, state, state_outcome):
        # takes in a state-binary pair and determines the magnitude the 1s in the binary split the state
        # and the magnitude the 0s in the binary split the state
        # binary is a length object_num binary vector
        # state is a length object_num discrete numpy array
        one_indices = np.nonzero(binary)[0]
        zero_indices = np.nonzero(1-binary)[0]
        one_check_names = [self.all_names[i] for i in one_indices]
        zero_check_names = [self.all_names[i] for i in zero_indices]
        
        state_tuple = tuple(state.tolist())
        # set the environment to the current state, we only need to do this once because any state
        # changed from getting the counterfactuals would be assigned for EVERY counterfactual
        # either by observational value, or by the counterfactual
        print(binary, state)
        def _evaluate_split(names, default=1):
            self.set_state(state)

            # get all the counterfactuals 
            # counterfactuals, cost = self._get_counterfactuals(names, frozen_relations=one_check_names if default == 1 else zero_check_names)
            counterfactuals, cost = self._get_counterfactuals(names, frozen_relations=one_check_names if default == 1 else zero_check_names)
            if len(counterfactuals) == 0: # the zero mask
                return default, 0
            print(counterfactuals, one_check_names, state)
            # state_outcome = counterfactuals[state_tuple]
            state_diff = np.sum([1 for outcome in counterfactuals.values() if outcome != state_outcome]).astype(float)
            return state_diff / len(counterfactuals), cost
        
        one_split_diff, onecost = _evaluate_split(one_check_names, default=1)
        zero_split_diff, zerocost = _evaluate_split(zero_check_names, default=0)
        return one_split_diff, zero_split_diff, onecost + zerocost





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
