import numpy as np
import gymnasium as gym

class Environment(gym.Env):
    def __init__(self, frameskip = 1, variant="", fixed_limits=False):
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
        self.self_reset = True
        self.num_actions = None # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "ABSTRACT_BASE" # required for an environment
        self.fixed_limits = False # uses normalization limits which are fixed across all objects
        self.discrete_actions = True
        self.frameskip = frameskip # no frameskip
        self.transpose = True # transposes the visual domain

        # spaces
        self.action_shape = (1,) # should be set in the environment, (1,) is for discrete action environments
        self.action_space = None # gym.spaces
        self.observation_space = None # raw space, gym.spaces
        self.pos_size = 2 # the dimensionality, should be set

        # state components
        self.frame = None # the image generated by the environment
        self.reward = Reward()
        self.done = Done()
        self.action = np.zeros(self.action_shape)
        self.extracted_state = None

        # running values
        self.itr = 0

        # factorized state properties
        self.all_names = [] # must be initialized, the names of all the objects including multi-instanced ones
        self.valid_names = list() # must be initialized, the names of all objects used in a particular trajectory. The environment should treat as nonexistent objects which are not part of this list
        self.num_objects = -1 # should be defined if valid, the number of objects (or virtual objects) in the flattened obs
        self.object_names = [] # must be initialized, a list of names that controls the ordering of things
        self.object_sizes = dict() # must be initialized, a dictionary of name to length of the state
        self.object_range = dict() # the minimum and maximum values for a given feature of an object
        self.object_dynamics = dict() # the most that an object can change in a single time step
        self.object_range_true = dict() # if using a fixed range, this stores the true range (for sampling)
        self.object_dynamics_true = dict() # if using a fixed dynamics range, this stores the true range (for sampling)
        self.object_instanced = dict() # name of object to max number of objects of that type
        self.object_proximal = dict() # name of object to whether that object has valid proximity
        self.object_name_dict = dict() # the string names to object classes
        self.instance_length = 0 # the total number of instances for the mask

        # proximity state components
        self.position_masks = dict()

    def step(self, action):
        '''
        self.save_path is the path to which to save files, and self.itr is the iteration number to be used for saving.
        The format of saving is: folders contain the raw state, names are numbers, contain 2000 raw states each
        obj_dumps contains the factored state
        empty string for save_path means no saving state
        matches the API of OpenAI gym by taking in action (and optional params)
        returns
            state as dict: next raw_state (image or observation) next factor_state (dictionary of name of object to tuple of object bounding box and object property)
            reward: the true reward from the environment
            done flag: if an episode ends, done is True
            info: a dict with additional info
        '''
        pass

    def reset(self):
        '''
        matches the API of OpenAI gym, resetting the environment
        returns:
            state as dict: next raw_state, next factor_state (dict with corresponding keys)
        '''
        pass

    def render(self, mode='human'):
        '''
        matches the API of OpenAI gym, rendering the environment
        returns None for human mode
        '''

    def close(self):
        '''
        closes and performs cleanup
        '''

    def seed(self, seed):
        '''
        numpy should be the only source of randomness, but override if there are more
        '''
        if seed < 0: seed = np.random.randint(10000)
        print("env seed", seed)
        np.random.seed(seed)


    def get_state(self):
        '''
        Takes in an action and returns:
            dictionary with keys:
                raw_state (dictionary of name of object to raw state)
                factor_state (dictionary of name of object to tuple of object bounding box and object property)
        '''
        pass

    def get_info(self): # returns the info, the most important value is TimeLimit.truncated, can be overriden
        return {"TimeLimit.truncated": False}

    def get_itr(self):
        return self.itr

    def run(self, policy, iterations = 10000):
        
        full_state = self.get_state()
        for self.itr in range(iterations):
            action = policy.act(full_state)
            if action == -1: # signal to quit
                break
            full_state = self.step(action)

    def set_from_factored_state(self, factored_state):
        '''
        from the factored state, sets the environment.
        If the factored state is not complete, then this function should do as good a reconstruction as possible
        might not be implemented for every environment
        '''
        pass

    def current_trace(self, names):
        targets = [self.object_name_dict[names.target]] if type(self.object_name_dict[names.target]) != list else self.object_name_dict[names.target]
        traces = list()
        for target in targets:
            if self.object_name_dict[names.primary_parent].name in target.interaction_trace:
                traces.append(1)
            else:
                traces.append(0)
        return traces

    def get_trace(self, factored_state, action, names):
        # gets the trace for a factored state, using the screen. If we don't want to screen to change, use a dummy screen here
        self.set_from_factored_state(factored_state)
        self.step(action)
        return self.current_trace(names)

    def get_full_current_trace(self):
        traces = dict()
        all_inter_names = [n for n in self.all_names if n not in {"Reward", "Done"}]
        for target in self.all_names:
            # if self.can_interact[target]:
            # traces[target] = np.zeros(len(all_inter_names)).tolist()
            target_traces = np.array([int((val in self.object_name_dict[target].interaction_trace) # a different name is in the trace
                                             or (val == target) # add self interactions
                                             ) for val in all_inter_names])
            traces[target] = target_traces
        return traces

    def get_full_trace(self, factored_state, action, outcome_variable=""):
        self.set_from_factored_state(factored_state)
        # print("stepping", factored_state["Ball"], factored_state["Block0"], factored_state["Block1"], factored_state["Block2"])
        self.step(action)
        factored_state = self.get_state()['factored_state']
        # print("stepped", factored_state["Ball"], factored_state["Block0"], factored_state["Block1"], factored_state["Block2"])
        all_inter_names = [n for n in self.all_names if n not in {"Reward", "Done"}]
        traces = self.get_full_current_trace()
        return traces

    def demonstrate(self):
        '''
        gives an image and gets a keystroke action
        '''
        return 0

    def toString(self, extracted_state):
        '''
        converts an extracted state into a string for printing. Note this might be overriden since self.objects is not a guaranteed attribute
        '''
        estring = "ITR:" + str(self.itr) + "\t"
        for i, obj in enumerate(self.objects):
            estring += obj.name + ":" + " ".join(map(str, extracted_state[obj.name])) + "\t" # TODO: attributes are limited to single floats
        if "VALID_NAMES" in extracted_state: # TODO: stores valid names in the factored state for now
            estring += "VALID_NAMES:" + " ".join(map(str, extracted_state['VALID_NAMES'])) + "\t"
        if "TRACE" in extracted_state: # TODO: stores valid names in the factored state for now
            estring += "TRACE:" + " ".join(map(str, extracted_state['TRACE'])) + "\t"
        # estring += "Reward:" + str(float(extracted_state["Reward"])) + "\t"
        # estring += "Done:" + str(int(extracted_state["Done"])) + "\t"
        return estring

    def valid_binary(self, valid_names):
        return np.array([(1 if n in valid_names else 0) for n in self.all_names])

    def name_indices(self, names):
        indices = list()
        for n in names:
            indices.append(self.all_names.find(n))
        return indices

class Done():
    def __init__(self):
        self.name = "Done"
        self.attribute = False
        self.interaction_trace = list()

    def get_state (self):
        return np.array([self.attribute])
    
    def interact (self, other):
        self.interaction_trace.append(other.name)

class Reward():
    def __init__(self):
        self.name = "Reward"
        self.attribute = 0.0
        self.interaction_trace = list()

    def get_state (self):
        return np.array([self.attribute])

    def interact (self, other):
        self.interaction_trace.append(other.name)

class Action():
    def __init__(self,discrete_actions, action_shape):
        self.name = "Action"
        self.discrete_actions = discrete_actions
        self.attr = np.array(0) if self.discrete_actions else np.zeros(action_shape)
        self.interaction_trace = list()

    def get_state(self):
        return np.array(self.attr).astype(float)
