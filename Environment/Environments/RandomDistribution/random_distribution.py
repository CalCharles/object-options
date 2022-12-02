# Screen
import sys, cv2
import numpy as np
from Environment.environment import Environment, Done, Reward
import imageio as imio
import os, copy, string
from Environment.Environments.RandomDistribution.random_specs import *
from Record.file_management import numpy_factored
from gym import spaces

def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = ''.join(letters[np.random.randint(len(letters))] for i in range(length))
    return result_str

class Action():
    def __init__(self, discrete, num_actions):
        self.name = "Action"
        self.discrete = discrete
        self.attribute = np.random.randint(num_actions) if discrete else (np.random.rand(num_actions) - .5) * 2
        self.num_actions = num_actions
        self.interaction_trace = list()

    def get_state(self):
        return np.array([self.attribute]) if self.discrete else self.attribute

    def step_state(self):
        return

class conditional_add_func():
    def __init__(self, parents, target, parent_size, target_size, use_target_bias=False, num_sets=10, conditional=False, conditional_weight=0, passive=None):
        self.parents = parents
        self.target = target
        self.target_size = target_size
        self.interaction_dynamics = rel_func(parents, target, parent_size, target_size, use_target_bias=False, num_sets=2, conditional=True, conditional_weight=conditional_weight) 
        self.add_dynamics = add_func(parents, target, parent_size, target_size, use_target_bias=True, num_sets=5, conditional=False, scale=2)
        self.passive = passive
        self.params = self.interaction_dynamics.params + self.add_dynamics.params

    def __call__(self, ps, ts):
        inter, _ = self.interaction_dynamics(ps, ts)
        print("target, inter", self.target, inter)
        if inter:
            return inter, self.add_dynamics(ps, ts)[1]
        else:
            if self.passive is None: # if we have a passive function, it replaces the interaction dynamics on non-interaction
                return False, np.zeros(self.target_size)
            else:
                return False, self.passive(ps, ts)[1]

class passive_func():
    def __init__(self, target, target_size, use_target_bias):
        self.parents = list()
        self.target = target
        self.target_bias = np.expand_dims(2 * (np.random.rand(target_size) - .5) * DYNAMICS_STEP / TARGET_REDUCE_FACTOR, axis=-1)
        self.target_matrix = 2 * (np.random.rand(target_size, target_size) - .5) * DYNAMICS_STEP / TARGET_REDUCE_FACTOR
        self.params = [self.target_bias, self.target_matrix]

    def __call__(self, ps, ts):
        return True, (np.matmul(self.target_matrix, np.expand_dims(ts,-1))[...,0] + self.target_bias)[...,0]

class add_func():
    def __init__(self, parents, target, parent_size, target_size, use_target_bias, num_sets=10, conditional=False, conditional_weight = 0, scale=1):
        self.parents = parents
        self.target = target
        self.scale = scale
        dstep = 1 if conditional else DYNAMICS_STEP
        prf = 1 if conditional else PARENT_REDUCE_FACTOR
        trf = 1 if conditional else TARGET_REDUCE_FACTOR
        self.parent_weight_sets = [2 * (np.random.rand(parent_size) - .5) * dstep / np.sqrt(num_sets + parent_size) * prf for k in range(num_sets)] # right now just creates 10 no matter what
        self.parent_bias = np.expand_dims(2 * (np.random.rand(parent_size) - .5) * dstep / np.sqrt(num_sets + parent_size) * prf, axis=-1)
        self.target_bias = np.expand_dims(2 * (np.random.rand(target_size) - .5) * dstep / np.sqrt(num_sets + target_size) * trf, axis=-1)
        self.target_weights = np.expand_dims(2 * (np.random.rand(target_size) - .5) * dstep / np.sqrt(num_sets + target_size) * trf, axis=-1)
        self.parent_weight_matrix = np.stack([np.zeros(len(self.parent_weight_sets[0])) for _ in range(len(self.target_weights))], axis=0)
        self.conditional = False
        self.conditional_weight = conditional_weight
        for ws in self.parent_weight_sets:
            for i, w in enumerate(ws):
                self.parent_weight_matrix[np.random.randint(len(self.target_weights))][i] += w
        self.params = [self.parent_weight_matrix, self.parent_bias, self.target_bias, self.target_weights]

    def __call__(self, ps, ts):
        ps = np.expand_dims(ps, -1)
        ts = np.expand_dims(ts, -1)
        # print(self.parent_weight_matrix.shape, ps.shape, self.parent_bias.shape, self.target_weights.shape, self.target_bias.shape, ts.shape)
        # print (self.parent_weight_matrix)
        # print(np.matmul(self.parent_weight_matrix, ps - self.parent_bias).shape, ps - self.parent_bias, (self.target_weights * ts).shape, self.target_weights.shape)
        # print((np.matmul(self.parent_weight_matrix, ps - self.parent_bias), self.target_weights * ts, self.target_bias))
        # sum_val = np.matmul(self.parent_weight_matrix, ps - self.parent_bias).squeeze()
        sum_val = (np.matmul(self.parent_weight_matrix, ps - self.parent_bias) * self.scale + self.target_weights * ts + self.target_bias)[...,0]
        if self.conditional:
            return np.sum(sum_val, axis=-1) > self.conditional_weight, None
        print(ts, "sumval", sum_val, "parent effect", np.matmul(self.parent_weight_matrix, ps - self.parent_bias) * self.scale, "taret_effect", self.target_weights * ts + self.target_bias)
        return True, sum_val

class rel_func():
    def __init__(self, parents, target, parent_size, target_size, use_target_bias, num_sets=10, conditional=False, conditional_weight=0):
        self.parents = parents
        self.target = target
        if conditional:
            self.parent_weight_sets = [2 * (np.random.rand(parent_size) - .5) / np.sqrt(num_sets + parent_size) for k in range(num_sets)] # right now just creates 10 no matter what
            self.target_weight_sets = [2 * (np.random.rand(target_size) - .5) / np.sqrt(num_sets + target_size) for k in range(num_sets)] # right now just creates 10 no matter what
        else:
            self.parent_weight_sets = [2 * (np.random.rand(parent_size) - .5) * DYNAMICS_STEP / np.sqrt(num_sets + parent_size) for k in range(num_sets)] # right now just creates 10 no matter what
            self.target_weight_sets = [2 * (np.random.rand(target_size) - .5) * DYNAMICS_STEP / np.sqrt(num_sets + parent_size) for k in range(num_sets)] # right now just creates 10 no matter what
        self.parent_bias = np.expand_dims(2 * (np.random.rand(parent_size) - .5) * DYNAMICS_STEP / np.sqrt(num_sets + target_size), axis=-1)
        self.conditional = conditional
        self.conditional_weight = conditional_weight

        self.parent_weight_matrix = np.stack([np.zeros(parent_size) for _ in range(target_size)], axis=0)
        for ws in self.parent_weight_sets:
            for i, w in enumerate(ws):
                self.parent_weight_matrix[np.random.randint(target_size)][i] += w

        self.target_weight_matrix = np.stack([np.zeros(target_size) for _ in range(parent_size)], axis=0)
        for ws in self.target_weight_sets:
            for i, w in enumerate(ws):
                self.target_weight_matrix[np.random.randint(parent_size)][i] += w
        self.params = [self.parent_weight_matrix, self.parent_bias, self.target_weight_matrix]


    def __call__(self, ps, ts):
        ps = np.expand_dims(ps, -1)
        ts = np.expand_dims(ts, -1)
        # print(ts, ps, np.matmul(self.target_weight_matrix, ts),self.parent_bias, np.matmul(self.parent_weight_matrix, ps - np.matmul(self.target_weight_matrix, ts) - self.parent_bias))
        rel_val = np.matmul(self.parent_weight_matrix, ps - np.matmul(self.target_weight_matrix, ts) - self.parent_bias)[...,0]
        if self.conditional:
            print("cond weight", np.sum(rel_val, axis=-1))
            return np.sum(rel_val, axis=-1) > self.conditional_weight, None
        return True, rel_val


object_relational_functions = ["add", "func", "rel", "const", "rotation"]
DYNAMICS_STEP = 0.02
OBJECT_MAX_DIM = 4
PARENT_REDUCE_FACTOR = 1.5
# PARENT_REDUCE_FACTOR = 1
TARGET_REDUCE_FACTOR = 0.5

class RandomDistObject():
    def __init__(self, name, state, lims):
        self.name = name
        self.state = state
        self.next_state = state
        self.lims = lims
        self.interaction_trace = list()

    def get_state(self):
        return self.state

    def step_state(self):
        self.state = np.clip(self.next_state, self.lims[0], self.lims[1])
        # print(self.next_state, self.lims[0], self.lims[1])

class RandomDistribution(Environment):
    def __init__(self, frameskip = 1, variant="default", fixed_limits=False):
        # generates "objects" as conditional distributions of each other
        self.variant = variant
        self.fixed_limits = fixed_limits
        self.discrete_actions, self.allow_uncontrollable, self.num_objects, self.multi_instanced, self.num_related, self.relate_dynamics, self.conditional, self.conditional_weight, self.distribution, self.noise_percentage, self.require_passive = variants[self.variant]
        
        self.set_objects()
        self.num_actions = self.discrete_actions # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "RandomDistribution" # required for an environment 
        self.discrete_actions = self.discrete_actions > 1
        self.frameskip = frameskip # no frameskip
        self.transpose = False # transposes the visual domain

        # spaces
        self.action_shape = (1,) if self.discrete_actions else (self.object_sizes["Action"], ) # should be set in the environment, (1,) is for discrete action environments
        self.action_space = spaces.Discrete(self.num_actions) if self.discrete_actions else spaces.Box(low=np.ones(self.object_sizes["Action"]) * -1, high=np.ones(self.object_sizes["Action"])) # gym.spaces
        self.observation_space = spaces.Box(low=np.concatenate([self.object_range[name][0] for name in self.object_names], axis=-1),
                                            high=np.concatenate([self.object_range[name][1] for name in self.object_names], axis=-1)) # raw space, gym.spaces
        self.pos_size = 1 # the dimensionality of position, set to 1 to allow more relationships

        # state components
        self.frame = None # the image generated by the environment
        self.reward = Reward()
        self.done = Done()
        self.action = Action(self.discrete_actions, self.num_actions if self.discrete_actions else self.object_sizes["Action"])

        # running values
        self.itr = 0


        # proximity state components
        self.position_masks = dict()

        self.extracted_state = self.reset()

    def set_objects(self): # creates objects based on their dictionary, and their relational connectivity
        # factorized state properties
        self.object_names = ["Action"] + [get_random_string(np.random.randint(7) + 3) for i in range(self.num_objects)] + ["Reward", "Done"] # must be initialized, a list of names that controls the ordering of things
        self.object_instanced = {name: np.random.randint(1, self.multi_instanced + 1) for name in self.object_names} # name of object to max number of objects of that type
        self.object_instanced["Action"], self.object_instanced["Reward"], self.object_instanced["Done"] = 1, 1, 1
        make_name = lambda x: x + str(i) if self.object_instanced[x] > 1 else x
        self.all_names = sum([[make_name(name) for i in range(self.object_instanced[name])] for name in self.object_names], start = list()) # must be initialized, the names of all the objects including multi-instanced ones
        self.object_sizes = {name: np.random.randint(2,OBJECT_MAX_DIM+1) for name in self.object_names} # must be initialized, a dictionary of name to length of the state
        self.object_sizes["Reward"], self.object_sizes["Done"] = 1,1
        self.object_range = {n: (- np.ones(self.object_sizes[n]), np.ones(self.object_sizes[n])) for n in self.object_names}
        self.object_mean = {n: (self.object_range[n][0] + self.object_range[n][1]) / 2 for n in self.object_names}
        self.object_var = {n: (self.object_range[n][1] - self.object_range[n][0]) for n in self.object_names}
        self.object_dynamics = {n: (np.ones(self.object_sizes[n])*-DYNAMICS_STEP, np.ones(self.object_sizes[n])*DYNAMICS_STEP) for n in self.object_names}
        self.object_proximal = {n: True for n in self.object_names} # name of object to whether that object has valid proximity
        self.object_proximal["Action"], self.object_proximal["Reward"], self.object_proximal["Done"] = True, True, True
        self.instance_length = len(self.all_names) # the total number of instances for the mask
        self.object_range_true = self.object_range
        self.object_dynamics_true = self.object_dynamics

        onames = self.object_names[:-2]
        nonames = self.object_names[1:-2]
        used = list()
        unused = [name for name in self.object_names[1:-2]]
        controllable = ["Action"]
        self.object_relational_sets, self.object_relational_functions = list(), list()
        print(self.object_sizes, self.object_instanced)
        
        self.passive_functions = dict()
        for name in self.object_names[1:-2]: # not actions or done/reward
            if self.require_passive:
                self.passive_functions[name] = passive_func(name, self.object_sizes[name], use_target_bias=True)
            else:
                self.passive_functions[name] = None # create a placeholder
        for i in range(self.num_related): # create relational links
            if self.allow_uncontrollable: parents = np.random.choice(onames, size=min(len(nonames) - 1, np.random.randint(1, 4)), replace = False).tolist()
            else:
                ronames = copy.deepcopy(onames)
                if i > 0: 
                    if "Action" in controllable: controllable.remove("Action")
                    if "Action" in ronames: ronames.remove("Action")
                ctrl_choice = np.random.choice(controllable)
                ronames.remove(ctrl_choice)
                print("ctrl options", ctrl_choice, controllable, ronames)
                parents = [ctrl_choice] + np.random.choice(ronames, size=min(len(nonames) - 1, np.random.randint(2)), replace = False).tolist()
            target = nonames[np.random.randint(len(nonames))]
            while target in parents or target in used:
                target = nonames[np.random.randint(len(nonames))]
            if not (self.relate_dynamics and self.conditional): used.append(target)
            if target in unused: unused.remove(target)
            if target not in controllable: controllable.append(target)
            self.object_relational_sets.append((parents, target))
            parent_size = np.sum([self.object_sizes[p] for p in parents])
            if self.conditional and i != 0:
                orf = conditional_add_func(parents,
                            target,
                            parent_size,
                            self.object_sizes[target],
                            use_target_bias = False,
                            conditional=True,
                            conditional_weight=self.conditional_weight,
                            passive=self.passive_functions[target],
                            )
            else:
                orf = add_func(parents,
                            target,
                            parent_size,
                            self.object_sizes[target],
                            use_target_bias = True,
                            conditional=False)
            print(orf.parents, orf.target, orf.params)
            self.object_relational_functions.append(orf)
        for target in unused:
            if self.require_passive:
                self.object_relational_functions.append(self.passive_functions[target])
        # error

    def reset(self):
        self.object_name_dict= dict()
        self.objects = list()
        for n in self.object_names:
            if n == "Action":
                self.object_name_dict["Action"] = Action(self.discrete_actions, self.num_actions if self.discrete_actions else self.object_sizes["Action"])
                self.objects.append(self.object_name_dict["Action"])
                self.action = self.object_name_dict["Action"]
            elif n == "Done":
                self.object_name_dict["Done"] = Done()
                self.objects.append(self.object_name_dict["Done"])
                self.done = self.object_name_dict["Done"]
            elif n == "Reward":
                self.object_name_dict["Reward"] = Reward()
                self.objects.append(self.object_name_dict["Reward"])
                self.reward = self.object_name_dict["Reward"]
            else:
                if self.object_instanced[n] > 1:
                    for i in range(self.object_instanced[n]):
                        next_obj = RandomDistObject(n + str(i), np.random.rand(self.object_sizes[n]) * self.object_var[n] + self.object_range[n][0])
                        self.object_name_dict[n + str(i)] = next_obj
                        self.objects.append(next_obj)
                else:
                    next_obj = RandomDistObject(n, np.random.rand(self.object_sizes[n]) * self.object_var[n] + self.object_range[n][0], self.object_range[n])
                    self.object_name_dict[n] = next_obj
                    self.objects.append(next_obj)
        return self.get_state()

    def get_state(self):
        state = dict()
        for n in self.object_names:
            if self.object_instanced[n] > 1:
                for i in range(self.object_instanced[n]):
                    state[n] = self.object_name_dict[n + str(i)].get_state()
            else:
                state[n] = self.object_name_dict[n].get_state()
        # print({n: self.object_name_dict[n].get_state() for n in self.object_names})
        return {"raw_state": None, "factored_state": {n: self.object_name_dict[n].get_state() for n in self.object_names}}

    def get_named_state(self, names):
        # print(names, [([n] if self.object_instanced[n] <= 1 else [n + str(i) for i in range(self.object_instanced[n])]) for n in names])
        instanced_names = sum([([n] if self.object_instanced[n] == 1 else [n + str(i) for i in range(self.object_instanced[n])]) for n in names], start=list())
        # print(names, self.object_name_dict, instanced_names)
        # print([self.object_name_dict[n].get_state() for n in instanced_names])
        return np.concatenate([self.object_name_dict[n].get_state() for n in instanced_names], axis=-1)

    def empty_interactions(self):
        for obj in self.objects:
            obj.interaction_trace = list()

    def step(self, action, render=False): 
        self.empty_interactions()
        for i in range(self.frameskip):
            self.done.attribute = False
            self.action.attribute = action
            for target in self.object_names:
                self.object_name_dict[target].next_state = self.object_name_dict[target].get_state()
            for i, orf in enumerate(self.object_relational_functions):
                # print("orf", orf.parents, orf.target)
                ps = self.get_named_state(orf.parents) if len(orf.parents) > 0 else None
                ts = self.get_named_state([orf.target])
                inter, nds = orf(ps, ts)
                if inter:
                    self.object_name_dict[orf.target].interaction_trace += orf.parents 
                # print(orf.target, nds)
                if self.relate_dynamics: 
                    self.object_name_dict[orf.target].next_state += np.clip(nds, -DYNAMICS_STEP, DYNAMICS_STEP)
                else:
                    self.object_name_dict[orf.target].next_state = nds
                # if i < 3: print(self.itr, self.done.attribute, orf.parents, orf.target, self.get_state()["factored_state"][orf.target])
            for obj in self.object_name_dict.values():
                # print("adding noise", obj.next_state)
                # if self.noise_percentage > 0: # TODO: it appears taking random actions is correlated with the random noise, so we removed this impl
                #     if self.distribution == "Gaussian":
                #         obj.next_state = obj.next_state + np.random.normal(scale=self.noise_percentage, size=obj.next_state.shape)
                    # print(obj.name, obj.next_state)
                if hasattr(obj, "step_state"): obj.step_state()
        self.itr += 1
        print(self.get_state()["factored_state"])
        if self.itr % 50 == 0:
            self.reset()
            self.done.attribute = True
            return self.get_state(), self.reward.attribute, self.done.attribute, {'Timelimit.truncated': True}
        return self.get_state(), self.reward.attribute, self.done.attribute, {'Timelimit.truncated': False}

    def set_from_factored_state(self, factored_state, seed_counter=-1, render=False):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        if seed_counter > 0:
            self.seed_counter = seed_counter
        for n in factored_state.keys():
            if n in self.object_name_dict:
                self.object_name_dict[n].state = copy.deepcopy(factored_state[n])
