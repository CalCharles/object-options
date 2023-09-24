# Screen
import sys, cv2, collections
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
        if self.discrete: return np.eye(self.num_actions)[np.array([[self.attribute]])]
        return self.attribute

    def step_state(self):
        return

object_relational_functions = ["add", "func", "rel", "const", "rotation"]
DYNAMICS_STEP = 0.01
DYNAMICS_CLIP = 0.02
OBJECT_MAX_DIM = 4
PARENT_REDUCE_FACTOR = 1.5
# PARENT_REDUCE_FACTOR = 1
TARGET_REDUCE_FACTOR = 1

class conditional_add_func():
    def __init__(self, parents, target, parent_size, target_size, use_bias=False, target_dependent=True, num_sets=1, conditional=False, conditional_weight=0, passive=None, dynamics_step =DYNAMICS_STEP):
        self.parents = parents
        self.target = target
        self.target_size = target_size
        self.use_bias = use_bias
        self.interaction_dynamics = rel_func(parents, target, parent_size, target_size, use_bias=False, target_dependent=target_dependent, num_sets=num_sets, conditional=True, conditional_weight=conditional_weight) 
        self.add_dynamics = add_func(parents, target, parent_size, target_size, use_bias=self.use_bias, dynamics_step=dynamics_step, target_dependent=target_dependent, num_sets=num_sets, conditional=False, scale=2)
        self.passive = passive
        self.params = self.interaction_dynamics.params + self.add_dynamics.params

    def __call__(self, ps, ts, require_passive=False):
        
        # print("target, inter", self.target, inter)
        if not require_passive:
            inter, _ = self.interaction_dynamics(ps, ts)
            if inter:
                return inter, self.add_dynamics(ps, ts)[1]
        if self.passive is None: # if we have a passive function, it replaces the interaction dynamics on non-interaction
            return False, np.zeros(self.target_size)
        else:
            return False, self.passive(ps, ts)[1]

class passive_func():
    def __init__(self, target, target_size, use_target_bias, dynamics_step= DYNAMICS_STEP, target_reduce=TARGET_REDUCE_FACTOR):
        self.parents = [target]
        self.target = target
        self.use_target_bias = use_target_bias
        self.target_bias = np.expand_dims(2 * (np.random.rand(target_size) - .5) * dynamics_step / target_reduce / np.sqrt(target_size), axis=-1)
        # self.target_bias = np.ones(target_size) * DYNAMICS_STEP
        self.target_matrix = 2 * (np.random.rand(target_size, target_size) - .5) * dynamics_step / target_reduce / np.sqrt(target_size)
        self.params = [self.target_bias, self.target_matrix]

    def __call__(self, ps, ts, require_passive=False):
        return True, (np.matmul(self.target_matrix, np.expand_dims(ts,-1))[...,0] + self.target_bias * float(self.use_target_bias))[...,0]

class add_func():
    def __init__(self, parents, target, parent_size, target_size, use_bias, target_dependent =True, num_sets=1, conditional=False, conditional_weight = 0, scale=1, passive=None, dynamics_step= DYNAMICS_STEP, target_reduce=TARGET_REDUCE_FACTOR, parent_reduce = PARENT_REDUCE_FACTOR):
        self.parents = parents
        self.target = target
        self.scale = scale
        self.use_bias = use_bias
        self.target_dependent = target_dependent
        dstep = 1 if conditional else dynamics_step
        prf = 1 if conditional else parent_reduce
        trf = 1 if conditional else target_reduce
        # self.parent_weight_sets = [2 * (np.random.rand(parent_size) - .5) * dstep / np.sqrt(num_sets + parent_size) * prf for k in range(num_sets)]
        self.parent_bias = np.expand_dims(2 * (np.random.rand(parent_size) - .5) * dstep / np.sqrt(num_sets + parent_size) * prf, axis=-1) * float(self.use_bias)
        self.target_bias = np.expand_dims(2 * (np.random.rand(target_size) - .5) * dstep / np.sqrt(num_sets + target_size) * trf, axis=-1) * float(self.use_bias)
        # self.target_weights = np.expand_dims(2 * (np.random.rand(target_size) - .5) * dstep / np.sqrt(num_sets + target_size) * trf, axis=-1)
        self.parent_weight_matrix = 2 * (np.random.rand(target_size, parent_size)-0.5)  * dstep / np.sqrt(parent_size) * prf
        self.target_weight_matrix = 2 * (np.random.rand(target_size, target_size)-0.5)  * dstep / np.sqrt(target_size) * trf
        self.conditional = False
        self.conditional_weight = conditional_weight
        # for ws in self.parent_weight_sets:
        #     for i, w in enumerate(ws):
        #         self.parent_weight_matrix[np.random.randint(len(self.target_weights))][i] += w
        self.params = [self.parent_weight_matrix, self.parent_bias, self.target_bias, self.target_weight_matrix]
        self.passive = passive

    def __call__(self, ps, ts, require_passive=False):
        # print(ps, ts)
        if require_passive:
            return False, self.passive(ps, ts)[1]
        ps = np.expand_dims(ps, -1)
        ts = np.expand_dims(ts, -1)
        # print(self.parent_weight_matrix.shape, ps.shape, self.parent_bias.shape, self.target_weights.shape, self.target_bias.shape, ts.shape)
        # print (self.parent_weight_matrix)
        # print(np.matmul(self.parent_weight_matrix, ps - self.parent_bias).shape, ps - self.parent_bias, (self.target_weights * ts).shape, self.target_weights.shape)
        # print((np.matmul(self.parent_weight_matrix, ps - self.parent_bias), self.target_weights * ts, self.target_bias))
        # sum_val = np.matmul(self.parent_weight_matrix, ps - self.parent_bias).squeeze()
        sum_val = (np.matmul(self.parent_weight_matrix, ps - self.parent_bias * float(self.use_bias)) * self.scale + (np.matmul(self.target_weight_matrix, ts) - self.target_bias * float(self.use_bias)) * float(self.target_dependent))[...,0]
        if self.conditional:
            return np.sum(sum_val, axis=-1) > self.conditional_weight, None
        # print(ts, "sumval", sum_val, "parent effect", np.matmul(self.parent_weight_matrix, ps - self.parent_bias) * self.scale, "taret_effect", self.target_weights * ts + self.target_bias)
        return True, sum_val

class rel_func():
    def __init__(self, parents, target, parent_size, target_size, use_bias, target_dependent=False, num_sets=1, conditional=False, conditional_weight=0):
        self.parents = parents
        self.target = target
        self.use_bias = use_bias
        # if conditional:
        #     self.parent_weight_sets = [2 * (np.random.rand(parent_size) - .5) / np.sqrt(num_sets + parent_size) for k in range(num_sets)] # right now just creates 1 no matter what
        #     self.target_weight_sets = [2 * (np.random.rand(target_size) - .5) / np.sqrt(num_sets + target_size) for k in range(num_sets)] # right now just creates 1 no matter what
        # else:
        #     self.parent_weight_sets = [2 * (np.random.rand(parent_size) - .5) * DYNAMICS_STEP / np.sqrt(num_sets + parent_size) for k in range(num_sets)] # right now just creates 1 no matter what
        #     self.target_weight_sets = [2 * (np.random.rand(target_size) - .5) * DYNAMICS_STEP / np.sqrt(num_sets + parent_size) for k in range(num_sets)] # right now just creates 1 no matter what
        self.parent_bias = 2 * (np.random.rand(parent_size) - .5)  * float(self.use_bias)
        self.target_bias = 2 * (np.random.rand(target_size) - .5)  * float(self.use_bias)
        self.conditional = conditional
        self.conditional_weight = conditional_weight
        self.target_dependent = target_dependent

        # self.parent_weight_matrix = np.stack([np.zeros(parent_size) for _ in range(target_size)], axis=0)
        # for ws in self.parent_weight_sets:
        #     for i, w in enumerate(ws):
        #         self.parent_weight_matrix[np.random.randint(target_size)][i] += w

        # self.target_weight_matrix = np.stack([np.zeros(target_size) for _ in range(parent_size)], axis=0)
        # for ws in self.target_weight_sets:
        #     for i, w in enumerate(ws):
        #         self.target_weight_matrix[np.random.randint(parent_size)][i] += w
        if self.target_dependent: self.weight_matrix = np.random.rand(1, target_size+ parent_size) / np.sqrt(parent_size + target_size)
        else: self.weight_matrix = np.random.rand(1, parent_size) / np.sqrt(parent_size)
        # self.target_weight_matrix = np.random.rand(parent_size, target_size) / np.sqrt(target_size)
        self.params = [self.weight_matrix, self.parent_bias, self.target_bias]
        print(self.conditional_weight)


    def __call__(self, ps, ts, require_passive=False):
        # print(ts, ps, np.matmul(self.target_weight_matrix, ts),self.parent_bias, np.matmul(self.parent_weight_matrix, ps - np.matmul(self.target_weight_matrix, ts) - self.parent_bias))
        if not hasattr(self, "target_dependent",) or self.target_dependent: rel_val = np.matmul(self.weight_matrix, np.expand_dims(np.concatenate([ts - self.target_bias, ps - self.parent_bias], axis =-1), axis=-1))[...,0]
        else: rel_val = np.matmul(self.weight_matrix, np.expand_dims(np.concatenate([ps - self.parent_bias], axis =-1), axis=-1))[...,0]
        if self.conditional:
            # print("cond weight", np.sum(rel_val, axis=-1))
            return np.sum(rel_val, axis=-1) > self.conditional_weight, None
        return True, rel_val



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

def get_object_name(n):
    return n.strip("0123456789")

class RandomDistribution(Environment):
    def __init__(self, frameskip = 1, variant="default", fixed_limits=False):
        # generates "objects" as conditional distributions of each other
        self.self_reset = True
        self.variant = variant
        self.fixed_limits = fixed_limits
        self.discrete_actions, self.allow_uncontrollable, self.num_objects, self.max_dim, self.min_dim, self.multi_instanced, self.num_related, self.max_control, self.relate_dynamics, self.conditional, self.conditional_weight, self.distribution, self.noise_percentage, self.require_passive, self.num_valid_min, self.num_valid_max = variants[self.variant]
        
        print(self.discrete_actions, self.allow_uncontrollable, self.num_objects, self.max_dim, self.min_dim, self.multi_instanced, self.num_related, self.max_control, self.relate_dynamics, self.conditional, self.conditional_weight, self.distribution, self.noise_percentage, self.require_passive, self.variant)
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

    def define_object_parameters(self):
        self.object_instanced = {name: np.random.randint(1, self.multi_instanced + 1) for name in self.object_names} # name of object to max number of objects of that type
        self.object_instanced["Action"], self.object_instanced["Reward"], self.object_instanced["Done"] = 1, 1, 1
        make_name = lambda x,i: x + str(i) if self.object_instanced[x] > 1 else x
        self.all_names = sum([[make_name(name, i) for i in range(self.object_instanced[name])] for name in self.object_names], start = list()) # must be initialized, the names of all the objects including multi-instanced ones
        self.object_sizes = {name: np.random.randint(self.min_dim,self.max_dim+1) for name in self.object_names} # must be initialized, a dictionary of name to length of the state
        self.object_sizes["Reward"], self.object_sizes["Done"] = 1,1
        self.object_range = {n: (- np.ones(self.object_sizes[n]), np.ones(self.object_sizes[n])) for n in self.object_names}
        self.object_mean = {n: (self.object_range[n][0] + self.object_range[n][1]) / 2 for n in self.object_names}
        self.object_var = {n: (self.object_range[n][1] - self.object_range[n][0]) for n in self.object_names}
        self.object_proximal = {n: True for n in self.object_names} # name of object to whether that object has valid proximity
        self.object_proximal["Action"], self.object_proximal["Reward"], self.object_proximal["Done"] = True, True, True
        self.instance_length = len(self.all_names) # the total number of instances for the mask
        self.object_range_true = self.object_range

    def set_objects(self): # creates objects based on their dictionary, and their relational connectivity
        # factorized state properties
        self.object_names = ["Action"] + [get_random_string(9) for i in range(self.num_objects)] + ["Reward", "Done"] # must be initialized, a list of names that controls the ordering of things
        self.define_object_parameters()

        onames = self.object_names[:-2]
        nonames = self.object_names[1:-2]
        used = list()
        unused = [name for name in self.object_names[1:-2]]
        controllable = ["Action"]
        self.object_relational_sets, self.object_relational_functions = list(), list()
        print(self.object_sizes, self.object_instanced)
        self.internal_statistics = dict()
        
        def create_parents(i):
            if self.allow_uncontrollable: 
                parents = np.random.choice(onames, size=min(len(onames), np.random.randint(1, self.max_control+1)), replace = False).tolist()
            else:
                ronames = copy.deepcopy(onames)
                if i > 0: 
                    if "Action" in controllable: controllable.remove("Action")
                    if "Action" in ronames: ronames.remove("Action")
                ctrl_choice = np.random.choice(controllable)
                ronames.remove(ctrl_choice)
                # print("ctrl options", ctrl_choice, controllable, ronames)
                parents = [ctrl_choice] + np.random.choice(ronames, size=min(len(nonames), np.random.randint(self.max_control)), replace = False).tolist()
            return parents
        
        self.passive_functions = dict()
        for name in self.object_names[1:-2]: # not actions or done/reward
            if self.require_passive:
                self.passive_functions[name] = passive_func(name, self.object_sizes[name], use_target_bias=True)
                self.internal_statistics[(" ".join(self.passive_functions[name].parents), self.passive_functions[name].target)] = 0
                self.internal_statistics[(" ".join(self.passive_functions[name].parents), self.passive_functions[name].target +"_clip")] = 0
            else:
                self.passive_functions[name] = None # create a placeholder
        partars = list() # the set of multi-edges
        for i in range(self.num_related): # create relational links
            target = nonames[np.random.randint(len(nonames))]
            while target in used or (target in controllable and len(controllable) == 2 and 'Action' in controllable):
                target = nonames[np.random.randint(len(nonames))]
            parents = create_parents(i)
            while target in parents:
                parents = create_parents(i)
                print(target, controllable, parents, used, onames)
            if not (self.relate_dynamics and self.conditional): used.append(target)
            if target in unused: unused.remove(target)
            if target not in controllable: controllable.append(target)
            partars.append((parents, target))
        self.target_counter = collections.Counter()
        for i in range(len(partars)):
            self.target_counter[partars[i][-1]] += 1
        for parents, target in partars:
            self.object_relational_sets.append((parents, target))
            parent_size = int(np.sum([self.object_sizes[p] for p in parents]))
            if self.conditional and (i != 0 or self.allow_uncontrollable):
                print(parent_size, self.object_sizes, parents)
                orf = conditional_add_func(parents,
                            target,
                            parent_size,
                            self.object_sizes[target],
                            use_bias = True,
                            conditional=True,
                            conditional_weight=self.conditional_weight,
                            passive=self.passive_functions[target],
                            dynamics_step = DYNAMICS_STEP / self.target_counter[target] if self.relate_dynamics else 1 / self.target_counter[target],
                            )
            else:
                orf = add_func(parents,
                            target,
                            parent_size,
                            self.object_sizes[target],
                            use_bias = True,
                            conditional=False,
                            dynamics_step = DYNAMICS_STEP / self.target_counter[target] if self.relate_dynamics else 1 / self.target_counter[target],
                            passive=self.passive_functions[target])
            print(orf.parents, orf.target, orf.params)
            self.object_relational_functions.append(orf)
            self.internal_statistics[(" ".join(orf.parents), orf.target)] = 0
            self.internal_statistics[(" ".join(orf.parents), orf.target + "_clip")] = 0
        print(unused)
        self.unused = unused
        for target in unused:
            if self.require_passive:
                self.object_relational_functions.append(self.passive_functions[target])
                self.internal_statistics[(" ".join([target]), target + "_clip")] = 0
        print(self.internal_statistics)

        # has to be set after we know how many ORFs have the object as target
        
        self.object_dynamics = dict()
        for n in self.object_names:
            orf_num = 0
            for orf in self.object_relational_functions:
                if orf.target == n:
                    total_parent_combinations = np.prod([self.object_instanced[p] for p in orf.parents])
                    orf_num += total_parent_combinations

            orf_num = max(1,orf_num)
            dynamics_step = DYNAMICS_CLIP * orf_num
            self.object_dynamics[n] = (np.ones(self.object_sizes[n])*-dynamics_step, np.ones(self.object_sizes[n])*dynamics_step)
        self.object_dynamics_true = self.object_dynamics

    def reset(self):
        self.object_name_dict= dict()
        self.objects = list()
        if self.num_valid_max > 0:
            num_valid = np.random.randint(self.num_valid_min, self.num_valid_max + 1 ) 
            if self.allow_uncontrollable:
                valid_choices = np.random.choice(np.arange(len(self.all_names) - 2), replace=False, size = (num_valid, ))
                self.valid_names = np.array(self.all_names)[valid_choices].tolist() + ["Done", "Reward"]
            else:
                valid_choices = np.random.choice(np.arange(len(self.all_names))[1:], replace=False, size = (num_valid - 1, ))
                self.valid_names = ["Action"] + np.array(self.all_names)[valid_choices].tolist() + ["Done", "Reward"]
        else:
            object_names = self.object_names
            self.valid_names = self.all_names
        # print(self.valid_names, self.allow_uncontrollable, valid_choices)
        for n in self.all_names:
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
                objn = get_object_name(n)
                next_obj = RandomDistObject(n, (np.random.rand(self.object_sizes[objn]) * self.object_var[objn] + self.object_range[objn][0])/ 2, self.object_range[objn])
                self.object_name_dict[n] = next_obj
                self.objects.append(next_obj)
        return self.get_state()

    def get_state(self):
        # state = dict()
        # for n in self.object_names:
        #     if self.object_instanced[n] > 1:
        #         for i in range(self.object_instanced[n]):
        #             state[n] = self.object_name_dict[n + str(i)].get_state()
        #     else:
        #         state[n] = self.object_name_dict[n].get_state()
        # print({n: self.object_name_dict[n].get_state() for n in self.object_names})
        return {"raw_state": None, "factored_state": {**{n: self.object_name_dict[n].get_state() for n in self.all_names}, **{"VALID_NAMES": self.valid_binary(self.valid_names)}}}

    def get_named_state(self, names):
        # print(names, [([n] if self.object_instanced[n] <= 1 else [n + str(i) for i in range(self.object_instanced[n])]) for n in names])
        instanced_names = sum([([n] if self.object_instanced[n] == 1 else [n + str(i) for i in range(self.object_instanced[n])]) for n in names], start=list())
        # print(names, self.object_name_dict, instanced_names)
        # print([self.object_name_dict[n].get_state() for n in instanced_names])
        return np.concatenate([self.object_name_dict[n].get_state() for n in instanced_names], axis=-1)
    
    def get_all_state(self, instanced_names):
        return np.concatenate([self.object_name_dict[n].get_state() for n in instanced_names], axis=-1)

    def empty_interactions(self):
        for obj in self.objects:
            obj.interaction_trace = list()

    def step(self, action, render=False, instant_update=False): 
        self.empty_interactions()
        for i in range(self.frameskip):
            self.done.attribute = False
            self.action.attribute = action
            updated = dict()
            for target in self.all_names:
                if self.relate_dynamics: 
                    self.object_name_dict[target].next_state = copy.deepcopy(self.object_name_dict[target].get_state())
                else:
                    if target in self.target_counter: 
                        self.object_name_dict[target].next_state = np.zeros(self.object_name_dict[target].get_state().shape)
                    else: self.object_name_dict[target].next_state = copy.deepcopy(self.object_name_dict[target].get_state())
            
            # print(self.get_state())
            for i, orf in enumerate(self.object_relational_functions):
                target_class = orf.target
                # print(orf.parents, orf.target)
                n, orf_average, orf_passive_average, clip_average = 0,0,0,0
                require_passive = False
                if self.num_valid_max > 0:
                    # don't use relations for nonexistent variables
                    possible_names = set([get_object_name(n) for n in self.valid_names])
                    if get_object_name(orf.target) not in possible_names: # missing a target
                        continue # move on to the next object relational function
                    if sum([(get_object_name(p) not in possible_names) for p in orf.parents]): # missing a parent
                        require_passive = True
                for tidx in range(self.object_instanced[target_class]): # for each instance of the target
                    target = target_class + str(tidx) if self.object_instanced[target_class] > 1 else target_class
                    if self.num_valid_max > 0: 
                        if target not in self.valid_names: continue
                        parent_mesh = list()
                        for p in orf.parents:
                            plist = list()
                            for pidx in range(self.object_instanced[p]):
                                pname = p + str(pidx) if self.object_instanced[p] > 1 else p
                                if pname in self.valid_names:
                                    plist.append(pidx)
                            parent_mesh.append(np.array(plist))
                        parent_mesh = np.array(np.meshgrid(*parent_mesh)).T.reshape(-1,len(parent_mesh))
                        if require_passive: parent_mesh = [[1]] # passive dynamics only require the target, ps will be ignored

                    # print("orf", orf.parents, orf.target)
                    else:
                        parent_nums = [self.object_instanced[p] for p in orf.parents]
                        parent_mesh = [np.arange(i) for i in parent_nums]
                        parent_mesh = np.array(np.meshgrid(*parent_mesh)).T.reshape(-1,len(parent_mesh))

                    for pmesh in parent_mesh: # for each combination of instances of the parents
                        instanced_names = [(p+str(i) if self.object_instanced[p] > 1 else p) for i,p in zip(pmesh, orf.parents)]

                        ps = self.get_all_state(instanced_names)
                        ts = self.get_all_state([target])
                        inter, nds = orf(ps, ts, require_passive)
                        if inter:
                            self.object_name_dict[target].interaction_trace += orf.parents
                        
                        orf_average = (int(inter) + (orf_average * n)) / (n + 1)
                        if self.require_passive and not inter:
                            orf_passive_average = (1 + (orf_passive_average * n)) / (n + 1)

                        clip_average = (int(np.any(np.abs(nds) > DYNAMICS_CLIP)) + (clip_average * n)) / (n + 1)
                            
                        # print(orf.parents, orf.target, inter)
                        # print(inter, orf.target, nds)
                        if self.relate_dynamics: 
                            self.object_name_dict[target].next_state += np.clip(nds, -DYNAMICS_CLIP, DYNAMICS_CLIP)
                        else:
                            self.object_name_dict[target].next_state += nds # adds together, but from zero, no clipping
                            # if i < 3: print(self.itr, self.done.attribute, orf.parents, orf.target, self.get_state()["factored_state"][orf.target])
                        if instant_update:
                            self.object_name_dict[target].state = self.object_name_dict[target].next_state
                        n += 1
                self.internal_statistics[ (" ".join(orf.parents), orf.target)] += orf_average
                self.internal_statistics[ (" ".join(orf.parents), orf.target  + "_clip")] += clip_average
                if self.require_passive and hasattr(orf, "passive"): self.internal_statistics[ (" ".join(orf.passive.parents), orf.passive.target)] += orf_passive_average
            for obj in self.object_name_dict.values():
                # print("adding noise", obj.next_state)
                # if self.noise_percentage > 0: # TODO: it appears taking random actions is correlated with the random noise, so we removed this impl
                #     if self.distribution == "Gaussian":
                #         obj.next_state = obj.next_state + np.random.normal(scale=self.noise_percentage, size=obj.next_state.shape)
                if hasattr(obj, "step_state"): obj.step_state()
        self.itr += 1
        # print(self.all_names)
        # print(self.get_full_current_trace())
        # error
        # print(self.get_state()["factored_state"])
        if self.itr % 1000 == 0:
            for k in self.internal_statistics.keys():
                print(k, self.internal_statistics[k] / self.itr)
        if self.itr % 50 == 0:
            self.reset()
            self.done.attribute = True
            return self.get_state(), self.reward.attribute, self.done.attribute, {'Timelimit.truncated': True, "valid_names": self.valid_names}
        return self.get_state(), self.reward.attribute, self.done.attribute, {'Timelimit.truncated': False, "valid_names": self.valid_names}

    def set_from_factored_state(self, factored_state, seed_counter=-1, render=False, valid_names=None):
        '''
        TODO: only sets the active elements, and not the score, reward and other features. This could be an issue in the future.
        '''
        if seed_counter > 0:
            self.seed_counter = seed_counter
        for n in factored_state.keys():
            if n in self.object_name_dict:
                self.object_name_dict[n].state = copy.deepcopy(factored_state[n])
        if valid_names is not None:
            self.valid_names = valid_names
            factored_state["VALID_NAMES"] = self.valid_binary(valid_names)

