import sys, cv2
import collections
import numpy as np
from Environment.environment import Environment, Done, Reward
import imageio as imio
import os, copy, string
from Environment.Environments.RandomDAG.random_DAG_specs import variants, parse_edges
from Record.file_management import numpy_factored
from gym import spaces
from Environment.Environments.RandomDistribution.random_distribution import RandomDistribution, conditional_add_func, passive_func, add_func, get_object_name,\
     Action, DYNAMICS_STEP, DYNAMICS_CLIP, OBJECT_MAX_DIM, PARENT_REDUCE_FACTOR, TARGET_REDUCE_FACTOR, passive_name, is_passive_name
        

class RandomDAG(RandomDistribution):
    def __init__(self, frameskip = 1, variant="default", fixed_limits=False, debug_mode=False):
        # generates "objects" as conditional distributions of each other
        self.self_reset = True
        self.variant = variant
        self.fixed_limits = fixed_limits
        self.debug_mode =debug_mode
        self.discrete_actions, self.allow_uncontrollable, self.graph_skeleton, self.num_nodes,\
            self.multi_instanced, self.min_dim, self.max_dim, self.instant_update, self.relate_dynamics,\
            self.conditional, self.conditional_weight, self.distribution, self.noise_percentage, self.require_passive,\
            self.num_valid_min, self.num_valid_max, self.intervention_state, self.intervention_rate, self.horizon = variants[self.variant]
        if self.debug_mode: 
            self.max_dim = self.min_dim = 1 # use one dimension to have obvious split points
            self.noise_percentage = 0
        self.set_objects()
        self.num_actions = self.discrete_actions # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "RandomDAG" # required for an environment 
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
        self.edge_list, self.object_names = parse_edges(self.graph_skeleton)
        if self.require_passive and self.instant_update: # create special passive variables to be used with the passive functions
            new_names = list()
            for name in self.object_names:
                if name not in ["Action", "Done", "Reward"]:
                    new_names.append(passive_name(name))
            self.object_names = self.object_names[:-2] + new_names + self.object_names[-2:]
        self.num_objects = len(self.object_names) - 3
        self.define_object_parameters()

        onames = self.object_names[:-2]
        nonames = self.object_names[1:-2]
        used = list()
        unused = [name for name in self.object_names[1:-2]]
        controllable = ["Action"]
        self.object_relational_sets, self.object_relational_functions = list(), list()
        print(self.object_sizes, self.object_instanced)
        self.internal_statistics = dict()
            
        
        self.passive_functions = dict()
        for name in self.object_names[1:-2]: # not actions or done/reward
            if is_passive_name(name):
                continue
            if self.require_passive:
                self.passive_functions[name] = passive_func(name, self.object_sizes[name], use_target_bias=not self.debug_mode, dynamics_step = DYNAMICS_STEP if self.relate_dynamics else 1)
                self.internal_statistics[(" ".join(self.passive_functions[name].parents), self.passive_functions[name].target)] = 0
                self.internal_statistics[(" ".join(self.passive_functions[name].parents), self.passive_functions[name].target +"_clip")] = 0
            else:
                self.passive_functions[name] = None # create a placeholder
        unused = copy.deepcopy(self.object_names[1:-2])
        self.target_counter = collections.Counter()
        for i in range(len(self.edge_list)):
            self.target_counter[self.edge_list[i][-1]] += 1
        for i in range(len(self.edge_list)): # create relational links, ordered in terms of priority
            parents = self.edge_list[i][:-1]
            target = self.edge_list[i][-1]

            if target in unused:
                unused.pop(unused.index(target))
            self.object_relational_sets.append((parents, target))
            parent_size = int(np.sum([self.object_sizes[p] for p in parents]))
            if self.conditional and (i != 0 or self.allow_uncontrollable):
                orf = conditional_add_func(parents,
                            target,
                            parent_size,
                            self.object_sizes[target],
                            use_bias = not self.debug_mode,
                            target_dependent = (not self.debug_mode) and (not self.instant_update or self.require_passive),
                            conditional=True,
                            conditional_weight=self.conditional_weight,
                            passive=self.passive_functions[target],
                            dynamics_step = DYNAMICS_STEP / self.target_counter[target] if self.relate_dynamics else 0.4 / self.target_counter[target],
                            )
            else:
                orf = add_func(parents,
                            target,
                            parent_size,
                            self.object_sizes[target],
                            use_bias = not self.debug_mode,
                            target_dependent = (not self.debug_mode) and (not self.instant_update or self.require_passive),
                            conditional=False,
                            passive=self.passive_functions[target],
                            dynamics_step = DYNAMICS_STEP / self.target_counter[target] if self.relate_dynamics else 0.4 / self.target_counter[target])
            print(orf.parents, orf.target, orf.params, self.instant_update)
            self.object_relational_functions.append(orf)
            self.internal_statistics[(" ".join(orf.parents), orf.target)] = 0
            self.internal_statistics[(" ".join(orf.parents), orf.target + "_clip")] = 0
        print(unused)
        self.unused = unused
        for target in unused:
            if self.require_passive and not is_passive_name(target):
                self.object_relational_functions.append(self.passive_functions[target])
                self.internal_statistics[(" ".join([target]), target + "_clip")] = 0
        print(self.internal_statistics)

        # has to be set after we know how many ORFs have the object as target
        
        self.object_dynamics = dict()
        self.target_last = dict()
        for n in self.object_names:
            orf_num = 0
            for i, orf in enumerate(self.object_relational_functions):
                if orf is not None and orf.target == n:
                    total_parent_combinations = np.prod([self.object_instanced[p] for p in orf.parents])
                    orf_num += total_parent_combinations
                    self.target_last[orf.target] = i

            orf_num = max(1,orf_num)
            dynamics_step = DYNAMICS_CLIP * orf_num
            if self.relate_dynamics:
                self.object_dynamics[n] = (np.ones(self.object_sizes[n])*-dynamics_step, np.ones(self.object_sizes[n])*dynamics_step)
            else:
                self.object_dynamics[n] = (-1 * np.ones(self.object_sizes[n]), np.ones(self.object_sizes[n]))
        self.object_dynamics_true = self.object_dynamics

    def reset(self): # placeholders
        state, info = super().reset()
        return state,info
    
    def get_state(self):
        return super().get_state()
    
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

    def step(self, action, render=False, intervening_except=None): 
        return super().step(action, render=render, instant_update = self.instant_update, intervention_state=self.intervention_state if np.random.rand() > 0.5 else None, intervening_except=intervening_except)
    
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