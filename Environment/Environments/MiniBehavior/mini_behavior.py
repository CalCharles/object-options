import numpy as np
import gymnasium as gym
import copy
from Environment.environment import Environment, Reward, Done, Action
from Environment.Environments.MiniBehavior.mini_behavior_specs import mini_variant_specs
from mini_behavior.wrappers.flatten_dict_observation import FlattenDictObservation


def init_object_state(room_size, discrete_obs, discrete_actions, action_space, env_state):
    object_sizes = {**{n: env_state[n].shape[0] for n in env_state.keys()}, **{"Action": 1 if discrete_actions else action_space.shape, "Reward": 1, "Done": 1}}
    object_range = {**{n: (np.zeros((object_sizes[n])), np.ones((object_sizes[n])) ) if discrete_obs else
                     (-1 * np.ones((object_sizes[n])) * room_size / 2 , np.ones((object_sizes[n])) * room_size / 2 ) for n in env_state.keys()}, 
                     **{"Action": (np.array([0]),np.array([action_space.n])) if discrete_actions else (action_space.low, action_space.high), "Reward": (np.array([-1]), np.array([1])), "Done": (np.array([0]), np.array([1]))}} # assumes continuous action space is box
    object_dynamics = {**{n: (-1 * np.array([1] * 2 + [10] * (object_sizes[n] - 2)), np.array([1] * 2 + [10] * (object_sizes[n] - 2)) ) for n in env_state.keys()}, # assumes only movement by 1
                     **{"Action": (np.array([-action_space.n]),np.array([action_space.n])) if discrete_actions else (action_space.low, action_space.high), "Reward": (np.array([-1]), np.array([1])), "Done": (np.array([0]), np.array([1]))}} # assumes continuous action space is box
    return object_sizes, object_range, object_dynamics

class MiniBehaviorObject():
    def __init__(self, name, env, room_size, discrete_obs):
        self.name = name
        self.env = env
        self.state = None
        self.interaction_trace = list()
        self.hroom_size = (room_size - 1) / 2.0
        self.discrete_obs = discrete_obs

    def get_state(self):
        state = self.state
        if state is not None and not self.discrete_obs: state = self.hroom_size * state
        return copy.deepcopy(state)

class MiniBehavior(Environment):
    def __init__(self, frameskip = 1, variant="", reward_variant="", fixed_limits=False):
        ''' Wraps a gym mini-behavior environment with the components necessary for this task
        All the below properties are set by the subclass
        '''
        # environment properties
        super().__init__(frameskip, variant, fixed_limits)
        env_specific_config = mini_variant_specs[variant]
        env_id = "MiniGrid-" + variant + "-v0"
        kwargs = {"evaluate_graph": True,
                  "discrete_obs": mini_variant_specs["discrete_obs"],
                  "room_size": env_specific_config["room_size"],
                  "max_steps": env_specific_config["max_steps"],
                  "use_stage_reward": env_specific_config["use_stage_reward"],
                  "random_obj_pose": env_specific_config["random_obj_pose"],
                  "task_name": reward_variant if len(reward_variant) != 0 else env_specific_config["reward_variant"]
                  }
        self.discrete_obs, self.room_size, self.max_steps, self.use_stage_reward, self.random_obj_pose = mini_variant_specs["discrete_obs"], env_specific_config["room_size"], env_specific_config["max_steps"], env_specific_config["use_stage_reward"], env_specific_config["random_obj_pose"]
        self.env = gym.make(env_id, **kwargs)
        self.env = FlattenDictObservation(self.env)
        self.env.set_render_mode(False)

        
        
        self.self_reset = True
        self.num_actions = self.num_actions = self.env.action_space.n if self.discrete_actions else -1 # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "mini_behavior" # required for an environment
        self.discrete_actions = type(self.env.action_space) == gym.spaces.Discrete

        # spaces
        self.action_space = self.env.action_space # gym.spaces
        self.action_shape = (1,) if self.discrete_actions else self.action_space.shape
        self.observation_space = self.env.observation_space
        self.pos_size = 2

        # state components
        self.reward = 0
        self.done = False
        self.action = Action(discrete_actions=self.discrete_actions, action_shape=self.action_shape)
        env_state = self.env.reset()[0]
        dos = self.env.dict_obs_space
        objs = self.env.objs
        agent = [self.env.agent_pos, self.env.agent_dir]
        self.breakpoints = self.env.breakpoints
        factored_state = {n: env_state[self.env.breakpoints[i]:self.env.breakpoints[i+1]] for n, i in zip(dos.keys(), range(len(self.env.breakpoints) - 1))}
        # running values
        self.itr = 0

        # factorized state properties
        self.all_names = list(dos.keys())
        # self.all_names.sort() # ensure consistent order
        self.all_names = ["Action"] + self.all_names + ["Reward", "Done"]
        self.valid_names = self.all_names # no invalids
        self.num_objects = len(self.all_names) 
        self.object_names = self.all_names # no multiinstancing
        self.object_sizes,self.object_range,self.object_dynamics = init_object_state(self.room_size, self.discrete_obs, self.discrete_actions, self.action_space, {n: factored_state[n] for n in factored_state.keys() if n not in ["Action", "Reward", "Done"]})
        self.object_range_true = self.object_range
        self.object_dynamics_true = self.object_dynamics
        self.object_instanced = {n: 1 for n in self.all_names}
        self.object_proximal = {n: True for n in self.all_names} if not self.discrete_obs else {n: False for n in self.all_names} # discrete obs proximity not supported
        self.object_name_dict = {n: MiniBehaviorObject(n, self.env, self.room_size, self.discrete_obs) for n in self.all_names if n not in ["Action", "Reward", "Done"]}
        self.object_name_dict["Action"] = self.action
        self.object_name_dict["Done"] = Done()
        self.object_name_dict["Reward"] = Reward()
        self.objects = [self.object_name_dict[n] for n in self.all_names]
        self.instance_length = len(self.all_names)
        state = self.reset()
        self.extracted_state = state["factored_state"]

        # proximity state components
        self.position_masks = {**{n: np.array([1,1] + [0] * (self.object_sizes[n] - 2)) for n in self.all_names if n not in ["Action", "Reward", "Done"]}, 
                               **{n: np.array([0] * (self.object_sizes[n])) for n in ["Action", "Reward", "Done"]}}
    
    def reset_traces(self):
        for o in self.object_name_dict.values():
            o.interaction_trace = list()

    def set_trace(self, info):
        if "factor_graph" in info:
            graph = info["factor_graph"]
            state_names = self.all_names[1:-2]
            for i, name in enumerate(state_names):
                self.object_name_dict[name].interaction_trace += [self.all_names[j] for j in range(len(graph[i])) if graph[i,j] == 1]

    def set_states(self, state, action=None, rew=None, done=None):
        state_names = self.all_names[1:-2]
        # print(state, state_names, self.all_names)
        for i, name in enumerate(state_names):
            self.object_name_dict[name].state = state[self.breakpoints[i]:self.breakpoints[i+1]]
            # print(name, state[self.breakpoints[i]:self.breakpoints[i+1]], self.breakpoints[i], self.breakpoints[i+1])
        if action is not None: self.object_name_dict["Action"].attr = np.array([action]) if self.discrete_actions else action
        if rew is not None: 
            self.object_name_dict["Reward"].attribute = rew
            self.reward = rew
        if done is not None: 
            self.object_name_dict["Done"].attribute = done
            self.done = done

    def step(self, action, render=False):
        self.reset_traces()
        self.object_name_dict["Done"].attribute, self.object_name_dict["Reward"].attribute = False, 0
        for i in range(self.frameskip):
            state, rew, done, trunc, info = self.env.step(action)
            self.set_states(state,action, rew, done or trunc)
            info["Timelimit.truncated"] = trunc
            self.set_trace(info)
            # print(action, self.object_name_dict["agent"].state)
            # self.set_from_factored_state(self.get_state()["factored_state"])
        trace = self.get_full_current_trace()
        bin_traces = np.stack([trace[n] for n in self.all_names]).flatten().astype(int)
        self.itr += 1
        if render: self.render()
        if self.done:
            self.env.reset()
        return self.get_state(bin_traces=bin_traces), self.reward, self.done, info

    def reset(self):
        state, info = self.env.reset()
        self.set_trace(info)
        self.set_states(state, None, None, None)
        trace = self.get_full_current_trace()
        bin_traces = np.stack([trace[n] for n in self.all_names]).flatten().astype(int)
        return self.get_state(bin_traces=bin_traces)

    def render(self, mode='human'):
        '''
        matches the API of OpenAI gym, rendering the environment
        returns None for human mode
        '''
        self.env.render()

    def close(self):
        '''
        closes and performs cleanup
        '''
        self.env.close()

    def seed(self, seed):
        '''
        numpy should be the only source of randomness, but override if there are more
        '''
        if seed < 0: seed = np.random.randint(10000)
        print("env seed", seed)
        np.random.seed(seed)
        # self.env.seed(seed)


    def get_state(self, bin_traces=None):
        extracted_state = {n: self.object_name_dict[n].get_state() for n in self.all_names}
        if bin_traces is not None: 
            extracted_state["TRACE"] = bin_traces
        else: # get the current trace and use it
            trace = self.get_full_current_trace()
            bin_traces = np.stack([trace[n] for n in self.all_names]).flatten().astype(int)
        raw_state = np.array([0])# self.env.get_state()["grid"].render(2)
        return {"raw_state": raw_state, "factored_state": extracted_state}

    def get_info(self): # TODO: also return the graph
        return {"TimeLimit.truncated": False, "itr": self.get_itr()}

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
        # TODO: need to figure out how this works
        for n in self.all_names[:-2]:
            self.env.objs[n] = factored_state[n]
