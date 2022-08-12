# action handler
import numpy as np
import copy
import gym
from tianshou.data import Batch, ReplayBuffer, to_torch_as, to_numpy
from Network.network_utils import pytorch_model

# Base action space handling
class PrimitiveActionMap():
    def __init__(self, args, filtered_active_set):
        self.action_space = args.environment.action_space
        self.discrete_actions = args.environment.discrete_actions
        self.discrete_control = np.arange(args.environment.action_space.n) if args.environment.discrete_actions else None
        self.control_max = args.environment.action_space.high if not args.environment.discrete_actions else args.environment.action_space.n
        self.control_min = args.environment.action_space.low if not args.environment.discrete_actions else 0
        self.filtered_active_set = filtered_active_set

    def sample(self):
        self.action_space.sample()

class ActionMap():
    def __init__(self, args, filtered_active_set, mapped_norm, mapped_select, discrete_primitive, round_values, no_scale_last, sample_angle):
        ''' ActionMap manages the mapped space, relative mapped space and the policy space
        The policy space is always [-1,1] for continuous, (1,) for discrete
        The mapped space covers the space of behaviors
        discrete_primitive is ONLY > 0 if the action space is the primitive space, and the primitive space is discrete,
            Then discrete primitive is the number of actions
        '''
        # parameters for the action mapping
        self.use_relative_action = args.use_relative_action
        self.relative_action_ratio = args.relative_action_ratio

        # input variables
        self.min_active_size = args.min_active_size
        self.filtered_active_set = filtered_active_set
        self.mapped_norm = mapped_norm
        self.mapped_select = mapped_select
        self.discrete_primitive = discrete_primitive
        self.discrete_actions, self.action_space = self.param_space(discrete_primitive)
        self.discrete_control = args.discrete_params # if the parameter is sampled from a discrete dict
        self.round_values = round_values
        self.no_scale_last = no_scale_last
        self.sample_angle = sample_angle
        # self.use_round = args.use_round

        # initialize the policy space
        if self.discrete_actions:
            self.policy_action_space = gym.spaces.Discrete(discrete_primitive if discrete_primitive != 0 else len(filtered_active_set))
            self.relative_action_space, self.relative_scaling = None, 1 # can't be relative and discrete
        else:
            policy_action_shape = (self.mapped_select.output_size(), )
            policy_min = -1 * np.ones(policy_action_shape) # policy space is always the same
            policy_max = 1 * np.ones(policy_action_shape)
            self.policy_action_space = gym.spaces.Box(policy_min, policy_max)

            # initialize the relative space
            self.relative_scaling = self.mapped_norm.mapped_norm[1] * self.relative_action_ratio
            self.relative_action_space = gym.spaces.Box(-self.mapped_norm.mapped_norm[1] * self.relative_action_ratio, self.mapped_norm.mapped_norm[1] * self.relative_action_ratio)

    def param_space(self, discrete_primitive):
        ''' This runs on initialization, deciding whether the parameter space should be discrete or continuous
        selection is based on if |active set| > args.min_active_size
        '''
        if discrete_primitive > 0:
            return True, gym.spaces.Discrete(discrete_primitive)
        elif len(self.filtered_active_set) < self.min_active_size and discrete_primitive == 0:
            return True, gym.spaces.Box(*self.mapped_norm.mapped_lim)
        else:
            return False, gym.spaces.Box(*self.mapped_norm.mapped_lim)

    def _convert_relative_action(self, state, act):
        return self.mapped_norm.clip(self.mapped_select(state['factored_state']) + act)

    def _reverse_relative_action(self, state, act):
        return act - self.mapped_select(state['factored_state'])

    def _round_action(self, action):
        new_action = list()
        for i in range(action.shape[-1]):
            idx = (np.abs(self.round_values[i] - action[i])).argmin()
            new_action.append(self.round_values[i][idx])
        return np.array(new_action)


    def map_action(self, act, batch):
        ''' 
        maps a policy space to the action space used by a parameter
        it does not apply exploration noise
        the action should be a vector, even if discrete
        '''
        act = pytorch_model.unwrap(act)
        if self.discrete_actions: 
            if not self.discrete_primitive: mapped_act = self._get_cont(act)
            else: mapped_act = act
        else:
            if self.use_relative_action: # mapped_act = act if relative actions are used
                mapped_act = act * self.relative_scaling
                # print("scaled", mapped_act)
                mapped_act = self._convert_relative_action(batch.full_state, mapped_act)
                # print("converted", mapped_act, self.mapped_select(state['factored_state']), state['factored_state']["Gripper"])
                if hasattr(self, "no_scale_last") and self.no_scale_last: mapped_act[...,-1] = (act[...,-1] + 1) / 2
            else: # rescale based on non-relative actions
                mapped_act = self.mapped_norm.reverse(act)
        # print(mapped_act)
        if hasattr(self, "sample_angle") and self.sample_angle: 
            mapped_act[...,2] = np.sin(act[...,2] * np.pi)
            mapped_act[...,3] = np.cos(act[...,2] * np.pi)
        if hasattr(self, "round_values") and self.round_values is not None: mapped_act = self._round_action(mapped_act) # TODO: remove hasattr
        return mapped_act

    def reverse_map_action(self, mapped_act, batch):
        '''
        gets a policy space action from a mapped action
        '''
        if self.use_relative_action: # converts relative actions maintaining value
            mapped_act = self._reverse_relative_action(batch.full_state, mapped_act)
            mapped_act = mapped_act / self.relative_scaling
        act = self._get_discrete(mapped_act)
        return act

    def _get_cont(self, act):
        return self.filtered_active_set[int(act.squeeze())].copy()

    def _get_discrete(self, act):
        def find_closest(a):
            closest = (-1, 99999999)
            for i in range(len(self.filtered_active_set)):
                dist = np.linalg.norm(a - np.array(self.filtered_active_set[i]))
                if dist < closest[1]:
                    closest = (i,dist)
            return closest[0]
        if self.discrete_actions:
            if self.discrete_primitive:
                return act
            return find_closest(act)
        return act

    def sample_policy_space(self):
        '''
        samples the policy action space
        '''
        return np.array(self.policy_action_space.sample())

    def sample(self):
        '''
        sample the mapped action space
        '''
        return self.action_space.sample() # maybe we should map a policy action space




'''
    I think these can be removed without issues, but they are in tianshou.policy.base so it would be good to make sure
    they don't get called
    def map_action(self, act: Union[Batch, np.ndarray]) -> Union[Batch, np.ndarray]:
        """COPIED FROM tianshou.policy.base: 
        """
        if isinstance(self.action_space, gym.spaces.Box) and \
                isinstance(act, np.ndarray):
            # currently this action mapping only supports np.ndarray action
            if self.algo_policy.action_bound_method == "clip":
                act = np.clip(act, -1.0, 1.0)  # type: ignore
            elif self.algo_policy.action_bound_method == "tanh":
                act = np.tanh(act)
            if self.algo_policy.action_scaling:
                assert np.min(act) >= -1.0 and np.max(act) <= 1.0, \
                    "action scaling only accepts raw action range = [-1, 1]"
                low, high = self.action_space.low, self.action_space.high
                act = low + (high - low) * (act + 1.0) / 2.0  # type: ignore
        return act

    def reverse_map_action(self, mapped_act):
        # reverse the effect of map_action, not one to one because information might be lost (ignores clipping)
        if self.algo_policy.action_scaling:
            low, high = self.action_space.low, self.action_space.high
            act = ((mapped_act - low) / (high - low)) * 2 - 1
        if self.algo_policy.action_bound_method == "tanh":
            act = np.arctanh(act)
        return act

'''