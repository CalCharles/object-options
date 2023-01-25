import numpy as np
import gym
import os
from gym import spaces
import robosuite
from robosuite.controllers import load_controller_config
import imageio
import copy
import cv2
from Environment.environment import Environment
from Environment.Environments.RoboPushing.robopushing_specs import *
from Record.file_management import numpy_factored, display_frame
from collections import deque
import robosuite.utils.macros as macros
macros.SIMULATION_TIMESTEP = 0.02

# control_freq, num_obstacles, standard_reward, goal_reward, obstacle_reward, out_of_bounds_reward, 
# joint_mode, hard_obstacles, planar_mode


DEFAULT = 0
JOINT_MODE = 1
HARD_MODE = 2
PLANAR_MODE = 3
DISCRETE_MODE = 4

a = [[-0.7,0,0],
   [0.7,0,0],
   [0,-0.7,0],
   [0,0.7,0],
   [0,0,-0.25],
   [0,0,.25]]
DISCRETE_MOVEMENTS = np.array(a).astype(float)

class RoboPushing(Environment):
    def __init__(self, variant="default", horizon=30, renderable=False, fixed_limits=False):
        super().__init__()
        self.self_reset = True
        self.fixed_limits = fixed_limits
        self.variant=variant
        control_freq, var_horizon, num_obstacles, standard_reward, goal_reward, obstacle_reward, out_of_bounds_reward, mode  = variants[variant]
        horizon = var_horizon if horizon < 0 else horizon
        self.mode = mode
        self.goal_reward = goal_reward
        controller = "JOINT_POSITION" if self.mode==JOINT_MODE else "OSC_POSE" # TODO: handles only two action spaces at the moment
        self.env = robosuite.make(
                "Push",
                robots=["Panda"],
                controller_configs=load_controller_config(default_controller=controller),
                has_renderer=False,
                has_offscreen_renderer=renderable,
                render_visual_mesh=renderable,
                render_collision_mesh=False,
                camera_names=["frontview"] if renderable else None,
                control_freq=control_freq,
                horizon=horizon,
                use_object_obs=True,
                use_camera_obs=renderable,
                hard_reset = False,
                num_obstacles=num_obstacles,
                standard_reward=float(standard_reward), 
                goal_reward=float(goal_reward), 
                obstacle_reward=float(obstacle_reward), 
                out_of_bounds_reward=float(out_of_bounds_reward),
                hard_obstacles=mode == HARD_MODE,
                keep_gripper_in_cube_plane=mode == PLANAR_MODE
            )
        # environment properties
        self.num_actions = -1 # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "RobosuitePushing" # required for an environment 
        self.discrete_mode = mode == DISCRETE_MODE
        self.discrete_actions = self.discrete_mode
        self.frameskip = control_freq
        self.timeout_penalty = -horizon
        self.planar_mode = mode == PLANAR_MODE

        # spaces
        if self.discrete_mode:
            self.action_shape = (1,)
            self.num_actions = 6 # up down left right forward backward
            self.action_space = spaces.Discrete(self.num_actions)
            self.action = 0
            limit = 1
        else:
            low, high = self.env.action_spec
            limit = 7 if self.mode == JOINT_MODE else 3
            self.action_shape = (limit,)
            self.action_space = spaces.Box(low=low[:limit], high=high[:limit])
            self.action = np.zeros(self.action_shape)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[9])
        self.renderable = renderable

        # running values
        self.timer = 0

        # state components
        self.reward = 0
        self.done = False
        self.extracted_state = None

        # factorized state properties
        self.object_names = ["Action", "Gripper", "Block", 'Obstacle',"Target", 'Done', "Reward"] # must be initialized, a list of names that controls the ordering of things
        self.object_sizes = {"Action": limit, "Gripper": 3, "Block": 3, "Obstacle": 3,"Target": 3, "Done": 1, "Reward": 1} # must be initialized, a dictionary of name to length of the state
        if self.discrete_mode:
            self.object_range = discrete_ranges
            self.object_dynamics = discrete_dynamics
            self.object_range_true = discrete_ranges
            self.object_dynamics_true = discrete_dynamics
            self.position_masks = discrete_position_masks
        else:
            self.object_range = ranges if not self.fixed_limits else ranges_fixed # the minimum and maximum values for a given feature of an object
            self.object_dynamics = dynamics if not self.fixed_limits else dynamics_fixed
            self.object_range_true = ranges
            self.object_dynamics_true = dynamics
            self.position_masks = position_masks

        # obstacles and objects
        self.num_obstacles = num_obstacles
        self.objects = ["Action", "Gripper", "Block"] + ["Obstacle" + str(i) for i in range(num_obstacles)] + ["Target", "Reward", "Done"]
        self.object_instanced = instanced
        self.object_instanced["Obstacle"] = num_obstacles
        self.all_names = sum([[(name + str(i) if instanced[name] > 1 else name) for i in range(instanced[name])] for name in self.object_names], start = [])
        self.instance_length = len(self.all_names)

        # position mask
        self.pos_size = 3

        self.full_state = self.reset()
        self.frame = self.full_state['raw_state'] # the image generated by the environment
        self.reward_collect = 0

    def set_named_state(self, obs_dict):
        obs_dict['Action'], obs_dict['Gripper'], obs_dict['Block'], obs_dict['Target'] = self.action, obs_dict['robot0_eef_pos'], obs_dict['cube_pos'], obs_dict['goal_pos']# assign the appropriate values
        if self.discrete_mode:
            obs_dict['Action'] = [self.action]
        for i in range(self.num_obstacles):
            # print("settin", obs_dict[f"obstacle{i}_pos"])
            obs_dict['Obstacle' + str(i)] = obs_dict[f"obstacle{i}_pos"]
        obs_dict['Reward'], obs_dict['Done'] = [self.reward], [self.done]

    def construct_full_state(self, factored_state, raw_state):
        self.full_state = {'raw_state': raw_state, 'factored_state': numpy_factored(factored_state)}
        return self.full_state

    def set_action(self, action):
        if self.mode == JOINT_MODE:
            use_act = action
        elif self.mode == PLANAR_MODE:
            use_act = np.concatenate([action[:2], [0,0,0,0]])
        elif self.mode == DISCRETE_MODE:
            use_act = np.concatenate([DISCRETE_MOVEMENTS[action], [0,0,0]])
        else:
            use_act = np.concatenate([action, [0, 0, 0]])
        return use_act

    def step(self, action, render=False): # render will NOT change renderable, so it will still render or not render
        # step internal robosuite environment
        self.action = action
        use_act = self.set_action(action)
        next_obs, self.reward, self.done, info = self.env.step(use_act)
        self.reward_collect += self.reward
        info["TimeLimit.truncated"] = False
        if self.done:
            info["TimeLimit.truncated"] = True
            print("end of episode:", self.reward_collect, ",", next_obs["cube_pos"], next_obs["goal_pos"])
        if self.reward == self.goal_reward: # don't wait at the goal, just terminate
            print("end of episode:", self.reward_collect, ",", next_obs["cube_pos"], next_obs["goal_pos"])
            self.done = True
            info["TimeLimit.truncated"] = False
        # set state
        self.set_named_state(next_obs) # mutates next_obs
        img = next_obs["frontview_image"][::-1] if self.renderable else None
        obs = self.construct_full_state(next_obs, img)
        # print(np.array([obs['factored_state']["Obstacle" + str(i)] for i in range(15)]))
        self.frame = self.full_state['raw_state']

        # step timers 
        self.itr += 1
        self.timer += 1

        if self.done:
            self.reset()
            self.timer = 0
        # print("step",self.env, np.array([obs['factored_state']["Obstacle" + str(i)] for i in range(15)]))
        return obs, self.reward, self.done, info

    def get_state(self, render=False):
        return copy.deepcopy(self.full_state)

    def get_trace(self, factored_state, action, object_names):
        return [1]

    def get_full_trace(self, factored_state, action, target_name):
        return np.ones(len(self.all_names))

    def current_trace(self, object_names):
        return [1]

    def reset(self):
        obs = self.env.reset()
        self.reward_collect = 0
        self.set_named_state(obs)
        self.frame = obs["frontview_image"][::-1] if self.renderable else None
        return self.construct_full_state(obs, self.frame)

    def render(self):
        return self.frame

    def toString(self, extracted_state):
        estring = "ITR:" + str(self.itr) + "\t"
        for i, obj in enumerate(self.objects):
            if obj not in ["Reward", "Done"]:
                estring += obj + ":" + " ".join(map(str, extracted_state[obj])) + "\t" # TODO: attributes are limited to single floats
            else:
                estring += obj + ":" + str(int(extracted_state[obj][0])) + "\t"
        return estring

