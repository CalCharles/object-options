import numpy as np
from Environment.environment import Environment
import gymnasium as gym
import copy
import cv2
from Environment.Environments.Pusher2D.pusher_specs import pusher_variants

def aabb_check(o1, o2, o1lw, o2lw):
    # print(o1[1] < o2[1] + o2lw[1],
    #     o1[1] + o1lw[1] > o2[1],
    #     o1[0] < o2[0] + o2lw[0],
    #     o1[0] + o1lw[0] > o2[0])
    return (o1[1] < o2[1] + o2lw[1] and
        o1[1] + o1lw[1] > o2[1] and
        o1[0] < o2[0] + o2lw[0] and
        o1[0] + o1lw[0] > o2[0])

class Action():
    def __init__(self,discrete_actions, action_shape):
        self.name = "Action"
        self.discrete_actions = discrete_actions
        self.attr = np.array(0) if self.discrete_actions else np.zeros(action_shape)
        self.interaction_trace = list()

    def get_state(self):
        return np.array(self.attr).astype(float)

class Object2D(): 
    def __init__(self, pos, attr, lw, name):
        self.name=name
        self.pos = pos
        self.length, self.width = lw
        self.lw = lw
        self.attr = attr
        self.midpoint = self.get_midpoint()
        self.interaction_trace = list()
        self.possible_next = np.zeros((2,))

    def get_midpoint(self):
        return np.array([self.pos[0] + (self.length / 2), self.pos[1] + (self.width / 2)])

    def get_state(self):
        self.midpoint = self.get_midpoint()
        return np.array(self.midpoint.tolist() + [self.attr]).astype(float)

class MobileObject2D(Object2D):
    def respond_obstacle(self, obstacle, use_next=False):
        # print(self.possible_next, obstacle.pos, aabb_check(self.possible_next, obstacle.pos, self.lw, obstacle.lw))
        opos = obstacle.possible_next if use_next else obstacle.pos
        if aabb_check(self.possible_next, opos, self.lw, obstacle.lw):
            self.interaction_trace.append(obstacle.name)
            self_vector = self.possible_next - self.pos
            first_edge = get_first_edge(self.pos, opos, self_vector, self.lw, obstacle.lw)
            response_dir = first_edge == 0 or first_edge == 2
            # print("first edge", first_edge)
            if first_edge == 1: # from the left
                self.possible_next[1] = opos[1] - self.width
            elif first_edge == 3: # from the right
                self.possible_next[1] = opos[1] + obstacle.width 
            elif first_edge == 0: # from below
                self.possible_next[0] = opos[0] - self.length
            else: # from above
                self.possible_next[0] = opos[0] + obstacle.length

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return None, None

    l2_diff = (line2[1][1] - line2[0][1], line2[1][0] - line2[0][0])
    start_diff = (line2[0][1] - line1[0][1], line2[0][0] - line1[0][0])
    t = det(l2_diff, start_diff) / div

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return t, (x, y)


def get_first_edge(move_loc, stationary_loc, vec, stlw, mvlw): # TODO

    top_segments = list()
    bot_segments = list()
    left_segments = list()
    right_segments = list()
    for i in range(int(mvlw[0] / 0.5) + 1): # size must be evenly divisible by 0.5
        for j in range(int(mvlw[1] / 0.5) + 1):
            if i == 0:
                point = move_loc + np.array([0, j * 0.5])
                bot_segments.append([point, point+vec])
            if j == 0:
                point = move_loc + np.array([i * 0.5, 0])
                left_segments.append([point, point+vec])
            if i == int(mvlw[0] / 0.5):
                point = move_loc + np.array([mvlw[0], j * 0.5])
                top_segments.append([point, point+vec])
            if j == int(mvlw[1] / 0.5):
                point = move_loc + np.array([i * 0.5, mvlw[1]])
                right_segments.append([ point, point+vec])
    # print(move_loc,int(mvlw[0] / 0.5) + 1, top_segments)

    s1 = stationary_loc + np.array([0, 0])
    s2 = stationary_loc + np.array([stlw[0], 0])
    s3 = stationary_loc + np.array([0, stlw[1]])
    s4 = stationary_loc + np.array([stlw[0], stlw[1]])

    ls1 = [s1,s2]
    ls2 = [s1,s3]
    ls3 = [s2,s4]
    ls4 = [s3,s4]


    ud_t = list()
    if vec[0] >= 0:
        for segment in top_segments: # if any intersections top to bottom
            t2, ls2_i = line_intersection(segment, ls2)
            if t2 is not None and t2 >= 0: ud_t.append(t2)
            # print("top line segments", t2, ls2_i, segment, ls2)
        rl_t = list()
        if vec[1] >= 0:
            for segment in right_segments: # if any intersections right to left
                t1, ls1_i = line_intersection(segment, ls1)
                if t1 is not None and t1 >= 0: rl_t.append(t1)
                # print("right line segments", t1, ls1_i, segment, ls1)
        else:
            for segment in left_segments: # if any intersections left to right
                t4, ls4_i = line_intersection(segment, ls4)
                if t4 is not None and t4 >= 0: rl_t.append(t4)
                # print("left line segments", t4, ls4_i, segment, ls4)
        first_ud = min(ud_t) if len(ud_t) > 0 else 10
        first_rl = min(rl_t) if len(rl_t) > 0 else 10
        if first_ud < first_rl:
            return 0
        else:
            return 1 if vec[1] >= 0 else 3
    else:
        for segment in bot_segments: # if any intersections top to bottom
            t3, ls3_i = line_intersection(segment, ls3)
            if t3 is not None and t3 >= 0: ud_t.append(t3)
            # print("bot line segments", t3, ls3_i, segment, ls3)
        rl_t = list()
        if vec[1] >= 0:
            for segment in right_segments: # if any intersections right to left
                t1, ls1_i = line_intersection(segment, ls1)
                if t1 is not None and t1 >= 0: rl_t.append(t1)
                # print("right line segments", t1, ls1_i, segment, ls1)
        else:
            for segment in left_segments: # if any intersections left to right
                t4, ls4_i = line_intersection(segment, ls4)
                if t4 is not None and t4 >= 0: rl_t.append(t4)
                # print("left line segments", t4, ls4_i, segment, ls4)
        first_ud = min(ud_t) if len(ud_t) > 0 else 10
        first_rl = min(rl_t) if len(rl_t) > 0 else 10
        if first_ud < first_rl:
            return 2
        else:
            return 1 if vec[1] >= 0 else 3



def check_edges(move_loc, stationary_loc, vec, stlw, mvlw): # TODO

    top_segments = list()
    bot_segments = list()
    left_segments = list()
    right_segments = list()
    for i in range(int(mvlw[0] / 0.5) + 1): # size must be evenly divisible by 0.5
        for j in range(int(mvlw[1] / 0.5) + 1):
            if i == 0:
                point = move_loc + np.array([0, j * 0.5])
                bot_segments.append([point, point+vec])
            if j == 0:
                point = move_loc + np.array([i * 0.5, 0])
                left_segments.append([point, point+vec])
            if i == int(mvlw[0] / 0.5):
                point = move_loc + np.array([mvlw[0], j * 0.5])
                top_segments.append([point, point+vec])
            if j == int(mvlw[1] / 0.5):
                point = move_loc + np.array([i * 0.5, mvlw[1]])
                right_segments.append([ point, point+vec])
    # print(move_loc,int(mvlw[0] / 0.5) + 1, top_segments)

    s1 = stationary_loc + np.array([0, 0])
    s2 = stationary_loc + np.array([stlw[0], 0])
    s3 = stationary_loc + np.array([0, stlw[1]])
    s4 = stationary_loc + np.array([stlw[0], stlw[1]])

    ls1 = [s1,s2]
    ls2 = [s1,s3]
    ls3 = [s2,s4]
    ls4 = [s3,s4]


    if vec[0] >= 0:
        for segment in top_segments: # if any intersections top to bottom
            ls1_i = line_intersection(segment, ls2)
        if vec[1] >= 0:
            for segment in right_segments: # if any intersections right to left
                ls1_i = line_intersection(segment, ls1)
        else:
            for segment in left_segments: # if any intersections left to right
                ls1_i = line_intersection(segment, ls4)
    else:
        for segment in bot_segments: # if any intersections bot to top
            ls1_i = line_intersection(segment, ls3)
        if vec[1] >= 0:
            for segment in right_segments: # if any intersections right to left
                ls1_i = line_intersection(segment, ls1)
        else:
            for segment in left_segments: # if any intersections left to right
                ls1_i = line_intersection(segment, ls1)
    # print(ls1_i)
    return ls1_i

class Pusher(MobileObject2D):
    def __init__(self, pos, attr, lw, name, size):
        super().__init__(pos, attr, lw, name)
        self.size = size

    def try_action(self, action_vec):
        self.interaction_trace.append("Action")
        self.possible_next = self.pos + action_vec
        # print("pusher next", self.possible_next, self.pos, action_vec)

    def correct_possible(self, block):
        # print("pusher correct", self.possible_next, block.possible_next, aabb_check(self.possible_next, block.possible_next, self.lw, block.lw))
        if aabb_check(self.possible_next, block.possible_next, self.lw, block.lw):
            self.interaction_trace.append(block.name)
            self.respond_obstacle(block, use_next=True)
        if self.possible_next[0] < 0: self.possible_next[0] = 0
        if self.possible_next[0] > self.size: self.possible_next[0] = self.size
        if self.possible_next[1] < 0: self.possible_next[1] = 0
        if self.possible_next[1] > self.size: self.possible_next[1] = self.size

class Block(MobileObject2D):
    def respond_pusher(self, pusher):
        self.possible_next = self.pos
        pusher_vector = pusher.possible_next - pusher.pos
        # print(check_edges(pusher.pos, self.pos, pusher_vector, pusher.lw, self.lw))
        if aabb_check(pusher.possible_next, self.pos, pusher.lw, self.lw):
            self.interaction_trace.append(pusher.name)
            self.interaction_trace.append("Action")
            pusher_vector = pusher.possible_next - pusher.pos
            first_edge = get_first_edge(pusher.pos, self.pos, pusher_vector, pusher.lw, self.lw)
            response_dir = first_edge == 1 or first_edge == 3
            if response_dir:
                if first_edge == 1: # right to left
                    response = pusher_vector[1] - (self.pos[1] - (pusher.pos[1] + pusher.width))
                if first_edge == 3: # left to right
                    response = pusher_vector[1] + (pusher.pos[1] - (self.pos[1] + self.width))
            else:
                if first_edge == 0: # top to bottom
                    response = pusher_vector[0] - (self.pos[0] - (pusher.pos[0] + pusher.length))
                if first_edge == 2: # bottom to top
                    response = pusher_vector[0] + (pusher.pos[0] - (self.pos[0] + self.length))
            self.possible_next = copy.deepcopy(self.pos)
            self.possible_next[int(response_dir)] = self.pos[int(response_dir)] + response
            # print("pushed", response, response_dir, first_edge,self.possible_next, self.pos)
        else: self.possible_next = self.pos

class Target(Object2D):
    def check_block(self, block):
        if aabb_check(self.pos, block.pos, self.lw, block.lw):
            self.interaction_trace.append(block.name)
            self.attr = 1
        else:
            self.attr = -1

action_table = [
                    np.array([0,0]),
                    np.array([0,1]),
                    np.array([1,0]),
                    np.array([0,-1]),
                    np.array([-1,0])
                ]


class Pusher2D(Environment):
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
        super().__init__(frameskip=frameskip, variant=variant, fixed_limits=fixed_limits)
        # environment properties
        self.self_reset = True
        self.discrete_actions, self.size, self.pusher_dims, self.block_dims, self.obstacle_dims, self.num_obstacles, self.max_steps = pusher_variants[variant]
        self.num_actions = 5 if self.discrete_actions else -1 # this must be defined, -1 for continuous. Only needed for primitive actions
        self.name = "Pusher2D" # required for an environment
        # self.action_table = [
        #                         np.array([0,0]),
        #                         np.array([0,1]),
        #                         np.array([1,0]),
        #                         np.array([0,-1]),
        #                         np.array([-1,0])
        #                     ]

        # spaces
        self.action_shape = (1,) if self.discrete_actions else (2,) # should be set in the environment, (1,) is for discrete action environments
        self.action_space = gym.spaces.Discrete(self.num_actions) if self.discrete_actions else gym.spaces.Box(low =np.array([-1,-1]), high = np.array([1,1])) # gym.spaces
        self.pos_size = 2 # the dimensionality, should be set

        # running values
        self.itr = 0

        # factorized state properties
        self.all_names = ["Action", "Pusher", "Block"] + ["Obstacle" + str(i) for i in range(self.num_obstacles)] + ["Target", "Reward", "Done"] # must be initialized, the names of all the objects including multi-instanced ones
        self.valid_names = copy.deepcopy(self.all_names) # must be initialized, the names of all objects used in a particular trajectory. The environment should treat as nonexistent objects which are not part of this list
        self.num_objects = len(self.all_names) # should be defined if valid, the number of objects (or virtual objects) in the flattened obs
        self.object_names = ["Action", "Pusher", "Block", "Obstacle", "Target", "Reward", "Done"] # must be initialized, a list of names that controls the ordering of things
        self.object_sizes = {"Action": self.action_shape[0], "Pusher": 3, "Block": 3, "Obstacle": 3, "Target": 3, "Reward": 1, "Done": 1} # must be initialized, a dictionary of name to length of the state
        self.object_range = {"Action": [np.array([0]), [np.array([self.action_shape[0]])]] if self.discrete_actions else [np.array([-1,-1,-1]), np.array([1,1,1])],
                             "Pusher": [np.array([0,0,-1]), np.array([self.size,self.size,1])],
                             "Block": [np.array([0,0,-1]), np.array([self.size,self.size,1])],
                             "Obstacle": [np.array([0,0,-1]), np.array([self.size,self.size,1])],
                             "Target": [np.array([0,0,-1]), np.array([self.size,self.size,1])],
                             "Reward": [np.array([-1]), np.array([1])],
                             "Done":  [np.array([0]), np.array([1])]} # the minimum and maximum values for a given feature of an object
        self.object_dynamics = {"Action": [np.array([-self.action_shape[0]]), [np.array([self.action_shape[0]])]] if self.discrete_actions else [np.array([-2,-2,-2]), np.array([2,2,2])],
                             "Pusher": [np.array([-1,-1,-2]), np.array([1,1,2])],
                             "Block": [np.array([-1,-1,-2]), np.array([1,1,2])],
                             "Obstacle": [np.array([-1,-1,-2]), np.array([1,1,2])],
                             "Target": [np.array([-1,-1,-2]), np.array([1,1,2])],
                             "Reward": [np.array([-2]), np.array([2])],
                             "Done":  [np.array([0]), np.array([1])]} # the most that an object can change in a single time step
        self.object_range_true = copy.deepcopy(self.object_range) # if using a fixed range, this stores the true range (for sampling)
        self.object_dynamics_true = copy.deepcopy(self.object_dynamics) # if using a fixed dynamics range, this stores the true range (for sampling)
        self.object_instanced = {"Action": 1, "Pusher": 1, "Block": 1, "Obstacle": self.num_obstacles, "Target": 1, "Reward": 1, "Done": 1} # name of object to max number of objects of that type
        self.object_proximal = {"Action": False, "Pusher": True, "Block": True, "Obstacle": True, "Target": True, "Reward": False, "Done": False} # name of object to whether that object has valid proximity
        self.instance_length = len(self.all_names) # the total number of instances for the mask
        self.observation_space = self.observation_space = gym.spaces.Box(low=-1, high=1, shape=[3*self.num_objects]) # raw space, gym.spaces

        # proximity state components
        self.position_masks = {
            "Action": np.array([0]),
            "Pusher": np.array([1,1,0]),
            "Block": np.array([1,1,0]),
            "Obstacle": np.array([1,1,0]),
            "Target": np.array([1,1,0]),
            "Done": np.array([0]),
            "Reward": np.array([0]),
        }
        self.itr = 0
        self.reset()
    
    def create_random_pos(self, scale = 1):
        return (self.size // 2) + ((np.random.rand(2) - 0.5) * self.size * scale)
    
    def reset(self):
        self.object_name_dict = {
            "Action": Action(self.discrete_actions, self.action_shape),
            "Pusher": Pusher(self.create_random_pos(), 0, self.pusher_dims, "Pusher", self.size),
            "Block": Block(self.create_random_pos(scale=0.5), 0, self.block_dims, "Block"),
            "Done": self.done,
            "Reward": self.reward,
        }
        self.action = self.object_name_dict["Action"]
        self.pusher = self.object_name_dict["Pusher"]
        self.block = self.object_name_dict["Block"]
        self.target = Target(self.create_random_pos(), 0, self.obstacle_dims, "Target")
        while (aabb_check(self.target.pos, self.block.pos, self.target.lw, self.block.lw)):
            self.target = Target(self.create_random_pos(), 0, self.obstacle_dims, "Target")
        self.object_name_dict["Target"] = self.target
        self.obstacles = list()
        for i in range(self.num_obstacles):
            next_obs = Object2D(self.create_random_pos(), 1, self.obstacle_dims, "Obstacle" + str(i))
            while ((aabb_check(next_obs.pos, self.pusher.pos, next_obs.lw, self.pusher.lw)) 
                    and (aabb_check(next_obs.pos, self.block.pos, next_obs.lw, self.block.lw))
                    and (aabb_check(next_obs.pos, self.target.pos, next_obs.lw, self.target.lw))):
                next_obs = Object2D(self.create_random_pos(), 1, self.obstacle_dims, "Obstacle" + str(i))
            self.object_name_dict["Obstacle" + str(i)] = next_obs
            self.obstacles.append(next_obs)
        self.itr = 0
        self.objects = [self.action, self.pusher, self.block, self.target] + self.obstacles + [self.done, self.reward]
        return self.get_state()

    def get_state(self, render=False):
        raw = self.render() if render else None
        return {"raw_state": raw, "factored_state": {name: val.get_state() for name, val in self.object_name_dict.items()}}

    def reset_interactions(self):
        for name in self.object_name_dict.keys():
            self.object_name_dict[name].interaction_trace = list()

    def convert_action(self, action):
        if self.discrete_actions:
            return action_table[action] + np.random.rand(2) * 0.2 
        else:
            return action

    def step(self, action, render=False):
        '''
            state as dict: next raw_state (image or observation) next factor_state (dictionary of name of object to tuple of object bounding box and object property)
            reward: the true reward from the environment
            done flag: if an episode ends, done is True
            info: a dict with additional info, especially TimeLimit.truncated
        '''
        for i in range(self.frameskip):
            self.action.attr = np.array(action)
            self.pusher.try_action(self.convert_action(action))
            for obstacle in self.obstacles:
                self.pusher.respond_obstacle(obstacle)
            self.block.respond_pusher(self.pusher)
            for obstacle in self.obstacles:
                self.block.respond_obstacle(obstacle)
            self.pusher.correct_possible(self.block)
            self.pusher.pos = self.pusher.possible_next
            self.block.pos = self.block.possible_next
            self.target.check_block(self.block)
            self.reward.attr = self.target.attr
        self.itr += 1
        self.done.attr = self.itr == self.max_steps
        if self.done.attr:
            self.reset()
        return self.get_state(render=render), self.reward.attr, self.done.attr, {"TimeLimit.truncated": self.itr == self.max_steps}

    def convert_pix(self, pos):
        return max(0, int(np.round(pos / (self.size) * 100)))

    def render(self, mode='human'):
        self.frame = np.zeros((100,100), dtype = 'uint8')
        for obstacle in self.obstacles:
            if obstacle.attr != 0:
                self.frame[self.convert_pix(obstacle.pos[0]):self.convert_pix(obstacle.pos[0])+self.convert_pix(obstacle.length), self.convert_pix(obstacle.pos[1]):self.convert_pix(obstacle.pos[1])+self.convert_pix(obstacle.width)] = int(.5 * 255)
                # print("obstacle", obstacle.pos, self.size, self.convert_pix(obstacle.pos[0]),self.convert_pix(obstacle.pos[0])+self.convert_pix(obstacle.length), self.convert_pix(obstacle.pos[1]),self.convert_pix(obstacle.pos[1])+self.convert_pix(obstacle.width))
        pusher, block, target = self.pusher, self.block, self.target
        self.frame[self.convert_pix(pusher.pos[0]):self.convert_pix(pusher.pos[0])+self.convert_pix(pusher.length), self.convert_pix(pusher.pos[1]):self.convert_pix(pusher.pos[1])+self.convert_pix(pusher.width)] = int(0.75 * 255)
        # print("pusher", self.action.attr, pusher.pos, self.size, self.convert_pix(pusher.pos[0]),self.convert_pix(pusher.pos[0])+self.convert_pix(pusher.length), self.convert_pix(pusher.pos[1]),self.convert_pix(pusher.pos[1])+self.convert_pix(pusher.width))
        # print("block", self.action.attr, block.pos, self.size, self.convert_pix(block.pos[0]),self.convert_pix(block.pos[0])+self.convert_pix(block.length), self.convert_pix(block.pos[1]),self.convert_pix(block.pos[1])+self.convert_pix(block.width))
        self.frame[self.convert_pix(block.pos[0]):self.convert_pix(block.pos[0])+self.convert_pix(block.length), self.convert_pix(block.pos[1]):self.convert_pix(block.pos[1])+self.convert_pix(block.width)] = int(1 * 255)
        self.frame[self.convert_pix(target.pos[0]):self.convert_pix(target.pos[0])+self.convert_pix(target.length), self.convert_pix(target.pos[1]):self.convert_pix(target.pos[1])+self.convert_pix(target.width)] = int(0.25 * 255)
        return self.frame

    def demonstrate(self):
        action = 0
        frame = self.render()
        frame = cv2.resize(frame, (frame.shape[0] * 3, frame.shape[1] * 3), interpolation = cv2.INTER_NEAREST)
        cv2.imshow('frame',frame)
        # print(self.get_state()["factored_state"])
        key = cv2.waitKey(1000)
        action = 0
        if key == ord('q'):
            action = -1
        elif key == ord('a'):
            action = 3
        elif key == ord('w'):
            action = 4
        elif key == ord('s'):
            action = 2
        elif key == ord('d'):
            action = 1
        elif key == ord(' '):
            action = 0
        return action

    def get_info(self): # returns the info, the most important value is TimeLimit.truncated, can be overriden
        return {"TimeLimit.truncated": self.itr == self.max_steps}

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
        self.pusher.pos = factored_state["Pusher"][:2]
        self.block.pos = factored_state["Block"][:2]
        self.target.pos = factored_state["Target"][:2]
        self.target.attr = factored_state["Target"][-1]
        for i in range(self.num_obstacles):
            self.obstacles[i].pos = factored_state["Obstacle" + str(i)][:2]
        self.reward.attr = self.target.attr
        self.done.attr = False


class GreedyTowards():
    def __init__(self, discrete_actions, action_shape):
        self.action_shape = action_shape
        self.discrete_actions = discrete_actions 

    def act(self, screen, angle=0):
        vec = screen.block.get_midpoint() - screen.pusher.get_midpoint()
        valid_actions = list()
        if self.discrete_actions: 
            if vec[0] > 0: valid_actions.append(2)
            elif vec[0] < 0: valid_actions.append(4)
            if vec[1] > 0: valid_actions.append(1)
            elif vec[1] < 0: valid_actions.append(3)
            if len(valid_actions) == 0: valid_actions = np.arange(5)
            return np.random.choice(valid_actions)
        else:
            rand_val = np.random.rand(2) * np.sign(vec) 
            # rand_val[0] = rand_val[0] * np.random.rand()
            # print("greedy vec", vec, rand_val, screen.block.pos, screen.pusher.pos)
            return rand_val

class RandGreedy():
    def __init__(self, discrete_actions, action_shape, random_rate):
        self.greedy = GreedyTowards(discrete_actions, action_shape)
        self.random_rate = random_rate
        self.discrete_actions = discrete_actions
        self.action_shape = action_shape
    
    def act(self, screen, angle=0): 
        greedy_act = self.greedy.act(screen)
        random_act = np.random.randint(5) if self.discrete_actions else 2*(np.random.rand(2) - 0.5)
        return random_act if np.random.rand() < self.random_rate else greedy_act

class RandGreedySticky():
    def __init__(self, discrete_actions, action_shape, random_rate):
        self.greedy = GreedyTowards(discrete_actions, action_shape)
        self.random_rate = random_rate
        self.discrete_actions = discrete_actions
        self.action_shape = action_shape
        self.sticky_counter = 1
        self.sticky_num = 3
        self.last_action = 2*(np.random.rand(2) - 0.5)
    
    def act(self, screen, angle=0):
        self.sticky_counter += 1
        greedy_act = self.greedy.act(screen)
        if self.sticky_counter % self.sticky_num == 0:
            random_act = np.random.randint(5) if self.discrete_actions else 2*(np.random.rand(2) - 0.5)
            random_act = random_act if np.random.rand() < self.random_rate else greedy_act
            self.last_action = random_act
            self.sticky_counter = 1
            return random_act
        else:
            random_act = np.random.rand(2) * np.sign(self.last_action)
            self.sticky_counter += 1
            return random_act