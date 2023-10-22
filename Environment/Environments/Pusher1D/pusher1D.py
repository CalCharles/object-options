# 1D pusher Domain
from Environment.environment import Environment
import numpy as np
import numpy as np
import gymnasium as gym
from Environment.environment import Environment
from Environment.Environments.ACDomains.ac_domain import ACDomain, ACObject

PUSHER_LEN = 3
def pusher_next(objects):
    nextpusher = objects["Pusher"].attribute + 1 # action_step
    objects["PusherNext"].attribute = objects["Pusher"].attribute
    if nextpusher != objects["Obstacle"].attribute: #and (0 <= nextpusher < PUSHER_LEN+1):
        objects["PusherNext"].attribute = nextpusher


class Pusher1D(ACDomain):
    def __init__(self, cf_states=False):
        self.all_names = ["Pusher", "Obstacle", "PusherNext"] # Action
        self.objects = {"Pusher": ACObject("Pusher", PUSHER_LEN+1),
                        "Obstacle": ACObject("Obstacle", PUSHER_LEN),
                        "PusherNext": ACObject("PusherNext", PUSHER_LEN)} # dict of name to value
        self.binary_relations = [pusher_next] # must get set prior to calling super (), the order follows the order of operations
        self.relation_outcome = ["PusherNext"]
        self.passive_mask = np.array([0,0])
        self.outcome_variable = "PusherNext"
        super().__init__(cf_states=cf_states)

# PUSHER_LEN = 3
# def pusher_next(objects):
#     nextpusher = objects["Pusher"].attribute + objects["Action"].attribute - 1 # action_step
#     objects["PusherNext"].attribute = objects["Pusher"].attribute
#     if nextpusher != objects["Obstacle"].attribute and (0 <= nextpusher <= PUSHER_LEN):
#         objects["PusherNext"].attribute = nextpusher


# class Pusher1D(ACDomain):
#     def __init__(self):
#         self.all_names = ["Action", "Pusher", "Obstacle", "PusherNext"] # Action
#         self.objects = {"Action": ACObject("Action", 3),
#                         "Pusher": ACObject("Pusher", PUSHER_LEN),
#                         "Obstacle": ACObject("Obstacle", PUSHER_LEN),
#                         "PusherNext": ACObject("PusherNext", PUSHER_LEN)} # dict of name to value
#         self.binary_relations = [pusher_next] # must get set prior to calling super (), the order follows the order of operations
#         self.relation_outcome = ["PusherNext"]
#         self.passive_mask = np.array([0,0])
#         self.outcome_variable = "PusherNext"
#         super().__init__()


        # self.pusher = [0]
        # self.obstacle = [0]
        # # self.action = 0
        # self.num_objects = 2
        # self.all_states = np.array(np.meshgrid([0,1,2], [0,1,2])).T.reshape(-1,2)
        # self.outcomes = list()
        # self.passive_mask = np.array([1,0])
        # for state in self.all_states:
        #     self.outcomes.append(self.step(state[0], state))
        #     print(state, self.outcomes[-1])
        # self.reset()
        # print(np.concatenate([self.all_states, np.array(self.outcomes)], axis=-1))

    # def get_state(self):
    #     return [self.pusher[0], self.obstacle[0]]

    # def reset(self):
    #     self.pusher = [np.random.randint(2)]
    #     self.obstacle = [np.random.randint(2)]

    # def step(self, action, state = None):
    #     # action_step = (action - 0.5) * 2
    #     if state is None:
    #         self.action = action
    #         nextpusher = self.pusher[0] + 1 # action_step
    #         if nextpusher != self.obstacle[0] and (0 <= nextpusher <= 2):
    #             self.pusher = [nextpusher]
    #         return self.get_state()
    #     else:
    #         nextpusher = state[0] + 1 # action_step
    #         if nextpusher != state[1] and (0 <= nextpusher <= 2):
    #             return np.array([nextpusher])
    #         return np.array([state[0]])
