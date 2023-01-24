import numpy as np
import time
from Baselines.HyPE.Policy.skill import Skill
from Option.action_map import PrimitiveActionMap
from State.object_dict import ObjDict

class PrimitiveExtractor():
    def get_parent(self, full_state):
        return full_state['factored_state']["Action"]

    def get_target(self, full_state):
        return full_state['factored_state']["Action"]

    def get_obs(self, full_state):
        return full_state['factored_state']["Action"]

    def get_diff(self, full_state, next_full_state):
        return full_state['factored_state']["Action"]

class PrimitiveSkill(Skill): # primitive discrete actions
    def __init__(self, environment):

        # parameters for saving
        self.name = "Action"
        self.train_epsilon = 0

        # primary models
        self.policy = None # policy to run during opion
        self.assignment_mode = False

        # assigns state extractor, rew_term_done, action_map, 
        self.extractor = PrimitiveExtractor() # extracts the desired state
        self.rew_term_done = None
        self.reward_model = ObjDict()
        self.num_skills = environment.num_actions
        self.action_space = environment.action_space
        self.temporal_extension_manager = None
        self.initiation_set = None # TODO: handle initiation states
        self.next_option = None
        self.interaction_model = None
        self.inline_trainer = ObjDict({'interaction_model': None})
        # cuda handling
        self.iscuda = False
        self.device = None

    def set_debug(self, debug):
        self.debug = debug

    def reset(self, full_state):
        return [True]

    def load_policy(self, load_dir):
        pass

    def update(self, act, chain, masks, update_policy=True):
        pass

    def cpu(self):
        self.iscuda = False

    def cuda(self, device=None):
        self.iscuda = True
        if device is not None:
            self.device=device
    
    def extended_action_sample(self, batch, state_chain, term, ext_terms, random=False):
        return (*self.sample_action_chain(batch, state_chain, random), True)

    def assign_interaction_model(self, interaction_model):
        self.interaction_model = interaction_model
        return interaction_model

    def sample_action_chain(self, batch, state, random=False): # param is an int denoting the primitive action, not protected (could send a faulty param)
        # start = time.time()
        sq_param = batch['assignment'].squeeze()
        if random: sq_param = self.action_map.sample()
        chain = [np.array(sq_param)]
        # print("primitive", time.time() -start)
        return sq_param, chain, None, list() # chain is the action as an int, policy batch is None, state chain is a list, resampled is True

    def terminate_chain(self, full_states, true_done=False, first=False):
        return [True]

    def zero_below_grads(self, top=False):
        pass