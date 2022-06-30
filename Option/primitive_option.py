import numpy as np
from Option.option import Option
from Option.action_map import PrimitiveActionMap
from State.object_dict import ObjDict

class PrimitiveSampler():
    def __init__(self, action_shape):
        self.mask = ObjDict({'active_mask': np.ones(action_shape)})

class PrimitiveStateExtractor():
    def get_obs(self, last_full_state, full_state, param, mask):
        return full_state['factored_state']["Action"]

class PrimitiveOption(Option): # primitive discrete actions
    def __init__(self, args, policy, environment):

        # parameters for saving
        self.name = "Action"
        self.train_epsilon = 0

        # primary models
        self.sampler = PrimitiveSampler(environment.action_shape) # samples params
        self.policy = None # policy to run during opion

        # assigns state extractor, rew_term_done, action_map, 
        self.state_extractor = PrimitiveStateExtractor() # extracts the desired state
        self.rew_term_done = None
        self.action_map = PrimitiveActionMap(args.environment, np.array([[i] for i in range (environment.num_actions)]) if environment.discrete_actions else list()) # object dict with action spaces
        self.terminate_reward = ObjDict()
        self.temporal_extension_manager = None
        self.initiation_set = None # TODO: handle initiation states
        self.next_option = None
        # cuda handling
        self.iscuda = False
        self.device = None


    def reset(self, full_state):
        return [True]

    def load_policy(self, load_dir):
        pass

    def update(self, act, chain, term_chain, masks, update_policy=True):
        pass

    def cpu(self):
        self.iscuda = False

    def cuda(self, device=None):
        self.iscuda = True
        if device is not None:
            self.device=device
    
    def extended_action_sample(self, batch, state_chain, term_chain, ext_terms, random=False, use_model=False):
        return (*self.sample_action_chain(batch, state_chain, random, use_model), True)

    def sample_action_chain(self, batch, state, random=False, use_model=False): # param is an int denoting the primitive action, not protected (could send a faulty param)
        sq_param = batch['param'].squeeze()
        if random: sq_param = self.action_map.sample()
        chain = [np.array(sq_param)]
        return sq_param, chain, None, list(), [np.ones(sq_param.shape)] # chain is the action as an int, policy batch is None, state chain is a list, resampled is True

    def terminate_reward_chain(self, state, next_state, param, chain, mask=None, needs_reward=False):
        return 1, [0], [1], True, True
