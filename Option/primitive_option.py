
class PrimitiveStateExtractor():
    def get_obs(self, full_state, param, mask):
        return full_state["Action"]

class PrimitiveOption(Option): # primitive discrete actions
    def __init__(self, args, policy):

        # parameters for saving
        self.name = "Action"

        # primary models
        self.sampler = None # samples params
        self.policy = None # policy to run during opion

        # assigns state extractor, rew_term_done, action_map, 
        self.state_extractor = PrimitiveStateExtractor() # extracts the desired state
        self.rew_term_done = None
        self.action_map = PrimitiveActionMap() # object dict with action spaces
        self.temporal_extension_manager = None
        self.initiation_set = None # TODO: handle initiation states
        # cuda handling
        self.iscuda = False
        self.device = None


    def reset(self, full_state):
        return [True]

    def save(self, save_dir, clear=False):
        return self

    def load_policy(self, load_dir):
        pass

    def update(self, buffer, done, last_state, act, chain, term_chain, param, masks, update_policy=True):
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
        chain = [sq_param]
        return sq_param, chain, None, list(), list() # chain is the action as an int, policy batch is None, state chain is a list, resampled is True

    def terminate_reward_chain(self, state, next_state, param, chain, mask=None, needs_reward=False):
        return 1, [0], [1], True, True
