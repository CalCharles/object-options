

class OptionControl():
    def __init__(self, **kwargs):
        self.term_as_done = kwargs["term_as_done"]
        self.true_done = kwargs["true_done"]
        self.target_select = kwargs["target_select"]

    def compute_done(self, term, true_done):
        return term * self.term_as_done + true_done * self.true_done

    def compute_rew_term_done(self, full_state, next_full_state, param, mask, true_done, true_reward):
        '''
        defined in subclass, outputs reward, termination and done computation
        '''
        return 0,0,0

class Reward():
    def __init__(self, **kwargs):
        pass
        
    def get_reward(self, inter, state, param, mask, true_reward=0, info=None):
        return 1