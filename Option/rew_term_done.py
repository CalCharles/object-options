

class OptionControl():
    def __init__(self, **kwargs):
        self.term_as_done = kwargs["term_as_done"]
        self.true_done = kwargs["true_done"]
        self.target_select = kwargs["target_select"]
        self.time_cutoff = kwargs["time_cutoff"]
        self.timer = 0

    def update(self):
        self.timer += 1

    def reset(self):
        self.timer = 0

    def check_timer(self):
        return self.timer >= self.time_cutoff

    def compute_done(self, term, true_done):
        return term * self.term_as_done + true_done * self.true_done

    def __call__(self, full_state, next_full_state, param, mask, true_done, true_reward):
        '''
        defined in subclass, outputs reward, termination and done and time cutoff computation
        '''
        return 0,0,0,0,0