
class RTD():
    def __init__(self, **kwargs): 
        self.compute_done = OptionControl(**kwargs)

    def update(self):
        self.compute_done.update()

    def compute_rew_term_done(self, full_state, next_full_state, param, mask, true_done, true_reward):
        return True, 0.0, True

    def __call__(self, full_state, next_full_state, param, mask, true_done, true_reward):
        term, rew, inter = self.compute_rew_term_done(full_state, next_full_state, param, mask, true_done, true_reward)
        term, rew, inter = term.squeeze(), rew.squeeze(), inter.squeeze()
        cutoff = self.compute_done.check_timer()
        ret_term = term or cutoff
        done = self.compute_done(term, true_done)
        done = done.squeeze()
        if ret_term: self.compute_done.reset()
        return ret_term, rew, done, inter, cutoff and not term

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
        return self.timer >= self.time_cutoff and self.time_cutoff > 0

    def __call__(self, term, true_done):
        return term * self.term_as_done + true_done * self.true_done