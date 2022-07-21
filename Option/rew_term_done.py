import numpy as np

class RTD():
    def __init__(self, **kwargs): 
        self.compute_done = OptionControl(**kwargs)
        self.interaction_model = kwargs["interaction_model"]
        self.between_terminate = kwargs["between_terminate"]
        self.since_last_terminate = self.between_terminate

    def update(self):
        self.compute_done.update()

    def compute_rew_term_done(self, full_state, next_full_state, param, mask, true_done, true_reward):
        return True, 0.0, True

    def __call__(self, inter_state, target, next_target, param, mask, true_done, true_reward, reset = True):
        # if reset is true, then include the timer to calculate terminations, otherwise, ignore the timer
        self.since_last_terminate += 1
        term, rew, inter = self.compute_rew_term_done(inter_state, target, next_target, param, mask, true_done, true_reward)
        term, rew, inter = term.squeeze(), rew.squeeze(), inter.squeeze()
        cutoff = self.compute_done.check_timer()
        # print(term, term or cutoff)
        if reset: 
            ret_term = term or cutoff
            if np.any(term) and self.since_last_terminate < self.between_terminate:
                ret_term = False
        else: ret_term = term
        # print(term, ret_term, reset, np.any(term), self.since_last_terminate, self.between_terminate)
        if np.any(ret_term): self.since_last_terminate = 0
        done = self.compute_done(term, true_done.squeeze())
        done = done.squeeze()
        # print(term, rew, inter, cutoff, ret_term, done)
        if reset and (ret_term or done): self.compute_done.reset()
        return ret_term, rew, done, inter, cutoff and not term

class OptionControl():
    def __init__(self, **kwargs):
        self.term_as_done = kwargs["term_as_done"]
        self.true_done = kwargs["true_done"]
        self.time_cutoff = kwargs["time_cutoff"]
        self.timer = 0

    def update(self):
        self.timer += 1

    def reset(self):
        self.timer = 0

    def check_timer(self):
        return self.timer >= self.time_cutoff and self.time_cutoff > 0

    def __call__(self, term, true_done):
        if true_done is None:
            return term * self.term_as_done
        return term * self.term_as_done + true_done * self.true_done