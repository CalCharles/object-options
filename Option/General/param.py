class BinaryParameterizedOptionControl(Reward):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epsilon_close = kwargs['epsilon_close']
        self.norm_p = kwargs['param_norm']
        self.constant_lambda = kwargs['constant_lambda']

    def compute_rew_term_done(self, full_state, next_full_state, param, mask, true_done, true_reward):
        state = self.target_select(next_full_state)
        inside = np.linalg.norm((state - param) * mask, ord = self.norm_p, axis=-1) <= self.epsilon_close
        term, rew = inside.copy(), inside.copy().astype(np.float64)
        return term, rew + self.constant_lambda, self.compute_done(term, true_done)