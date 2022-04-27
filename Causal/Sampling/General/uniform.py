class UniformSampler(Sampler):
    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        weights = np.random.rand(size = self.masking.shape)
        return (self.masking.limits[0] + self.masking.range * weights) * self.masking.active_mask