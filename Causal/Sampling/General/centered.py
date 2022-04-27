class CenteredSampler(Sampler):
    def __init__(self, **kwargs):
        self.distance = kwargs["sample_distance"]
        self.clipping_ratio = kwargs["sample_clipping"]
        self.schedule_counter = 0
        self.schedule = kwargs["sample_schedule"]
        super().__init__(**kwargs)
        self.current_distance = .1 if self.schedule > 0 else kwargs["sample_distance"]

    def update(self):
        if self.schedule > 0:
            self.schedule_counter += 1
            self.current_distance = self.distance - (self.distance - self.current_distance) * np.exp(-(self.schedule_counter + 1)/self.schedule)

    def sample(self, full_state):
        '''
        samples a new value: full_state
        '''
        target = self.target_select(full_state["factored_state"])
        limit_ranges = np.min(self.masking.limits[1], target + self.current_distance, axis=-1) - np.max(target - self.current_distance, self.masking.limits[0], axis=-1) 
        return (target + limit_ranges * weights) * self.masking.active_mask