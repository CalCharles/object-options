class Sampler():
    def __init__(self, **kwargs):
        self.mask = kwargs["mask"]
        self.target_selector = kwargs["target_select"]
        self.test_sampler = True if "test_sampler" in kwargs and kwargs["test_sampler"] else False

    def update(self):
        # if there is a schedule for the sampler values, updates the schedule
        return

    def sample(self, full_state):
        '''
        samples a new value according to the sampling function
        only returns a single sample. Full state is used to bring in relevant information to the sampling
        '''
        return