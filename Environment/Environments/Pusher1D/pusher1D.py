# 1D pusher Domain

class Pusher1D(Environment):
    def __init__(self):
        self.pusher = [0]
        self.obstacle = [0]
        self.action = 0
        self.num_objects = 3
        self.all_states = np.array(np.meshgrid([0,1], [0,1,2], [0,1,2])).T.reshape(-1,3)
        self.outcomes = list()
        for state in self.all_states:
            self.outcomes.append(self.step(state[0], state))
        self.reset()
    
    def get_state(self):
        return [self.action, self.pusher[0], self.obstacle[0]]

    def reset(self):
        self.pusher = [np.random.randint(3)]
        self.obstacle = [np.random.randint(3)]

    def step(self, action, state = None):
        action_step = (action - 0.5) * 2
        if state is None:
            self.action = action
            nextpusher = self.pusher[0] + action_step
            if nextpusher != self.obstacle[0] and (0 <= nextpusher < 3):
                self.pusher = [nextpusher]
            return self.get_state()
        else:
            nextpusher = state[1] + action_step
            if nextpusher != state[2] and (0 <= nextpusher < 3):
                return np.array([nextpusher])
            return np.array([state[1]])
