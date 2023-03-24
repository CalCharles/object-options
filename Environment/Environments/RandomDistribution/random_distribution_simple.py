import numpy as np
import copy

def get_state(names, factored_state):
    return np.concatenate([factored_state[n] for n in names], axis=-1)

def rand(*args):
    # randomizes between -1,1
    return (np.random.rand(*args) - 0.5) * 2

class add_func():
    # implements (Ax + By)
    def __init__(self, parents, target, feature_dim, step_size):
        self.parents = parents
        self.target = target
        self.A = rand(feature_dim * len(parents), feature_dim) * step_size / np.sqrt(feature_dim)
        self.B = rand(feature_dim, feature_dim) * step_size / np.sqrt(feature_dim)
        print("A,B")
        print(self.A)
        print(self.B)
    
    def __call__(self, factored_state): 
        x = np.expand_dims(get_state(self.parents, factored_state), -1)
        y = np.expand_dims(get_state([self.target], factored_state), -1)
        return np.matmul(self.A.T, x) + np.matmul(self.B, y)

class rel_func():
    # implements c^top (Ax + By) > tau
    def __init__(self, parents, target, feature_dim, step_size, tau):
        self.parents = parents
        self.target = target
        self.tau = tau
        self.add = add_func(parents, target, feature_dim, step_size)
        self.c = rand(feature_dim)
        print("c", self.c)

    def __call__(self, factored_state):
        x = get_state(self.parents, factored_state)
        y = get_state([self.target], factored_state)
        return float(np.sum(self.c * self.add(factored_state)) > self.tau)

class passive_func():
    # implements an add function
    def __init__(self, parents, target, feature_dim, forward_step, idxes):
        # parents is list of target names
        # target is target name
        self.parents = parents
        self.target = target
        print(self.parents, self.target)
        self.add = add_func(parents, target, feature_dim, forward_step)
        self.idxes = idxes

    def __call__(self, factored_state):
        x = get_state(self.parents, factored_state)
        y = get_state([self.target], factored_state)
        return 1, self.add(factored_state)

class cond_func():
    # implements c^top (Ax + By) > tau = i, y' = (Cx + Dy)i
    def __init__(self, parents, target, feature_dim, forward_step, rel_step, tau, idxes):
        # parents is list of parent names
        # target is target name
        # feature dim is size of padded feature
        # step size is the step size
        self.parents = parents
        self.target = target
        print(self.parents, self.target)
        self.rel = rel_func(parents, target, feature_dim, rel_step, tau)
        self.add = add_func(parents, target, feature_dim, forward_step)
        self.idxes = idxes

    def __call__(self, factored_state):
        x = get_state(self.parents, factored_state)
        y = get_state([self.target], factored_state)
        i = self.rel(factored_state)
        return i, i * self.add(factored_state)

class RandomDistribution():
    def __init__(self, num_cond, max_parents, num_objects, tau, passive, noise=0.0001, episode_length = 50):
        self.names = ["Action"]
        self.feature_dim = 4
        self.forward_step = 0.03
        self.action_dim = self.feature_dim
        self.noise = noise
        tau = tau - 0.5
        self.dims = [self.feature_dim] # action always 5D
        for i in range(num_objects):
            self.names.append(str(chr(ord('a')+int(i // 26)) + chr(ord('A')+int(i % 26))))
            self.dims.append(np.random.randint(2,5))
        self.conditions = list()

        # always add action dependance
        if num_cond > 0:
            target = np.random.choice(self.names)
            while target == "Action": target = np.random.choice(self.names)
            parent_names = copy.deepcopy(self.names)
            parent_names.pop(self.names.index(target))
            parent_names.pop(parent_names.index("Action"))  
            parent_names = np.random.choice(parent_names, size = np.random.randint(0,num_objects-2), replace=False).tolist()
            parent_names = ["Action"] + parent_names
            self.conditions.append(cond_func(parent_names, target, self.feature_dim, self.forward_step, 1, -100, self.get_indices(parent_names+ [target])))
        # add other conditions
        for i in range(num_cond-1):
            target = np.random.choice(self.names)
            while target == "Action": target = np.random.choice(self.names)
            parent_names = copy.deepcopy(self.names)
            parent_names.pop(self.names.index(target))
            parent_names = np.random.choice(parent_names, size = np.random.randint(1,num_objects-1), replace=False).tolist()
            self.conditions.append(cond_func(parent_names, target, self.feature_dim, self.forward_step, 1, tau, self.get_indices(parent_names+ [target])))
        if passive:
            for n in self.names:
                if n != "Action": self.conditions.append(passive_func([n], n, self.feature_dim, self.forward_step, self.get_indices([target])))
        self.episode_length = episode_length
        self.reset()
        self.stats = np.zeros(len(self.conditions))


    def reset(self):
        self.counter = 0
        self.states = {n: rand(self.feature_dim) / 2 for n in self.names}
        for i, n in enumerate(self.names):
            self.states[n][self.dims[i]:] = 0
        return self.get_state()

    def get_state(self):
        return copy.deepcopy(self.states)

    def get_indices(self, names):
        idxes = list()
        for i, n in enumerate(self.names):
            if n in names: idxes.append(i)
        return idxes

    def step(self, action):
        self.states["Action"] = action
        state = self.get_state()
        traces = {n: np.zeros(len(self.names)) for n in self.names}
        for i,n in enumerate(self.names):
            traces[n][i] = 1
        
        # dynamic state gets added, with some random jitter
        dyn_state = {n: np.random.normal(0,1,size=self.feature_dim) * self.noise for n in self.names}
        # apply the causal functions
        for i, cond in enumerate(self.conditions):
            trace, dyn_val = cond(state)
            dyn_state[cond.target] += dyn_val.squeeze()
            if trace: traces[cond.target][cond.idxes] = 1
            self.stats[i] += trace 
        
        # update the state
        for i, n in enumerate(self.names):
            self.states[n] += dyn_state[n]
            self.states[n][self.dims[i]:] = 0
        self.counter += 1
        done = False
        if self.counter == self.episode_length:
            self.reset()
            done = True

        return self.get_state(), -1, done, { "trace": traces, "stats": self.stats, "truncate": done}
    
if __name__ == "__main__":
    # number of random conditions, max_parents per random condition (could be less)
    # num_objects, threshold for an interaction 0.5 gives the decision plane at 0.0, passive zero if False, otherwise there are constant passive dyanmics
    env = RandomDistribution(3, 3, 5, 0.5, True)
    for i in range(10000):
        action = rand(env.action_dim)
        state, reward, done, info = env.step(action)
        print(state)
        print(info["trace"])
        print(info["stats"] / (i+1))