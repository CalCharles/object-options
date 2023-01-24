import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import sys, glob, copy, os, collections, time
import numpy as np
from Network.network_utils import pytorch_model, get_parameters, set_parameters
import cma, cv2
import time
from tianshou.data import Batch, ReplayBuffer, to_numpy, to_torch_as
import torch.nn as nn

class CMAES_optimizer(nn.Module):
    def __init__(self, num_population, init_var, models, num_actions, gamma, elitism, dist_fn, reward_classes = None, needs_init=False):
        super().__init__()
        self.optimizers = []
        self.solutions = []
        xinit = pytorch_model.unwrap(get_parameters(models[0]))

        # xinit = (np.random.rand(train_models.currentModel().networks[0].count_parameters())-0.5)*args.init_var # initializes [-1,1]
        sigma = init_var
        cmaes_params = {"popsize": num_population} # might be different than the population in the model...
        cmaes = cma.CMAEvolutionStrategy(xinit, sigma, cmaes_params)
        self.optimizer = cmaes
        self.num_population = num_population
        solutions = cmaes.ask()
        # self.assign_solutions(models)
        self.models = nn.ModuleList(models)
        self.mean, self.best = copy.deepcopy(self.models[0]), copy.deepcopy(self.models[0])
        self.policy_index = 0
        self.dist_fn = torch.distributions.Categorical
        self.deterministic = False
        self.num_actions = num_actions
        self.gamma = gamma
        self.elitism = elitism
        self.using_best = False

        self.updating = False

    def next_policy(self):
        self.policy_index = (self.policy_index + 1) % self.num_population
        print("next_policy", self.policy_index, self.using_best)

    def first_policy(self):
        self.policy_index = 0

    def neg_policy(self):
        self.policy_index = -1

    def set_eps(self, epsilon):
        self.epsilon = epsilon
        if epsilon < 0: # right now use negative epsilon for deterministic
            self.deterministic = True
        else: self.deterministic = False

    def assign_solutions(self, solutions, exclude=[]):
        for j in range(self.num_population):
            if j not in exclude:
                set_parameters(self.models[j], solutions[j])

    def post_process_fn(self, batch, use_buffer, indice):
        # needed to match tianshou learning algorithms, but unneeded for cmaes
        return

    def get_returns(self, batch, num_samples):
        returns = list()
        for k in range(int(len(batch) // num_samples)):
            sample_batch = batch[k*num_samples:(k+1)*num_samples]
            sample_returns = list()
            total_return = 0
            gamma = self.gamma
            for i, b in enumerate(sample_batch):
                # print(k, b.network_index)
                total_return += b.rew #  * np.pow(gamma, i) rather than discounting, use a negative constant reward
                if np.any(b.done):
                    if not np.any(b.truncated): sample_returns.append(total_return) # we drop truncated trajectories
                    total_return = 0
                # print(b.done, np.any(b.truncated), sample_returns)
            if len(sample_returns) == 0:
                sample_returns.append(total_return) # take the total return
            returns.append(np.sum(sample_returns))
        print("returns", returns, len(returns), len(batch), num_samples)
        return np.array(returns)

    def learn(self, batch, **kwargs):
        # batch is a sequential dataset with num_population * num_samples per population values
        models = self.models
        # print(len(batch), kwargs["policy_iters"])
        returns = self.get_returns(batch, kwargs["policy_iters"])
        sorted_returns, best_indexes = self.sort_best(returns)
        # print("learning", len(self.models), len(returns), self.policy_index)
        solutions = [pytorch_model.unwrap(get_parameters(m)) for m in models]
        print("num sol ret", len(solutions), len(returns))
        self.optimizer.tell(solutions, -1*returns)
        solutions = self.optimizer.ask()
        exclude = list() if self.elitism <= 1 else best_indexes[len(best_indexes) - self.elitism:len(best_indexes) - 1]
        self.assign_solutions(solutions, exclude=exclude)
        best = self.optimizer.result[0]
        mean = self.optimizer.result[int(self.num_population // 2)]
        set_parameters(models[0], best) # elitism for the best model
        set_parameters(self.best, best)
        set_parameters(self.mean, mean)
        return Batch(solutions=solutions)

    def sort_best(self, returns):
        returns = []
        idxes = np.argsort(np.array(returns))
        idxes = idxes[-self.num_population:]
        return np.array(returns)[idxes], np.array(idxes)

    def post_process_fn(self, batch, use_buffer, indice):
        return batch

    def process_fn(self, batch, use_buffer, indice):
        # we don't need a process function
        return batch

    def exploration_noise(self, act, batch):
        if np.random.rand() < self.epsilon:
            return pytorch_model.wrap([np.random.randint(self.num_actions)], cuda=act.is_cuda)
        return act

    def __call__(self, batch, state = None, input="obs", **kwargs):
        if self.using_best: model = self.best
        else: model = self.models[self.policy_index]
        logits, hidden = model(batch.obs, state=state)
        if isinstance(logits, tuple):
            dist = self.dist_fn(*logits)
        else:
            dist = self.dist_fn(logits)
        if self.deterministic:
            act = [logits.argmax(-1)]
        else:
            act = dist.sample()
        return Batch(logits=logits, act=act, state=hidden, dist=dist)
