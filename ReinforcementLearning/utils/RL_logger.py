import logging
from Network.network_utils import pytorch_model
import numpy as np
import os
import time
from Record.file_management import create_directory
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from Record.logging import Logger

class RLLogger(Logger):

    def __init__(self, name, record_rollouts, log_interval, maxlen, log_filename=""):
        super().__init__(log_filename)
        self.record_rollouts = record_rollouts
        if len(record_rollouts) != 0:
            full_logdir = os.path.join(create_directory(record_rollouts+ "/logs"))
            self.tensorboard_logger = SummaryWriter(log_dir=full_logdir)
        self.name = name
        self.log_interval = log_interval
        self.maxlen = maxlen

        self.reset()

    def reset(self):
        # resets the logger values
        self.total_steps = 0
        self.total_episodes = 0
        self.term_count = 0
        self.time = time.time()
        self.current_episodes = deque(maxlen=self.maxlen)
        self.current_steps = deque(maxlen=self.maxlen)
        self.current_term = deque(maxlen=self.maxlen)
        self.reward = deque(maxlen=self.maxlen)
        self.success = deque(maxlen=self.maxlen)
        self.total_losses = dict()
        
    def log_results(self, result):
        # adds to the running totals 
        self.total_steps += result["n/st"]
        self.total_episodes += result["n/ep"]
        self.term_count += result["n/tr"]

        # appends to the rolling totals
        self.current_steps.append(result["n/st"])
        self.current_episodes.append(result["n/ep"])
        self.current_term.append(result["n/tr"])

        self.reward.append(result["rews"])
        self.success.append(result["n/h"])

    def log_losses(self, losses):
        # losses may differ for values 
        for k in losses.keys():
            if k not in self.total_losses:
                self.total_losses[k] = deque(maxlen=self.maxlen)
            self.total_losses[k].append(losses[k])

    def log_frames(self, frame):
        # TODO: logs images of interest
        pass

    def print_log(self, i, force=False): # force forces the printout of log values
        print(self.log_interval)
        if i % self.log_interval == 0 or force:
            # first line logs the rolling totals
            log_string = self.name + f': Iters: {i}, Steps: {self.total_steps}, Episodes: {self.total_episodes}, FPS: {self.total_steps/(time.time()-self.time)}'
            # gives per step, per termination and per episode returns
            log_string += f'\nReturn (step, term, ep): {np.sum(self.reward)/np.sum(self.current_steps)}, {np.sum(self.reward)/np.sum(self.current_term)}, {np.sum(self.reward)/max(1, np.sum(self.current_episodes))}'
            # gives hit rates (rate of reaching goal, regardless of negative rewards)
            log_string += f'\nHit (step, term, ep): {np.sum(self.success)/np.sum(self.current_steps)}, {np.sum(self.success)/np.sum(self.current_term)}, {np.sum(self.success)/max(1, np.sum(self.current_episodes))}'
            if len(self.record_rollouts) != 0:
                # adds to the tensorboard logger for graphing
                self.tensorboard_logger.add_scalar("Return/"+self.name, np.sum(self.reward)/np.sum(self.current_episodes), i)
                self.tensorboard_logger.add_scalar("Success/"+self.name, np.sum(self.success)/np.sum(self.current_term), i)
                # log the loss values
                if self.log_losses: 
                    for k in self.total_losses.keys():
                        self.tensorboard_logger.add_scalar("Loss/" + k, np.mean(self.total_losses[k]), i)
                        log_string += f'\nLoss {k}: {np.mean(self.total_losses[k])}'
                self.tensorboard_logger.flush()
            logging.info(log_string)
            print(log_string)
        return np.sum(self.reward) / max(1, np.sum(self.current_episodes)), np.sum(self.success) / max(1, np.sum(self.current_episodes)), 