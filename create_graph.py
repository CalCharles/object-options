# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!
from Record.graph_read_functions import *

import glob
import json
import os
import math
import argparse
import copy
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})
from configs.graph_paths import name_keys



color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'  # blue-teal
]

# performance_factor

def compute_error_bars(results, step_size, maxrange, key):
    means = list()
    stds = list()
    steps = list()
    min_std = 1
    results = [list(zip(*r)) for r in results]

    for i in [j*step_size for j in range(int(maxrange // step_size + 1))]:
        at_step = list()
        at_mean = list()
        for r in results:
            r = r[key]
            if len(r) > 0:
                s,v = r[0]
                while (s < i):
                    # print(i, s,v)
                    at_step.append(s)
                    # apply necessary value transformations
                    at_mean.append(v)
                    r.pop(0)
                    if len(r) <= 0:
                        break
                    s,v = r[0]
        if len(at_step) > 0:
            steps.append(np.mean(at_step))
            means.append(np.mean(at_mean))
            stds.append(np.std(at_mean) + min_std)
    # apply necessary extensions
    return steps, means, stds

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--name', default='break')
    parser.add_argument('--target', default='plot.svg')
    args = parser.parse_args()


    filenames, ranges, pltting, color = name_keys[args.name]
    yrng, xlim = ranges
    if args.name.find("cdl") != -1:
        results = [read_iterations_cdl(filename) for filename in filenames]
    elif args.name.find("hype") != -1:
        results = [read_iterations(filename) for filename in filenames]
    elif args.name.find("full") != -1:
        results = [read_full_inter(filename) for filename in filenames]
    rkeys = list(results[0][1].keys()) # they should all have the same keys

    def plot(results, name, ci, key):
        steps, meanvals, stdvs = compute_error_bars(results, 50000, xlim, key)
        steps = np.array(steps)
        returns = np.array(meanvals)
        error = np.array(stdvs) / 4
        # print(len(steps), len(returns))
        plt.plot(steps, returns, label=name, color=color_defaults[ci])
        plt.fill_between(steps, returns+error, returns-error, alpha=0.1, color=color_defaults[ci])
        print(steps, returns)
        print("mean std", np.max(returns), error[-1])
        if len(returns.shape) > 0:
            return np.min(returns), np.max(returns)
        return None, None
    if not pltting:
        for k in rkeys:
            print([np.array(r[1]).shape for r in results])
            minlen = min([len(r[1]) for r in results])
            mean, std = np.mean(np.array([r[1][:minlen] for r in results])), np.std(np.array([r[1][:minlen] for r in results]))
            print("mean", mean)
            print("std", std)
            plt.plot([0, xlim], [mean, mean], linewidth =2, color = color_defaults[color])
            plt.fill_between([0, xlim], mean+std, mean-std, alpha=0.1, color=color_defaults[color])
    else:
        for key in rkeys:
            minrtHO, maxrtHO = plot(results, args.name + "_" + key, key, color)
    plt.xlim(0, xlim)
    plt.ylim(yrng[0], yrng[1])
    # plt.ylim(0, 270)
    plt.xlabel('Number of Timesteps')
    plt.ylabel('Average Rewards per Episode')
    plt.title("Plot")
    # plt.legend(loc=2)
    # plt.figure(figsize = (600, 200))
    plt.savefig(args.target)