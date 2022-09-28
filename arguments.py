import argparse
from State.object_dict import ObjDict
from Hyperparam.read_config import read_config
import numpy as np
import os

def get_command_args():
    parser = argparse.ArgumentParser(description='Construct an environment')
    # environment parameters
    parser.add_argument('--record-rollouts', default = "",
                        help='base directory to save results')
    parser.add_argument('--record-recycle', type=int, default=-1,
                        help='max number of frames to keep, default: -1 not used')
    parser.add_argument('--save-dir', default = "",
                        help='save the trained network here')
    parser.add_argument('--env',
                        help='environment to run on')
    parser.add_argument('--render', action='store_true', default=False,
                        help='run the pushing gripper domain')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='amount of frameskip, 1=no frameskip')
    parser.add_argument('--variant', default="default",
                        help='environment variant to use')
    parser.add_argument('--time-cutoff', type=int, default=-1,
                        help='time cutoff for environment resets, defaults -1 (no cutoff)')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed for the environment, set to a random number in contsruction')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random motion')
    # debugging parameters
    parser.add_argument('--run-test', default = "",
                        help='the name of the test to run')
    parser.add_argument('--collect-mode', action='store_true', default=False,
                        help='in collect mode, collects data for a test')
    # torch parameters
    parser.add_argument('--config', default="",
                        help='config file to read for hyperparameters, overrides args')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the gpu device to run on')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='no cuda')
    parser.add_argument('--torch-seed', type=int, default=-1,
                        help='seed for torch, set to a random number in contsruction')
    # shared train args
    parser.add_argument('--dummy', default="",
                        help='trains in dummy mode, for running baselines or running final layer options')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='usually included, trains the network')
    parser.add_argument('--num-frames', type=int, default=1000,
                        help='number of frames to run')
    parser.add_argument('--train-edge', type=str, nargs='+', default=list(),
                        help='pair of objects to train interaction on, [source, [additional], target] (default: list')
    parser.add_argument('--train-test-ratio', type=float, default=0.9,
                    help='ratio of training samples to testing ones')
    parser.add_argument('--train-iters', type=int, default=0,
                        help='number of iterations (shared for inter vs option)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='the number of values to sample in a batch')
    # masking hyperparameters
    parser.add_argument('--min-sample-difference', type=float, default=1.0,
                help='samples with less than this value will be considered the same')
    parser.add_argument('--var-cutoff', type=float, nargs='+', default=list(),
                        help='the normalized cutoff variance for an attribute to be considered active')
    parser.add_argument('--num-samples', type=int, default=0,
                        help='number of samples to take for identifying active components')
    parser.add_argument('--sample-grid', action='store_true', default=False,
                        help='samples parent values from a grid (rather than a fixed number of uniformly random samples)')
    parser.add_argument('--dynamics-difference', action='store_true', default=False,
                        help='uses the diffrence of dyanmics values rather than the difference of base values')

    # peripheral arguments
    parser.add_argument('--load-intermediate', action ='store_true', default=False,
                        help='load the passive model/interaction to skip passive model training')
    parser.add_argument('--save-intermediate', action ='store_true', default=False,
                        help='save the passive model to skip training later')
    parser.add_argument('--load-rollouts', default = "",
                        help='load data from here')
    parser.add_argument('--load-dir', default = "",
                        help='load saved values from here')
    parser.add_argument('--refresh', action='store_true', default=False,
                        help='creates a new graph from scratch in masking')

    # active passive args
    parser.add_argument('--predict-dynamics', action='store_true', default=False,
                        help='predicts the change in state instead of the state itself')
    parser.add_argument('--interaction-testing', type=float, nargs='+', default=list(),
                        help='interaction binary cutoff, require Active greater than, omit passive less than, require difference between P-A  (default: empty list)')
    parser.add_argument('--proximity-epsilon', type=float, default=-1,
                        help='the minimum distance for two objects to be considered "close"')
    parser.add_argument('--passive-iters', type=int, default=0,
                        help='number of passive iterations to run')
    parser.add_argument('--compare-trace', action ='store_true', default=False,
                        help='evaluates comparisons with the trace values')
    # passive args
    parser.add_argument('--passive-logging-interval', type=int, default=0,
                        help='number of interaction-only trace iterations to run')
    parser.add_argument('--pretrain-active', action ='store_true', default=False,
                        help='trains the active model along with the passive model on non-active states')
    # trace training args
    parser.add_argument('--interaction-pretrain', type=int, default=0,
                        help='number of interaction-only trace training steps to run')
    # combined training args
    parser.add_argument('--weighting', type=float, nargs='+', default=[0,0,-1,0],
                        help='4-tuple of weighting values: passive_error_cutoff, passive_error_upper, weighting_ratio, weighting schedule (default: weighting[2] = -1 for no weighting)')
    parser.add_argument('--active-log-interval', type=int, default=100,
                        help='prints logs every n iterations')
    parser.add_argument('--interaction-schedule', type=int, default=-1,
                        help='halves the interaction lambda every n timesteps')
    parser.add_argument('--inline-iters', type=int, nargs='+', default=list(),
                        help='3-tuple of max number, starting number, doubling n for interaction training iterations per active model step')
    parser.add_argument('--interaction-weighting', type=float, nargs='+', default=list(),
                        help='2-tuple of starting interaction weighting (for passive error) lambda, schedule to double')
    parser.add_argument('--intrain-passive', action ='store_true', default=False,
                        help='trains the passive model during the active model training')
    # network arguments 
    parser.add_argument('--net-type', default = "mean",
                        help='determines the architecture of the network')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=list(),
                        help='the sizes of intermediate layers between num-inputs and num-outputs (default: empty')
    parser.add_argument('--layer-norm', action ='store_true', default=False,
                        help='takes the layer norm between all layers')
    parser.add_argument('--init-form', default = "",
                        help='The kind of initialization for the weights of the network')
    parser.add_argument('--activation', default = "relu",
                        help='The activate function for intermediate layers of the network')
    parser.add_argument('--activation-final', default = "none",
                        help='The activation function for final layer of the network')
    parser.add_argument('--scale-logits', type=float, default=-1,
                        help='scales the final output by scale-logits if positive, -1 if unused')
    # pair network arguments
    parser.add_argument('--post-channel', action ='store_true', default=False,
                        help='has a channel to transmit information late')
    parser.add_argument('--drop-first', action ='store_true', default=False,
                        help='no first_obj appended to each (but included in the post-layer)')
    parser.add_argument('--reduce-function', default = "mean",
                        help='defines what function is used to reduce the pointnet points')
    parser.add_argument('--aggregate-final', action ='store_true', default=False,
                        help='combines all of the values at the end')
    # optimizer arguments
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, not used if actor and critic learning rate used for algo (default: 1e-6)')
    parser.add_argument('--alt-lr', type=float, default=1e-4,
                        help='alternative learning rate, for critic, or interaction model (default: 1e-4)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    # policy args
    parser.add_argument('--learning-type', default = "dqn",
                        help='how the policy learns, options: dqn, sac, rainbow, ddpg, ppo')
    parser.add_argument('--epsilon-random', type=float, default=0.10,
                        help='rate for taking random actions (default: 0.1)')
    parser.add_argument('--epsilon-schedule', type=int, default=-1,
                        help='rate epsilon random decays (-1 not used)')
    parser.add_argument('--num-atoms', type=int, default=51,
                        help='number of atoms for rainbow')
    parser.add_argument('--grad-epoch', type=int, default=10,
                        help='number of grad epochs for learning per iterations')
    parser.add_argument('--sample-form', default = "merged",
                        help='how the data is sampled, merged, her, base')
    parser.add_argument('--discount-factor', type=float, default=0.99,
                        help='also gamma in RL, the future discount factor')
    parser.add_argument('--lookahead', type=int, default=2,
                        help='number of steps for RL lookahead')
    parser.add_argument('--max-critic', type=int, default=-1,
                        help='max value the critic can take, not really implemented except for rainbow')
    parser.add_argument('--reward-normalization', action ='store_true', default=False,
                        help='normalizes the reward values')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='tau value for dqn momentum (default: 0.005)')
    parser.add_argument('--sac-alpha', type=float, default=0.2,
                        help='alpha value for sac')
    # state setting
    parser.add_argument('--single-obs-setting', type=int, nargs='+', default=[0,0,0,0,0,0],
                        help='6-tuple of "param", "parent", "additional", "target", "inter", "diff"')
    parser.add_argument('--relative-obs-setting', type=int, nargs='+', default=[0,0,0,0,0],
                        help='4-tuple of "parent_relative", "parent_additional_relative", "additional_relative", "parent_param", "param_relative"')
    # rew term arguments
    parser.add_argument('--term-form', default = "param",
                        help='the termination/reward function type (comb, term)')
    parser.add_argument('--term-as-done', action ='store_true', default=False,
                        help='if a termination occurs, sends a done signal')
    parser.add_argument('--trunc-true', action ='store_true', default=False,
                        help='truncates the true reward signal')
    parser.add_argument('--true-done', action ='store_true', default=False,
                        help='if a true done occurs, sends a done signal')
    parser.add_argument('--epsilon-close', type=float, default=-1,
                        help='minimum distance for term/reward, in unnormalized units (default: -1)')
    parser.add_argument('--param-norm', type=int, default=1,
                        help='the norm used to compute distance (default: 1)')
    parser.add_argument('--between-terminate', type=int, default=1,
                        help='the minimum amount of time between valid terminations (default: 1)')
    # reward parameters
    parser.add_argument('--constant-lambda', type=float, default=0,
                        help='reward given at every state (default: 0)')
    parser.add_argument('--param-lambda', type=float, default=-1,
                        help='reward given for getting to the correct goal state (default: -1)')
    parser.add_argument('--inter-lambda', type=float, default=-1,
                        help='reward given for getting an interaction (default: -1)')
    # term arguments
    parser.add_argument('--interaction-as-termination', action ='store_true', default=False,
                        help='treats interactions as termination signals')
    parser.add_argument('--use-binary', action ='store_true', default=False,
                        help='uses the interaction binary instead of the interaction model')
    # termination manager arguments
    parser.add_argument('--temporal-extend', type=int, default=-1,
                        help='the number of steps before requiring a resampled action (default: -1)')
    # policy logging options
    parser.add_argument('--log-interval', type=int, default=-1,
                        help='iterations between logs (default: 0)')
    parser.add_argument('--train-log-maxlen', type=int, default=0,
                        help='the maximum number of train iterations to store in the logging rolling averages (default: 0)')
    parser.add_argument('--test-log-maxlen', type=int, default=0,
                        help='the maximum number of test iterations to store in the logging rolling averages (default: 0)')
    parser.add_argument('--initial-trials', type=int, default=0,
                        help='the total number of episodes to trial with random actions for comparison (default: 0)')
    parser.add_argument('--test-trials', type=int, default=0,
                        help='the total number of episodes to trial with random actions every log-interval iterations (default: 0)')
    parser.add_argument('--max-terminate-step', type=float, nargs=2, default=(1, 30),
                        help='terminates after reaching either mts[0] terminations or mts[1] steps (default: (0.9, 0.999))')
    # inline interaction training
    parser.add_argument("--interaction-config", default="",
                        help='location of config file for interaction training (overriden by other inline args)')
    parser.add_argument("--inpolicy-iters", type=int, default=5000,
                        help='numbe of iterations of training for inpolicy training')
    parser.add_argument("--inpolicy-schedule", type=int, default=-1,
                        help='how often to run inpolicy training')
    parser.add_argument("--inpolicy-times", type=int, default=-1,
                        help='number of times  to do inpolicy training (saves time, -1 not used)')
    parser.add_argument("--policy-intrain-passive", action='store_true', default=False,
                        help='trains the passive model along with the interaction one')
    parser.add_argument("--intrain-weighting", type=float, nargs='+', default=[-13, 1, 1, -1],
                        help='weighting values for binary cutoffs for passive error weighting')
    parser.add_argument("--save-inline", type=float, default=False,
                        help='whether to save the intrained values')
    parser.add_argument("--policy-inline-iters", type=int, nargs='+', default=[5, 1, 1000],
                        help='inline iters for training the interaction network')
    parser.add_argument("--reset-weights", type=int, nargs='+', default=[0,0,0],
                        help='resets the weights of networks: interaction, active, passive')

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    return args

def get_args():
    args = ObjDict()
    args.command = get_command_args()
    config = args.command.config
    if len(args.command.config) > 0:
        args = read_config(args.command.config)
        args.config = config
        args.config_name = os.path.split(config)[1][:-5]
    if args.environment.seed == -1: args.environment.seed = np.random.randint(100000) # randomly assign seed
    if args.torch.torch_seed == -1: args.torch.torch_seed = np.random.randint(100000) # randomly assign seed
    return args