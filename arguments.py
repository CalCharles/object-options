import argparse
from State.object_dict import ObjDict
from Hyperparam.read_config import read_config
import numpy as np

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
                        help='number of frames to run')
    parser.add_argument('--demonstrate', action='store_true', default=False,
                        help='get the data from demonstrations or from random motion')
    # torch parameters
    parser.add_argument('--config', default="",
                        help='config file to read for hyperparameters, overrides args')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the gpu device to run on')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='no cuda')
    # shared train args
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

    # peripheral arguments
    parser.add_argument('--load-intermediate', action ='store_true', default=False,
                        help='load the passive model/interaction to skip passive model training')
    parser.add_argument('--save-intermediate', action ='store_true', default=False,
                        help='save the passive model to skip training later')
    parser.add_argument('--load-rollouts', default = "",
                        help='load data from here')

    # active passive args
    parser.add_argument('--predict-dynamics', action='store_true', default=False,
                        help='predicts the change in state instead of the state itself')
    parser.add_argument('--interaction-testing', type=float, nargs='+', default=list(),
                        help='interaction value, difference between P,A, Active greater than, passive less than  (default: empty list)')
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
    parser.add_argument('--weighting', type=float, nargs='+', default=list(),
                        help='4-tuple of weighting values: passive_error_cutoff, passive_error_upper, weighting_ratio, weighting schedule (default: empty for no weighting)')
    parser.add_argument('--active-log-interval', type=int, default=100,
                        help='prints logs every n iterations')
    parser.add_argument('--interaction-schedule', type=int, default=-1,
                        help='halves the interaction lambda every n timesteps')
    parser.add_argument('--inline-iters', type=int, nargs='+', default=list(),
                        help='3-tuple of max number, starting number, doubling n for interaction training iterations per active model step')
    parser.add_argument('--interaction-weighting', type=float, nargs='+', default=list(),
                        help='2-tuple of starting interaction weighting lambda, schedule to double')
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
    if args.environment.seed == -1: args.environment.seed = np.random.randint(10000) # randomly assign seed
    return args