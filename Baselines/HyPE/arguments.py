import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description='RL')
    # parser.add_argument('--algo', default='a2c',
    #                     help='algorithm to use: a2c, ppo, evo')

    # environment hyperparameters
    parser.add_argument('--true-environment', action='store_true', default=False,
                        help='triggers the baseline methods')
    parser.add_argument('--frameskip', type=int, default=1,
                        help='number of frames to skip (default: 1 (no skipping))')
    # # optimization hyperparameters
    parser.add_argument('--lr', type=float, default=7e-4,
                        help='learning rate (default: 1e-6)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RMSprop/Adam optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999),
                        help='Adam optimizer betas (default: (0.9, 0.999))')
    parser.add_argument('--weight-decay', type=float, default=0.00,
                        help='Adam optimizer l2 norm constant (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    # cost function hyperparameters
    parser.add_argument('--return-form', default='true',
                        help='determines what return equation to use. true is true returns, gae is gae (not implemented), value uses the value function')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy-coef', type=float, default=1e-2,
                        help='entropy loss term coefficient (default: 1e-7)')
    parser.add_argument('--high-entropy', type=float, default=0,
                        help='high entropy (for low frequency) term coefficient (default: 1)')
    parser.add_argument('--value-loss-coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    # model hyperparameters
    parser.add_argument('--num-layers', type=int, default=1,
                    help='number of layers for network. When using basis functions, defines independence relations (see ReinforcementLearning.basis_models.py)')
    parser.add_argument('--factor', type=int, default=4,
                        help='decides width of the network')
    parser.add_argument('--optim', default="Adam",
                        help='optimizer to use: Adam, RMSprop, Evol')
    parser.add_argument('--activation', default="relu",
                        help='activation function for hidden layers: relu, sin, tanh, sigmoid')
    parser.add_argument('--init-form', default="uni",
                    help='initialization to use: uni, xnorm, xuni, eye')
    parser.add_argument('--model-form', default="",
                        help='choose the model form, which is defined in Models.models')
    # state hyperparameters
    parser.add_argument('--normalize', action='store_true', default=False,
                        help='Normalized inputs for the neural network/function approximator')
    parser.add_argument('--num-stack', type=int, default=4,
                        help='number of frames to stack (default: 4)')
    # target hypothesis
    parser.add_argument('--target-tau', type=float, default=0.5,
                        help='mixture value for target network (default: 0.5)')
    # distributional RL parameters
    parser.add_argument('--value-bounds', type=float, nargs=2, default=(0, 10),
                        help='bounds for the possible value of a state (default: (0, 10))')
    parser.add_argument('--num-value-atoms', type=int, default=51,
                        help='number of atoms in distributional RL (default: 51)')
    # distributional regularization parameters
    parser.add_argument('--dist-interval', type=int, default=-1,
                        help='decides how often distributional interval is computed')
    parser.add_argument('--exp-beta', type=float, default=0.1,
                        help='beta value in exponential distribution')
    parser.add_argument('--dist-coef', type=float, default=1e-5,
                        help='the coefficient used for determining the loss value of the distribution')
    parser.add_argument('--correlate-steps', type=int, default=-1,
                    help='decides how many steps are used to compute correlate diversity enforcement (default -1)')
    parser.add_argument('--diversity-interval', type=int, default=2,
                        help='decides how often correlate diversity error is computed')

    # novelty search hyperparameters
    parser.add_argument('--novelty-decay', type=int, default=5000,
                        help='number of updates after which novelty rewards are halved')
    parser.add_argument('--novelty-wrappers', default=[], nargs='+',
                    help='the different novelty definitions, which are defined in RewardFunctions.novelty_wrappers, empty uses no wrappers (default)')
    parser.add_argument('--visitation-magnitude', type=float, default=.01,
                        help='the highest magnitude reward from novelty') # TODO: if multiple, don't share parameters
    parser.add_argument('--visitation-lambda', type=float, default=1,
                        help='laplace regularization of novelty decay term')
    parser.add_argument('--novelty-hash-order', type=int, default=20,
                        help='the number of possible values for tiles. Uses initial min max values, which might not be great (default: 20)')
    # offline learning parameters
    parser.add_argument('--grad-epoch', type=int, default=1,
                        help='number of gradient epochs in offline learning (default: 1, 4 good)')

    #PPO parameters
    parser.add_argument('--clip-param', type=float, default=0.2,
                    help='ppo clip parameter (default: 0.2)')

    # Evolution parameters 
    parser.add_argument('--base-form',  default="",
                        help='base network form for population model')
    parser.add_argument('--select-ratio', type=float, default=0.25,
                    help='percentage of population selected in evolution(default: 0.25)')
    parser.add_argument('--num-population', type=int, default=20,
                        help='size of the population (default: 20)')
    parser.add_argument('--sample-duration', type=int, default=-1,
                        help='number of time steps to evaluate a subject of the population (default: -1)')
    parser.add_argument('--sample-schedule', type=int, default=-1,
                        help='number of updates to increase the duration by a factor of duration (default: 10)')
    parser.add_argument('--retest-schedule', type=int, default=-1,
                        help='if nonnegative, increases every n steps (default: -1)')
    parser.add_argument('--elitism', action='store_true', default=False,
                        help='keep the best performing networks')
    parser.add_argument('--evo-gradient', type=float, default=-1,
                        help='take a step towards the weighted mean, with weight as given, (default -1, not used)')
    parser.add_argument('--variance-lr', type=float, default=-1,
                        help='adjusts the learning rate based on the variance of the weights, (default -1, not used)')
    parser.add_argument('--reassess-num', type=int, default=-1,
                        help='number of best sampled networks, (default -1, not used)')
    parser.add_argument('--reentry-rate', type=float, default=0.0,
                        help='rate of randomly re-entering a best network, to update the performance, (default 0.0)')
    parser.add_argument('--retest', type=int, default=1,
                        help='number of times a network is sampled, (default 1)')
    parser.add_argument('--reward-stopping', action='store_true', default=False,
                        help='if getting a reward causes a stopping behavior, (default False)')
    parser.add_argument('--OoO-eval', action='store_true', default=False,
                        help='out of order execution of networks, (default False)')
    parser.add_argument('--weight-sharing', type=int, default=-1,
                        help='uses the best networks for weight sharing for n steps, (default -1 for not used)')
    # Evolution Gradient parameters
    parser.add_argument('--sample-steps', type=int, default=2000,
                    help='number of time steps to run to evaluate full population (default: 2000)')
    parser.add_argument('--base-learner',  default="",
                        help='base learning algorithm for running the gradient component')
    parser.add_argument('--evo-lr', type=float, default=.05,
                        help='the learning rate for the evolutionary steps (default .05, not used)')
    parser.add_argument('--init-var', type=float, default=1.0,
                        help='the initial variance (default 1.0)')
    # Stein Variational policy gradient hyperparameters
    parser.add_argument('--stein-alpha', type=float, default=.05,
                        help='the learning rate for the stein steps (default .05, not used)')
    parser.add_argument('--kernel-form',  default="",
                        help='name of kernel function used (defined in ReinforcementLearning.kernels')
    # option time determination
    parser.add_argument('--swap-form', default="dense",
                    help='choose how often to check for new actions, where dense is every time step, and "reward" is when the proxy environment gets reward')

    # basis function parameters
    parser.add_argument('--period', type=float, default=1,
                help='length of period over which fourier basis is applied')
    parser.add_argument('--scale', type=float, default=1,
                help='scaling term for magnitudes, which can be useful in multilayer to exacerbate differences')
    parser.add_argument('--order', type=int, default=40,
                        help='decides order of the basis functions (related to number of basis functions)')
    parser.add_argument('--connectivity', type=int, default=1,
                        help='decides the amount the basis functions are connected (1, 2, 12, 22, 3)')
    # Transformer Network parameters
    parser.add_argument('--key-dim', type=int, default=1,
                        help='decides the amount the basis functions are connected (1, 2, 12, 22, 3)')
    parser.add_argument('--value-dim', type=int, default=1,
                        help='decides the amount the basis functions are connected (1, 2, 12, 22, 3)')
    parser.add_argument('--post-transform-form', default='none',
                        help='has the same inputs as model-form, the model after the transform. (default: none for none)')
    parser.add_argument('--num-heads', type=int, default=1,
                        help='number of heads for multiheaded attention (or multiple copies of any network) (default: 1)')
    parser.add_argument('--attention-form', default='vector',
                        help='has the same inputs as model-form, the model after the transform. (default: none for none)')
    # Adjustment model variables
    parser.add_argument('--adjustment-form', default='none',
                        help='has the same inputs as model-form, the model for the base')
    parser.add_argument('--correction-form', default='none',
                        help='how the adjustment form relates to the base: none, linear, inbiaspost, inbiaspre, biaspost, biaspre')
    parser.add_argument('--freeze-initial', action ='store_true', default=False,
                        help='freeze the weights of the loaded network, do not use with no-adjustment')
    parser.add_argument('--keep-initial-output', action ='store_true', default=False,
                        help='use the output parameters from the loaded network (requires that dimensions of hidden layer match')

    # Behavior policy parameters
    parser.add_argument('--greedy-epsilon', type=float, default=0.1,
                    help='percentage of random actions in epsilon greedy')
    parser.add_argument('--min-greedy-epsilon', type=float, default=0.1,
                    help='minimum percentage of random actions in epsilon greedy (if decaying)')
    parser.add_argument('--greedy-epsilon-decay', type=float, default=-1,
                    help='greedy epsilon decays by half every n updates (-1 is for no use)')
    parser.add_argument('--behavior-policy', default='',
                        help='defines the behavior policy, as defined in BehaviorPolicies.behavior_policies')

    # pretraining arguments TODO: not implemented
    parser.add_argument('--pretrain-iterations', type=int, default=-1,
                    help='number of time steps to run the pretrainer using PPO on optimal demonstration data, -1 means not used (default: -1)')
    parser.add_argument('--pretrain-target', type=int, default=0,
                    help='pretrain either with actions (0) or outputs (1) (default: -1)')
    # Reinforcement model settings
    parser.add_argument('--optimizer-form', default="",
                        help='choose the optimizer form, which is defined in ReinforcementLearning.learning_algorithms')
    parser.add_argument('--state-forms', default=[""], nargs='+',
                    help='the different relational functions, which are defined in Environment.state_definition')
    parser.add_argument('--state-names', default=[""], nargs='+',
                    help='should match the number of elements in state-forms, contains the names of nodes used')
    # Hindsight learning parameter
    parser.add_argument('--base-optimizer', default="",
                        help='choose the optimizer form, which is defined in ReinforcementLearning.learning_algorithms')
    # Learning settings
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1,
                        help='how many training CPU processes to use (default: 16)')
    parser.add_argument('--lag-num', type=int, default=2,
                        help='lag between states executed and those used for learning, to delay for reward computation TODO: 1 is the minimum for reasons... (default: 1)')
    parser.add_argument('--num-steps', type=int, default=1,
                        help='number of reward checks before update (default: 1)')
    parser.add_argument('--num-grad-states', type=int, default=-1,
                        help='number of forward steps used to compute gradient, -1 for not used (default: -1)')
    parser.add_argument('--reward-check', type=int, default=5,
                        help='steps between a check for reward, (default 1)')
    parser.add_argument('--num-update-model', type=int, default=1,
                        help='number of gradient steps before switching options (default: 3)')
    parser.add_argument('--changepoint-queue-len', type=int, default=30,
                        help='number of steps in the queue for computing the changepoints')
    parser.add_argument('--num-iters', type=int, default=int(2e3),
                        help='number of iterations for training (default: 2e3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--warm-up', type=int, default=10,
                        help='num updates before changing model (default: 200 (1000 timesteps))')
    parser.add_argument('--reward-swapping', action='store_true', default=False,
                        help='if getting a reward only causes a change in policy, (default False)')
    parser.add_argument('--done-swapping', type=int, default=9999999999,
                        help='after n updates, the episode is determined by the episode from the true environment, (default 99999)')

    # Replay buffer settings
    parser.add_argument('--match-option', action='store_true', default=False,
                        help='use data only from the option currently learning (default False')
    parser.add_argument('--buffer-steps', type=int, default=-1,
                        help='number of buffered steps in the record buffer, -1 implies it is not used (default: -1)')
    parser.add_argument('--buffer-clip', type=int, default=20,
                        help='backwards return computation (strong effect on runtime')
    parser.add_argument('--weighting-lambda', type=float, default=1e-2,
                        help='lambda for the sample weighting in prioritized replay (default = 1e-2)')
    parser.add_argument('--prioritized-replay', default="",
                        help='different prioritized replay schemes, (TD (Q TD error), return, recent, ""), default: ""')
    # Trace settings
    parser.add_argument('--trace-len', type=int, default=-1,
                        help='number of states in a trace trajectory (default -1)')
    parser.add_argument('--trace-queue-len', type=int, default=-1,
                        help='number of trace trajectories in the trace queue (default -1)')
    # dilated settings
    parser.add_argument('--dilated-stack', type=int, default=4,
                        help='number of states to keep when one dilated index is added (default 4)')
    parser.add_argument('--dilated-queue-len', type=int, default=-1,
                        help='number of states in the dilation index queue (multiply with dilated-stack) (default -1 not used)')
    parser.add_argument('--target-stack', type=int, default=10,
                        help='states to keep around the pretrain-target (default 10)')

    # logging settings
    parser.add_argument('--log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 10)')
    parser.add_argument('--save-dir', default='',
                        help='directory to save data when adding edges')
    parser.add_argument('--save-graph', default='graph',
                        help='directory to save graph data. Use "graph" to let the graph specify target dir, empty does not train')
    parser.add_argument('--save-recycle', type=int, default=-1,
                        help='only saves the last n timesteps (-1 if not used)')
    parser.add_argument('--record-rollouts', default="",
                        help='path to where rollouts are recorded (when adding edges, where data was recorded to compute min/max)')
    parser.add_argument('--changepoint-dir', default='./data/optgraph/',
                        help='directory to save/load option chain')
    parser.add_argument('--unique-id', default="0",
                        help='a differentiator for the save path for this network')
    parser.add_argument('--save-past', type=int, default=-1,
                    help='save past, saves a new net at the interval, -1 disables, must be a multiple of save-interval (default: -1)')
    parser.add_argument('--save-models', action ='store_true', default=False,
                        help='Saves environment and models to option chain directory if true')
    parser.add_argument('--display-focus', action ='store_true', default=False,
                        help='shows an image with the focus at each timestep like a video')
    parser.add_argument('--single-save-dir', default="",
                        help='saves all images to a single directory with name all')
    # Option Chain Parameters
    parser.add_argument('--base-node', default="Action",
                        help='The name of the lowest node in the option chain (generally should be Action)')

    # changepoint parameters
    parser.add_argument('--past-data-dir', default='',
                        help='directory to load data for computing minmax')
    parser.add_argument('--segment', action='store_true', default=False,
                    help='if true, the reward function gives reward for a full segment, while if false, will apply a single sparse reward')
    parser.add_argument('--remove-outliers', type=int, default=-1,
                        help='removes outliers prior to clustering, that is, removes |x| > remove value, -1 is not used')
    parser.add_argument('--transforms', default=[''], nargs='+',
                    help='Different transforms to be used to reduce a segment or window to a single value. Options are in RewardFunctions.dataTransforms.py')
    parser.add_argument('--train-edge', default='',
                        help='the edge to be trained, of the form Object->Object, for changepoints, just Object')
    parser.add_argument('--determiner', default='',
                        help='defines the determiner to use, using strings as defined in RewardFunctions.changepointDeterminers')
    parser.add_argument('--reward-form', default='',
                        help='defines the kind of reward function to use, as defined in RewardFunctions.changepointReward, also: dense, x, bounce, dir, move_dir[0,1,2,3,4, all] for different cardinal directions')
    parser.add_argument('--changepoint-name', default='changepoint',
                        help='name to save changepoint related values')
    parser.add_argument('--champ-parameters', default=["Paddle"], nargs='+',
                        help='parameters for champ in the order len_mean, len_sigma, min_seg_len, max_particles, model_sigma, dynamics model enum (0 is position, 1 is velocity, 2 is displacement). Pre built Paddle and Ball can be input as "paddle", "ball"')
    parser.add_argument('--window', type=int, default=3,
                        help='A window over which to compute changepoint statistics')
    parser.add_argument('--min-cluster', type=int, default=15,
                        help='A number defining the number of elements in a cluster')
    parser.add_argument('--focus-dumps-name', default='object_dumps.txt',
                    help='the name of the dump file used for CHAMP')

    # environmental variables
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu number to use (default: 0)')
    parser.add_argument('--num-frames', type=int, default=10e4,
                        help='number of frames to use for the training set (default: 10e6)')
    parser.add_argument('--env', default='SelfBreakout',
                        help='environment to train on (default: SelfBreakout)')
    parser.add_argument('--train', action ='store_true', default=False,
                        help='trains the algorithm if set to true')
    # load variables
    parser.add_argument('--load-weights', action ='store_true', default=False,
                        help='load the options for the existing network')
    # parametrized options parameters
    parser.add_argument('--parameterized-option', type=int, default=0,
                        help='parametrization enumerator,as defined in multioption, default no parametrization (default: 0)')
    parser.add_argument('--parameterized-form', default='basic',
                        help='has the same inputs as model-form, the model for the base')

    # parser.add_argument('--load-networks', default=[], nargs='+',
    #                     help='load weights from the network')
    # DP-GMM parameters
    parser.add_argument('--cluster-model', default="DPGMM",
                    help='Which clustering model to use, current DPGMM or Filtered DPGMM')
    parser.add_argument('--dp-gmm', default=["default"], nargs='+',
                    help='parameters for dirichlet process gaussian mixture model, in order number of components, maximum iteration number, prior, covariance type and covariance prior')
            
    # testing parameters
    parser.add_argument('--visualize', action='store_true', default=False,
                        help='show an image of the output')


    args = parser.parse_args()
    if args.dp_gmm[0] == 'default':
        args.dp_gmm = [10, 6000, 100, 'diag', 1e-10]
    elif args.dp_gmm[0] == 'ataripaddle':
        args.dp_gmm = [10, 6000, 100, 'diag', 1e-10]
    elif args.dp_gmm[0] == 'block':
        args.dp_gmm = [10, 6000, 1, 'diag', 10]
    elif args.dp_gmm[0] == 'atariball':
        args.dp_gmm = [10, 6000, 1e-10, 'diag', 20]
    elif args.dp_gmm[0] == 'far':
        args.dp_gmm = [10, 6000, 1e-10, 'diag', 20]
    elif args.dp_gmm[0] == 'further':
        args.dp_gmm = [10, 6000, 1e-30, 'diag', 20]
    if args.champ_parameters[0] == "Paddle":
        args.champ_parameters = [3, 5, 1, 100, 100, 2, 1e-2, 3]
    elif args.champ_parameters[0] == "PaddleLong":
        args.champ_parameters = [3, 30, 1, 100, 100, 2, 1e-2, 3]
    elif args.champ_parameters[0] == "Block":
        args.champ_parameters = [3, 30, 1, 100, 100, 2, 1, 3]
    elif args.champ_parameters[0] == "PaddleAtari":
        args.champ_parameters = [3, 5, 1, 100, 100, 2, 1, 3]
    elif args.champ_parameters[0] == "Ball": 
        args.champ_parameters = [15, 10, 1, 100, 100, 2, 1e-2, 3] 
    elif args.champ_parameters[0] == "BallAtari":
        args.champ_parameters = [15, 7, 1, 100, 100, 2, .5, 3]
    else:
        args.champ_parameters = [float(p) for p in args.champ_parameters]
    if len(args.behavior_policy) == 0:
        print(args.optimizer_form in ["DQN", "SARSA", "TabQ", "Dist", "DDPG"])
        if args.optimizer_form in ["DQN", "SARSA", "TabQ", "Dist", "DDPG"]:
            args.behavior_policy = "egq"
        elif args.optimizer_form in ["PPO", "A2C", "PG", "Evo", "GradEvo", "CMAES", "SVPG", "Hind", "SAC", ""]:
            args.behavior_policy = "esp"


    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args

def get_edge(edge):
    head = edge.split("->")[0]
    tail = edge.split("->")[1]
    head = head.split(",")
    return head, tail[0]
