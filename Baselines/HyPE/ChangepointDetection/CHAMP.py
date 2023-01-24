# CHAMP algorithm Python Version
import numpy as np
import scipy as sci
import scipy.special as special
import copy, os, sys, pickle
from Record.file_management import load_from_pickle, save_to_pickle
from Baselines.HyPE.ChangepointDetection.DynamicsModels import *
from Baselines.HyPE.ChangepointDetection.ChangepointDetectorBase import ChangepointDetector
DEBUG = False

class CHAMP_parameters():
    def __init__(self, len_mean, len_sigma, min_seg_len, max_particles, resamp_particles, data_order, model_sigma):
        self.LEN_MEAN = int(len_mean)         
        self.LEN_SIG = int(len_sigma)       
        self.MIN_SEG_LEN = int(min_seg_len)         
        self.MAX_PARTICLES = int(max_particles)          
        self.RESAMP_PARTICLES = int(resamp_particles)
        self.DATA_ORDER = int(data_order)
        self.MODEL_SIGMA = model_sigma

# INTERNAL CHAMP CODE #
class Particle():
    def __init__(self, prev_MAP, pos, model_class, model_sigma):
        self.prev_MAP = prev_MAP
        self.pos = pos
        self.MAP = -np.inf
        self.nMAP = -np.inf
        self.fitter = model_class(sigma=model_sigma)

def gaussCDF(t, mean, sigma):
    return 0.5 * special.erfc(-(t-mean)/(sigma*np.sqrt(2)))

# Calc cdf of log len dist by truncating normal gauss cdf: cdf(t) - cdf(a)
# NOTE: This only is accurate up to a normalizing constant introduced by truncation.
def logOneMinLenCDF(t, mean, sigma, min_seg_len):
    p = gaussCDF(t, mean, sigma) - gaussCDF(min_seg_len-1, mean, sigma)
    return np.log(1-p)

def logLenPDF(t, mean, sigma):
    # Integrate between t-1 and t, so that the CDF is consistent later for cdf(t) = pdf(t) + pdf(t-1) + ... + pdf(MIN_SEG_LEN) 
    return np.log( gaussCDF(t, mean, sigma) - gaussCDF(t-1.0, mean, sigma) )

def add_new_cp(t, data, prev_max_MAP, particles, params, model_classes, max_path_indices, max_path_models, max_MAP, online=False):
    l_prior = np.log(1.0 / len(model_classes))
    if t >= (2*params.MIN_SEG_LEN)-1:
        prev = prev_max_MAP[0]
        prev_max_MAP.pop(0)
        if DEBUG:
            print ("Adding particles for p=%i with prev: %f\n" % (t-params.MIN_SEG_LEN,prev))
        
        # Create new particles for a changepoint at time t-MIN_SEG_LEN, one for each model in Q
        for i in range(len(model_classes)):
            particles.append(Particle(max_MAP,t-params.MIN_SEG_LEN,model_classes[i], params.MODEL_SIGMA))
    
    # Compute fit probs for all particles
    for i in range(len(particles)):
        p = particles[i]
        seg_len = t-p.pos
                    
        # Fit the model and calc the data likelihood
        p.fitter.fitSegment(data, p.pos, t)
        
        # p_tjq is the prob of the CP **PRIOR** to time t occuring at j (p_pos)
        # p->MAP is the prob of the MAP CP occuring at time j (p_pos)
        p_ME = p.fitter.params.modelEvidence
        if(np.isnan(p_ME) or p_ME == -np.inf):
            p_tjq = -np.inf
        else:
            p_tjq = logOneMinLenCDF(seg_len-1, params.LEN_MEAN, params.LEN_SIG, params.MIN_SEG_LEN) + p_ME + l_prior + p.prev_MAP
        
        if(np.isnan(p_tjq) or p_tjq == -np.inf):
            p.MAP = -np.inf
        else:
            p.MAP = (p_tjq + logLenPDF(seg_len, params.LEN_MEAN, params.LEN_SIG)
             - logOneMinLenCDF(seg_len-1, params.LEN_MEAN, params.LEN_SIG, params.MIN_SEG_LEN))
        if DEBUG:
            print("position: ", p.pos)
            p.fitter.params.printParams()
                          
    # Update global stats and viterbi path
    max_val = -np.inf
    max_particle = None
    for i in range(len(particles)):
        if particles[i].MAP > max_val: 
            max_particle = particles[i]
            max_val = max_particle.MAP

    if max_particle != None: 
        max_MAP = max_particle.MAP
        max_path_indices.append(max_particle.pos)
        copy_model = copy.deepcopy(max_particle.fitter)
        max_path_models.append(copy_model)
        if DEBUG:
            print("MAX " , t , " pos: " , max_particle.pos,
                " model: " ,max_particle.fitter.params.getModelName(),
                " map: ", max_particle.MAP, "\n")
    else:
        max_path_indices.append(-1)
        max_path_models.append(None)
    
    # If there are more than MAX_PARTICLES, resample down to RESAMP_PARTICLES
    particles = resampleParticles(particles, params.MAX_PARTICLES, params.RESAMP_PARTICLES)

    # Keep track of MIN_SEG_LEN many of the previous max_MAP values to create particles with later.
    prev_max_MAP = [max_MAP] + prev_max_MAP
    return prev_max_MAP, particles, max_MAP


def detectChangepoints(model_classes, params, data):
    
    # Initialize particle filter and viterbi stats
    max_MAP = np.log(1.0 / len(model_classes))
    particles = []
    for i in range(len(model_classes)):
        particles.append(Particle(max_MAP,-1,model_classes[i], params.MODEL_SIGMA))
    max_path_indices = [] # list of indexes
    max_path_models = [] # list of corresponding models
    prev_max_MAP = []
    
    # Process each time step, approximating the distribution of the most recent changepoint *BEFORE* time t
    # NOTE: By starting at MIN_SEG_LEN-1, we ensure the first segment will be >= MIN_SEG_LEN
    for t in range(params.MIN_SEG_LEN-1, len(data) - params.DATA_ORDER):
        # Only create new particle for first time when a CP there would divide data in 2 halves, each of MIN_SEG_LEN
        prev_max_MAP, particles, max_MAP = add_new_cp(t, data, prev_max_MAP, particles, params, model_classes, max_path_indices, max_path_models, max_MAP)
    
    # Now max_path contains the final viterbi path, so trace it 
    curr_cp = len(data)-params.DATA_ORDER - 1
    path_index = curr_cp - params.MIN_SEG_LEN + 1  # This isn't a CP number, but an index into the max_path
    segments = []
    seg_idexes = []
    if DEBUG:
        print("mpi", list(enumerate(max_path_indices)))
        print("\nFINAL CHANGEPOINTS:\n")
    while(curr_cp > -1):
        # Print CP info
        if max_path_models[path_index] is None:
            curr_cp = max_path_indices[path_index] 
            continue
        if DEBUG:
            print("start: " , (max_path_indices[path_index]+1),  "   model: " , (max_path_models[path_index].params.getModelName()) 
                 , "   len: " , (curr_cp - (max_path_indices[path_index])), "\n")
            max_path_models[path_index].params.printParams()
            print("diffs: ", max_path_models[path_index].params.diff)
            if max_path_models[path_index].params.diff > 1:
                max_path_models[path_index].params.printParams()
        seg_idexes.append(max_path_indices[path_index]+1)
        
        # Add a ModelSegment
        segments.insert(0,max_path_models[path_index].params)
        
        # Go to previous CP in chain
        curr_cp = max_path_indices[path_index] 
        path_index = curr_cp - params.MIN_SEG_LEN + 1
    
    return segments, seg_idexes


# Normalize the MAP values (into nMAP) of the vector of particles. These are NOT log.
def normalizeMAP(particles):
    # Find the max log
    max_log = max(particles, key=lambda x: x.MAP).MAP
    
    total = 0
    # Factor out (subtract) the max log and un-log them. Some may still underflow, but they are basically zero anyway.
    for i in range(len(particles)):
        particles[i].nMAP = np.exp(particles[i].MAP - max_log)
        # print(particles[i].MAP - max_log, max_log, particles[i].MAP)
        if not np.isnan(particles[i].nMAP) and particles[i].nMAP != -np.inf:
            total += particles[i].nMAP
    
    for i in range(len(particles)):
        particles[i].nMAP = particles[i].nMAP / total

# Resample down to resamp_particles if there are more than max_particles, using Stratified Optimal Resampling.
def resampleParticles(particles, max_particles, resamp_particles):
    # Get ready to perform resampling if there are too many particles 
    if len(particles) > max_particles:
        # Normalize MAP values of all particles
        normalizeMAP(particles)
        
        # Throw out particles with a p->nMAP value of -INFINITY or NAN
        particles = [p for p in particles if (not np.isnan(p.nMAP) and not p.nMAP == -np.inf)]
    
    # Only continue with resampling if the renormalizing step didn't get rid of enough
    if len(particles) > max_particles: 
        # Sort the particles by normalized p->MAP values, smallest to largest
        particles.sort(key = lambda p: p.MAP)
        particles.reverse()
        
        # Calculate alpha
        alpha = calculateAlpha(particles, resamp_particles)
        
        # Keep particles that have weight >= alpha
        new_particles = [p for p in particles if p.nMAP >= alpha]
        particles = [p for p in particles if p.nMAP < alpha]
            
        # Choose random u from uniform dist [0, alpha]
        u = np.random.rand() * alpha
        
        # Resample using SOR(3)
        reduced_particles = []
        for p in particles:
            u -= p.nMAP
            if(u <= 0):
                new_particles.append(p)
                u = u + alpha
            else:
                reduced_particles.append(p)
        
        # If underflow caused too few particles to be resampled, choose randomly
        while(len(new_particles) < resamp_particles):
            ind = np.random.randint(len(reduced_particles))
            new_particles.append(reduced_particles[ind])
            reduced_particles.pop(ind)
        return new_particles
    return particles

def calculateAlpha(particles, M): 
    # Implementation of algo from Fearnhead & Clifford (2003)
    # Look for smallest kappa = w_i that satisfies sum[min(w_i/kappa , 1)] <= M
    N = len(particles)
    for i in range(N-M-1, N):   # i is the cutoff index
        A_i = N-i-1          # Number of elements > w_i
        B_i = 0           # Sum of elements <= w_i
        for j in range(i+1):
            B_i += particles[j].nMAP
        
        kappa = particles[i].nMAP
        if(kappa == 0):
            continue             # Account for underflow
        stat = (1.0/kappa)*B_i + A_i        # Check if sum[min(w_i/kappa , 1)] <= M
        # print(kappa, B_i, A_i, stat, i)
        if(stat <= M):
             alpha = B_i / (M - A_i)      # Prob mass of w_i <= kappa div by size of set
             print ("i %i  kappa %f  A_i %i  B_i %f  Stat: %f  M: %i  Alpha: %f\n" % (i, kappa, A_i, B_i, stat, M, alpha))
             return alpha
    # print("ERROR: calculateAlpha(): No suitible alpha value found\n")
    # error
    return -1

# EXTERNAL CHAMP INTERFACE

def online_changepoint_detection(datapt, particles, prev_max_MAP, MAP_indexes, history, t, max_path_indices, max_path_models, params, model_classes, reset_len = 500):
    '''
    Performs online changepoint detection TODO: implement
    '''
    if len(history) > reset_len or len(history) == 0:
        MAP_indexes.append(prev_max_MAP)
        particles = []
        prev_max_MAP = []
        t = params.MIN_SEG_LEN-1
        max_MAP = np.log(1.0 / len(model_classes))
        for i in range(len(model_classes)):
            particles.append(Particle(max_MAP,-1,model_classes[i], params.MODEL_SIGMA))
        history = datapt
    else:
        data = np.concatenate((history, datapt), axis=0) 
    prev_max_MAP, particles, max_MAP = add_new_cp(t, data, prev_max_MAP, particles, params, model_classes, max_path_indices, max_path_models, max_MAP, online=True)
    
    return prev_max_MAP, particles, t

def generate_changepoints(model_classes, params, data):
    '''
    Splits into 200 frame segments and searches for changepoints
    '''
    models = []
    changepoints = []
    for i in range(max(1, int(np.ceil(len(data) / 200)))):
        if len(data) / 200 > 1:
            print(i * 200)
        seg_models, cpts = detectChangepoints(model_classes, params, data[i*200:(i+1)*200])
        cpts = np.array(cpts, dtype = np.int64)
        cpts += 200*i
        cpts = cpts[::-1]
        changepoints.append(cpts)
        models += seg_models
    # print(changepoints)
    changepoints = np.concatenate(changepoints) 
    return models, changepoints


class CHAMPDetector(ChangepointDetector):
    def __init__(self, train_edge, champ_parameters):
        super(CHAMPDetector, self).__init__(train_edge)
        if champ_parameters[7] == 0:
            self.model_class = LinearDynamicalPositionFitter
        elif champ_parameters[7] == 1:
            self.model_class = LinearDynamicalVelocityFitter
        elif champ_parameters[7] == 2:
            self.model_class = LinearDynamicalDisplacementFitter
        elif champ_parameters[7] == 3:
            self.model_class = LinearDisplacementFitter
        print(champ_parameters)
        self.params = CHAMP_parameters(champ_parameters[0], champ_parameters[1], champ_parameters[2], champ_parameters[3], champ_parameters[4], champ_parameters[5], champ_parameters[6]) # paddle parameters (also change sigma in DynamicsModels)

    def generate_changepoints(self, data, save_dict=False):
        seg_models, changepoints = generate_changepoints([self.model_class], self.params, data)

        if save_dict:
            changepoints = np.array(changepoints[::-1], dtype=np.int64)
            for cpt, seg_model in zip(changepoints, seg_models):
                print(cpt)
                print("model: \n", seg_model.A)
                print("data: \n", seg_model.data)
                print("predictions: \n", seg_model.predictions)
                print("diff: ", seg_model.diff)
                print("log likelihood: ", seg_model.logLikelihood)
            print(changepoints)
            # correlate_data(seg_models, changepoints, action_data, args.num_frames, data)
            # correlate_data(seg_models, changepoints, paddle_data, args.num_frames, data, prox_distance=5)
            cp_dict = {cp : m for cp, m in zip(changepoints, seg_models)}
            # print(cp_dict)
            cp_dict[-2] = args.num_frames
            # print(cp_dict)
            save_to_pickle(os.path.join(args.record_rollouts, 'changepoints-' + self.head + '.pkl'), cp_dict)
            # with open(os.path.join(args.record_rollouts, 'changepoints-' + self.head + '.pkl'), 'wb') as fid:
            #     pickle.dump(cp_dict, fid)
        return seg_models, changepoints