# computes the EM algorithm for a simple linear case
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from collections import deque
from torch.utils.tensorboard import SummaryWriter

# TODO: adapt to affine
# TODO: adapt to state specific data distributions
class GeneralizedLinearEnvironment():
    '''
    Creates an environment where num_distributions generate equal amounts of linear/affine data
    using n variables, where the output is determined using a hyperplane switching function
    '''
    def __init__(self, dim, output_dim, num_distributions, num_variables, rate =0.5, switching=False, affine=False):
        self.used_binaries = list()
        self.linear_models = list()
        for n in range(num_distributions):
            linear_model = list()
            bin = tuple([np.random.choice([0,1], p = [1-rate,rate]) for i in range(num_variables)])
            while bin in self.used_binaries or np.sum(bin) == 0:
                bin = tuple([np.random.choice([0,1], p = [1-rate,rate]) for i in range(num_variables)])
            self.used_binaries.append(bin)
            for b in bin:
                if b:
                    linear_model.append(np.random.rand(dim, output_dim))
            self.linear_models.append(linear_model)
        self.num_variables = num_variables
        self.num_distributions = num_distributions
        self.dim = dim
        self.output_dim = output_dim
        print(self.linear_models, self.used_binaries, self.dim, self.output_dim)

    def generate_data(self, num_data):
        data = list()
        for i in range(self.num_variables):
            data.append(np.random.rand(int(num_data), self.dim))
        output = list()
        binaries= list()
        asmts = list()
        for i, (bin, models) in enumerate(zip(self.used_binaries, self.linear_models)):
            midx = 0
            amount_data = int(num_data // self.num_distributions)
            new_output = np.zeros((amount_data, self.output_dim))
            for j, b in enumerate(bin):
                if b:
                    new_output += np.matmul(data[j][i*amount_data:(i+1)*amount_data], models[midx])
                    midx += 1
            output.append(new_output)
            binaries.append(np.array([list(bin) for k in range(amount_data)]))
            asmt = np.zeros((amount_data, self.num_distributions))
            asmt[:,i] = 1
            asmts.append(asmt)
        output = np.concatenate(output, axis=0)
        binaries = np.concatenate(binaries, axis=0)
        data = np.concatenate(data, axis=-1)[:len(binaries)]
        asmts = np.concatenate(asmts, axis=0)
        return data, output, binaries, asmts

def regress_residuals(X,y,weights):
    regr = LinearRegression()
    regr.fit(X, y, weights)
    # print(regr)
    return np.linalg.norm(regr.predict(X) - y, axis=-1) / y.shape[-1]

def sigmoid(z):
    return 1/(1 + np.exp(-z))

UPWEIGHT = 10
INIT_LAMBDA = 1

class GeneralizedLinearEMAlgorithm():
    def __init__(self, dim, num_distributions, logdir):
        self.dim = dim
        self.num_distributions = num_distributions
        self.writer = SummaryWriter(logdir=logdir)

    
    def init_assignments(self, data):
        assn = (INIT_LAMBDA + np.random.rand(len(data[0]), self.num_distributions) )
        assn = (assn / np.expand_dims(np.sum(assn, axis=-1), axis=-1)).transpose(1,0)
        print(assn)
        return assn
        # return (np.random.rand(len(data[0]), self.num_distributions) / self.num_distributions).transpose(1,0) 
    
    def E_step(self, data, assignments):
        regr = list()
        for i in range(self.num_distributions):
            X = data[0]
            # X = X * np.concatenate([np.ones((X.shape[0], self.dim)), np.zeros((X.shape[0], self.dim))], axis=-1) # mask out the y component
            y = data[1]
            weights = assignments[i]
            # print("r1")
            regr.append(regress_residuals(X,y,weights))
        return np.stack(regr, axis=0)

    def M_step(self, data, residuals):
        # assigns based on the magnitude of the differential of residuals
        asmts = list()
        for i in range(self.num_distributions):
            # asmt = sigmoid(UPWEIGHT * (residuals[1] - residuals[0]))
            asmt = residuals[i] - np.min(residuals, axis=0)
            asmt = np.exp(UPWEIGHT * -residuals[i]) / np.sum(np.exp(UPWEIGHT*-residuals), axis = 0)
            asmts.append(asmt)
        # asmt = sigmoid(UPWEIGHT * (residuals[1] - residuals[0]))
        # asmts = [asmt, 1-asmt]
        return np.stack(asmts,axis=0)

    def run(self, data, num_iters):
        assignments = self.init_assignments(data)
        print(assignments.shape)
        losses = {"assignment": list(), }
        last_30_residuals = deque(maxlen=30)
        for i in range(num_iters):
            residuals = self.E_step(data, assignments)
            assignments = self.M_step(data, residuals)
            idxes = np.random.randint(len(data[0]), size = (3,)).astype(int)
            last_30_residuals.append(np.mean(residuals[0]*assignments))
            print(data[2][idxes])
            print(assignments[:,idxes])
            # print(residuals[0][idxes], residuals[1][idxes])
            print("assignment accuracy",i,  (np.sum(np.abs(data[3] - (1-assignments.transpose(1,0))))) / len(data[3]), np.mean(residuals[0]*assignments), np.mean(residuals[1] * (1-assignments)))
            if len(last_30_residuals) == 30 and (np.abs(last_30_residuals[-1] - last_30_residuals[0]) < 1e-10):
                print("resetting!!!")
                assignments = self.init_assignments(data)

def load_factored_data(buffer, target_name, target_diff=True):
    '''
    Takes in a tianshou buffer and then loads the data in the format for GLEMA 
    '''
    return

if __name__ == "__main__":
    LOAD_DATA = False

    if LOAD_DATA:
        environment, record = initialize_environment(args.environment, args.record)
        extractor, normalization = regenerate(True, environment, all=all_train)
        target_name = args.full_inter.train_names[0]
        
        data = train_all_buffer
        em = GeneralizedLinearEMAlgorithm(DIM, NUM_DIST, LOGDIR)
        em.run(data, 5000)
    else:
        # environment variables
        DIM= 10
        OUTPUT_DIM = 10
        NUM_DIST= 3
        NUM_VAR = 3
        RATE =0.5
        SWITCH=False
        AFFINE=False
        LOGDIR = "logs/lem/"
        env = GeneralizedLinearEnvironment(DIM,OUTPUT_DIM, NUM_DIST, NUM_VAR, RATE, SWITCH, AFFINE)
        data = env.generate_data(10000)
        em = GeneralizedLinearEMAlgorithm(DIM, NUM_DIST, LOGDIR)
        em.run(data, 5000)