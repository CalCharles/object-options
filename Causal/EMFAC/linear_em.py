# computes the EM algorithm for a simple linear case
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from collections import deque

class LinearEnvironment():
    def __init__(self, dim, output_dim):
        self.A1 = np.random.rand(dim, output_dim)
        self.A2 = np.random.rand(dim, output_dim)
        self.B2 = np.random.rand(dim, output_dim)
        self.dim = dim
        self.output_dim = output_dim
        print(self.A1, self.A2, self.B2, self.dim, self.output_dim)

    def generate_data(self, num_data):
        xdata1 = np.random.rand(int(num_data // 2), self.dim)
        ydata1 = np.random.rand(int(num_data // 2), self.dim)
        xdata2 = np.random.rand(int(num_data // 2), self.dim)
        ydata2 = np.random.rand(int(num_data // 2), self.dim)
        print(xdata1.shape, self.A1.shape)
        output1 = np.matmul(xdata1, self.A1)
        output2 = np.matmul(xdata2, self.A2) + np.matmul(ydata2, self.B2)
        input_data1 = np.concatenate([xdata1, ydata1], axis=-1)
        input_data2 = np.concatenate([xdata2, ydata2], axis=-1)
        input_data = np.concatenate([input_data1, input_data2], axis =0)
        output_data = np.concatenate([output1, output2], axis =0)
        binaries = np.concatenate([np.zeros(int(num_data// 2)), np.ones(int(num_data// 2))])
        return input_data, output_data, binaries

def regress_residuals(X,y,weights):
    regr = LinearRegression()
    regr.fit(X, y, weights)
    # print(regr)
    return np.linalg.norm(regr.predict(X) - y, axis=-1)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

UPWEIGHT = 10# 0.05

class LinearEMAlgorithm():
    def __init__(self, dim):
        self.dim = dim
    
    def init_assignments(self, data):
        # return np.ones(len(data[0])) * 0.5
        return np.random.rand(len(data[0])).clip(0.3,0.7)
    
    def E_step(self, data, assignments):
        X = data[0]
        # X = X * np.concatenate([np.ones((X.shape[0], self.dim)), np.zeros((X.shape[0], self.dim))], axis=-1) # mask out the y component
        y = data[1]
        weights = assignments
        # print("r1")
        regr1 = regress_residuals(X,y,weights)
        X = data[0]
        # print("r2")
        regr2 = regress_residuals(X, y, 1-weights)
        return regr1, regr2

    def M_step(self, data, residuals):
        # assigns based on the magnitude of the differential of residuals
        asmt = sigmoid(UPWEIGHT * (residuals[1] - residuals[0]))
        return asmt

    def run(self, data, num_iters):
        assignments = self.init_assignments(data)
        last_30_residuals = deque(maxlen=30)
        for i in range(num_iters):
            residuals = self.E_step(data, assignments)
            assignments = self.M_step(data, residuals)
            idxes = np.random.randint(len(data[0]), size = (10,)).astype(int)
            last_30_residuals.append(np.mean(residuals[0]*assignments))
            print(data[2][idxes], assignments[idxes])#, residuals[0][idxes], residuals[1][idxes])
            print("assignment accuracy",i,  (np.sum(np.abs(data[2] - (1-assignments)))) / len(data[2]), np.mean(residuals[0]*assignments), np.mean(residuals[1] * (1-assignments)))
            if len(last_30_residuals) == 30 and (np.abs(last_30_residuals[-1] - last_30_residuals[0]) < 1e-10):
                print("resetting!!!")
                assignments = self.init_assignments(data)
if __name__ == "__main__":
    DIM= 10
    OUTPUT_DIM = 10
    env = LinearEnvironment(DIM,OUTPUT_DIM)
    data = env.generate_data(10000)
    em = LinearEMAlgorithm(DIM)
    em.run(data, 5000)