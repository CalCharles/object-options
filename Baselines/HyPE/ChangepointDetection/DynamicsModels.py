
    # std::vector< std::vector<double> > calcFinalSegStats(double **data, const int start, const int end)
    # {
    #     std::vector< std::vector<double> > empty;
    #     return empty;
    # }
import sys
import numpy as np
sys.path.append("./Changepoint")


class ModelParams():

    def getModelName(self):
        return "Model"

    def printParams(self):
        print ("Raw ModelParams")

class Gauss1DParams(ModelParams): 
 
    def __init__(self, sigma=1):
        self.mu = 0
        self.sigma = sigma
        self.logLikelihood = 0
        self.modelEvidence = 0

    def getModelName(self):
        return "Gaussian"

    def printParams(self):
        print("Mu: %f\n"% self.mu)
        print("Sigma: %f\n"% self.sigma)
        print("Log Likelihood: %f\n"%self.logLikelihood)
        print("Model Evidence: %f\n\n"%self.modelEvidence)
        
    # def fillParams(ModelSegment &seg)
    # {
    #     seg.param_names.push_back("mu");
    #     seg.param_names.push_back("sigma");
    #     seg.param_names.push_back("log_likelihood");
    #     seg.param_names.push_back("model_evidence");
        
    #     seg.model_params.push_back(mu);
    #     seg.model_params.push_back(sigma);
    #     seg.model_params.push_back(logLikelihood);
    #     seg.model_params.push_back(modelEvidence);
    # }    

class ModelFitter():
    def fitSegment(self, data, start, end):
        print("not supposed to be here")


class Gauss1DFitter(ModelFitter):
    # Assumes mean 0, fits the variance
    def __init__(self, sigma=.01):
        self.params = Gauss1DParams(sigma=sigma) # Gauss1DParams

    def fitSegment(self, data, start, end):
        
        n = end-start
        print(start, end)
        
        self.params.mu = 0.0
        self.params.sigma = np.sqrt(np.sum(np.square(data[start+1:end+1]) / n))
        diff = 0
        diff = np.sum(np.square(data[start+1:end+1] - self.params.mu))
        
        term1 = (-n/2.0) * np.log(2.0*np.pi)
        term2 = (-n/2.0) * np.log(self.params.sigma**2)
        term3 = -(diff/(2*(self.params.sigma**2)))
        
        self.params.logLikelihood = term1 + term2 + term3
        self.params.modelEvidence = self.params.logLikelihood - np.log(n)  # LL - 1/2 (2 ln n)

class LinearDynamicalParams(ModelParams): 
 
    def __init__(self, sigma=.01):
        self.A = None
        self.sigma = sigma # initialize sigma to 1 for now (.01 for paddle)
        self.logLikelihood = 0
        self.modelEvidence = 0
        self.data = None
        self.diff = 0
        self.mode = -1

    def getModelName(self):
        return "LinearStateParams"

    def printParams(self):
        print("A: ", self.A)
        print("Sigma: %f"% self.sigma)
        print("Log Likelihood: %f"%self.logLikelihood)
        print("Model Evidence: %f"%self.modelEvidence)
        print("Diff: %f"% self.diff)
        print("din: ", self.datain)
        print("dpred: ", self.datapred)
        print("predictions: ", self.predictions)

        print("Mode: ", self.mode)

class LinearDynamicalPVFitter(ModelFitter):
    def __init__(self, sigma=.01):
        self.params = LinearDynamicalParams(sigma=sigma) # LinearStateParams

    def fitSegment(self, data, start, end=-1000):
        '''
        assume input data of the form: [Pos[0], Pos[1] ... Pos[N], Pos[N+1]]^T 
        '''
        if end == -1000: # -1000 is a magic number
            end = len(data) - 2
        data = np.squeeze(data)
        Xdot = (data[start+2:end+2] - data[start+1:end+1])[:,:2]
        X = data[start+1:end+1]
        self.params.data = X
        self.params.datain = X
        self.params.datapred = Xnext

        Xinv_r = np.linalg.pinv(X, rcond=1e-3)
        self.params.A = np.dot(Xinv_r, Xdot)

        deltas = np.dot(data[start+1:end+1], self.params.A)
        deltas = np.hstack((deltas, deltas, np.zeros((len(deltas), 1))))
        predictions = deltas + data[start+1:end+1]
        diff = np.sum(np.abs(data[start+2:end+2] - predictions))
        self.params.predictions = predictions
        self.params.diff = diff
        
        n = end - start
        term1 = (-n/2.0) * np.log(2.0*np.pi)
        term2 = (-n/2.0) * np.log(self.params.sigma**2)
        term3 = -(diff/(2*self.params.sigma**2))

        self.params.logLikelihood = term1 + term2 + term3
        self.params.modelEvidence = self.params.logLikelihood - np.log(n)  # LL - 1/2 (2 ln n)

class LinearDynamicalPositionFitter(ModelFitter):
    def __init__(self, sigma=.01):
        self.params = LinearDynamicalParams(sigma=sigma) # LinearStateParams

    def fitSegment(self, data, start, end):
        '''
        assume input data of the form: [Pos[0], Pos[1] ... Pos[N], Pos[N+1]]^T 
        '''
        if end == -1000: # -1000 is a magic number
            end = len(data) - 2

        data = np.squeeze(data)
        X = data[start+1:end+1]
        Xnext = data[start+2:end+2]
        self.params.data = data[start+1:end+2]
        self.params.datain = X
        self.params.datapred = Xnext

        Xinv_r = np.linalg.pinv(X, rcond=1e-3)
        self.params.A = np.dot(Xinv_r, Xnext)

        predictions = np.dot(data[start+1:end+1], self.params.A)
        diff = np.sum(np.abs(self.params.datapred - predictions))
        self.params.predictions = predictions
        self.params.diff = diff
        n = end - start
        term1 = (-n/2.0) * np.log(2.0*np.pi)
        term2 = (-n/2.0) * np.log(self.params.sigma**2)
        term3 = -(diff/(2*self.params.sigma**2))
        # print(n, term1, term2, term3, self.params.sigma)
        # error

        self.params.logLikelihood = term1 + term2 + term3
        self.params.modelEvidence = self.params.logLikelihood - np.log(n)  # LL - 1/2 (2 ln n)

class LinearDynamicalDisplacementFitter(ModelFitter):
    def __init__(self, sigma=.01):
        self.params = LinearDynamicalParams(sigma=sigma) # LinearStateParams

    def fitSegment(self, data, start, end):
        '''
        assume input data of the form: [Pos[0], Pos[1] ... Pos[N], Pos[N+1]]^T 
        '''
        if end == -1000: # -1000 is a magic number
            end = len(data) - 2

        data = np.squeeze(data)
        X = data[start+1:end+1]
        Xnext = data[start+2:end+2]
        self.params.data = data[start+1:end+2]
        self.params.datain = X
        self.params.datapred = Xnext - X
        
        Xinv_r = np.linalg.pinv(X, rcond=1e-10)
        # print(Xinv_r)
        self.params.A = np.dot(Xinv_r, self.params.datapred)

        predictions = np.dot(data[start+1:end+1], self.params.A)
        diff = np.sum(np.abs(self.params.datapred - predictions))
        self.params.predictions = predictions
        self.params.diff = diff
        
        n = end - start
        term1 = (-n/2.0) * np.log(2.0*np.pi)
        term2 = (-n/2.0) * np.log(self.params.sigma**2)
        term3 = -(diff/(2*self.params.sigma**2))

        self.params.logLikelihood = term1 + term2 + term3
        self.params.modelEvidence = self.params.logLikelihood - np.log(n)  # LL - 1/2 (2 ln n)


class LinearDynamicalVelocityFitter(ModelFitter):
    def __init__(self, sigma=.01):
        self.params = LinearDynamicalParams(sigma=sigma) # LinearStateParams

    def fitSegment(self, data, start, end=-1000):
        '''
        assume input data of the form: [Pos[0], Pos[1] ... Pos[N], Pos[N+1], Pos[N+2]]^T 
        '''
        if end == -1000: # -1000 is a magic number
            end = len(data) - 3

        data = np.squeeze(data)
        Xdot = (data[start+2:end+3] - data[start+1:end+2])[:,:2]
        self.params.data = Xdot
        self.params.datain = Xdot[:-1]
        self.params.datapred = Xdot[1:]

        Xinv_r = np.linalg.pinv(Xdot[:-1], rcond=1e-5)
        self.params.A = np.dot(Xinv_r, Xdot[1:])
        predictions = np.dot(Xdot[:-1], self.params.A)
        diff = np.sum(np.abs(Xdot[1:] - predictions))
        self.params.predictions = predictions
        
        n = end - start
        # print (n, diff)
        term1 = (-n/2.0) * np.log(2.0*np.pi)
        term2 = (-n/2.0) * np.log(self.params.sigma**2)
        term3 = -(diff/(2*self.params.sigma**2))

        self.params.diff = diff
        self.params.logLikelihood = term1 + term2 + term3
        self.params.modelEvidence = self.params.logLikelihood - np.log(n)  # LL - 1/2 (2 ln n)

class LinearDisplacementFitter(ModelFitter):
    def __init__(self, sigma=.01):
        self.params = LinearDynamicalParams(sigma=sigma) # LinearStateParams

    def fitSegment(self, data, start, end):
        '''
        assume input data of the form: [Pos[0], Pos[1] ... Pos[N], Pos[N+1]]^T 
        '''
        if end == -1000: # -1000 is a magic number
            end = len(data) - 2

        data = np.squeeze(data)
        X = data[start+1:end+1]
        Xnext = data[start+2:end+2]
        self.params.data = data[start+1:end+2]
        self.params.datain = X
        self.params.datapred = Xnext
        
        # print(Xinv_r)
        self.params.A = np.mean(Xnext - X, axis=0)

        predictions = data[start+1:end+1] + self.params.A
        diff = np.sum(np.abs(self.params.datapred - predictions))
        self.params.predictions = predictions
        self.params.diff = diff
        
        n = end - start
        term1 = (-n/2.0) * np.log(2.0*np.pi)
        term2 = (-n/2.0) * np.log(self.params.sigma**2)
        term3 = -(diff/(2*self.params.sigma**2))

        self.params.logLikelihood = term1 + term2 + term3
        self.params.modelEvidence = self.params.logLikelihood - np.log(n)  # LL - 1/2 (2 ln n)
