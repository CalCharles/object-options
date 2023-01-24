import sklearn as sk
import sklearn.mixture as mix
import numpy as np

class ClusterModel():
    def ___init__(self):
        pass

    def fit(self, datas):
        pass

    def predict(self, datas):
        pass

    def mean(self):
        pass

class BayesianGaussianMixture(ClusterModel):
    def __init__(self, dp_gmm):
        self.dp_gmm = dp_gmm
        self.model = None # for functionality, @method fit needs to be run

    def fit(self, data):
        cov_prior = [float(self.dp_gmm[4]) for _ in range(data.shape[1])]
        # mean_prior = [self.dp_gmm[5] for _ in range(data.shape[1])]
        mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)
        rng = mmax - mmin
        mmean = np.mean(data, axis=0)
        mean_prior = [0 for i in range(data.shape[1])]
        # mean_prior = [mmin + (rng/data.shape[1] * i) for i in range(data.shape[1])]
        # print(mmin, mmax, mean_prior)
        # error
        print(self.dp_gmm)
        self.model = mix.BayesianGaussianMixture(n_components=int(self.dp_gmm[0]), max_iter=int(self.dp_gmm[1]), 
                                        weight_concentration_prior=float(self.dp_gmm[2]), covariance_type=self.dp_gmm[3], 
                                        covariance_prior=cov_prior, mean_prior=mean_prior) # uses a dirichlet process GMM to cluster
        print(data)
        print(self.model.fit(data))
        print("means", self.model.means_)
        return self.model

    def predict(self, data):
        return self.model.predict(data)

    def mean(self):
        return self.model.means_