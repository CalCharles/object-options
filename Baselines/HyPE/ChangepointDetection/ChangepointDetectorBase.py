#Changepoint Detector Base:
import numpy as np
from Record.file_management import load_from_pickle, save_to_pickle
# from changepointCorrelation import correlate_data


class ChangepointDetector():
    def __init__(self, train_edge):
        self.head,self.tail = train_edge[-1], train_edge[0]

    def generate_changepoints(self, data, save_dict=False):
        '''
        generates changepoints on data, formatted [num elements, dimension of elements] 
        returns a tuple of: 
            models (a model over the changepoints, TODO: not sure what that should be standardized yet, but at least should contain data in segment)
            changepoints: a vector of values at which changepoints occur. TODO: soft changepoints??
        '''
        pass
