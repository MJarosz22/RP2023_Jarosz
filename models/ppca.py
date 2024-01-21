import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
import scipy as sp
from scipy.stats import entropy

# Implementation based on Medium article by Oliver K. Ernst, PhD
# https://medium.com/practical-coding/the-simplest-generative-model-you-probably-missed-c840d68b704
class PPCA:
    
    def __init__(self, q, data):
        self.q = q
        self.data = data
        
    def sample_hidden_given_visible(
        self,
        weight_ml : np.array, 
        mu_ml : np.array,
        var_ml : float,
        visible_samples : np.array
        ) -> np.array:

        
        m = np.transpose(weight_ml) @ weight_ml + var_ml * np.eye(self.q)

        cov = var_ml * np.linalg.inv(m) @ np.eye(self.q)
        act_hidden = []
        for data_visible in visible_samples:
            mean = np.linalg.inv(m) @ np.transpose(weight_ml) @ (data_visible - mu_ml)
            sample = np.random.multivariate_normal(mean,cov,size=1)
            act_hidden.append(sample[0])
    
        return np.array(act_hidden)
        
    def sample_visible_given_hidden(
        self,
        weight_ml : np.array, 
        mu_ml : np.array,
        var_ml : float,
        hidden_samples : np.array
        ) -> np.array:

        d = weight_ml.shape[0]

        act_visible = []
        for data_hidden in hidden_samples:
            mean = weight_ml @ data_hidden + mu_ml
            cov = var_ml * np.eye(d)
            sample = np.random.multivariate_normal(mean,cov,size=1)
            act_visible.append(sample[0])
    
        return np.array(act_visible)
    
    def generate(self, no_samples):
        d = self.data.shape[1]


        mu_ml = np.mean(self.data,axis=0)


        data_cov = np.cov(self.data,rowvar=False)
        # Variance
        lambdas, eigenvecs = np.linalg.eig(data_cov)
        idx = lambdas.argsort()[::-1]   
        lambdas = lambdas[idx]
        eigenvecs = - eigenvecs[:,idx]

        var_ml = (1.0 / (d-self.q)) * sum([lambdas[j] for j in range(self.q,d)])


        # Weight matrix
        uq = eigenvecs[:,:self.q]


        lambdaq = np.diag(lambdas[:self.q])

        weight_ml = uq @ np.sqrt(lambdaq - var_ml * np.eye(self.q))
        
        act_hidden = self.sample_hidden_given_visible(
            weight_ml=weight_ml,
            mu_ml=mu_ml,
            var_ml=var_ml,
            visible_samples=self.data
            )
        
        mean_hidden = np.full(self.q,0)
        cov_hidden = np.eye(self.q)

        samples_hidden = np.random.multivariate_normal(mean_hidden,cov_hidden,size=no_samples)
        
        synthetic_data = self.sample_visible_given_hidden(
            weight_ml=weight_ml,
            mu_ml=mu_ml,
            var_ml=var_ml,
            hidden_samples=samples_hidden
            )
        return synthetic_data