import torch
from torch import nn
import torch.distributions as dist
import numpy as np


def kl_div(true, synthetic): 
    true_data_mean = true.mean(axis=0)
    model_mean = synthetic.mean(axis=0)
    true_data_covariance = np.cov(true, rowvar=False)
    model_covariance = np.cov(synthetic, rowvar=False)
    
    # Convert 'data_mean' to a PyTorch tensor
    data_mean_tensor = torch.Tensor(true_data_mean)
     
    # Convert 'true_data_covariance' to a PyTorch tensor
    true_data_covariance_tensor = torch.Tensor(true_data_covariance)
     
    # Convert 'model_mean' and 'model_covariance' to PyTorch tensors
    model_mean_tensor = torch.Tensor(model_mean)
    model_covariance_tensor = torch.Tensor(model_covariance)
     
    # Create multivariate normal distributions
    true_distribution = dist.MultivariateNormal(data_mean_tensor, covariance_matrix=true_data_covariance_tensor)
    estimated_distribution = dist.MultivariateNormal(model_mean_tensor, covariance_matrix=model_covariance_tensor)
     
    # Compute KL divergence
    kl_divergence = dist.kl.kl_divergence(true_distribution, estimated_distribution)
    
    return kl_divergence.detach()