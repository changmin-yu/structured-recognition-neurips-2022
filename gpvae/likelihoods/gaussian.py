import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from utils.networks import MLP
from .base import Likelihood

class NNHomoGaussian(Likelihood):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), sigma=None, sigma_grad=True, min_sigma=1e-3, 
                 nonlinearity=nn.ReLU):
        super().__init__()
        self.network = MLP(in_dim, out_dim, hidden_dims, nonlinearity)

        self.min_sigma = min_sigma
        if sigma is None:
            self.log_sigma = nn.Parameter(torch.zeros((out_dim, )), requires_grad=sigma_grad)
        else:
            self.log_sigma = nn.Parameter(torch.ones((out_dim, )) * sigma, requires_grad=sigma_grad)
        
    def forward(self, x):
        mu = self.network(x)
        sigma = self.log_sigma.exp().clamp_min(self.min_sigma)
        out_dist = Normal(mu, sigma)
        return out_dist

class NNHeteroGaussian(Likelihood):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), min_sigma=1e-3, init_sigma=None, 
                 nonlinearity=nn.ReLU):
        super().__init__()
        self.out_dim = out_dim
        self.min_sigma = min_sigma
        self.network = MLP(in_dim, out_dim*2, hidden_dims, nonlinearity)
        
        if init_sigma is not None:
            self.network.network[-1].bias.data[out_dim:] = torch.log(torch.exp(torch.tensor(init_sigma))-1)
        
    def forward(self, x):
        out = self.network(x)
        mu = out[..., :self.out_dim]
        sigma = out[..., self.out_dim:]
        sigma = F.softplus(sigma) + self.min_sigma
        
        out_dist = Normal(mu, sigma)
        return out_dist