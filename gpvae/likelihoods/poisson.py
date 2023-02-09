import torch
import torch.nn as nn

from torch.distributions import Poisson
from gpvae.utils.networks import MLP
from .base import Likelihood

class PoissonCount(Likelihood):
    def __init__(self):
        super().__init__()
    
    def forward(self, rate):
        assert torch.all(rate >= 0)
        return Poisson(rate)

class NNPoissonCount(Likelihood):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), nonlinearity=nn.ReLU):
        super().__init__()
        self.network = MLP(in_dim, out_dim, hidden_dims, nonlinearity)
        self.likelihood = PoissonCount()
    
    def forward(self, x):
        rate = self.network(x)
        rate = torch.exp(rate)
        return self.likelihood(rate)