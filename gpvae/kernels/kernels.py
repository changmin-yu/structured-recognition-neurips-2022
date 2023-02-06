import torch
import torch.nn as nn
import numpy as np

class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return NotImplementedError
    

class SEKernel(Kernel):
    def __init__(self, lengthscale=1., scale=1.):
        super().__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        
    def forward(self, x1, x2, diag=False):
        x1 = x1.unsqueeze(-1) if len(x1.shape) == 1 else x1
        x2 = x2.unsqueeze(-1) if len(x2.shape) == 1 else x2
        
        assert x1.shape[-1] == x2.shape[-1], 'Inputs must be of same dimension'
        
        if not diag:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(0)
        else:
            assert x1.shape == x2.shape, 'Inputs must be of same shape'
        
        sd = torch.square(x1 - x2).clamp_min_(0.0)
        sd = (sd / (self.lengthscale ** 2)).sum(-1)
        cov = self.scale ** 2 * torch.exp(-sd)
        
        return cov


class PeriodicKernel(Kernel):
    def __init__(self, lengthscale=1., period=1., scale=1.):
        super().__init__()
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale), requires_grad=True)
        self.log_period = nn.Parameter(torch.tensor(period).log(), requires_grad=True)
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=True)
        
    def forward(self, x1, x2, diag=False):
        x1 = x1.unsqueeze(-1) if len(x1.shape) == 1 else x1
        x2 = x2.unsqueeze(-1) if len(x2.shape) == 1 else x2
        
        assert x1.shape[-1] == x2.shape[-1], 'Inputs must be of same dimension'
        
        if not diag:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(0)
        else:
            assert x1.shape == x2.shape, 'Inputs must be of same shape'
            
        ad = torch.abs(x1 - x2)
        ad = 2 * torch.square(torch.sin((np.pi * ad / self.log_period.exp())))
        ad = (ad / (self.lengthscale ** 2)).sum(-1)
        cov = self.scale ** 2 * torch.exp(-ad)
        
        return cov