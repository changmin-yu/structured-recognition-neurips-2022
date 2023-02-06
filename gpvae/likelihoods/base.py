import torch.nn as nn

class Likelihood(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, z):
        raise NotImplementedError
    
    def log_likelihood(self, z, x):
        px_z = self(z)
        return px_z.log_prob(x)