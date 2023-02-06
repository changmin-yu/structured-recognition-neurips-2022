import torch
import torch.nn as nn

from .kernels import Kernel

class KernelList(nn.ModuleList):
    def __init__(self, kernels):
        super().__init__(kernels)
    
    def forward(self, x1, x2, diag=False, embed=True):
        covs = [k.forward(x1, x2, diag) for k in self]
        if diag and embed:
            covs = torch.stack([cov.diag_embed() for cov in covs])
        else:
            covs = torch.stack(covs)
        
        return covs


class AdditiveKernel(Kernel):
    def __init__(self, *args):
        super().__init__()
        
        self.kernels = KernelList(args)
        
    def forward(self, x1, x2, diag=False):
        cov = self.kernels.forward(x1, x2, diag, embed=False).sum(0)
        return cov
    

class MultiplicativeKernel(Kernel):
    def __init__(self, *args):
        super().__init__()
        self.kernels = KernelList(args)
    
    def forward(self, x1, x2, diag=False):
        cov = self.kernels.forward(x1, x2, diag, embed=False).prod(0)
        return cov