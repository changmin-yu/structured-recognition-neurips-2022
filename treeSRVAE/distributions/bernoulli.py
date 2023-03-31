from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.tensor(1e-10).to(DEVICE)

def standard_to_natural(p):
    return torch.log(p/(1-p+1e-10))

def natural_to_standard(eta):
    return 1./(1+torch.exp(-eta))