from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

def standard_to_natural(alpha):
    return alpha - 1

def natural_to_standard(alpha_nat):
    return alpha_nat + 1

def expected_log_pi(alpha_nat):
    return torch.digamma(alpha_nat) - torch.digamma(torch.sum(alpha_nat, dim=-1, keepdim=True))