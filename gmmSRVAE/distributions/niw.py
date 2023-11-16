from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from helpers.general_utils import outer
import distributions.mniw as mniw
'''
Normal-Inverse-Wishart prior distribution over the (mu, Sigma) of Gaussian distributions
p(mu, Sigma| mu_{0}, beta, W, v) = N(mu|mu_{0}, (beta * Sigma)^{-1})) * W(Sigma|W, v)
'''
def standard_to_natural(beta, m, C, v):
    K, D = m.shape
    assert beta.shape == (K, )
    v_hat = v + D + 2
    b = beta.unsqueeze(-1) * m
    A = C + outer(b, m)
    return A, b, beta, v_hat

def natural_to_standard(A, b, beta, v_hat):
    # K, D = b.shape
    # assert beta.shape == (K, )
    
    m = b / beta.unsqueeze(-1)
    K, D = m.shape
    C = A - outer(b, m)
    v = v_hat - D - 2
    return beta, m, C, v

def expectation(beta, m, C, v):
    exp_m = m
    C_inv = torch.linalg.inv(C)
    C_inv_sym = (C_inv + C_inv.transpose(-1, -2))/2.
    exp_C = torch.linalg.inv(C_inv_sym * v.unsqueeze(-1).unsqueeze(-1))
    return exp_m, exp_C

'''
A new function of univariate NIW distribution
'''

# def standard_to_natural(nu, S, m, kappa):
#     A, B, C, d = mniw.standard_to_natural(nu, S, m[:, None], torch.atleast_2d(kappa))
#     return (C, B.ravel(), A[0, 0], d)