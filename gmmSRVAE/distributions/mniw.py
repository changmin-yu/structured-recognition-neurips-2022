from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

'''
Here we are dealing with multivariate-case of the Normal-Inverse-Wishart distribution
'''

def symmetrise(A):
    return (A + torch.transpose(A, -2, -1))/2.

def standard_to_natural(nu, S, M, K):
    # here K in the **Inverse** of the second parameter of the gaussian distribution in the canonical parameterisation of NIW distribution
    # I.e., K is \lambda^{-1} https://en.wikipedia.org/wiki/Normal-inverse-Wishart_distribution
    # this might change in the later iterations
    Kinv = torch.linalg.inv(K)
    A = Kinv.clone()
    B = torch.linalg.matmul(Kinv, M.T)
    C = S + torch.linalg.matmul(M, B)
    d = nu.clone()
    return (A, B, C, d)

def natural_to_standard(natparam):
    A, B, C, d = natparam
    nu = d
    Kinv = A.clone()
    K = symmetrise(torch.linalg.inv(Kinv))
    M = torch.linalg.matmul(K, B).transpose(-2, -1)
    S = C - torch.linalg.matmul(M, B)
    return nu, S, M, K
