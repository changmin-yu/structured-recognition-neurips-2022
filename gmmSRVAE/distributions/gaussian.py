from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from helpers.general_utils import logdet

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def standard_to_natural(mu, sigma):
    eta_2 = -0.5 * torch.linalg.inv(sigma)
    eta_1 = -2 * torch.matmul(eta_2, mu.unsqueeze(-1))
    eta_1 = eta_1.squeeze(-1)
    return eta_1, eta_2

def natural_to_standard(eta1, eta2):
    sigma = torch.linalg.inv(-2 * eta2)
    mu = torch.matmul(sigma, eta1.unsqueeze(-1))
    mu = mu.squeeze(-1)
    return mu, sigma

def log_prob_nat(x, eta1, eta2, weights=None):
    N, D = x.shape
    
    if len(eta1.shape) != 3:
         # eta1.shape = (N, K, D), where K is the number of mixture components
        raise AssertionError(f'eta1 must be of shape (N, K, D), but received input shape {eta1.shape}')
    
    logprob = torch.einsum('nd,nkd->nk', x, eta1)
    logprob = logprob + torch.einsum('nkd,nd->nk', torch.einsum('nd,nkde->nke', x, eta2), x)
    logprob = logprob - D/2. * torch.log(2*torch.tensor(np.pi))
    
    logprob = logprob + 1./4 * torch.einsum('nkdi,nkdi->nk', torch.linalg.solve(eta2, eta1.unsqueeze(-1)), eta1.unsqueeze(-1))
    logprob = logprob + 0.5 * logdet(-2 * eta2 + 1e-20 * torch.eye(D).unsqueeze(0).unsqueeze(0).to(DEVICE))
    
    if weights is not None:
        logprob = logprob + torch.log(weights).unsqueeze(0)
    
    # logsumexp for numerical stability
    max_logprob, _ = torch.max(logprob, dim=1, keepdim=True)
    normaliser = max_logprob + torch.logsumexp(logprob-max_logprob, dim=1, keepdim=True)
    
    return logprob-normaliser

def log_prob_nat_per_sample(x, eta1, eta2):
    N, K, S, D = x.shape
    assert eta1.shape == (N, K, D)
    assert eta2.shape == (N, K, D, D)
    
    logprob = torch.einsum('nksd,nksd->nks', torch.einsum('nkij,nksj->nksi', eta2, x), x)
    logprob = logprob + torch.einsum('nki,nksi->nks', eta1, x)
    logprob = logprob + 1./4*torch.einsum('nkdi,nkd->nki', torch.linalg.solve(eta2, eta1.unsqueeze(-1)), eta1)
    logprob = logprob - D/2. * torch.log(2*torch.tensor(np.pi))
    logprob = logprob + 1./2 * logdet(-2.*eta2+1e-20*torch.eye(D).to(DEVICE)).unsqueeze(-1)
    
    return logprob # (N, K, S)