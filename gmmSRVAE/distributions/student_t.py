import torch
import numpy as np
from helpers.general_utils import logdet

def logprob_full_scale(y, mu, sigma, v):
    '''
    Student-t log-probability with full scale matrix sigma
    
    Args: 
        y: (N, K, S, D)
        mu: (K, D)
        sigma: (K, D, D)
        v: (K, )
    
    Returns: 
        log S(y|mu,sigma, v)
    '''
    N, K, S, D = y.shape
    assert mu.shape == (K, D)
    assert sigma.shape == (K, D, D)
    assert v.shape == (K, )
    mu = torch.tile(mu.unsqueeze(0).unsqueeze(2), (N, 1, S, 1))
    sigma = torch.tile(sigma.unsqueeze(0).unsqueeze(2), (N, 1, S, 1, 1))
    v = v.unsqueeze(0).unsqueeze(2) # (1, K, 1)
    
    err = (y-mu).unsqueeze(-1)
    squared_term = torch.einsum('nksdi,nksdi->nks', err, torch.linalg.solve(sigma, err)) # (N, K, S)
    logprob = torch.lgamma(0.5*(v+D)) - torch.lgamma(0.5*v)
    logprob -= 0.5 * D * torch.log(np.pi)
    logprob -= 0.5 * logdet(sigma)
    logprob -= 0.5 * (v+D) * torch.log1p(squared_term/v)
    return logprob

def logprob_smm(y, mu, sigma, v, log_pi):
    N, D = y.shape
    K, D_ = mu.shape
    assert D == D_
    y = torch.tile(y.unsqueeze(1).unsqueeze(2), (1, K, 1, 1))
    logprob = logprob_full_scale(y, mu, sigma, v)
    logprob = logprob.view((N, K))
    logprob = logprob + log_pi.unsqueeze(0)
    return logprob

def logprob_per_sample(y, mu, sigma, v):
    return logprob_full_scale(y, mu, sigma, v)
