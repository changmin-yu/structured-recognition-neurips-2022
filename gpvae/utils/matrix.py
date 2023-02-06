import torch

def add_diagonal(x, val=1., device=torch.device('cpu')):
    assert x.shape[-1] == x.shape[-2], "x must be (batch of) square matrix"
    d = (torch.ones(x.shape[-2], device=device) * val).diag_embed()
    return x + d