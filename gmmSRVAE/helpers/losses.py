from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

def weighted_mse(y_true, y_pred, res_pred):
    '''
    Args:
        y_true: (N, D)
        y_pred: (N, K, S, D)
        res_pred: (N, K)
    
    Returns:
        \sum_{k=1}^{K}1/N r_nk * \sum_{n=1}^{N} 1/S * \sum_{s=1}^{S} 1/D * \sum_{d=1}^{D} (y_nd-\hat{y}_ndk)^2 
    '''
    N, K, S, D = y_pred.shape
    assert y_true.shape == (N, D)
    assert res_pred.shape == (N, K)
    
    mse = torch.mean(torch.sum(torch.square(y_true.unsqueeze(1).unsqueeze(2)-y_pred), dim=-1), dim=-1)
    return torch.mean(torch.sum(mse * res_pred, dim=-1))

def imputation_mse(y_true, y_pred, res_pred, missing_data_mask):
    N, K, S, D = y_pred.shape
    assert y_true.shape == (N, D)
    assert res_pred.shape == (N, K)
    assert missing_data_mask.shape == (N, D)
    
    y_true_missing = y_true * missing_data_mask
    y_pred_missing = y_pred * missing_data_mask.unsqueeze(1).unsqueeze(2)
    
    mse = torch.mean(torch.square(y_true_missing.unsqueeze(1).unsqueeze(2)-y_pred_missing), dim=-1)
    res_mse = mse * res_pred.unsqueeze(-1)
    return torch.sum(res_mse) / N

def generate_missing_data_mask(y, noise_ratio=0.3, mask_type='random', seed=0):
    N, D = y.shape
    mask_size = N * D
    mask = np.zeros((mask_size), dtype=bool)
    
    if mask_type == 'random':
        num_noise = int(mask_size * noise_ratio)
        random_state = np.random.RandomState(seed)
        mask_idx = random_state.choice(np.arange(mask_size), size=num_noise, replace=False)
        mask[mask_idx] = True
    
    else:
        side = np.sqrt(D)
        assert side.is_integer()
        side = int(side)
        half_side = side // 2
        mask = mask.reshape((N, side, side))
        if mask_type == 'quarter':
            # remove lower left
            mask[:, half_side:side, :half_side] = True
        elif mask_type == 'lower_half':
            mask[:, half_side:side, :] = True
        elif mask_type == 'left_half':
            mask[:, :, :half_side] = True
        else:
            raise NotImplementedError(f'The input mask type "{mask_type}" is not recognised.')
    return torch.FloatTensor(mask).reshape(N, D)