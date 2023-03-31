from sqlite3 import Time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
from itertools import product
import os
import pickle
import matplotlib.pyplot as plt

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from utils import sigmoid, to_multi_hot, sample_tree

def make_dataloader(dataset, ratio_train=None, ratio_val=None, binary=False, datapath='./data', 
                   batch_size=128, test_bs=-1, n_towers=1, n_threads=0, seed_split=0, seed_minibatch=0, 
                   dtype=torch.float32, noise_level=0.1, bar_settings=None, num_samples=5000, 
                   grid_dim=10, b=2.2, prior_tree=None, permutation=None, seed_generation=0, video_generation_kwargs=None,
                   ghmm_kwargs=None, **kwargs):
    val_loader = None
    labels = None
    if dataset == 'bar-test':
        data = make_bar_test_data(num_samples, b, grid_dim, prior_tree=prior_tree, permutation=permutation, seed_generation=seed_generation)
    elif 'video' in dataset:
        assert video_generation_kwargs is not None
        raise NotImplementedError
    elif dataset == 'GaussianHMM':
        assert ghmm_kwargs is not None
        raise NotImplementedError
    else:
        raise Exception(f'dataset "{dataset}" does not exist')
    
    N = data.shape[0]
    if labels is not None:
        labels_oh = np.zeros((N, len(np.unique(labels))))
        labels_oh[np.arange(N), labels] = 1
    
    ratio_test = 1-ratio_train
    if labels is None:
        X_train, X_test = train_test_split(data, test_size=ratio_test, random_state=seed_split)
        y_train, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(data, labels_oh, test_size=ratio_test, random_state=seed_split)

    if ratio_val is not None:
        assert ratio_train > ratio_val
        ratio_test -= ratio_val
        ratio_val_train = ratio_val / (ratio_train+ratio_val)
        
        if labels is None:
            X_train, X_val = train_test_split(X_train, test_size=ratio_val_train, random_state=seed_split)
        else:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=ratio_val_train, random_state=seed_split)
        print(f'N: {N:6d}\n\t N_train: {X_train.shape[0]:6d}\n\t N_valid: {X_val.shape[0]:6d}\n\t N_test: {X_test.shape[0]:6d}')
    else:
        print(f'N: {N:6d}\n\t N_train: {X_train.shape[0]:6d}\n\t N_valid: {0}\n\t N_test: {X_test.shape[0]:6d}')
    
    if dataset == 'noisy-pinwheel':
        X_train = perturb_pinwheel(X_train, noise_ratio=noise_level, noise_mean=0, noise_std=10, seed=seed_split)
    
    if dataset == 'auto':
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train) * 5.
        X_test = scaler.transform(X_test) * 5.
        if ratio_val is not None:
            X_val = scaler.transform(X_val) * 5.
    elif dataset not in ['pinwheel', 'noisy-pinwheel', 'bar-test', 'video_fhmm', 'video_vae', 'GaussianHMM']:
        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        if ratio_val is not None:
            X_val = scaler.transform(X_val)
    elif dataset == 'video_fhmm':
        train_loader = None
        test_loader = None
        val_loader = None
        return train_loader, test_loader, val_loader
    elif dataset == 'GaussianHMM':
        train_loader = None
        test_loader = None
        val_loader = None
        return train_loader, test_loader, val_loader
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    if ratio_val is not None:
        X_val = torch.FloatTensor(X_val)
    if labels is not None:
        y_train = torch.FloatTensor(y_train)
        y_test = torch.FloatTensor(y_test)
        if ratio_val is not None:
            y_val = torch.FloatTensor(y_val)
    
    if labels is not None:
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    else:
        train_dataset = torch.utils.data.TensorDataset(X_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size if batch_size >0 else train_dataset.__len__(), 
                                                shuffle=True, num_workers=n_threads)
    if labels is not None:
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    else:   
        test_dataset = torch.utils.data.TensorDataset(X_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                shuffle=False, num_workers=n_threads)
    if ratio_val is not None:
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, 
                                                    shuffle=False, num_workers=n_threads)
        
    return train_loader, test_loader, val_loader
        
            
def make_pinwheel_data(radial_std, tangential_std, num_classes, num_per_class, rate):
    rads = np.linspace(0, 2*np.pi, num_classes, endpoint=False)
    np.random.seed(1)
    features = np.random.randn(num_classes * num_per_class, 2) * np.array([radial_std, tangential_std])
    features[:, 0] += 1.
    labels = np.repeat(np.arange(num_classes), num_per_class)
    angles = rads[labels] + rate * np.exp(features[:, 0])
    rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
    rotations = rotations.T.reshape((-1, 2, 2))
    
    feats = 10 * np.einsum('ti,tij->tj', features, rotations)
    data = np.random.permutation(np.hstack([feats, labels[:, None]]))
    return data[:, :2], data[:, 2].astype(np.int)

def perturb_pinwheel(x, noise_ratio=0.1, noise_mean=0.0, noise_std=10, seed=0):
    np.random.seed(seed)
    N, D = x.shape
    num_perturb = int(noise_ratio * N)
    noise_idx = np.random.permutation(np.arange(N))[:num_perturb]
    x[noise_idx, :] += np.random.normal(noise_mean, noise_std, size=(num_perturb, D))
    return x

def make_bar_test_data(num_samples, b, N, idx=None, savepath=None, prior_tree=None, permutation=None, 
                       seed_generation=0):
    np.random.seed(seed_generation)
    if not os.path.exists('./data'):
        os.makedirs('./data')
    if savepath is None:
        tree_handle = 'Tree' if prior_tree is not None else 'Uniform'
        savepath = f'./data/bar_test_{tree_handle}_{N}_{num_samples}_{b}_{seed_generation}.pkl'
    if os.path.exists(savepath):
        with open(savepath, 'rb') as f:
            bar_samples = pickle.load(f)
        f.close()
        print(f'loaded data with {N}_{num_samples}_{b}')
        return bar_samples
    if prior_tree is None:
        num_samples_per_cfg = num_samples // (N**2)
        if idx is None:
            idx = product(np.arange(N), np.arange(N, 2*N))
            idx = np.array(list(idx))
        z = to_multi_hot(2*N, idx)
    else:
        assert permutation is not None
        z = sample_tree(prior_tree, permutation, num_samples)
    W = np.zeros((N**2, 2*N))
    num_cfgs = z.shape[0]
    for i in range(N):
        for j in range(N):
            W[i*N+j, N+i] = 2 * b
            W[i*N+j, j] = 2 * b
    logits = sigmoid(W.dot(z.T) - np.ones((N**2, num_cfgs)) * b)
    if prior_tree is None:
        bar_samples = np.vstack([np.hstack([np.random.binomial(1, logits[l, i], size=(num_samples_per_cfg, )).reshape(-1, 1) for l in range(N**2)]) for i in range(num_cfgs)])
    else:
        bar_samples = np.vstack([np.array([np.random.binomial(1, logits[l, i]) for l in range(N**2)]).reshape(1, -1) for i in range(num_samples)])
    bar_samples = bar_samples.reshape(-1, N**2)
    bar_samples = bar_samples[np.random.permutation(len(bar_samples)), :]
    with open(savepath, 'wb') as f:
        pickle.dump(bar_samples, f)
    print(f'generated data with {N}_{num_samples}_{b}_{seed_generation}')
    return bar_samples