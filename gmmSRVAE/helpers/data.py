import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision
from torchvision import datasets
import torch.nn.functional as F

def make_dataloader(dataset, ratio_train=None, ratio_val=None, binary=False, datapath='./datasets', 
                   batch_size=128, test_bs=-1, n_towers=1, n_threads=0, seed_split=0, seed_minibatch=0, 
                   dtype=torch.float32, noise_level=0.1):
    val_loader = None
    if dataset in ['mnist', 'fashion-mnist']:
        if ratio_train is not None:
            print(f'Warning: ratio_train is predefined for the {dataset} dataset.')
        if dataset == 'mnist':
            train_dataset = datasets.MNIST(f'{datapath}/mnist/train', train=True, download=True,  
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize((0.1307,), (0.3081,))]), # empirical mean and std of mnist dataset
                                           target_transform=torchvision.transforms.Compose([
                                               lambda x:torch.LongTensor([x]), # or just torch.tensor
                                               lambda x:F.one_hot(x,10)]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size if batch_size > 0 else train_dataset.__len__(), shuffle=True)
            test_dataset = torchvision.datasets.MNIST(f'{datapath}/mnist/test', train=False, download=True, 
                                                      transform=torchvision.transforms.Compose([
                                                          torchvision.transforms.ToTensor(), 
                                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))]), 
                                                      target_transform=torchvision.transforms.Compose([
                                                          lambda x:torch.LongTensor([x]), # or just torch.tensor
                                                          lambda x:F.one_hot(x,10)]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs if test_bs > 0 else test_dataset.__len__(), shuffle=False)
        elif dataset == 'fashion-mnist':
            train_data = datasets.FashionMNIST(f'{datapath}/fashion-mnist/train', train=True, download=True, 
                                                transform=torchvision.transforms.Compose([
                                                    torchvision.transforms.ToTensor(),
                                                ]))
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=False)
            n_samples_seen = 0.
            mean = 0
            std = 0
            for train_batch, _ in train_loader:
                run_batch_size = train_batch.shape[0]
                train_batch = train_batch.view(run_batch_size, -1)
                this_mean = torch.mean(train_batch, dim=1)
                this_std = torch.sqrt(
                    torch.mean((train_batch - this_mean[:, None]) ** 2, dim=1))
                mean += torch.sum(this_mean, dim=0)
                std += torch.sum(this_std, dim=0)
                n_samples_seen += run_batch_size

            mean /= n_samples_seen
            std /= n_samples_seen
            
            train_dataset = datasets.FashionMNIST(f'{datapath}/fashion-mnist/train', train=True, download=True, 
                                                  transform=torchvision.transforms.Compose([
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((mean, ), (std, ))]), 
                                                  target_transform=torchvision.transforms.Compose([
                                                      lambda x:torch.LongTensor([x]), # or just torch.tensor 
                                                      lambda x:F.one_hot(x,10)]))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size if batch_size > 0 else train_dataset.__len__(), shuffle=True)
            test_dataset = datasets.FashionMNIST(f'{datapath}/fashion-mnist/test', train=False, download=True, 
                                                 transform=torchvision.transforms.Compose([
                                                     torchvision.transforms.ToTensor(),
                                                     torchvision.transforms.Normalize(mean.view(1), std.view(1))]), 
                                                 target_transform=torchvision.transforms.Compose([
                                                     lambda x:torch.LongTensor([x]), # or just torch.tensor
                                                     lambda x:F.one_hot(x,10)]))
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs if test_bs > 0 else test_dataset.__len__(), shuffle=False)

    else:
        if binary:
            raise NotImplementedError
        
        labels = None
        if dataset == 'geyser':
            path = datapath + '/geyser'
            # data = pd.read_csv(path, sep=' ', header=None).values[:, [1, 2]]
            data = pd.read_csv(path, sep=' ', header=None)
            data = np.vstack([data.values[:107, 1:], data.values[107:, :-1]])[:, 1:]
            labels = (data[:, 1] > 3.).astype(np.int)
        elif dataset == 'pinwheel' or dataset == 'noisy-pinwheel':
            num_classes = 5
            data, labels = make_pinwheel_data(0.3, 0.05, num_classes, 2000, 0.25)
        elif dataset == 'aggregation':
            path = datapath + '/Aggregation.txt'
            data_all = pd.read(path, sep='\t', header=None).values
            data = data_all[:, :2]
            labels = data_all[:, -1].astype(np.int) - 1
        elif dataset == 'auto':
            path = datapath + '/Auto/auto-mpg.csv'
            data = pd.read_csv(path, sep=',', header=None).values
            data = data[data[:, 3]!='?']
            labels = data[:, 1]
            data = data[:, [0, 2, 3, 4, 5, 6]].astype(np.float32)
            labels[labels==3] = 0
            labels[labels==4] = 1
            labels[labels==5] = 2
            labels[labels==6] = 3
            labels[labels==8] = 4
            labels = labels.astype(np.int)
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
        elif dataset not in ['pinwheel', 'noisy-pinwheel']:
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            if ratio_val is not None:
                X_val = scaler.transform(X_val)
        
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        if ratio_val is not None:
            X_val = torch.FloatTensor(X_val)
        if labels is not None:
            y_train = torch.FloatTensor(y_train)
            y_test = torch.FloatTensor(y_test)
            if ratio_val is not None:
                y_val = torch.FloatTensor(y_val)
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size if batch_size >0 else train_dataset.__len__(), 
                                                   shuffle=True, num_workers=n_threads)
        test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_bs if test_bs > 0 else test_dataset.__len__(), 
                                                  shuffle=False, num_workers=n_threads)
        if ratio_val is not None:
            val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_bs if test_bs > 0 else val_dataset.__len__(), 
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