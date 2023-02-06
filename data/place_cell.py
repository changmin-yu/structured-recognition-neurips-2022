import numpy as np
import os
import pandas as pd
import pickle

def load(session_id, test_ratio=0.05, num_points=6000, binsize=0.1, normalise=True):
    logdir = os.path.join('data', 'place_cell', f'{session_id}.pkl')
    with open(logdir, 'rb') as f:
        all_sc, behavioral_df, all_ft = pickle.load(f)
    f.close()
    first_non_stationary_ind = (behavioral_df.speed.to_numpy() != 0).argmax(axis=0)
    last_non_stationary_ind = (behavioral_df.speed.shape[0]-1) - np.argmax(behavioral_df.speed[::-1]!=0, axis=0)
    behavioral_df = behavioral_df.iloc[first_non_stationary_ind:last_non_stationary_ind]
    all_sc = all_sc[first_non_stationary_ind:last_non_stationary_ind]
    all_ft = all_ft[first_non_stationary_ind:last_non_stationary_ind]
    num_data = min(num_points, len(all_sc))
    behavioral_df = behavioral_df.iloc[:num_data]
    all_sc = all_sc[:num_data]
    all_ft = all_ft[:num_data]
    num_test = int(num_data * test_ratio)
    test_ind_start = np.random.choice(num_data-num_test)
    test_inds = np.arange(test_ind_start, test_ind_start+num_test)
    # test_inds = np.random.choice(np.arange(num_data), size=(num_test, ), replace=False)
    x = behavioral_df.index.to_numpy()
    if normalise:
        x = x / (x.max() + 0.5)
        max_time = (num_data + 0.5) * binsize
    else:
        max_time = 1.
    all_ft_new = []
    offset = first_non_stationary_ind * binsize#  + binsize/2
    for i in range(len(all_ft)):
        all_ft_new.append([])
        for j in range(len(all_ft[i])):
            if len(all_ft[i][j]) == 0:
                all_ft_new[i].append(all_ft[i][j])
            else:
                all_ft_new[i].append(tuple((np.array(all_ft[i][j])-offset)/max_time))
    all_ft = np.array(all_ft_new)
    train_df = pd.DataFrame(all_sc, index=x)
    test_df = pd.DataFrame(all_sc[test_inds], index=x[test_inds])
    all_ft_test = all_ft[test_inds]
    max_counts = np.array([[len(all_ft[i, j]) for i in range(num_data)] for j in range(all_ft.shape[1])]).max()
    return behavioral_df, train_df, test_df, (all_ft, all_ft_test, max_counts)