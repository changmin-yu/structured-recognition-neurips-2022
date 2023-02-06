import torch
import copy
from torch.utils.data import Dataset

class TupleDataset(Dataset):
    """An iterable over a dataset of tuples (x, y).

    :param x: (tensor) input data.
    :param y: (tensor) output data.
    :param missing: (bool, optional) whether data contains missing data.
    """
    def __init__(self, x, y, missing=False):
        super().__init__()

        assert len(x) == len(y), 'x and y must be the same length.'

        if len(x.shape) == 1:
            # Ensure inputs are 2-dimensional.
            self.x = x.unsqueeze(1)
        else:
            self.x = x

        if missing:
            self.y = copy.deepcopy(y)
            self.m = torch.ones_like(y).fill_(True)

            # Identify nan values and replace with 0.
            m_idx = torch.isnan(y)
            self.m[m_idx] = False
            self.y[m_idx] = 0.
        else:
            self.y = y

        self.missing = missing

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.missing:
            m = self.m[idx]
            return x, y, m, idx
        else:
            return x, y, idx

    def dataset(self):
        idx = list(range(len(self)))

        return self.__getitem__(idx)