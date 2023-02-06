import numpy as np

def mse(mean, data):
    return ((mean - data) ** 2).mean()

def smse(mean, data):
    return mse(mean, data) / mse(data.mean(), data)