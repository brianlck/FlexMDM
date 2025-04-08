# A file of bregman divergences
import torch

def mse(x, y):
    return torch.sum((x - y) ** 2)


