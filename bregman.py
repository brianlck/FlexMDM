# A file of bregman divergences
import torch

def mse(x, y):
    return torch.mean((x - y) ** 2)


