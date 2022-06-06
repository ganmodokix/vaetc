import torch
from torch import nn
from torch.nn import functional as F

def kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

    z_dim = x.shape[1]

    return torch.exp(-((x[:,None,:] - y[None,:,:]) ** 2).mean(dim=2) / z_dim)
    
def mmd(x, y):

    return kernel(x, x).mean() + kernel(y, y).mean() - 2 * kernel(x, y).mean()

def hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    
    kxx = kernel(x, x)
    kyy = kernel(y, y)

    return (kxx * kyy).mean() + kxx.mean() * kyy.mean() - 2 * (kxx.mean(dim=1) * kyy.mean(dim=1)).mean()
