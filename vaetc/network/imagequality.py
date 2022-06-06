import torch
from torch import nn
from torch.nn import functional as F

def mse(x: torch.Tensor, x2: torch.Tensor):
    batch_size = x.shape[0]
    return ((x - x2) ** 2).view(batch_size, -1).sum(dim=1)

def cossim(x: torch.Tensor, x2: torch.Tensor):

    batch_size = x.shape[0]
    r  = F.normalize((x  - 0.5).view(batch_size, -1), p=2, dim=1)
    r2 = F.normalize((x2 - 0.5).view(batch_size, -1), p=2, dim=1)
    return (r * r2).sum(dim=1)

def ssim_window(window_size: int, sigma: float = 1.5, dtype=None, device=None, requires_grad=False):

    x = torch.linspace(-window_size / 2.0, window_size / 2.0, window_size, dtype=dtype, device=device, requires_grad=requires_grad)
    x = (-x ** 2 / 2.0 / sigma ** 2).exp()
    x = x / x.sum()

    x = x[:,None] @ x[None,:]

    return x

def ssim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

    batch_size = x1.shape[0]
    num_channels = x1.shape[1]

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    window_size = 11
    window = ssim_window(window_size, dtype=x1.dtype, device=x1.device)
    window = window[None,None,...].tile(3, 1, 1, 1)
    m1 = F.conv2d(x1, window, padding=window_size//2, groups=num_channels)
    m2 = F.conv2d(x2, window, padding=window_size//2, groups=num_channels)
    s1 = m1 ** 2
    s2 = m2 ** 2
    mm = m1 * m2

    ss1 = F.conv2d(x1 ** 2, window, padding=window_size//2, groups=num_channels) - s1
    ss2 = F.conv2d(x2 ** 2, window, padding=window_size//2, groups=num_channels) - s2
    cov = F.conv2d(x1 * x2, window, padding=window_size//2, groups=num_channels) - mm

    ssim_map = (2 * mm + c1) * (2 * cov + c2) / (s1 + s2 + c1) / (ss1 + ss2 + c2)

    return ssim_map.view(batch_size, -1).sum(dim=1)
