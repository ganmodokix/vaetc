import torch
from torch import nn
from torch.nn import functional as F

def mse(x: torch.Tensor, x2: torch.Tensor):
    batch_size = x.shape[0]
    return ((x - x2) ** 2).view(batch_size, -1).mean(dim=1)

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

def ssim_window_stats(x1: torch.Tensor, x2: torch.Tensor, window_size: int = 11):
    """ Returns μ1, μ2, σx^2, σy^2, σxy
    """

    batch_size = x1.shape[0]
    num_channels = x1.shape[1]

    window = ssim_window(window_size, dtype=x1.dtype, device=x1.device)
    window = window[None,None,...].tile(3, 1, 1, 1)
    m1 = F.conv2d(x1, window, padding=window_size//2, groups=num_channels)
    m2 = F.conv2d(x2, window, padding=window_size//2, groups=num_channels)

    ss1 = F.conv2d(x1 ** 2, window, padding=window_size//2, groups=num_channels) - m1 ** 2
    ss2 = F.conv2d(x2 ** 2, window, padding=window_size//2, groups=num_channels) - m2  **2
    cov = F.conv2d(x1 * x2, window, padding=window_size//2, groups=num_channels) - m1 * m2

    return m1, m2, ss1, ss2, cov

def ssim_comparisons(m1: torch.Tensor, m2: torch.Tensor, ss1: torch.Tensor, ss2: torch.Tensor, cov: torch.Tensor, c1: float, c2: float, c3: float):

    eps = 1e-8 # to prevent nan grad
    s1s2 = (ss1 * ss2).clamp_min(eps) ** 0.5

    l = (2 * m1 * m2 + c1) / (m1 ** 2 + m2 ** 2 + c1)
    c = (2 * s1s2 + c2) / (ss1 + ss2 + c2)
    s = (cov + c3) / (s1s2 + c3)

    return l, c, s

def ssim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

    batch_size = x1.shape[0]
    num_channels = x1.shape[1]
    window_size = 11

    m1, m2, ss1, ss2, cov = ssim_window_stats(x1, x2, window_size=window_size)

    value_range = max(x1.max().item(), x2.max().item())
    k1, k2 = 0.01, 0.03
    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2
    # c3 = c2 / 2

    # l, c, s = ssim_comparisons(m1, m2, ss1, ss2, cov, c1, c2, c3)
    # ssim_map = l ** alpha + c ** beta + s ** gamma
    ssim_map = (2 * m1 * m2 + c1) * (2 * cov + c2) / (m1 ** 2 + m2 ** 2 + c1) / (ss1 + ss2 + c2)

    return ssim_map.view(batch_size, -1).mean(dim=1)

def msssim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

    batch_size = x1.shape[0]
    window_size = 11

    # best values in [Wang+, 2003]
    alpha = 0.1333
    betas = [0.0447, 0.2856, 0.3001, 0.2363, 0.1333]
    gammas = [0.0447, 0.2856, 0.3001, 0.2363, 0.1333]
    m = len(gammas)

    value_range = max(x1.max().item(), x2.max().item())
    k1, k2 = 0.01, 0.03
    c1 = (k1 * value_range) ** 2
    c2 = (k2 * value_range) ** 2
    c3 = c2 / 2

    h1, h2 = x1, x2
    cs, ss = [], []
    eps = 1e-8
    for i, (beta, gamma) in enumerate(zip(betas, gammas)):
        m1, m2, ss1, ss2, cov = ssim_window_stats(h1, h2, window_size=window_size)
        l, c, s = ssim_comparisons(m1, m2, ss1, ss2, cov, c1, c2, c3)
        c = c.reshape(batch_size, -1).mean(dim=-1)
        s = s.reshape(batch_size, -1).mean(dim=-1)
        cs += [c.clamp_min(eps) ** beta]
        ss += [s.clamp_min(eps) ** gamma]
        if i+1 < m:
            h1 = F.avg_pool2d(h1, kernel_size=2, padding=(h1.shape[2] % 2, h1.shape[3] % 2))
            h2 = F.avg_pool2d(h2, kernel_size=2, padding=(h2.shape[2] % 2, h2.shape[3] % 2))

    # note that the l in the last level is used
    l = l.view(batch_size, -1).mean(dim=-1) ** alpha
    c = torch.stack(cs, dim=0).prod(dim=0)
    s = torch.stack(ss, dim=0).prod(dim=0)
    
    return l * c * s