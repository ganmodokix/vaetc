from sys import stderr
import numpy as np
import torch

from vaetc.network.losses import kl_gaussian, neglogpxz_gaussian

def elbo(x: np.ndarray, x2: np.ndarray, mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:

    return distortion(x, x2) + rate(mean, logvar)

def rate(mean: np.ndarray, logvar: np.ndarray) -> np.ndarray:

    return kl_gaussian(
        torch.tensor(mean).cuda(),
        torch.tensor(logvar).cuda()
    ).detach().cpu().numpy()

def distortion(x: np.ndarray, x2: np.ndarray, sigma = 1) -> np.ndarray:

    se = squared_error(x, x2)

    return 0.5 * (se + np.log(np.pi * 2 * sigma))

def mean_squared_error(x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    num_pixels = x[0].size
    return squared_error(x, x2) / num_pixels

def psnr(x: np.ndarray, x2: np.ndarray) -> np.ndarray:

    if max(x.max(), x2.max()) > 100:
        print(f"WARNING: data max {x.max()} and {x2.max()}; the inputs may be in [0, 255] while they should be in [0, 1]", file=stderr)
        max_value = 255.0
    else:
        max_value = 1.0

    mse = mean_squared_error(x, x2)
    return np.log10(max_value) * 20 - np.log10(mse) * 10

def squared_error(x: np.ndarray, x2: np.ndarray) -> np.ndarray:
    batch_size = x.shape[0]
    return ((x.reshape(batch_size, -1) - x2.reshape(batch_size, -1)) ** 2).sum(axis=1)