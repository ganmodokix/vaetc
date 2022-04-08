import numpy as np
from scipy.linalg import sqrtm

import torch
from torch import nn

from torchvision.models import inception_v3
from torchvision import transforms

from tqdm import tqdm

from vaetc.utils.debug import debug_print

@torch.no_grad()
def features_mean_and_cov(x_data: np.ndarray, features, transform, batch_size=64) -> tuple[np.ndarray, np.ndarray]:

    fs = []
    ibs = range(0, x_data.shape[0], batch_size)
    for ib in tqdm(ibs):
        
        x = x_data[ib:ib+batch_size]
        x = torch.tensor(x, dtype=torch.float)
        
        x = transform(x)
        x = x.cuda()

        x = features._transform_input(x)
        f = features(x)
        fs += [f.detach()]

    f = torch.cat(fs, dim=0)
    mean = f.mean(dim=0)
    cov = f.T.cov()

    return mean.detach().cpu().numpy(), cov.detach().cpu().numpy()


def fid(x_real: np.ndarray, x_gen: np.ndarray, eps = 1e-6) -> float:

    features = inception_v3(pretrained=True)
    features.dropout = nn.Sequential()
    features.fc = nn.Sequential()
    features = features.cuda()
    features.eval()

    transform = nn.Sequential(
        nn.AdaptiveAvgPool2d([299, 299]),
    )

    x_real.resize([x_real.shape[0], 3, 64, 64])
    x_gen .resize([x_gen .shape[0], 3, 64, 64])

    mean_real, cov_real = features_mean_and_cov(x_real, features, transform)
    mean_gen , cov_gen  = features_mean_and_cov(x_gen , features, transform)

    # time-consuming part
    debug_print("Calculating matrix sqrt...")
    covmean, _ = sqrtm(cov_real @ cov_gen, disp=False)

    if not np.isfinite(covmean).all():
        debug_print("Matrix sqrt failed, retrying adding εI...")
        offset = np.eye(covmean.shape[0]) * eps
        covmean = sqrtm((cov_real + offset) @ (cov_gen + offset))
    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError(f"Not invertible; imaginary value {m} found in diagonal")
        else:
            covmean = covmean.real

    mean_discrepancy = np.sum((mean_real - mean_gen) ** 2)
    cov_discrepancy = np.trace(cov_real) + np.trace(cov_gen) - 2 * np.trace(covmean)

    return float(mean_discrepancy + cov_discrepancy)