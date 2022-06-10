import numpy as np
from scipy.stats import gamma
from scipy.special import gammaln

import torch

def gnppf(p: np.ndarray, mu: np.ndarray, logvar: np.ndarray, logbeta: np.ndarray) -> np.ndarray:
    
    beta = np.exp(logbeta)
    invbeta = np.exp(-logbeta)
    logs2 = logvar + gammaln(invbeta) - gammaln(3 * invbeta)
    s = np.exp(logs2 * 0.5)

    return mu + np.sign(p - 0.5) * gamma.ppf(2 * np.abs(p - 0.5), invbeta, scale=s ** beta) ** invbeta

def randgn(logbeta: np.ndarray, **factory_kwargs) -> torch.Tensor:
    """
    Samples from Generalized Normal Distribution (μ=0, σ=1, beta)
    Using Inverse CDF (μ=0, σ=1, beta)
    https://cran.r-project.org/web/packages/gnorm/vignettes/gnormUse.html
    """

    mu = 0
    logvar = 0

    if torch.is_tensor(logbeta):
        return logbeta.detach().cpu().numpy()

    p = np.random.random(size=factory_kwargs["size"])
    q = gnppf(p, mu, logvar, logbeta)
    return torch.tensor(q, **{k: v for k, v in factory_kwargs.items() if k != "size"})