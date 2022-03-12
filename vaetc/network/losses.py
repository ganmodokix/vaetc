import torch

from torch import nn
from torch.nn import functional as F

def kl_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL divergence of from the standard normal distribution to a factorized normal distribution

    Args:
        mean (torch.Tensor): :math:`\\boldsymbol{\mu}`
        logvar (torch.Tensor): :math:`\\log \\boldsymbol{\sigma^2}`

    Returns:
        torch.Tensor: :math:`\mathbb{KL}\\left(\mathcal{N}(\\boldsymbol{\mu}, \\boldsymbol{\\sigma^2})||\mathcal{N}(\\mathbf{0}, \\boldsymbol{I})\\right)`
    """

    var = torch.exp(logvar)

    return 0.5 * torch.sum(mean ** 2 + var - logvar - 1, dim=-1)

def neglogpxz_gaussian(x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss in a von Mises-Fisher posterior

    Args:
        x (torch.Tensor): :math:`\\mathbf{x}`
        x2 (torch.Tensor): :math:`\\mathbf{\\hat{x}}`

    Note:
        The normalization constant is not included.
    
    Returns:
        torch.Tensor: :math:`-\\log\\mathcal{N}(\\mathbf{x}|\\mathbf{\\hat{x}}, \\boldsymbol{I})` (without const.)
    """

    xf = x.view(x.shape[0], -1)
    x2f = x2.view(x2.shape[0], -1)
    return 0.5 * torch.sum((xf - x2f) ** 2, dim=-1)

def neglogpxz_von_mises_fisher(x, x2):
    """
    Reconstruction loss in a Gaussian posterior

    Args:
        x (torch.Tensor): :math:`\\mathbf{x}`
        x2 (torch.Tensor): :math:`\\mathbf{\\hat{x}}`

    Note:
        The normalization constant is not included.
    
    Returns:
        torch.Tensor: :math:`-\\log\\mathrm{vMF}(\\mathbf{x}|\\mathbf{\\hat{x}}, \\frac{1}{2})` (without const.)
    """

    return -0.5 * (F.normalize(x, dim=1) * F.normalize(x2, dim=1)).sum(dim=1)

def neglogpxz_bernoulli(x, x2):
    """
    Reconstruction loss in a Bernoulli posterior

    Args:
        x (torch.Tensor): :math:`\\mathbf{x}`
        x2 (torch.Tensor): :math:`\\mathbf{\\hat{x}}`
    
    Returns:
        torch.Tensor: :math:`-\\log\\mathrm{Bernoulli}(\\mathbf{x}|\\mathbf{\\hat{x}})` (without const.)
    """

    x = x.view(x.shape[0], -1)
    x2 = x2.view(x2.shape[0], -1)
    return (torch.xlogy(x, x2) + torch.xlogy(1 - x, 1 - x2)).sum(dim=1)

def neglogpxz_continuous_bernoulli(x, x2):
    """
    Reconstruction loss in a continuous Bernoulli posterior
    (https://proceedings.neurips.cc/paper/2019/hash/f82798ec8909d23e55679ee26bb26437-Abstract.html)

    Args:
        x (torch.Tensor): :math:`\\mathbf{x}`
        x2 (torch.Tensor): :math:`\\mathbf{\\hat{x}}`
    
    Returns:
        torch.Tensor: :math:`-\\log\\mathcal{CB}(\\mathbf{x}|\\mathbf{\\hat{x}})` (without const.)
    """

    EPS = 1e-5

    x2f = x2.view(x2.shape[0], -1)
    m = 1 - 2 * x2f
    c = torch.where(torch.abs(m) < EPS, 2., 2 * torch.arctanh(m) / m)
    logc = c.log()

    return neglogpxz_bernoulli(x, x2) + logc.sum(dim=1)

def sgvb_gaussian(x, x2, mean, logvar):
    """ Stochastic gradient variational Bayes (SGVB) estimator to approximate the evidence lower bound (ELBO),
    where a standard Gaussian prior and diagonal Gaussian :math:`p(\\mathbf{x}|\\mathbf{z})` and :math:`q(\\mathbf{z}|\\mathbf{x})`

    Args:
        x (torch.Tensor): :math:`\\mathbf{x}`
        x2 (torch.Tensor): :math:`\\mathbf{\\hat{x}}`
        mean (torch.Tensor): :math:`\\boldsymbol{\mu}`
        logvar (torch.Tensor): :math:`\\log \\boldsymbol{\sigma^2}`

    Returns
        torch.Tensor: :math:`-\\frac{p(\\mathbf{x},\\mathbf{z})}{q(\\mathbf{z}|\\mathbf{x})}`
    """

    return neglogpxz_gaussian(x, x2) + kl_gaussian(mean, logvar)
