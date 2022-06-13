import math
from typing import Union
from cv2 import log
import torch

from torch import is_tensor, nn
from torch.nn import functional as F

from .gn import randgn

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

def neglogpxz_gaussian(x: torch.Tensor, x2: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Reconstruction loss in a von Mises-Fisher posterior

    Args:
        x (torch.Tensor): :math:`\\mathbf{x}`
        x2 (torch.Tensor): :math:`\\mathbf{\\hat{x}}`
        sigma (torch.Tensor | float): :math:`\\sigma`
        log_sigma (torch.Tensor | float): :math:`\\log \\sigma`

    Note:
        The normalization constant is not included.
    
    Returns:
        torch.Tensor: :math:`-\\log\\mathcal{N}(\\mathbf{x}|\\mathbf{\\hat{x}}, \\sigma \\boldsymbol{I})` (without const.)
    """

    batch_size = x.shape[0]

    if "loggamma" in kwargs:
        loggamma = kwargs["loggamma"]
        if torch.is_tensor(loggamma):
            gamma = torch.exp(loggamma)
        else:
            gamma = math.exp(gamma)
    elif "gamma" in kwargs:
        gamma = kwargs["gamma"]
        if torch.is_tensor(gamma):
            loggamma = torch.log(gamma)
        else:
            loggamma = math.log(gamma)
    else:
        loggamma = 0
        gamma = 1
    
    gamma = kwargs["gamma"] if "gamma" in kwargs else 1
    return 0.5 * ((x - x2) ** 2 / gamma + math.log(math.pi * 2) + loggamma).view(batch_size, -1).sum(dim=1)

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

    batch_size = x.shape[0]
    return -0.5 * (F.normalize(x.view(batch_size, -1), dim=1) * F.normalize(x2.view(batch_size, -1), dim=1)).sum(dim=1)

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

def neglogpxz_gn(x: torch.Tensor, x2: torch.Tensor, logbeta):
    
    logbeta_tensor = torch.tensor(logbeta, dtype=x.dtype, device=x.device)
    logvar = torch.tensor(1., device=x.device, dtype=x.dtype)

    beta = logbeta_tensor.exp()
    invbeta = logbeta_tensor.neg().exp()
    loggib = torch.special.gammaln(invbeta) # log Γ(1/β_e)
    logg3ib = torch.special.gammaln(invbeta) # log Γ(3/β_e)
    logs2 = logvar + loggib - logg3ib # log α^2

    return (((x - x2).abs() / (0.5 * logs2).exp()) ** beta - logbeta_tensor + math.log(2) + logs2 + loggib).view(x.shape[0], -1).sum(dim=1)

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

def kl_gn_gaussian(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    logbeta: torch.Tensor
):
    """ Generalized Normal(Gaussian)
    KL(p(x)||N(x|0,I)),
    where i.i.d p(x_i) = GN(x|mean, std*Γ(1/beta)/Γ(3/beta), beta)
    """

    beta = logbeta.exp()
    invbeta = (-logbeta).exp()

    loggib = torch.special.gammaln(invbeta) # logΓ(1/β)
    logg3ib = torch.special.gammaln(3 * invbeta) # logΓ(3/β)
    logs2 = logvar + loggib - logg3ib

    neg_entropy = -invbeta + logbeta - math.log(2) - logs2 * 0.5 - loggib

    return neg_entropy + 0.5 * (mean ** 2 + logvar.exp() + math.log(2 * math.pi))

def kl_gn_gn(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    logbeta_enc: torch.Tensor,
    logbeta_prior: torch.Tensor,
    num_sampling: int = 5
):
    """
    [Bouhlel & Dziri, 2019 (https://ieeexplore.ieee.org/document/8707051)]
    """

    beta_enc = logbeta_enc.exp()
    beta_prior = logbeta_prior.exp()
    invbeta_enc = (-logbeta_enc).exp()
    invbeta_prior = (-logbeta_prior).exp()

    loggib_enc = torch.special.gammaln(invbeta_enc) # logΓ(1/β_e)
    logg3ib_enc = torch.special.gammaln(invbeta_enc) # logΓ(3/β_e)
    logs2_enc = logvar + loggib_enc - logg3ib_enc
    
    loggib_prior = torch.special.gammaln(invbeta_prior) # logΓ(1/β_p)

    neg_entropy_enc = -invbeta_enc + logbeta_enc - math.log(2) - logs2_enc * 0.5 - loggib_enc

    eps = randgn(logbeta_enc, device=logbeta_enc.device, size=[mean.shape[0], num_sampling, mean.shape[1]])
    z = mean[:,None,:] + (logvar * 0.5).exp()[:,None,:] * eps

    return (neg_entropy_enc - math.log(2) - loggib_prior + (z.abs() ** beta_prior).mean(dim=1)).sum(dim=1)
    

def kl_uniform_gn(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    logbeta_prior: torch.Tensor,
    num_sampling: int = 100
):

    beta_prior = logbeta_prior.exp()
    invbeta_prior = (-logbeta_prior).exp()

    loggib_prior = torch.special.gammaln(invbeta_prior) # logΓ(1/β_p)

    neg_entropy_enc = logvar * 0.5 + math.log(2 * 3 ** 0.5)

    eps = (torch.rand(size=[mean.shape[0], num_sampling, mean.shape[1]], device=mean.device) * 2 - 1) * 3 ** 0.5
    z = mean[:,None,:] + logvar.exp()[:,None,:] * eps

    return (neg_entropy_enc - math.log(2) - loggib_prior + (z.abs() ** beta_prior).mean(dim=1)).sum(dim=1)

def gaussian_cdf(z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return (((z - mean) / (logvar * 0.5).exp() / 2 ** 0.5).erf() + 1) * 0.5

def kl_uniform_gaussian(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

    neg_entropy_enc = logvar * 0.5 + math.log(2 * 3 ** 0.5)
    support_radius = (logvar * 0.5).exp() * 3 ** 0.5
    support_left = mean - logvar * support_radius
    support_right = mean + logvar * support_radius
    neg_cross_entropy = (gaussian_cdf(support_right, mean, logvar) - gaussian_cdf(support_left, mean, logvar)) / support_radius / 2

    return (neg_entropy_enc - neg_cross_entropy).sum(dim=1)