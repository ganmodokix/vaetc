import math
from numpy import histogramdd

import torch

from .reparam import reparameterize

@torch.jit.script
def log_gaussian_density(
    mean1: torch.Tensor, logvar1: torch.Tensor,
    mean2: torch.Tensor, logvar2: torch.Tensor
):
    """The log-density of log-Gaussian (for minibatch weighted sampling)

    Args:
        mean1  : :math:`\\mu` of sampling batch outside the log with size :math:`(B_1, L)`
        logvar1: :math:`\\log\\sigma^2` of sampling batch outside the log with size :math:`(B_1, L)`
        mean2  : :math:`\\mu` of sampling batch inside the log with size :math:`(B_2, L)`
        logvar2: :math:`\\log\\sigma^2` of sampling batch inside the log with size :math:`(B_2, L)`

    Returns:
        torch.Tensor :math:`\\log(z_k^{(j)}|x^{(i)})` with size (B1, B2, L)
    """

    z1 = reparameterize(mean1, logvar1) # (B1, L)

    invstd2 = torch.exp(logvar2 * -0.5) # (B2, L)

    z1      = z1     [:,None,:] # (B1, 1, L)
    mean2   = mean2  [None,:,:] # (B2, L, 1)
    invstd2 = invstd2[None,:,:] # (B2, L, 1)
    logvar2 = logvar2[None,:,:] # (B2, L, 1)
    log2pi  = math.log(math.pi * 2)

    core = ((z1 - mean2) * invstd2) ** 2 + logvar2 + log2pi

    return -core

@torch.jit.script
def entropy_sampling(mean: torch.Tensor, logvar: torch.Tensor) -> float:
    """ Entropy by sampling (https://openreview.net/forum?id=HJgK0h4Ywr)

    Args:
        mean (numpy.ndarray): mean of Gaussian, shape (B, L)
        logvar (numpy.ndarray): log-varianve of Gaussian, shape (B, L)
        
    Returns:
        float: :math:`H(\\mathbf{z})`
    """

    assert mean.ndim == 2
    assert logvar.ndim == 2
    assert mean.shape == logvar.shape

    batch_size = mean.shape[0]
    assert batch_size >= 2

    split_size = batch_size // 2
    mean1   = mean  [:split_size]
    logvar1 = logvar[:split_size]
    mean2   = mean  [split_size:]
    logvar2 = logvar[split_size:]

    size1 = split_size
    size2 = batch_size - split_size
    
    logcore = log_gaussian_density(mean1, logvar1, mean2, logvar2) # (B1, B2, L)
    logcore = torch.sum(logcore, dim=2) # (B1, B2)

    entropy = torch.logsumexp(logcore, dim=1) # (B1, )
    entropy = entropy - math.log(size2) # (B1, )
    entropy = -torch.mean(entropy) # ()

    return entropy

QUANTIZATION_RANGE = (-4, 4)
QUANTIZATION_BINS = 100
def entropy_quantization(mean: torch.Tensor, logvar: torch.Tensor, bins: int = QUANTIZATION_BINS):
    """ Entropy by quantization (https://openreview.net/forum?id=HJgK0h4Ywr)

    Args:
        mean (torch.Tensor): mean of :math:`z_i` with shape :math:`(B, L)`
        logvar (torch.Tensor): logvar of :math:`z_i` with shape :math:`(B, L)`
    
    Note:
        The sum is *not* the joint entropy of the entire :math:`\\mathbf{z}`
    
    Returns:
        :math:`H(z_i)` with shape (L, )
    """

    assert bins >= 1
    
    zmin, zmax = QUANTIZATION_RANGE
    poss = torch.linspace(zmin, zmax, bins+1, device=mean.device)
    a = poss[None,None, :-1]
    b = poss[None,None,1:  ]
    mean   = mean  [:,:,None]
    logvar = logvar[:,:,None]

    invstd = torch.exp(logvar * -0.5)
    invsqrt2 = 2 ** -0.5

    eps = torch.tensor(1e-10, device=mean.device)
    ca = (a - mean)  * invstd * invsqrt2
    cb = (b - mean)  * invstd * invsqrt2

    # qsin[n,i,j] = Q(s_j|x_n)
    qsin = 0.5 * (torch.erf(cb) - torch.erf(ca)) # (B, L, bins)

    # Q(s_j) = Mean of Q(s_j|x_n) w.r.t. n
    qsi = torch.mean(qsin, dim=0) # (L, bins)

    hzi = torch.sum(-torch.xlogy(qsi, torch.maximum(qsi, eps)), dim=1) # (L, )

    return hzi

def entropy_histogram(z: torch.Tensor, bins: int = QUANTIZATION_BINS):
    """ Entropy by histogram (https://openreview.net/forum?id=HJgK0h4Ywr)

    Args:
        z (torch.Tensor): sampled latent codes, shape (B, L)
    
    Returns:
        :math:`H(\\mathbf{z}) via quantization
    """

    assert z.ndim == 2

    batch_size, z_dim = z.shape
    assert z_dim <= 2, "it suffers from curse of dimension"

    eps = 1e-7

    zmin, zmax = torch.min(z, dim=0).values[None,:] - eps, torch.max(z, dim=0).values[None,:] + eps

    if z.shape[1] == 1:
        
        histogram = torch.histc(z, bins=bins, min=zmin, max=zmax)

    else:
        
        zpos = ((z - zmin) / (zmax - zmin) * bins).type(torch.int)
        zpos = torch.clip(zpos, 0, bins-1)
        histogram = torch.zeros(size=(bins, ) * z_dim, dtype=int, device=z.device)

        unique_zpos = torch.unique(zpos, dim=0)
        for zpos_i in unique_zpos:
            match_table = zpos == zpos_i[None,...]
            match_table_unflattened = match_table.view(zpos.shape[0],-1)
            histogram[tuple(zpos_i)] += match_table_unflattened.all(dim=1).count_nonzero()
        
        # for i in range(batch_size):
        #     histogram[tuple(zpos[i])] += 1
    
    prob = histogram.type(torch.float) / batch_size

    darea = torch.prod((zmax - zmin) / bins)
    ent = torch.sum(-torch.xlogy(prob, prob / darea))

    return ent

def entropy_conditioned(logvar: torch.Tensor) -> torch.Tensor:
    """ Conditioned entropy.

    Args:
        logvar: :math:`\\log\\sigma_i^2`, shape (B, L)

    Returns:
        :math:`H(z_i|x)`, shape (L, )
    """

    return ((logvar + math.log(math.pi * 2) + 1) * 0.5).mean(dim=0)

@torch.jit.script
def entropy_binary(binary: torch.Tensor) -> torch.Tensor:

    batch_size, t_dim = binary.shape

    p = torch.count_nonzero(binary, dim=0) / batch_size

    return -torch.xlogy(p, p) - torch.xlogy(1-p, 1-p)

def entropy_joint_binary(
    mean: torch.Tensor, logvar: torch.Tensor,
    binary: torch.Tensor, threshold: float = 0.5
):
    """ Joint entropy of Gaussian variables and Bernoulli variables by quantization.

    Args:
        mean: :math:`\\mu_i`, shape (B, L)
        logvar: :math:`\\log \\sigma_i^2`, shape (B, L)
        binary: :math:`y_k`, shape (B, C)
        threshold (:class:`float`, optional): threshold of :math:`y`

    Returns:
        :math:`H(z_i, y_k)` with shape (L, C)
    """

    assert mean.shape == logvar.shape
    assert mean.shape[0] == binary.shape[0]
    
    batch_size, z_dim = mean.shape
    batch_size, t_dim = binary.shape

    mask = (binary >= threshold).T # (t_dim, batch_size)

    ent_joint = []
    for k in range(t_dim):

        n1 = torch.count_nonzero(mask[k])
        n0 = batch_size - n1
        
        if min(n0, n1) == 0:

            ent = entropy_quantization(mean, logvar)
            ent_joint += [ent]

        else:

            ent1 = entropy_quantization(mean[mask[k]], logvar[mask[k]])
            ent0 = entropy_quantization(mean[~mask[k]], logvar[~mask[k]])
            ent_joint += [(ent0 * n0 + ent1 * n1) / (n0 + n1)]

    ent_joint = torch.stack(ent_joint, dim=1)
    return ent_joint