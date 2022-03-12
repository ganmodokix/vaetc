from typing import Tuple

import itertools
import math

import torch
from tqdm import tqdm

# WIP: entropy一括importで書き直す
from .entropy import entropy_sampling, entropy_quantization, entropy_histogram
from .entropy import entropy_conditioned, entropy_binary, entropy_joint_binary
from .entropy import QUANTIZATION_RANGE, QUANTIZATION_BINS

def misjed(mean: torch.Tensor, logvar: torch.Tensor, bins: int = QUANTIZATION_BINS) -> torch.Tensor:
    """ MISJED by [Do and Tran, 2020 (https://openreview.net/forum?id=HJgK0h4Ywr)]

    Args:
        mean: :math:`\\mu_i`, shape (B, L)
        logvar: :math:`\\log\\sigma_i^2`, shape (B, L)

    Returns:
        (L, L)-sized torch.Tensor: normalized :math:`\mathrm{MISJED}(z_i, z_j)`
    """

    assert mean.shape == logvar.shape
    assert mean.ndim == 2
    assert bins >= 1

    batch_size, z_dim = mean.shape

    h = entropy_quantization(mean, logvar, bins=bins) # (L, )

    hbar = torch.zeros(size=(z_dim, z_dim), device=mean.device)

    # diagonal
    hbar[range(z_dim), range(z_dim)] = h

    # non-diagonal
    ij = [(i, j) for i, j in itertools.product(range(z_dim), range(z_dim)) if i < j]
    for i, j in tqdm(ij):
        hbar[i,j] = hbar[j,i] = entropy_histogram(mean[:,(i, j)], bins=bins)

    unnormalized = h[:,None] + h[None,:] - hbar # (L, L)

    vmin = 0
    vmax = math.log(bins) * 2
    normalized = (unnormalized - vmin) / max(1e-12, vmax - vmin)

    return normalized

def informativeness(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """ Informativeness by [Do and Tran, 2020 (https://openreview.net/forum?id=HJgK0h4Ywr)]

    Args:
        mean: :math:`\\mu_i`, shape (B, L)
        logvar: :math:`\\log\\sigma_i^2`, shape (B, L)
    
    Returns:
        :math:`I(x,z_i)`
    """

    batch_size, z_dim = mean.shape

    ent_zi = entropy_quantization(mean, logvar)
    ent_notzi = torch.zeros(size=(z_dim, ), device=mean.device)
    for i in range(z_dim):
        not_i = torch.arange(z_dim) != i
        ent_notzi[i] = entropy_sampling(mean[:,not_i], logvar[:,not_i])

    ent_zi_on_x = entropy_conditioned(logvar)
    mut_x_zi = ent_zi - ent_zi_on_x # informativeness

    return mut_x_zi

def windin(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """ WINDIN by [Do and Tran, 2020 (https://openreview.net/forum?id=HJgK0h4Ywr)]

    Args:
        mean: :math:`\\mu_i`, shape (B, L)
        logvar: :math:`\\log\\sigma_i^2`, shape (B, L)
        
    Returns:
        :math:`\\mathrm{WINDIN}(z_i)`
    """

    batch_size, z_dim = mean.shape

    bin_width = (QUANTIZATION_RANGE[1] - QUANTIZATION_RANGE[0]) / QUANTIZATION_BINS

    # informativeness
    ent_zi = entropy_quantization(mean, logvar) # H(z_i)
    ent_notzi = torch.zeros(size=(z_dim, ), device=mean.device) # H(z_!=i)
    for i in range(z_dim):
        not_i = torch.arange(z_dim) != i
        ent_notzi[i] = entropy_sampling(mean[:,not_i], logvar[:,not_i]) - math.log(bin_width)
    
    ent_z = entropy_sampling(mean, logvar) - math.log(bin_width) # H(z)
    mut_zi_znoti = ent_zi + ent_notzi - ent_z # I(z_i, z_!=i) = H(z_i) + H(z_!=i) - H(z)

    ent_zi_on_x = entropy_conditioned(logvar) # H(z_i|x)
    mut_x_zi = ent_zi - ent_zi_on_x # informativeness I(x, z_i) = H(z_i) - H(z_i|x)

    indin_zi = mut_x_zi - mut_zi_znoti
    
    eps = torch.tensor(1e-10, device=mean.device)
    rho = mut_x_zi / torch.maximum(mut_x_zi.sum(), eps)
    windin_zi = rho * indin_zi

    return torch.sum(windin_zi)

def mut_binary(
    mean: torch.Tensor, logvar: torch.Tensor,
    binary: torch.Tensor
):
    """ Entropies and mutual information

    Args:
        mean: (B, L) np.ndarray
        logvar: (B, L) np.ndarray
        binary: (B, T) np.ndarray
    
    Returns:
        A tuple of `(ent_zi, ent_yk, ent_zi_yk, mut_zi_yk)`

            `ent_zi`: :math:`H(z_i)` (L, ) torch.Tensor

            `ent_yk`: :math:`H(y_k)` (T, ) torch.Tensor

            `ent_zi_yk`: :math:`H(z_i, y_k)` (L, T) torch.Tensor

            `mut_zi_yk`: :math:`I(z_i, y_k)` (L, T) torch.Tensor
    """

    t_dim = binary.shape[1]

    ent_zi = entropy_quantization(mean, logvar)
    ent_yk = entropy_binary(binary)
    ent_zi_yk = entropy_joint_binary(mean, logvar, binary)
    mut_zi_yk = ent_zi[:,None] + ent_yk[None,:] - ent_zi_yk

    return ent_zi, ent_yk, ent_zi_yk, mut_zi_yk


def rmig_jemmig(
    mean: torch.Tensor, logvar: torch.Tensor,
    binary: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ (R)MIG and JEMMIG by [Do and Tran, 2020 (https://openreview.net/forum?id=HJgK0h4Ywr)]

    Args:
        mean: (B, L)
        logvar: (B, L)
        binary: (B, T)
    
    Returns:
        a tuple of (RMIG, JEMMIG, Normalized JEMMIG), 3x (T, )-shaped tensor.
    
    Note:
        You should take the mean for the entire model.
    """

    t_dim = binary.shape[1]
    ent_zi, ent_yk, ent_zi_yk, mut_zi_yk = mut_binary(mean, logvar, binary)

    idx = torch.argsort(mut_zi_yk, dim=0)
    iastr = idx[-1]
    jcirc = idx[-2]
    mut_ziastr_yk = mut_zi_yk[iastr,range(t_dim)]
    mut_zjcirc_yk = mut_zi_yk[jcirc,range(t_dim)]
    rmig_k = mut_ziastr_yk - mut_zjcirc_yk
    ent_ziastr_yk = ent_zi_yk[iastr,range(t_dim)]
    jemmig_k = ent_ziastr_yk - rmig_k

    a, b = QUANTIZATION_RANGE
    ent_u = math.log(QUANTIZATION_BINS)
    normalized_jemmig_k = 1.0 - jemmig_k / (ent_u + ent_yk)

    return rmig_k, jemmig_k, normalized_jemmig_k

def mig_sup(
    mean: torch.Tensor, logvar: torch.Tensor,
    binary: torch.Tensor
) -> torch.Tensor:
    """ MIG-sup by [Li+, 2020 (https://openreview.net/forum?id=SJxpsxrYPS)]

    Args:
        mean: (B, L)
        logvar: (B, L)
        binary: (B, T)

    Returns:
        MIG-sup; (L, )-shaped
    
    Note:
        You should take the mean for the entire model
    """

    batch_size, z_dim = mean.shape
    batch_size, t_dim = binary.shape
    
    ent_zi, ent_yk, ent_zi_yk, mut_zi_yk = mut_binary(mean, logvar, binary)

    idx = torch.argsort(mut_zi_yk, dim=1)
    k1st = idx[:,-1]
    k2nd = idx[:,-2]
    mut_zi_yk1st = mut_zi_yk[range(z_dim),k1st]
    mut_zi_yk2nd = mut_zi_yk[range(z_dim),k2nd]
    mig_sup_i = mut_zi_yk1st - mut_zi_yk2nd

    return mig_sup_i

def modularity(
    mean: torch.Tensor, logvar: torch.Tensor,
    binary: torch.Tensor
) -> torch.Tensor:
    """ Modularity Score by [Ridgeway and Mozer, 2018 (https://proceedings.neurips.cc/paper/2018/hash/2b24d495052a8ce66358eb576b8912c8-Abstract.html)]

    Args:
        mean: (B, L)
        logvar: (B, L)
        binary: (B, T)

    Returns:
        modularity score of :math:`z_i`, shape (L, )
    """

    batch_size, z_dim = mean.shape
    batch_size, t_dim = binary.shape
    
    ent_zi, ent_yk, ent_zi_yk, mut_zi_yk = mut_binary(mean, logvar, binary)

    idx = torch.argsort(mut_zi_yk, dim=1)
    k1st = idx[:,-1]
    mut2_sum_i = torch.sum(mut_zi_yk ** 2, dim=1)
    mut2_zi_yk1st = mut_zi_yk[range(z_dim), k1st] ** 2

    EPS = 1e-4
    modularity_i = (mut2_sum_i - mut2_zi_yk1st) / (EPS + mut2_zi_yk1st ** 2 * (t_dim - 1))
    return modularity_i

def dcimig(
    mean: torch.Tensor, logvar: torch.Tensor,
    binary: torch.Tensor
) -> torch.Tensor:
    """ DCIMIG (3CharM) by [Sepliarskaia+, 2019 (https://arxiv.org/abs/1910.05587)]
    """

    batch_size, z_dim = mean.shape
    batch_size, t_dim = binary.shape
    
    ent_zi, ent_yk, ent_zi_yk, mut_zi_yk = mut_binary(mean, logvar, binary)

    idx = torch.argsort(mut_zi_yk, dim=1)
    k1st = idx[:,-1]
    k2nd = idx[:,-2]
    mut_zi_yk1st = mut_zi_yk[range(z_dim),k1st]
    mut_zi_yk2nd = mut_zi_yk[range(z_dim),k2nd]
    gap_i = mut_zi_yk1st - mut_zi_yk2nd # also MIG-sup
    scatter = [[torch.zeros(size=(), device=mean.device)] for _ in range(t_dim)]
    for i in range(z_dim):
        scatter[k1st[i]].append(gap_i[i])
    
    s_i = torch.stack([torch.max(torch.stack(tensor_list)) for tensor_list in scatter])

    EPS = 1e-4
    dcimig = torch.sum(s_i) / (EPS + torch.sum(ent_yk))
    return dcimig
