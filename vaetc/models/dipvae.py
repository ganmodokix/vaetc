from typing import Optional, Tuple

import math

import torch

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

def cov(x: torch.Tensor) -> torch.Tensor:
    """ Sample covariance.

    Args:
        x (torch.Tensor): shape (B, L)
    
    Returns:
        torch.Tensor: shape (L, L)
    """
    
    # E[x x^T]
    exxt = torch.mean(x[:,None,:] * x[:,:,None], dim=0)

    # E[x] E[x]^T
    ex = torch.mean(x, dim=0)
    exext = ex[:,None] * ex[None,:]

    return exxt - exext

def dip_losses(cov_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    # cov_matrix: (L, L)

    diag_part = torch.diagonal(cov_matrix) # (L, )

    on_diag  = torch.diagflat(diag_part) # (L, L)
    off_diag = cov_matrix - on_diag # (L, L)

    return (off_diag ** 2).sum(), ((diag_part - 1) ** 2).sum()

def dip_i_losses(mean: torch.Tensor) -> torch.Tensor:

    cov_mean = cov(mean)
    
    return dip_losses(cov_mean)

def dip_ii_losses(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

    cov_mean = cov(mean)
    cov_eps  = logvar.exp().mean(dim=0).diagflat()
    
    return dip_losses(cov_mean + cov_eps)

class DIPVAEI(VAE):
    """ DIP-VAE-I
    [Kumar+, 2018 (https://openreview.net/forum?id=H1kG7GZAW)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.ld  = float(hyperparameters["ld"])
        self.lod = float(hyperparameters["lod"])

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))
        loss_lod, loss_ld = dip_i_losses(mean)

        # Total loss
        loss = loss_ae + loss_reg + loss_lod * self.lod + loss_ld * self.ld

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "loss_lod": loss_lod,
            "loss_ld": loss_ld,
        })

class DIPVAEII(DIPVAEI):
    """ DIP-VAE-II
    [Kumar+, 2018 (https://openreview.net/forum?id=H1kG7GZAW)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))
        loss_lod, loss_ld = dip_ii_losses(mean, logvar)

        # Total loss
        loss = loss_ae + loss_reg + loss_lod * self.lod + loss_ld * self.ld

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "loss_lod": loss_lod,
            "loss_ld": loss_ld,
        })