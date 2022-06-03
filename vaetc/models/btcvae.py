from typing import Optional

import math

import torch

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

def total_correlation(z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

    log_qz_core = log_gaussian(z[:,None,:], mean[None,:,:], logvar[None,:,:])

    log_qz = (log_qz_core.sum(dim=2).logsumexp(dim=1) - math.log(mean.shape[0])).mean()
    log_qz_factorized = (log_qz_core.logsumexp(dim=1) - math.log(mean.shape[0])).sum(dim=1).mean()

    return log_qz - log_qz_factorized

def log_gaussian(z: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    
    norm_const = 2. * math.pi

    inv_sigma = torch.exp(-logvar)
    zm = z - mean

    return -0.5 * (zm ** 2 * inv_sigma + logvar + norm_const)

class BetaTCVAE(VAE):
    """ :math:`\\beta`-TCVAE
    [Chen+, 2018 (https://proceedings.neurips.cc/paper/2018/hash/1ee3dfcd8a0645a25a35977997223d22-Abstract.html)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))
        loss_tc  = torch.mean(total_correlation(z, mean, logvar))

        # Total loss
        loss = loss_ae + loss_reg + loss_tc * (self.beta - 1)

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "loss_tc": loss_tc
        })