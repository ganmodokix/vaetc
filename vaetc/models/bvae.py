from typing import Optional

import math

import torch

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class BetaVAE(VAE):
    """ :math:`\\beta`-VAE
    [Higgins+, 2016 (https://openreview.net/forum?id=Sy2fzU9gl)]"""

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        # Total loss
        loss = loss_ae + loss_reg * self.beta

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
        })