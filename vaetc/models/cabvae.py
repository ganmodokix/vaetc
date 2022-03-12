from typing import Optional

import math

import torch

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class CyclicalAnnealingBetaVAE(VAE):
    """ Cyclical Annealing :math:`\\beta`-VAE
    [Fu+, 2019 (https://arxiv.org/abs/1903.10145)]"""

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        # Cyclical Annealing
        if progress is not None:
            schedule = min(1, 1 if progress >= 15/16 else progress * 8 % 2)
            annealed_beta = self.beta * schedule
        else:
            annealed_beta = self.beta

        # Total loss
        loss = loss_ae + loss_reg * annealed_beta

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "annealed_beta": annealed_beta,
        })