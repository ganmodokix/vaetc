from typing import Optional

import math

import torch

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class BetaAnnealedVAE(VAE):
    """ :math:`\\beta`-Annealed VAE
    [Sankarapandian et al., NeurIPS 2020 Workshop (https://arxiv.org/abs/2107.10667)]
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        # Cyclical Annealing
        t = progress if progress is not None else 1.0
        t = 1 - t
        annealed_beta = self.beta * t

        # Total loss
        loss = loss_ae + loss_reg * annealed_beta

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "annealed_beta": annealed_beta,
        })