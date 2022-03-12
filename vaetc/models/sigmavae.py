from typing import Optional

import math

import torch
from torch import nn
from torch.nn import functional as F

from .utils import detach_dict
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

def softclip(x, xmin):
    return xmin + F.softplus(x - xmin)

class SigmaVAE(VAE):
    """ :math:`\\sigma`-VAE
    [Rybkin+, 2021 (http://proceedings.mlr.press/v139/rybkin21a.html)]"""

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.log_sigma = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):
        
        # Optimal σ
        if progress is not None:
            log_sigma = ((x - x2) ** 2).mean().sqrt().log()
            log_sigma = softclip(log_sigma, -6)
            self.log_sigma.data = log_sigma

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        beta = torch.exp(self.log_sigma.data * 2) * 2

        # Total loss
        loss = loss_ae + loss_reg

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "sigma": self.log_sigma.exp(),
            "equivalent_beta": beta,
        })