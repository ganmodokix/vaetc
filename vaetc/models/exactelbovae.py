from typing import Optional

import math

import torch
from torch import nn

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

LOG2PI = math.log(math.pi * 2)

class ExactELBOVAE(VAE):
    """ Don't Blame the ELBO!
    [Lucas+, 2019 (https://proceedings.neurips.cc/paper/2019/hash/7e3315fe390974fcf25e44a9445bd821-Abstract.html)]
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.log_sigma2 = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        # Total loss
        sigma2 = self.log_sigma2.exp()
        loss = loss_reg + loss_ae / sigma2 + 0.5 * (LOG2PI + self.log_sigma2)

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "sigma": (0.5 * self.log_sigma2).exp(),
        })