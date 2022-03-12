from typing import Optional

import torch

from .utils import detach_dict
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class AnnealedVAE(VAE):
    """ Annealed VAE
    [Burgess+, 2017 (https://arxiv.org/abs/1804.03599)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.gamma = float(hyperparameters["gamma"])

        self.max_capacity = float(hyperparameters["max_capacity"])
        assert self.max_capacity >= 0

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = neglogpxz_gaussian(x, x2)
        loss_reg = kl_gaussian(mean, logvar)

        # Cyclical Annealing
        if progress is not None:
            capacity = min(1, progress / 0.9) * self.max_capacity
        else:
            capacity = self.max_capacity

        # Total loss
        loss = loss_ae + self.gamma * torch.abs(loss_reg - capacity)

        loss = loss.mean()
        loss_ae = loss_ae.mean()
        loss_reg = loss_reg.mean()

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "capacity": capacity
        })