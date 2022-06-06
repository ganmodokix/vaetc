from typing import Optional, Tuple

import torch

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from vaetc.network.kernel import mmd

from .vae import VAE

class MMDVAE(VAE):
    """ InfoVAE with the MMD divergence
    [Zhao+, 2017 (https://arxiv.org/abs/1706.02262)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.w_alpha = float(hyperparameters["alpha"])
        self.w_lambda = float(hyperparameters["lambda"])

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        z_true = torch.randn_like(z).to(self.device)
        loss_mmd = mmd(z_true, z)

        # Total loss
        loss = loss_ae \
            + loss_reg * (1 - self.w_alpha) \
            + loss_mmd * (self.w_alpha + self.w_lambda - 1)

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_mmd": loss_mmd,
        })
