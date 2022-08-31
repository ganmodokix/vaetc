from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vaetc.models.utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian

from .vae import VAE

class SWAE(VAE):
    """ Sliced Wasserstein Autoencoder
    [Kolouri+, 2019 (https://openreview.net/forum?id=H1xaJn05FQ)] """

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.coeff_lambda = float(hyperparameters["lambda"])
        self.num_projections = int(hyperparameters.get("num_projections", 100))

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        batch_size = x.shape[0]
        
        loss_ae = neglogpxz_gaussian(x, x2).mean()

        zp = self.sample_prior(batch_size).to(device=z.device)

        th = F.normalize(torch.randn(size=[batch_size, self.num_projections, self.z_dim], device=z.device), dim=2, p=2, eps=1e-12)
        zth = (z [:,None,:] * th).sum(dim=2)
        zpth = (zp[:,None,:] * th).sum(dim=2)
        zth_sorted, _ = torch.sort(zth, dim=0)
        zpth_sorted, _ = torch.sort(zpth, dim=0)
        loss_sw = ((zth_sorted - zpth_sorted).abs() ** 2).sum()

        loss = loss_ae + self.coeff_lambda * loss_sw

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_sw": loss_sw,
        })