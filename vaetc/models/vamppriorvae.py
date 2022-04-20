import math
from typing import Optional

import torch
from torch import nn

from .vae import VAE
from vaetc.data.utils import IMAGE_SHAPE
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .utils import detach_dict

class VampPriorVAE(VAE):
    """ VAE + VampPrior [Tomczak and Welling, 2017]
    (https://proceedings.mlr.press/v84/tomczak18a.html)
    """

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.num_pseudo_inputs = int(hyperparameters["num_pseudo_inputs"])
        assert self.num_pseudo_inputs > 0

        self.pseudo_inputs = nn.Parameter(torch.rand(size=[self.num_pseudo_inputs, *IMAGE_SHAPE]), requires_grad=True)

    def sample_prior(self, batch_size: int) -> torch.Tensor:

        indices = torch.randint(low=0, high=self.num_pseudo_inputs - 1, size=[batch_size, ]).cuda()
        mean_vp, logvar_vp = self.enc_block(self.pseudo_inputs)

        mean = mean_vp[indices]
        logvar = logvar_vp[indices]

        return self.reparameterize(mean, logvar)

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))

        log2pi = math.log(math.pi * 2)
        logpz = -0.5 * ((z - mean) / logvar.exp() + logvar + log2pi).sum(dim=1)
        mean_vp, logvar_vp = self.enc_block(self.pseudo_inputs)
        logqz = -0.5 * (
            (z[:,None,:] - mean_vp[None,:,:]) / logvar_vp[None,:,:].exp()
            + logvar_vp[None,:,:]
            + log2pi
        ).logsumexp(dim=1).sum(dim=1)
        loss_reg = torch.mean(logpz - logqz)

        # Total loss
        loss = loss_ae + loss_reg

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
        })