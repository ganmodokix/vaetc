import math
from typing import Optional

import torch
from torch import kl_div, nn
from vaetc.models.utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian

from .vae import VAE

LOG2PI = math.log(math.pi * 2)

class IWAE(VAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.num_samples = int(hyperparameters["num_samples"])
        assert self.num_samples > 0

    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        x = x.cuda()
        batch_size = x.shape[0]

        mean, logvar = self.encode_gauss(x)

        meank = mean[:,None,:].tile(1, self.num_samples, 1).view(batch_size * self.num_samples, self.z_dim)
        logvark = logvar[:,None,:].tile(1, self.num_samples, 1).view(batch_size * self.num_samples, self.z_dim)

        zk = self.reparameterize(meank, logvark)
        x2k = self.decode(zk).view(batch_size, self.num_samples, *x.shape[1:])
        zk = zk.view(batch_size, self.num_samples, self.z_dim)

        meank = meank.view(batch_size, self.num_samples, self.z_dim)
        logvark = logvark.view(batch_size, self.num_samples, self.z_dim)

        rec = -0.5 * ((x[:,None,...] - x2k) ** 2 + LOG2PI + 1).view(batch_size, self.num_samples, -1).sum(dim=2)
        reg = -0.5 * torch.sum(meank ** 2 + logvark.exp() - logvark - 1, dim=-1)
        elbo = rec + reg
        loss = -(elbo.logsumexp(dim=1) - math.log(batch_size)).mean()

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()

        return detach_dict({
            "loss": loss
        })
        

    def eval_batch(self, batch):
        return self.step_batch(batch)

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)