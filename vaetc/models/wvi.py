from typing import Optional
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from vaetc.models.utils import detach_dict

from .vae import VAE

class WVI(VAE):
    """ Wasserstein Variational Inference
    [Ambrogioni+, 2018 (https://papers.nips.cc/paper/2018/hash/2c89109d42178de8a367c0228f169bf8-Abstract.html)]"""

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.w1 = float(hyperparameters.get("w1", 1))
        self.w2 = float(hyperparameters.get("w2", 1))
        self.w3 = float(hyperparameters.get("w3", 1))
        self.w4 = float(hyperparameters.get("w4", 1))
        self.w5 = float(hyperparameters.get("w5", 0))

        if self.w5 != 0:
            raise NotImplementedError("f-divergences not implemented")

    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        x2 = x.cuda()
        batch_size = x2.shape[0]

        mean2, logvar2 = self.enc_block(x2)
        z2 = self.reparameterize(mean2, logvar2)

        z1 = self.sample_prior(batch_size)
        x1 = self.dec_block(z1)

        gpz1 = x1
        gpz2 = self.dec_block(z2)

        hqx1, _ = self.enc_block(x1)
        hqx2 = mean2

        loss_ae = ((x1 - x2) ** 2).view(batch_size, -1).sum(dim=1).mean()
        loss_pb = ((gpz1 - gpz2) ** 2).view(batch_size, -1).sum(dim=1).mean()
        loss_la = (((z1 - hqx1) - (z2 - hqx2)) ** 2).sum(dim=1).mean()
        loss_oa = ((x2 - gpz2) ** 2).view(batch_size, -1).sum(dim=1).mean()
        # loss_fd = # not used
        
        loss = self.w1 * loss_ae \
             + self.w2 * loss_pb \
             + self.w3 * loss_la \
             + self.w4 * loss_oa

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()
        
        return detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_pb": loss_pb,
            "loss_la": loss_la,
            "loss_oa": loss_oa,
        })

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)