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
        self.sinkhorn_iterations = int(hyperparameters.get("sinkhorn_iterations", 20))
        self.eps = float(hyperparameters.get("eps", 1e-2))

        if self.w5 != 0:
            raise NotImplementedError("f-divergences not implemented")

    def losses(self, x1, z1, x2, z2, gpz1, gpz2, hqx1, hqx2):

        batch_size = x1.shape[0]

        x1 = x1[:,None,...]
        x2 = x2[None,:,...]
        gpz1 = gpz1[:,None,...]
        gpz2 = gpz2[None,:,...]
        hqx1 = hqx1[:,None,...]
        hqx2 = hqx2[None,:,...]

        c_ae = ((x1 - x2) ** 2).view(batch_size, batch_size, -1).sum(dim=2)
        c_pb = ((gpz1 - gpz2) ** 2).view(batch_size, batch_size, -1).sum(dim=2)
        c_la = (((z1 - hqx1) - (z2 - hqx2)) ** 2).sum(dim=2)
        c_oa = ((x2 - gpz2) ** 2).view(batch_size, batch_size, -1).sum(dim=2)
        # c_fd = # not used

        return c_ae, c_pb, c_la, c_oa

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
        
        c_ae, c_pb, c_la, c_oa = self.losses(x1, z1, x2, z2, gpz1, gpz2, hqx1, hqx2)
        c_ae_1, c_pb_1, c_la_1, c_oa_1 = self.losses(x1, z1, x1, z1, gpz1, gpz1, hqx1, hqx1)
        c_ae_2, c_pb_2, c_la_2, c_oa_2 = self.losses(x2, z2, x2, z2, gpz2, gpz2, hqx2, hqx2)

        c_ae = c_ae - (c_ae_1 + c_ae_2) / 2
        c_pb = c_pb - (c_pb_1 + c_pb_2) / 2
        c_la = c_la - (c_la_1 + c_la_2) / 2
        c_oa = c_oa - (c_oa_1 + c_oa_2) / 2
        
        cost_matrix = self.w1 * c_ae \
             + self.w2 * c_pb \
             + self.w3 * c_la \
             + self.w4 * c_oa

        # sinkhorn
        k = (-cost_matrix / self.eps).exp()
        n, m = k.shape
        r = torch.ones_like(k[:,0:1]) / n
        c = torch.ones_like(k[0:1,:]).T / m
        u = r
        for t in self.sinkhorn_iterations:
            a = k.T @ u
            b = c / a
            u = m / (cost_matrix @ b)
        v = c / (k.T @ u)
        loss = (u * ((k * cost_matrix) @ v)).sum()

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()
        
        return detach_dict({
            "loss": loss,
            "c_ae": c_ae,
            "c_pb": c_pb,
            "c_la": c_la,
            "c_oa": c_oa,
        })

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)