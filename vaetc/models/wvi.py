import math
from typing import Optional
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from vaetc.models.utils import detach_dict
from vaetc.network.ot import sinkhorn, sinkhorn_log

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
        self.use_unbiased_estimator = bool(hyperparameters.get("use_unbiased_estimator", False))
        self.detach_sinkhorn = bool(hyperparameters.get("detach_sinkhorn", True))

        if self.w5 != 0:
            raise NotImplementedError("f-divergences not implemented")

    def divergences(self, x1, z1, x2, z2, gpz1, gpz2, hqx1, hqx2):

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

    def sinkhorn(self, cost_matrix: torch.Tensor) -> torch.Tensor:

        n, m = cost_matrix.shape
        r = torch.ones(size=[n, 1], device=cost_matrix.device, dtype=cost_matrix.dtype) / n
        c = torch.ones(size=[m, 1], device=cost_matrix.device, dtype=cost_matrix.dtype) / m

        return sinkhorn(r, c, cost_matrix, self.sinkhorn_iterations, self.eps)

    def sinkhorn_log(self, cost_matrix: torch.Tensor) -> torch.Tensor:

        n, m = cost_matrix.shape
        logr = torch.zeros(size=[n, 1], device=cost_matrix.device, dtype=cost_matrix.dtype) - math.log(n)
        logc = torch.zeros(size=[m, 1], device=cost_matrix.device, dtype=cost_matrix.dtype) - math.log(m)

        return sinkhorn_log(logr, logc, cost_matrix, self.sinkhorn_iterations, self.eps, ab_log=True)

    def loss_by_sinkhorn(self, x1, z1, x2, z2, gpz1, gpz2, hqx1, hqx2):
        
        c_ae, c_pb, c_la, c_oa = self.divergences(x1, z1, x2, z2, gpz1, gpz2, hqx1, hqx2)
        
        cost_matrix = self.w1 * c_ae \
             + self.w2 * c_pb \
             + self.w3 * c_la \
             + self.w4 * c_oa

        s = self.sinkhorn_log(cost_matrix)
        if self.detach_sinkhorn:
            s = s.detach()
        loss = (s * cost_matrix).sum()

        return loss

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
        
        loss_pq = self.loss_by_sinkhorn(x1, z1, x2, z2, gpz1, gpz2, hqx1, hqx2)
        if self.use_unbiased_estimator:
            loss_pp = self.loss_by_sinkhorn(x1, z1, x1, z1, gpz1, gpz1, hqx1, hqx1)
            loss_qq = self.loss_by_sinkhorn(x2, z2, x2, z2, gpz2, gpz2, hqx2, hqx2)
            loss = loss_pq - (loss_pp + loss_qq) / 2
        else:
            loss = loss_pq

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()
        
        loss_dict = detach_dict({
            "loss": loss,
            "loss_pq": loss_pq,
        })

        if self.use_unbiased_estimator:

            loss_dict |= detach_dict({
                "loss_pp": loss_pp,
                "loss_qq": loss_qq,
            })

        return loss_dict

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)