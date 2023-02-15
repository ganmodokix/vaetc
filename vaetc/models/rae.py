import math
from typing import Optional

import torch
from torch import nn

from .vae import VAE
from vaetc.data.utils import IMAGE_SHAPE
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from vaetc.network.ot import sinkhorn, sinkhorn_log
from .utils import detach_dict

class HierarchicalRAE(VAE):
    """ Regularized Autoencoder (RAE) with hierarchical FGW [Xu et al., 2020]
    (https://proceedings.mlr.press/v119/xu20e.html, https://arxiv.org/abs/2002.02913) """ 

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        
        self.beta = float(hyperparameters.get("beta", 0.1))
        self.gamma = float(hyperparameters.get("gamma", 1.0))
        self.n_components = int(hyperparameters.get("n_components", 10))

        self.component_mean = nn.Parameter(0.01 * torch.randn(size=(self.n_components, self.z_dim)), requires_grad=True)
        self.component_logvar = nn.Parameter(-2 + 0.01 * torch.randn(size=(self.n_components, self.z_dim)), requires_grad=True)

        self.proximal_iteration = int(hyperparameters.get("proximal_iteration", 50))
        assert self.proximal_iteration > 0

    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        batch_size = x.shape[0]
        x: torch.Tensor = x.cuda()

        mean, logvar = self.enc_block(x)
        z = self.reparameterize(mean, logvar)
        x2: torch.Tensor = self.dec_block(z)
        
        loss_rec = (x - x2).square().view(batch_size, -1).sum(dim=1).mean()

        component_sigma = (self.component_logvar * 0.5).exp()
        batch_sigma = (logvar * 0.5).exp()

        EPS = 1e-6
        
        sse_mean = (mean[None,:,:] - self.component_mean[:,None,:]).square().sum(dim=2)
        sse_var = (batch_sigma[None,:,:] - component_sigma[:,None,:]).square().sum(dim=2)
        dpq = sse_mean + sse_var + EPS # (K, N)

        sse_mean = (self.component_mean[None,:,:] - self.component_mean[:,None,:]).square().sum(dim=2)
        sse_var = (component_sigma[None,:,:] - component_sigma[:,None,:]).square().sum(dim=2)
        dp = sse_mean + sse_var + EPS # (K, K)

        sse_mean = (mean[None,:,:] - mean[:,None,:]).square().sum(dim=2)
        sse_var = (batch_sigma[None,:,:] - batch_sigma[:,None,:]).square().sum(dim=2)
        dq = sse_mean + sse_var + EPS # (N, N)

        f1_st = (dp ** 2).sum(dim=1) / self.n_components # (K, )
        f2_st = (dq ** 2).sum(dim=1) / batch_size # (N, )
        d = f1_st[:,None] + f2_st[None,:] # (K, N)
        p = torch.ones_like(dp[:,0]) / self.n_components
        q = torch.ones_like(dq[:,0]) / batch_size
        tran = p[:,None] * q[None,:]
        a = p
        for j in range(self.proximal_iteration):
            dpdq = d - 2 * dp @ tran @ dq.T
            cost = dpq * (1 - self.beta) + dpdq * self.beta
            cost = torch.clamp(cost, min=0)

            alpha = 0.1 * cost.abs().max()
            phi = (-cost / alpha).exp() * cost
            b = q / (phi.T @ a)
            a = p / (phi @ b)
            tran = torch.diag(a) @ phi @ torch.diag(b)

        tran = tran.detach()
        dpdq = d - 2 * dp @ tran @ dq.T
        cost = dpq * (1 - self.beta) + dpdq * self.beta
        loss_hfgw = (cost * tran).sum()

        loss = loss_rec + self.gamma * loss_hfgw

        if training:
            
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()

        return detach_dict({
            "loss": loss,
            "loss_rec": loss_rec,
            "loss_hfgw": loss_hfgw,
        })

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)
    
    def sample_prior(self, batch_size: int) -> torch.Tensor:

        eps = torch.randn((batch_size, self.z_dim), device=self.device)
        idx = torch.randint(low=0, high=self.n_components, size=(batch_size, ), device=self.device)
        mean = self.component_mean[idx]
        std = (self.component_logvar[idx] * 0.5).exp()
        return mean + std * eps