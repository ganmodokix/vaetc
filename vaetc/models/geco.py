import math
from typing import Optional

import torch
from torch import nn

from .vae import VAE
from vaetc.data.utils import IMAGE_SHAPE
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .utils import detach_dict

class GECO(VAE):
    """ Generalized ELBO with Constrained Optimization, GECO
    [Rezende & Viola, 2018 (http://bayesiandeeplearning.org/2018/papers/33.pdf, https://arxiv.org/abs/1810.00597)] """ 

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.tolerance = float(hyperparameters["tolerance"])
        self.momentum = float(hyperparameters["momentum"])
        self.lbd_step = int(hyperparameters["lbd_step"])
        assert self.lbd_step > 0

        # lagrange multiplier(s)
        self.log_lambda = nn.Parameter(torch.tensor(0.0), requires_grad=False)

        self.running_constraint = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.running_batches = nn.Parameter(torch.tensor(0), requires_grad=False)

    def constraint(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        re_constraint = ((x - x2) ** 2).view(batch_size, -1).sum(dim=1)

        return re_constraint - self.tolerance ** 2
        
    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        x = x.cuda()

        mean, logvar = self.enc_block(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.dec_block(z)

        batch_constraint = self.constraint(x, x2)
        batch_avg_constraint = batch_constraint.mean()
        if self.running_batches.item() == 0:
            self.running_constraint.fill_(batch_avg_constraint.item())
        else:
            prev = self.running_constraint.item()
            curr = batch_avg_constraint.item()
            self.running_constraint.fill_(prev * self.momentum + curr * (1.0 - self.momentum))

        loss_constraint = batch_constraint + (self.running_constraint - batch_constraint).detach()
        loss_constraint = loss_constraint.mean()
        loss_reg = torch.clip(kl_gaussian(mean, logvar), min=0).mean()

        loss = loss_constraint * self.log_lambda.exp() + loss_reg

        if training:
            
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()
            
            if self.running_batches.item() % self.lbd_step == 0 and self.running_batches.item() > 200:
                self.log_lambda.fill_(self.log_lambda.item() + max(math.log(0.9), min(math.log(1.05), loss_constraint.item())))

        self.running_batches.fill_(self.running_batches.item() + 1)

        return detach_dict({
            "loss": loss,
            "loss_constraint": loss_constraint,
            "loss_reg": loss_reg,
            "equivalent_beta": (-self.log_lambda).exp(),
        })

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)