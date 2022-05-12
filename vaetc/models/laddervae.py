from typing import Dict, List, Tuple, Optional

import torch
from torch import batch_norm, nn
from vaetc.network.cnn import ConvEncoder
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from vaetc.data.utils import IMAGE_CHANNELS, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_SHAPE
from .vae import VAE
from vaetc.network.blocks import SigmoidInverse

class LadderVAE(VAE):
    """ LadderVAE
    [Sønderby+, 2016 (https://arxiv.org/abs/1602.02282)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.num_layers = int(hyperparameters["num_layers"])
        self.batchnorm_latent = bool(hyperparameters.get("batchnorm_latent", False))
        self.warmup = bool(hyperparameters.get("warmup", False))

        # hyperparameters
        assert self.z_dim // self.num_layers > 0
        self.ladder_dims = [self.z_dim // self.num_layers] * (self.num_layers - 1) \
            + [self.z_dim - self.z_dim // self.num_layers * (self.num_layers - 1)]

        del self.dec_block
        del self.enc_block
        self.enc_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    SigmoidInverse(),
                    nn.Conv2d(IMAGE_CHANNELS, 8, 4, 2, 1),
                    nn.SiLU(True),
                    nn.Conv2d(8, 16, 4, 2, 1),
                    nn.SiLU(True),
                    nn.Conv2d(16, 32, 4, 2, 1),
                    nn.SiLU(True),
                    nn.Conv2d(32, 64, 4, 2, 1),
                    nn.SiLU(True),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.SiLU(True),
                    nn.Flatten(),
                    nn.Linear(128 * 2 * 2, 256),
                    nn.SiLU(True),
                )
            ] + [
                nn.Sequential(
                    nn.Linear(256, 256),
                    nn.SiLU(True),
                    nn.Linear(256, 256),
                    nn.SiLU(True),
                )
            ] * (self.num_layers - 1)
        )
        self.enc_means = nn.ModuleList([
            (
                nn.Sequential(
                    nn.Linear(256, self.ladder_dims[i]),
                    nn.BatchNorm1d(self.ladder_dims[i])
                )
                if self.batchnorm_latent else
                nn.Linear(256, self.ladder_dims[i])
            )
            for i in range(self.num_layers)
        ])
        self.enc_logvars = nn.ModuleList([
            (
                nn.Sequential(
                    nn.Linear(256, self.ladder_dims[i]),
                    nn.BatchNorm1d(self.ladder_dims[i])
                )
                if self.batchnorm_latent else
                nn.Linear(256, self.ladder_dims[i])
            )
            for i in range(self.num_layers)
        ])
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.ladder_dims[0], 64 * 4 * 4),
                nn.SiLU(True),
                nn.Unflatten(dim=1, unflattened_size=[64, 4, 4]),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.SiLU(True),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.SiLU(True),
                nn.ConvTranspose2d(16, 8, 4, 2, 1),
                nn.SiLU(True),
                nn.ConvTranspose2d(8, IMAGE_CHANNELS, 4, 2, 1),
                nn.Sigmoid(),
            )
        ] + [
            nn.Sequential(
                nn.Linear(self.ladder_dims[i], 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
            )
            for i in range(1, self.num_layers)
        ])
        self.dec_means = nn.ModuleList([
            nn.Linear(256, self.ladder_dims[i])
            for i in range(self.num_layers)
        ])
        self.dec_logvars = nn.ModuleList([
            nn.Linear(256, self.ladder_dims[i])
            for i in range(self.num_layers)
        ])

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr)
        }

    def encode_ladder(self, x: torch.tensor) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:

        num_ladders = len(self.ladder_dims)

        d = [x]
        for i in range(num_ladders):
            d += [self.enc_blocks[i](d[-1])]

        means_hat = [None]
        logvars_hat = [None]
        for i in range(num_ladders):
            means_hat += [self.enc_means[i](d[i+1])]
            logvars_hat += [self.enc_logvars[i](d[i+1])]
        
        means_q = [None] * (num_ladders + 1)
        logvars_q = [None] * (num_ladders + 1)
        zs = [None] * (num_ladders + 1)

        means_p = [None] * (num_ladders + 1)
        logvars_p = [None] * (num_ladders + 1)

        means_q[num_ladders] = means_p[num_ladders] = means_hat[num_ladders]
        logvars_q[num_ladders] = logvars_p[num_ladders] = logvars_hat[num_ladders]

        for i in range(num_ladders, 0, -1):

            if i+1 <= num_ladders:
                
                h = self.dec_blocks[i](zs[i+1])
                means_p[i] = self.dec_means[i-1](h)
                logvars_p[i] = self.dec_logvars[i-1](h)
                
                logvars_q[i] = -torch.stack([-means_p[i], -means_hat[i]], dim=0).logsumexp(dim=0)
                means_q[i] = (means_hat[i] * (-logvars_hat[i]).exp() + means_p[i] * (-logvars_p[i]).exp()) * logvars_q[i].exp()

            zs[i] = self.reparameterize(means_q[i], logvars_q[i])

        return means_p, logvars_p, means_q, logvars_q

    def encode_gauss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        means_p, logvars_p, means_q, logvars_q = self.encode_ladder(x)

        mean = torch.concat(means_q[1:], dim=1)
        logvar = torch.concat(logvars_q[1:], dim=1)

        return mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        return self.dec_blocks[0](z[:,:self.ladder_dims[0]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.decode(z)

        return z, mean, logvar, x2

    def sample_prior(self, batch_size: int) -> torch.Tensor:

        means_p = [None] * (self.num_layers + 1)
        logvars_p = [None] * (self.num_layers + 1)
        zs = [None] * (self.num_layers + 1)

        zs[self.num_layers] = torch.randn(size=[batch_size, self.ladder_dims[-1]], device="cuda")
        for i in range(self.num_layers, 1, -1):
            h = self.dec_blocks[i-1](zs[i])
            means_p[i-1] = self.dec_means[i-2](h)
            logvars_p[i-1] = self.dec_logvars[i-2](h)
            # zs[i-1] = self.reparameterize(means_p[i-1], logvars_p[i-1])
            zs[i-1] = means_p[i-1]
        
        return torch.cat(zs[1:], dim=1)

    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        x = x.cuda()

        means_p, logvars_p, means_q, logvars_q = self.encode_ladder(x)
        z1 = self.reparameterize(means_p[1], logvars_p[1])
        x2 = self.dec_blocks[0](z1)

        mean_p = torch.concat(means_p[1:-1], dim=1)
        logvar_p = torch.concat(logvars_p[1:-1], dim=1)
        mean_q = torch.concat(means_q[1:-1], dim=1)
        logvar_q = torch.concat(logvars_q[1:-1], dim=1)

        kl1 = (logvar_p - logvar_q + (logvar_q.exp() + (mean_q - mean_p) ** 2) / 2 * (-logvar_p).exp() - 1/2).sum(dim=1)
        kl2 = kl_gaussian(means_q[-1], logvars_q[-1])

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl1 + kl2)

        # Total loss
        beta = min(1.0, progress * 4) if self.warmup and training else 1.0
        loss = loss_ae + loss_reg * beta

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()

        return detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "beta": beta,
        })


    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)

    def eval_batch(self, batch):
        return self.step_batch(batch)