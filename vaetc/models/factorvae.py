from typing import Dict, Tuple, Optional
import itertools

import torch
from torch import nn

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class Discriminator(nn.Module):
    """ Discriminator for FactorVAE. """

    def __init__(self, z_dim: int):

        super().__init__()

        self.z_dim = int(z_dim)

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # [1-D(x), D(x)] = [p_fake, p_real]
        return self.net(x)

@torch.jit.script
def permute_dims(z: torch.Tensor) -> torch.Tensor:
    
    batch_size = z.shape[0]
    z_dim = z.shape[1]
    
    permute_indices = torch.rand_like(z).argsort(dim=0)
    keep_elements = torch.arange(z_dim)[None,:]
    zs_permuted = z[permute_indices,keep_elements]

    return zs_permuted

class FactorVAE(VAE):
    """ FactorVAE
    [Kim and Mnih, 2018 (http://proceedings.mlr.press/v80/kim18b.html)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        self.lr_disc = float(hyperparameters["lr_disc"])
        self.gamma = float(hyperparameters["gamma"])

        # network layers
        self.enc_block = ConvGaussianEncoder(z_dim=self.z_dim)
        self.dec_block = ConvDecoder(z_dim=self.z_dim)
        self.disc_block = Discriminator(z_dim=self.z_dim)

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:

        main_parameters = itertools.chain(
            self.enc_block.parameters(),
            self.dec_block.parameters(),
        )
        disc_parameters = self.disc_block.parameters()
        
        return {
            "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "disc": torch.optim.Adam(disc_parameters, lr=self.lr_disc, betas=(0.5, 0.9)),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.decode(z)
        logits_fake = self.disc_block(z)

        return z, mean, logvar, x2, logits_fake

    def loss(self, x, z, mean, logvar, x2, logits_fake):

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))
        loss_tc  = torch.mean(logits_fake[:,0] - logits_fake[:,1])

        # Total loss
        loss = loss_ae + loss_reg + loss_tc * self.gamma

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "loss_tc": loss_tc,
        })

    def loss_disc(self, logits_real: torch.Tensor, logits_fake: torch.Tensor):

        return -torch.mean(logits_real[:,1] + logits_fake[:,0])

    def step_batch(self, batch, optimizers = None, training: bool = False):

        x, t = batch

        x = x.to(self.device)
        
        if training:
            self.zero_grad()
        z, mean, logvar, x2, logits_fake = self(x)
        loss, loss_dict = self.loss(x, z, mean, logvar, x2, logits_fake)
        if training:
            loss.backward()
            optimizers["main"].step()

        if training:
            self.zero_grad()
        z_fake = z.detach()
        logits_fake = self.disc_block(z_fake)
        z_real = permute_dims(z_fake)
        logits_real = self.disc_block(z_real)
        loss_disc = self.loss_disc(logits_real, logits_fake)
        if training:
            loss_disc.backward()
            optimizers["disc"].step()

        loss_dict["loss_disc"] = float(loss_disc.detach().cpu().numpy())

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch, training=False)