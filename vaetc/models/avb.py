from turtle import forward
from typing import Dict, Tuple, Optional
import itertools

import torch
from torch import nn

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.network.reparam import reparameterize
from vaetc.data.utils import IMAGE_SHAPE

from .utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class Discriminator(nn.Module):
    """ Discriminator for AVB. """

    def __init__(self, x_shape: tuple[int, int, int], z_dim: int):

        super().__init__()

        self.x_shape = tuple(map(int, x_shape))
        self.z_dim = int(z_dim)

        padding_mode = "zeros" if torch.are_deterministic_algorithms_enabled() else "replicate"
        self.features = nn.Sequential(
            nn.Conv2d(self.x_shape[0], 64, 4, 2, 1, padding_mode=padding_mode),
            nn.Dropout2d(0.2, True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, padding_mode=padding_mode),
            nn.Dropout2d(0.5, True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, padding_mode=padding_mode),
            nn.Dropout2d(0.5, True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, padding_mode=padding_mode),
            nn.Dropout2d(0.5, True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, padding_mode=padding_mode),
            nn.Dropout2d(0.5, True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1, padding_mode=padding_mode),
            nn.ReLU(True),
            nn.Flatten()
        )

        self.logits = nn.Sequential(
            nn.Linear(self.z_dim + 64, 256),
            nn.Dropout(0.2, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 256),
            nn.Dropout(0.2, True),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        
        # [1-D(x), D(x)] = [p_fake, p_real]
        return self.logits(torch.cat([self.features(x), z], dim=1))

class Reparameterizer(nn.Module):

    def __init__(self, z_dim) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.SiLU(True),
            nn.Linear(256, 256),
            nn.SiLU(True),
            nn.Linear(256, z_dim),
            nn.SiLU(True),
            nn.BatchNorm1d(z_dim, affine=False),
        )

    def forward(self, eps: torch.Tensor):
        return self.net(eps)

class AdversarialVariationalBayes(VAE):
    """ Adversarial Variational Bayes
    [Mescheder+, 2017 (https://dl.acm.org/doi/10.5555/3305890.3305928)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        self.lr_disc = float(hyperparameters["lr_disc"])
        self.beta = float(hyperparameters["beta"])

        # network layers
        self.enc_block = ConvGaussianEncoder(z_dim=self.z_dim)
        self.dec_block = ConvDecoder(z_dim=self.z_dim)
        self.disc_block = Discriminator(x_shape=IMAGE_SHAPE, z_dim=self.z_dim)
        self.noise_block = Reparameterizer(z_dim=self.z_dim)

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
        logits_fake = self.disc_block(x, z)

        return z, mean, logvar, x2, logits_fake

    def reparameter(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:

        eps = torch.randn_like(mean)
        eps = self.noise_block(eps)
        return mean + eps * (logvar * 0.5).exp()

    def encode(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode_gauss(x)
        return self.reparameter(mean, logvar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameter(mean, logvar)
        x2 = self.decode(z)

        logits_fake = self.disc_block(x, z)

        return z, mean, logvar, x2, logits_fake

    def loss(self, x, z, mean, logvar, x2, logits_fake):

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(logits_fake[:,0] - logits_fake[:,1])

        # Total loss
        loss = loss_ae + loss_reg * self.beta

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
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
        z_real = torch.randn_like(z_fake)
        logits_fake = self.disc_block(x, z_fake)
        logits_real = self.disc_block(x, z_real)
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