import itertools
from functools import reduce
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .vae import VAE
from vaetc.data.utils import IMAGE_SHAPE
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import kl_gaussian, neglogpxz_gaussian
from vaetc.network.cnn import ConvDecoder, ConvGaussianEncoder

from .utils import detach_dict

class Discriminator(nn.Module):
    
    def __init__(self, in_shape: tuple[int, int, int]) -> None:
        super().__init__()

        self.in_shape = tuple(map(int, in_shape))
        self.in_channels, self.in_height, self.in_width = self.in_shape

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.in_channels, 32, 4, 2, 1),
                nn.SiLU(False),
                nn.BatchNorm2d(32),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.SiLU(False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.SiLU(False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 4, 2, 1),
                nn.SiLU(False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Flatten(),
                nn.Linear((self.in_height // 16) * (self.in_height // 16) * 64, 256),
                nn.SiLU(False),
                nn.Linear(256, 2),
                nn.LogSoftmax(dim=1),
            ),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ outputs hidden, [1-p(real), p(real)] """

        h = [x]
        for layer in self.layers:
            h += [layer(h[-1])]

        return h[1:-1], h[-1]

class VAEGAN(VAE):
    """ VAE-GAN [Larsen+, 2015]
    (https://arxiv.org/abs/1512.09300)
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.lr_disc = float(hyperparameters["lr_disc"])
        self.beta = float(hyperparameters["beta"])
        self.gamma = float(hyperparameters["gamma"])

        self.disc_block = Discriminator(IMAGE_SHAPE)

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        enc_parameters = self.enc_block.parameters()
        dec_parameters = self.dec_block.parameters()
        disc_parameters = self.disc_block.parameters()
        
        return {
            "enc": torch.optim.Adam(enc_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "dec": torch.optim.Adam(dec_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "disc": torch.optim.Adam(disc_parameters, lr=self.lr_disc, betas=(0.5, 0.9)),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.decode(z)

        zp = torch.randn_like(z)
        xp = self.decode(zp)

        h, logit = self.disc_block(x)
        h2, logit2 = self.disc_block(x2)
        hp, logitp = self.disc_block(xp)

        return z, mean, logvar, x2, h, h2, hp, logit, logit2, logitp

    def loss(self, x, z, mean, logvar, x2, h, h2, hp, logit, logit2, logitp, progress: Optional[float] = None):

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        loss_ae_disc = []
        for hi, h2i in zip(h, h2):
            batch_size = hi.shape[0]
            loss_ae_disc_i = neglogpxz_gaussian(hi, h2i)
            loss_ae_disc += [loss_ae_disc_i]
        loss_ae_disc = torch.stack(loss_ae_disc, dim=1).mean(dim=1)
        loss_ae_disc = torch.mean(loss_ae_disc)

        loss_gan = -torch.mean(logit[:,1] + logit2[:,0] + logitp[:,0])

        # Total loss to validate
        loss = loss_ae + loss_ae_disc + loss_reg + loss_gan

        loss_enc = loss_ae + loss_ae_disc + loss_reg * self.beta
        loss_dec = self.gamma * (loss_ae + loss_ae_disc) - loss_gan
        loss_disc = loss_gan

        return loss, loss_enc, loss_dec, loss_disc, detach_dict({
            "loss": loss,
            "loss_enc": loss_enc,
            "loss_dec": loss_dec,
            "loss_disc": loss_disc,
            "loss_ae": loss_ae,
            "loss_ae_disc": loss_ae_disc,
            "loss_reg": loss_reg,
            "loss_gan": loss_gan,
        })

    def step_batch(self, batch, optimizers = None, training: bool = False, progress = None):

        x, t = batch

        x = x.to(self.device)

        args = self(x)
        loss, loss_enc, loss_dec, loss_disc, loss_dict = self.loss(x, *args, progress)

        if training:
            
            # grad_enc
            optimizers["enc"].zero_grad()
            loss_enc.backward(retain_graph=True)
            
            # grad_dec
            optimizers["dec"].zero_grad()
            loss_dec.backward(retain_graph=True)
            
            # grad_disc
            optimizers["disc"].zero_grad()
            loss_disc.backward()
            
            optimizers["dec"].step()
            optimizers["enc"].step()
            optimizers["disc"].step()

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch, training=False)