import itertools
from functools import reduce
from typing import Optional

import torch
from torch import convolution, nn
from torch.nn import functional as F

from .vae import VAE
from vaetc.data.utils import IMAGE_SHAPE
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import kl_gaussian, neglogpxz_gaussian
from vaetc.network.cnn import ConvDecoder, ConvGaussianEncoder

from .utils import detach_dict

class Discriminator(nn.Module):
    """ estimates p_real(x,z) / [p_real(x,z) + p_fake(x,z)]
    """
    
    def __init__(self, x_shape: tuple[int, int, int], z_dim: int) -> None:
        super().__init__()

        self.x_shape = tuple(map(int, x_shape))
        self.z_dim = int(z_dim)

        self.x_channels, self.x_height, self.x_width = self.x_shape

        self.disc_x = nn.Sequential(
            nn.Conv2d(self.x_channels, 16, 4, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(16),
            nn.Dropout2d(0.2),

            nn.Conv2d( 16,  32, 4, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.5),
            
            nn.Conv2d( 32,  64, 4, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.5),
            
            nn.Conv2d( 64, 128, 4, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.5),
            
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.SiLU(),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.SiLU(),
            
            nn.Flatten(),
        )

        self.disc_z = nn.Sequential(
            nn.Linear(self.z_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Dropout(0.5),
        )

        self.disc_entire = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ outputs hidden, [1-p(real), p(real)] """

        hx = self.disc_x(x)
        hz = self.disc_z(z)

        h = torch.cat([hx, hz], dim=1)
        return self.disc_entire(h)

class ALI(VAE):
    """ ALI [Dumoulin+, 2017] / BiGAN []
    "the authors independently examine the same model" [Dumoulin+, 2017]

    [Dumoulin+, 2017]
    (https://openreview.net/forum?id=B1ElR4cgg)
    (https://ishmaelbelghazi.github.io/ALI/)

    [Donahue+, 2017]
    (https://openreview.net/forum?id=BJtNZAFgg)
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # refrain from replacing
        # self.enc_block = ConvGaussianEncoder(self.z_dim, inplace=False)
        # self.dec_block = ConvDecoder(self.z_dim, inplace=False)
        self.disc_block = Discriminator(IMAGE_SHAPE, self.z_dim)

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        main_parameters = itertools.chain(self.enc_block.parameters(), self.dec_block.parameters())
        disc_parameters = self.disc_block.parameters()
        
        return {
            "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.5, 0.999)),
            "disc": torch.optim.Adam(disc_parameters, lr=self.lr, betas=(0.5, 0.999)),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = reparameterize(mean, logvar)

        zp = torch.randn_like(z)
        xpmean = self.decode(zp)
        # xp = xpmean + torch.randn_like(x)
        xp = xpmean # p(x|z) is degenerate because of stability
        
        logit_q = self.disc_block(x, z)
        logit_p = self.disc_block(xp, zp)

        return z, mean, logvar, logit_q, zp, xpmean, xp, logit_p

    def loss_disc(self, x, z, mean, logvar, logit_q, zp, xpmean, xp, logit_p, progress: Optional[float] = None):

        loss_disc = -torch.mean(logit_q[:,1] + logit_p[:,0])

        return loss_disc, detach_dict({
            "loss_disc": loss_disc,
        })


    def loss_gen(self, x, z, mean, logvar, logit_q, zp, xpmean, xp, logit_p, progress: Optional[float] = None):

        loss_gen = -torch.mean(logit_q[:,0] + logit_p[:,1])

        return loss_gen, detach_dict({
            "loss_gen": loss_gen,
        })

    def step_batch(self, batch, optimizers = None, training: bool = False, progress = None):

        x, t = batch

        x = x.to(self.device)

        args = self(x)
        z, mean, logvar, logit_q, zp, xpmean, xp, logit_p = args
        loss_disc, loss_disc_dict = self.loss_disc(x, *args)

        # does_train_D = loss_gen.item() < 3.5

        if training:
            self.zero_grad()
            loss_disc.backward()
            optimizers["disc"].step()
        
        args = self(x)
        z, mean, logvar, logit_q, zp, xpmean, xp, logit_p = args
        loss_gen, loss_gen_dict = self.loss_gen(x, *args)

        if training:
            self.zero_grad()
            loss_gen.backward()
            optimizers["main"].step()

        loss_dict = {
            "loss": loss_disc_dict["loss_disc"] + loss_gen_dict["loss_gen"],
            **loss_disc_dict,
            **loss_gen_dict
        }

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch, training=False)