from typing import Dict, Tuple, Optional

import torch
from torch import nn
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class VariationalLadderAutoEncoders(VAE):
    """ VLadderAE
    [Zhao+, 2017 (https://proceedings.mlr.press/v70/zhao17c.html)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        assert self.z_dim > 6
        self.ladder_dims = (self.z_dim // 4, self.z_dim // 4, self.z_dim // 4, self.z_dim - self.z_dim // 4 * 3)
        self.cnn_channels = (3, 32, 32, 64, 64)

        # network layers
        self.enc_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_features, out_features, 4, 2, 1),
                nn.LeakyReLU(0.2, True),
            )
            for in_features, out_features in zip(self.cnn_channels[:-1], self.cnn_channels[1:])
        ])
        self.enc_features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveMaxPool2d([4, 4]),
                nn.Flatten(),
                nn.Linear(in_features * 4 * 4, 64),
                nn.LeakyReLU(0.2, True),
            )
            for in_features in self.cnn_channels[1:]
        ])
        self.enc_means = nn.ModuleList([
            nn.Linear(64, dims)
            for dims in self.ladder_dims
        ])
        self.enc_logvars = nn.ModuleList([
            nn.Linear(64, dims)
            for dims in self.ladder_dims
        ])
        del self.enc_block

        self.dec_headers = nn.ModuleList([nn.Linear(self.ladder_dims[-1], self.cnn_channels[-1] * 8 * 8)] + [
            nn.Sequential(
                nn.Linear(dims, 256),
                nn.LeakyReLU(0.2, True),
                nn.Linear(256, 64 * 8 * 8),
                nn.LeakyReLU(0.2, True),
                nn.Unflatten(1, [64, 8, 8]),
                nn.AdaptiveAvgPool2d([8 * 2 ** i, 8 * 2 ** i]),
            )
            for i, dims in enumerate(self.ladder_dims[-2::-1])
        ])
        self.dec_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_features + (64 if i != 0 else 0), out_features, 4, 2, 1),
                nn.LeakyReLU(0.2, True) if i+1 < len(self.ladder_dims) else nn.Sigmoid(),
            )
            for i, in_features, out_features in zip(range(len(self.cnn_channels) - 1), self.cnn_channels[-1:0:-1], self.cnn_channels[-2::-1])
        ])
        del self.dec_block

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr)
        }

    def encode_gauss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        ml = []
        h = x
        for i in range(len(self.ladder_dims)):
            h = self.enc_blocks[i](h)
            g = self.enc_features[i](h)
            meani = self.enc_means[i](g)
            logvari = self.enc_logvars[i](g)
            ml += [(meani, logvari)]

        mean, logvar = list(zip(*ml))
        mean = torch.cat(mean, dim=1)
        logvar = torch.cat(logvar, dim=1)

        return mean, logvar
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        batch_size, z_dim = z.shape

        zs = []
        ib = 0
        for dims in self.ladder_dims:
            ie = ib + dims
            zs += [z[:,ib:ie]]
            ib = ie

        h = None
        for i, zi in enumerate(zs[::-1]):

            if h is None:
                h = self.dec_headers[i](zi)
                h = h.view(batch_size, 64, 8, 8)
            else:
                g = self.dec_headers[i](zi)
                h = torch.cat([h, g], dim=1)
                h = self.dec_blocks[i](h)

        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.decode(z)

        return z, mean, logvar, x2

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        # Total loss
        loss = loss_ae + loss_reg

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
        })

    def train_batch(self, batch, optimizers, progress: float):

        x, t = batch

        x = x.to(self.device)
        
        self.zero_grad()
        z, mean, logvar, x2 = self(x)
        loss, loss_dict = self.loss(x, z, mean, logvar, x2, progress)
        loss.backward()
        optimizers["main"].step()

        return loss_dict

    def eval_batch(self, batch):

        x, t = batch

        x = x.to(self.device)
        z, mean, logvar, x2 = self(x)
        loss, loss_dict = self.loss(x, z, mean, logvar, x2)

        return loss_dict