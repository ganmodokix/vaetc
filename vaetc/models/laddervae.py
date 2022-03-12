from typing import Dict, Tuple, Optional

import torch
from torch import nn
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class LadderVAE(VAE):
    """ LadderVAE
    [Sønderby+, 2016 (https://arxiv.org/abs/1602.02282)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        assert self.z_dim > 6
        self.ladder_dims = (self.z_dim // 4, self.z_dim // 4, self.z_dim // 4, self.z_dim - self.z_dim // 4 * 3)
        self.cnn_channels = (3, 32, 32, 64, 64)

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
        z = reparameterize(mean, logvar)
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