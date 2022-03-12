from typing import Dict, Tuple, Optional
import torch
from torch import nn

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

from .abstract import GaussianEncoderAutoEncoderRLModel
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian

class CVAE(GaussianEncoderAutoEncoderRLModel):
    """ Conditional VAE
    [Kingma+, 2014 (https://proceedings.neurips.cc/paper/2014/hash/d523773c6b194f37b938d340d5d02232-Abstract.html)] """

    def __init__(self, hyperparameters: dict):

        super().__init__()

        # hyperparameters
        self.z_dim = int(hyperparameters["z_dim"])
        assert self.z_dim > 0, "# of the dimensions of z must be a positive integer"
        self.t_dim = int(hyperparameters["t_dim"])
        assert self.t_dim > 0, "# of the dimensions of t must be a positive integer"
        self.lr = float(hyperparameters["lr"])

        # network layers
        self.enc_block = ConvGaussianEncoder(z_dim=self.z_dim, in_features=3 + self.t_dim)
        self.dec_block = ConvDecoder(z_dim=self.z_dim + self.t_dim)

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr)
        }

    @property
    def inputs_include_targets(self):
        return True

    def encode_gauss(self, x: torch.Tensor, t: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if t is None:
            t = torch.zeros((x.shape[0], self.t_dim)).to(self.device)

        t_tiled = t[:,:,None,None].tile(1, 1, x.shape[2], x.shape[3])
        xt = torch.cat([x, t_tiled], dim=1)
        mean, logvar = self.enc_block(xt)

        return mean, logvar
    
    def decode(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:

        if t is None:
            t = torch.zeros((x.shape[0], self.t_dim)).to(self.device)
        
        h = torch.cat([x, t], dim=-1)
        h = self.dec_block(h)

        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x, t)
        z = reparameterize(mean, logvar)
        x2 = self.decode(z, t)

        return z, mean, logvar, x2

    def loss(self, x, t, z, mean, logvar, x2, progress: Optional[float] = None):

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
        t = t.to(self.device)
        
        self.zero_grad()
        z, mean, logvar, x2 = self(x, t)
        loss, loss_dict = self.loss(x, t, z, mean, logvar, x2, progress)
        loss.backward()
        optimizers["main"].step()

        return loss_dict

    def eval_batch(self, batch):

        x, t = batch

        x = x.to(self.device)
        t = t.to(self.device)

        z, mean, logvar, x2 = self(x, t)
        loss, loss_dict = self.loss(x, t, z, mean, logvar, x2)

        return loss_dict