import math
from typing import Dict, Tuple, Optional
import torch
from torch import nn

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import kl_gaussian, neglogpxz_bernoulli, neglogpxz_continuous_bernoulli, neglogpxz_gaussian, neglogpxz_von_mises_fisher
from vaetc.network.imagequality import mse, cossim, ssim

from .utils import detach_dict

from .abstract import GaussianEncoderAutoEncoderRLModel

def neglogpxz(x: torch.Tensor, x2: torch.Tensor, distribution: str, *args, **kwargs):

    if distribution == "gaussian":
        return neglogpxz_gaussian(x, x2, **kwargs)
    elif distribution == "bernoulli":
        return neglogpxz_bernoulli(x, x2)
    elif distribution == "von_mises_fisher":
        return neglogpxz_von_mises_fisher(x, x2)
    elif distribution == "continuous_bernoulli":
        return neglogpxz_continuous_bernoulli(x, x2)
    elif distribution == "compound":
        return (mse(x, x2) - cossim(x, x2) - ssim(x, x2)) * x.nelement() / x.shape[0]

class VAE(GaussianEncoderAutoEncoderRLModel):
    """ Variational Autoencoder
    [Kingma and Welling, 2013 (https://openreview.net/forum?id=33X9fd2-9FyZd)]
    """

    def __init__(self, hyperparameters: dict):

        super().__init__()

        # hyperparameters
        self.z_dim = int(hyperparameters["z_dim"])
        assert self.z_dim > 0, "# of the dimensions of z must be a positive integer"
        self.lr = float(hyperparameters["lr"])
        self.decoder_distribution = str(hyperparameters.get("decoder_distribution", "gaussian"))

        # network layers
        self.enc_block = ConvGaussianEncoder(z_dim=self.z_dim)
        self.dec_block = ConvDecoder(z_dim=self.z_dim)

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr)
        }

    def encode_gauss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.enc_block(x)
    
    def decode(self, x: torch.Tensor) -> torch.Tensor:

        return self.dec_block(x)

    def sample_prior(self, batch_size: int) -> torch.Tensor:
        
        z = torch.randn(size=[batch_size, self.z_dim], device=self.device)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        mean, logvar = self.encode_gauss(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.decode(z)

        return z, mean, logvar, x2

    def reconstruction_term(self, x, x2, **kwargs):
        
        return neglogpxz(x, x2, distribution=self.decoder_distribution)

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):
        """
        the arguments between x and progress must be *self.forward(x)
        """

        # losses
        loss_ae  = torch.mean(self.reconstruction_term(x, x2))
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
        args = self(x)
        loss, loss_dict = self.loss(x, *args, progress)
        loss.backward()
        optimizers["main"].step()

        return loss_dict

    def eval_batch(self, batch):

        x, t = batch

        x = x.to(self.device)
        args = self(x)
        loss, loss_dict = self.loss(x, *args)

        return loss_dict