import math
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.stats import gamma
from scipy.special import gammaln

import torch
from torch import nn

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.network.losses import kl_gaussian, kl_gn_gaussian, kl_gn_gn, kl_uniform_gaussian, kl_uniform_gn
from vaetc.network.losses import neglogpxz_bernoulli, neglogpxz_continuous_bernoulli, neglogpxz_gaussian, neglogpxz_von_mises_fisher
from vaetc.network.losses import neglogpxz_gn
from vaetc.network.imagequality import mse, cossim, ssim
from vaetc.network.gn import randgn

from .utils import detach_dict

from .abstract import GaussianEncoderAutoEncoderRLModel

def neglogpxz(x: torch.Tensor, x2: torch.Tensor, distribution: str, *args, **kwargs):
    """ -log p(x|z) parameterized by x2(z) """

    loggamma = kwargs.get("loggamma", 1.0)
    if not torch.is_tensor(loggamma):
        loggamma = torch.tensor(loggamma, device=x.device)

    if distribution == "gaussian":
        return neglogpxz_gaussian(x, x2, loggamma=loggamma)
    elif distribution == "bernoulli":
        return neglogpxz_bernoulli(x, x2)
    elif distribution == "von_mises_fisher":
        return neglogpxz_von_mises_fisher(x, x2)
    elif distribution == "continuous_bernoulli":
        return neglogpxz_continuous_bernoulli(x, x2)
    elif distribution == "mse-cossim":
        num_pixels = x.nelement() / x.shape[0]
        negsim = mse(x, x2) - cossim(x, x2)
        return (negsim / loggamma.exp() + loggamma + math.log(2 * math.pi)) * num_pixels
    elif distribution == "mse-ssim":
        num_pixels = x.nelement() / x.shape[0]
        negsim = mse(x, x2) - ssim(x, x2)
        return (negsim / loggamma.exp() + loggamma + math.log(2 * math.pi)) * num_pixels
    elif distribution == "-cossim-ssim":
        num_pixels = x.nelement() / x.shape[0]
        negsim = - cossim(x, x2) - ssim(x, x2)
        return (negsim / loggamma.exp() + loggamma + math.log(2 * math.pi)) * num_pixels
    elif distribution == "-cossim":
        num_pixels = x.nelement() / x.shape[0]
        negsim = - cossim(x, x2)
        return (negsim / loggamma.exp() + loggamma + math.log(2 * math.pi)) * num_pixels
    elif distribution == "-ssim":
        num_pixels = x.nelement() / x.shape[0]
        negsim = - ssim(x, x2)
        return (negsim / loggamma.exp() + loggamma + math.log(2 * math.pi)) * num_pixels
    elif distribution == "mse-cossim-ssim":
        num_pixels = x.nelement() / x.shape[0]
        negsim = mse(x, x2) - cossim(x, x2) - ssim(x, x2)
        return (negsim / loggamma.exp() + loggamma + math.log(2 * math.pi)) * num_pixels
    elif distribution == "laplace":
        return neglogpxz_gn(x, x2, 0)
    elif distribution == "generalized_gaussian":
        if "logbeta" in kwargs:
            logbeta = kwargs["logbeta"]
        elif "beta" in kwargs:
            logbeta = kwargs["beta"]
            if torch.is_tensor(logbeta):
                logbeta = logbeta.log()
            else:
                logbeta = np.log(logbeta)
        return neglogpxz_gn(x, x2, logbeta=logbeta)
    else:
        raise NotImplementedError(f"Decoder distribution '{distribution}' not implemented")

def sample_reparam_noise(mean: torch.Tensor, logvar: torch.Tensor, distribution: str, *args, **kwargs):
    """ sample ε with mean 0 and cov I """

    if distribution == "laplace":
        distribution = "generalized_gaussian"
        kwargs["beta"] = 1

    if distribution == "gaussian":

        return torch.randn_like(mean)

    elif distribution == "generalized_gaussian":

        if "logbeta" in kwargs:
            logbeta = kwargs["logbeta"]
        elif "beta" in kwargs:
            logbeta = np.log(kwargs["beta"])
        else:
            raise RuntimeError("Generalized Gaussian specified but no beta/logbeta is given")
        
        return randgn(logbeta, size=mean.shape, device=mean.device, dtype=mean.dtype)

    elif distribution == "uniform":

        return (torch.rand_like(mean) * 2 - 1) * 3 ** 0.5

    else:

        raise NotImplementedError(f"Encoder distribution {distribution} not implemented")

def kl_encoder_prior(mean: torch.Tensor, logvar: torch.Tensor, encoder_distribution: str, prior_distribution: str, *args, **kwargs):
    """ KL(enc||prior), assuming that the encoder is factorized conditioned on x and the marginal prior is factorized and has mean 0 and cov I """

    device = mean.device
    dtype = float

    if prior_distribution == "gaussian":

        if encoder_distribution == "gaussian":

            return kl_gaussian(mean, logvar)

        elif encoder_distribution == "generalized_gaussian":

            if "encoder_logbeta" in kwargs:
                encoder_logbeta = torch.tensor(kwargs["encoder_logbeta"], dtype=dtype, device=device)
            elif "encoder_beta" in kwargs:
                encoder_logbeta = torch.tensor(kwargs["encoder_beta"], dtype=dtype, device=device).log()
            else:
                raise RuntimeError("neither encoder_logbeta nor encoder_beta")
            
            return kl_gn_gaussian(mean, logvar, encoder_logbeta)

        elif encoder_distribution == "uniform":

            return kl_uniform_gaussian(mean, logvar)

    elif prior_distribution == "generalized_gaussian":

        if "prior_logbeta" in kwargs:
            prior_logbeta = torch.tensor(kwargs["prior_logbeta"], dtype=dtype, device=device)
        elif "prior_beta" in kwargs:
            prior_logbeta = torch.tensor(kwargs["prior_beta"], dtype=dtype, device=device).log()
        else:
            raise RuntimeError("neither prior_logbeta nor prior_beta")

        if encoder_distribution == "gaussian":
            encoder_distribution = "generalized_gaussian"
            kwargs["encoder_beta"] = 2
        elif encoder_distribution == "laplace":
            encoder_distribution = "generalized_gaussian"
            kwargs["encoder_beta"] = 1

        if encoder_distribution == "generalized_gaussian":

            if "encoder_logbeta" in kwargs:
                encoder_logbeta = torch.tensor(kwargs["encoder_logbeta"], dtype=float, device=mean.device)
            elif "encoder_beta" in kwargs:
                encoder_logbeta = torch.tensor(kwargs["encoder_beta"], dtype=float, device=mean.device).log()

            return kl_gn_gn(mean, logvar, encoder_logbeta, prior_logbeta)

        elif encoder_distribution == "uniform":

            return kl_uniform_gn(mean, logvar, prior_logbeta)

    raise NotImplementedError(f"encoder '{encoder_distribution}' vs. prior '{prior_distribution}' not implemented")

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
        
        self.encoder_distribution = str(hyperparameters.get("encoder_distribution", "gaussian"))
        self.decoder_distribution = str(hyperparameters.get("decoder_distribution", "gaussian"))
        self.prior_distribution = str(hyperparameters.get("prior_distribution", "gaussian"))
        
        self.kl_shape_parameters = {}
        for key in ["encoder_logbeta", "encoder_beta", "prior_logbeta", "prior_beta"]:
            if key in hyperparameters:
                self.kl_shape_parameters[key] = hyperparameters[key]
        
        self.decoder_shape_parameters = {}
        if "decoder_logbeta" in hyperparameters:
            self.decoder_shape_parameters["logbeta"] = hyperparameters["decoder_logbeta"]
        if "decoder_beta" in hyperparameters:
            self.decoder_shape_parameters["beta"] = hyperparameters["decoder_beta"]

        self.decoder_variance = hyperparameters.get("decoder_includes_variance", 1.0)
        if self.decoder_variance == "trainiable":
            self.decoder_loggamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

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

    def reconstruction_term(self, x, x2):

        decoder_parameters = {
            **self.decoder_shape_parameters
        }

        if self.decoder_variance == "trainable":
            decoder_parameters["loggamma"] = self.decoder_loggamma
        else:
            decoder_parameters["loggamma"] = math.log(self.decoder_variance)
        
        return neglogpxz(x, x2, distribution=self.decoder_distribution, **decoder_parameters)

    def regularization_term(self, mean, logvar):

        return kl_encoder_prior(
            mean, logvar,
            encoder_distribution=self.encoder_distribution,
            prior_distribution=self.prior_distribution,
            **self.kl_shape_parameters
        )

    def reparameterize(self, mean, logvar):

        if self.encoder_distribution == "gaussian":
            
            eps = torch.randn_like(mean)
            return mean + (logvar * 0.5).exp() * eps

        elif self.encoder_distribution == "uniform":
            
            eps = (torch.rand_like(mean) * 2 - 1) * 3 ** 0.5
            return mean + (logvar * 0.5).exp() * eps

        elif self.encoder_distribution in ["laplace", "generalized_gaussian"]:
            
            if self.encoder_distribution == "laplace":
                logbeta = 0
            else:
                logbeta = self.kl_shape_parameters["encoder_logbeta"] if "encoder_logbeta" in self.kl_shape_parameters else self.kl_shape_parameters["encoder_beta"]

            eps = randgn(logbeta, dtype=mean.dtype, size=mean.shape, device=mean.device)
            return mean + (logvar * 0.5).exp() * eps

        raise NotImplementedError()

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):
        """
        the arguments between x and progress must be *self.forward(x)
        """

        # losses
        loss_ae  = torch.mean(self.reconstruction_term(x, x2))
        loss_reg = torch.mean(self.regularization_term(mean, logvar))

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