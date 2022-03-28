from typing import Dict, List, Tuple, Optional
import torch
from torch import nn

from vaetc.network.reparam import reparameterize

REPARAMETERIZE_GAUSS_IN_ENCODE = True
class RLModel(nn.Module):
    """ The abstract class of representation learning models with at least an encoder.

    Note:
        This class does not require the :meth:`decode()` method like :class:`AutoEncoderRLModel`.

    """

    def __init__(self, hyperparameters: dict = {}):
        """ 
        Args:
            hyperparameters (dict): Hyperparameters of the model.
        """
        super().__init__()
        
    @property
    def device(self):
        """ The device of the parameters. """
        return "cuda" if next(self.parameters()).is_cuda else "cpu"

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """ Build optimizers for training.

        Returns:
            dict[str, torch.optim.Optimizer]
        """
        raise NotImplementedError()

    @property
    def inputs_include_targets(self):
        """ bool: Whether the model is supervised."""
        return False

    def encode(self, *args: torch.Tensor) -> torch.Tensor:
        """ Encode a latent representation :math:`\\mathbf{z}` from an observed (given image) :math:`\\mathbf{x}`.
        
        Args:
            *args (torch.Tensor): `x` if unsupervised, or `(x, t)` if supervised

        Note:
            :attr:`inputs_include_targets` decides whether the model is supervised

        Returns:
            torch.Tensor: The encoded latent code `z`
        """
        raise NotImplementedError()

    def forward(self, *args):
        """ Run to build the entire model """
        raise NotImplementedError()

    def train_batch(self, batch, optimizers, progress: float) -> dict[str, list[float]]:
        """ Train with a batch (x, t)

        Args:
            batch (tuple[torch.Tensor]): (y, x) if suprvised, (x, ) otherwise.

        Returns:
            dict[str, list[float]]: the loss values
        """
        raise NotImplementedError()

    def eval_batch(self, batch) -> dict[str, list[float]]:
        """ Evaluate losses without training

        Args:
            batch (tuple[torch.Tensor]): (y, x) if suprvised, (x, ) otherwise.

        Returns:
            dict[str, list[float]]: the loss values

        """
        raise NotImplementedError()

class AutoEncoderRLModel(RLModel):
    """ The abstract class of represnetation learning models
    with a decoder in addition to an encoder. """

    def decode(self, *args) -> torch.Tensor:
        """ Decode latent codes :math:`\\mathbf{z}` into reconstruction :math:`\\mathbf{\\hat{x}}`
        
        Args:
            *args (torch.Tensor): mainly :math:`\\mathbf{z}`
        
        Returns:
            torch.Tensor: reconstruction :math:`\\mathbf{\\hat{x}}`
        """

        raise NotImplementedError()

class GaussianEncoderAutoEncoderRLModel(AutoEncoderRLModel):
    """ The abstract class of autoencoding-based representation
    learning models whose encoder outputs a Gaussian distribution
    :math:`q(\\mathbf{z}|\\mathbf{x}) = \\mathcal{N}(\\mathbf{z}|\\boldsymbol{\mu}(\\mathbf{x}), \\boldsymbol{\sigma(\\mathbf{x})^2I})`.
    
    Note:
        In concrete classes, :meth:`encode_gauss` should be implemented, not :meth:`encode`.
    """

    def encode_gauss(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Outputs the mean and the variance logit of :math:`q(\\mathbf{z}|\\mathbf{x})`.

        Args:
            *args (torch.Tensor): (y, x) if suprvised, (x, ) otherwise.

        Returns:
            (torch.Tensor, torch.Tensor): (mean :math:`\\boldsymbol{\mu}`, logvar :math:`\\boldsymbol{\sigma}`)
        """

        raise NotImplementedError()

    def reparameterize(self, mean, logvar):
        """ Apply the reparameterization trick

        Args:
            mean (torch.Tensor): μ in the encoder output
            mean (torch.Tensor): logσ^2 in the encoder output

        Returns:
            torch.Tensor: z
        """
        
        if REPARAMETERIZE_GAUSS_IN_ENCODE:
            z = reparameterize(mean, logvar)
        else:
            z = mean

        return z

    def encode(self, *args: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Encode and sample :math:`\\mathbf{z}` by the reparameterization trick (https://arxiv.org/abs/1312.6114).

        Args:
            *args (torch.Tensor): (y, x) if suprvised, (x, ) otherwise.

        Note:
            This method should not be overridden; override :meth:`encode_gauss` instead.

        Returns:
            (torch.Tensor, torch.Tensor): :math:`\\mathbf{z}`.
        """

        mean, logvar = self.encode_gauss(*args)
        z = self.reparameterize(mean, logvar)

        return z