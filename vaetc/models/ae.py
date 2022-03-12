from typing import Dict, Tuple, Optional
import itertools

import torch
from torch import nn

from .abstract import AutoEncoderRLModel
from vaetc.network.cnn import ConvEncoder, ConvDecoder

from .utils import detach_dict

class AutoEncoder(AutoEncoderRLModel):
    """ (Convolutional) Autoencoder.
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        self.z_dim = int(hyperparameters["z_dim"])
        assert self.z_dim > 0, "# of latent variables must be positive"
        self.lr = float(hyperparameters["lr"])

        # network layers
        self.enc_block = ConvEncoder(z_dim=self.z_dim)
        self.dec_block = ConvDecoder(z_dim=self.z_dim)

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999)),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_block(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_block(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode(x)
        x2 = self.decode(z)

        return z, x2

    def loss(self, x, z, x2):

        loss = 0.5 * torch.mean(torch.sum((x - x2) ** 2, dim=[1, 2, 3]))

        return loss, detach_dict({
            "loss": loss,
        })

    def step_batch(self, batch, optimizers = None, training: bool = False):

        x, t = batch

        x = x.to(self.device)
        
        self.zero_grad()
        z, x2 = self(x)
        loss, loss_dict = self.loss(x, z, x2)
        if training:
            loss.backward()
            optimizers["main"].step()

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch, training=False)