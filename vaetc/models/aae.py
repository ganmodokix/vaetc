from typing import Dict, Tuple, Optional
import itertools

import torch
from torch import nn

from .abstract import AutoEncoderRLModel
from vaetc.network.cnn import ConvEncoder, ConvDecoder
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

class Discriminator(nn.Module):
    """ Discriminator.
        Recieves :math:`\\mathbf{z}` as its input and output logits
        :math:`[\\log(p), \\log(1-p)]^T`, where :math:`p=Prob(\\mathbf{z} \\sim p(\\mathbf{z}))`
    """

    def __init__(self, z_dim: int):

        super().__init__()

        self.z_dim = int(z_dim)

        self.net = nn.Sequential(
            nn.Linear(self.z_dim, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 1000),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1000, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        logits: torch.Tensor = self.net(x)

        # logit softmax
        softmaxed_logit = logits - logits.logsumexp(dim=-1, keepdim=True)

        return softmaxed_logit

class AAE(AutoEncoderRLModel):
    """ Adversarial Autoencoders [Makhzani+, 2015 (https://arxiv.org/abs/1511.05644)]
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        self.z_dim = int(hyperparameters["z_dim"])
        assert self.z_dim > 0, "# of latent variables must be positive"
        self.coef_lambda = float(hyperparameters["lambda"])
        self.lr = float(hyperparameters["lr"])
        self.lr_disc = float(hyperparameters["lr_disc"])

        # network layers
        self.enc_block = ConvEncoder(z_dim=self.z_dim)
        self.dec_block = ConvDecoder(z_dim=self.z_dim)
        self.disc_block = Discriminator(z_dim=self.z_dim)

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        main_parameters = itertools.chain(
            self.enc_block.parameters(),
            self.dec_block.parameters())
        disc_parameters = self.disc_block.parameters()
        return {
            "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "disc": torch.optim.Adam(disc_parameters, lr=self.lr_disc, betas=(0.5, 0.9)),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_block(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_block(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.encode(x)
        x2 = self.decode(z)

        return z, x2

    def loss(self, x, z, x2, logits_fake):

        # losses
        loss_ae  = torch.mean(torch.sum((x - x2) ** 2, dim=[1, 2, 3]))
        loss_adv = -torch.mean(logits_fake[:,0] - logits_fake[:,1])

        # Total loss
        loss = loss_ae + loss_adv * self.coef_lambda

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_adv": loss_adv,
        })

    def loss_disc(self, logits_real: torch.Tensor, logits_fake: torch.Tensor):

        return -torch.mean(logits_real[:,0] + logits_fake[:,1])

    def step_batch(self, batch, optimizers = None, training: bool = False):

        x, t = batch

        x = x.to(self.device)
        
        self.zero_grad()
        z, x2 = self(x)
        logits_fake = self.disc_block(z)
        loss, loss_dict = self.loss(x, z, x2, logits_fake)
        if training:
            loss.backward()
            optimizers["main"].step()

        self.zero_grad()
        z_fake = z.detach()
        logits_fake = self.disc_block(z_fake)
        z_real = torch.randn_like(z_fake).to(self.device)
        logits_real = self.disc_block(z_real)
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