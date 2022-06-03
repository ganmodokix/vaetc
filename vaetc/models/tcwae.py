import itertools
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from vaetc.models.utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian

from .vae import VAE
from .btcvae import total_correlation
from .infovae import mmd
from .factorvae import Discriminator, permute_dims

class TCWAE(VAE):

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.estimator_type = str(hyperparameters["estimator_type"])
        if self.estimator_type not in ["mws", "gan"]:
            raise ValueError(f"Estimator '{self.estimator_type}' not supported")

        if self.estimator_type == "gan":
            self.lr_disc = float(hyperparameters["lr_disc"])
            self.disc_block = Discriminator(self.z_dim)

        self.beta = float(hyperparameters["beta"])
        self.gamma = float(hyperparameters["gamma"])

    def forward(self, x):

        mean, logvar = self.enc_block(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.dec_block(z)

        if self.estimator_type == "gan":
            logit = self.disc_block(z)
            return z, mean, logvar, x2, logit

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        main_parameters = itertools.chain(
            self.enc_block.parameters(),
            self.dec_block.parameters(),
        )

        if self.estimator_type == "mws":
            
            return {
                "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.9, 0.999)),
            }

        elif self.estimator_type == "gan":

            disc_parameters = self.disc_block.parameters()
            
            return {
                "main": torch.optim.Adam(main_parameters, lr=self.lr, betas=(0.9, 0.999)),
                "disc": torch.optim.Adam(disc_parameters, lr=self.lr_disc, betas=(0.5, 0.9)),
            }

    def step_batch(self, batch, optimizers=None, progress=None, training=False):
        
        x, t = batch
        x = x.cuda()
        batch_size = x.shape[0]

        # MAIN STEP
        
        mean, logvar = self.enc_block(x)
        z = self.reparameterize(mean, logvar)
        x2 = self.dec_block(z)

        loss_ae = neglogpxz_gaussian(x, x2).mean()
        loss_w = mmd(z, self.sample_prior(batch_size))

        if self.estimator_type == "mws":

            loss_tc = total_correlation(z, mean, logvar).mean()

        elif self.estimator_type == "gan":

            logits_fake = self.disc_block(z)
            loss_tc = (logits_fake[:,0] - logits_fake[:,1]).mean()

        loss = loss_ae + loss_w * self.beta + loss_tc * self.gamma

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()

        # ADVERSARIAL STEP (in the TCWAE-GAN case)
        if self.estimator_type == "gan":

            z = z.detach()
            zp = permute_dims(z).detach()

            logits_fake = self.disc_block(z)
            logits_real = self.disc_block(zp)
            loss_disc = -(logits_real[:,1] + logits_fake[:,0]).mean()

            if training:
                self.zero_grad()
                loss_disc.backward()
                optimizers["disc"].step()

        # RETURNS
        if self.estimator_type == "mws":
            
            return detach_dict({
                "loss": loss,
                "loss_ae": loss_ae,
                "loss_w": loss_w,
                "loss_tc": loss_tc,
            })

        elif self.estimator_type == "gan":

            return detach_dict({
                "loss": loss,
                "loss_ae": loss_ae,
                "loss_w": loss_w,
                "loss_tc": loss_tc,
                "loss_disc": loss_disc
            })

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)