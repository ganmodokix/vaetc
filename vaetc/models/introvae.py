import itertools
from functools import reduce
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from .vae import VAE
from vaetc.data.utils import IMAGE_SHAPE
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import kl_gaussian, neglogpxz_gaussian, sgvb_gaussian
from vaetc.network.cnn import ConvDecoder, ConvGaussianEncoder

from .utils import detach_dict

class IntroVAE(VAE):
    """ IntroVAE [Huang+, 2018]
    (https://proceedings.neurips.cc/paper/2018/hash/093f65e080a295f8076b1c5722a46aa2-Abstract.html)
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.alpha = float(hyperparameters["alpha"])
        self.beta = float(hyperparameters["beta"])
        self.margin = float(hyperparameters["margin"])

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        enc_parameters = self.enc_block.parameters()
        dec_parameters = self.dec_block.parameters()
        
        return {
            "enc": torch.optim.Adam(enc_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "dec": torch.optim.Adam(dec_parameters, lr=self.lr, betas=(0.9, 0.999)),
        }

    def loss_enc(self, x, z, mean, logvar, x2):

        loss_ae = torch.mean(self.reconstruction_term(x, x2))

        zp = torch.randn_like(z)
        xp = self.decode(zp)
        
        mean2, logvar2 = self.encode_gauss(x2.detach())
        meanp, logvarp = self.encode_gauss(xp.detach())

        loss_reg = torch.mean(kl_gaussian(mean, logvar))
        loss_reg2 = torch.mean(F.relu(self.margin - kl_gaussian(mean2, logvar2)))
        loss_regp = torch.mean(F.relu(self.margin - kl_gaussian(meanp, logvarp)))
        loss_adv = loss_reg + (loss_reg2 + loss_regp) * self.alpha

        loss_enc = loss_adv + self.beta * loss_ae

        return loss_enc, detach_dict({
            "loss_enc": loss_enc,
            "loss_enc_ae": loss_ae,
            "loss_enc_reg": loss_reg,
            "loss_enc_reg2": loss_reg2,
            "loss_enc_regp": loss_regp,
        })

    def loss_dec(self, x, z, mean, logvar, x2):

        loss_ae = torch.mean(self.reconstruction_term(x, x2))

        zp = torch.randn_like(z)
        xp = self.decode(zp)
        
        mean2, logvar2 = self.encode_gauss(x2)
        meanp, logvarp = self.encode_gauss(xp)

        loss_reg2 = torch.mean(F.relu(self.margin - kl_gaussian(mean2, logvar2)))
        loss_regp = torch.mean(F.relu(self.margin - kl_gaussian(meanp, logvarp)))
        loss_adv = self.alpha * (loss_reg2 + loss_regp)

        loss_dec = loss_adv + self.beta * loss_ae

        return loss_dec, detach_dict({
            "loss_dec": loss_dec,
            "loss_dec_ae": loss_ae,
            "loss_dec_reg2": loss_reg2,
            "loss_dec_regp": loss_regp,
            "loss_dec_adv": loss_adv,
        })


    def step_batch(self, batch, optimizers = None, training: bool = False, progress = None):

        x, t = batch

        x = x.to(self.device)

        # enc (E) training
        if training:
            self.zero_grad()
        args = self(x)
        loss, loss_enc_dict = self.loss_enc(x, *args)
        if training:
            loss.backward()
            optimizers["enc"].step()

        # dec (G) training
        if training:
            self.zero_grad()
        args = self(x)
        loss, loss_dec_dict = self.loss_dec(x, *args)
        if training:
            loss.backward()
            optimizers["dec"].step()

        # SGVB for eval
        z, mean, logvar, x2 = args
        loss = torch.mean(sgvb_gaussian(x, x2, mean, logvar))
        loss_dict = {
            "loss": float(loss.detach().item()),
            **loss_enc_dict,
            **loss_dec_dict,
        }

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch, training=False)