import math

import torch
from torch import nn
from torch.nn import functional as F

from .vae import VAE
from vaetc.data.utils import IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_SHAPE, IMAGE_WIDTH
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import kl_gaussian, neglogpxz_gaussian, sgvb_gaussian
from vaetc.network.cnn import ConvDecoder, ConvGaussianEncoder

from .utils import detach_dict

class SoftIntroVAE(VAE):
    """ Soft-IntroVAE [Daniel+, 2021]
    (https://openaccess.thecvf.com/content/CVPR2021/html/Daniel_Soft-IntroVAE_Analyzing_and_Improving_the_Introspective_Variational_Autoencoder_CVPR_2021_paper.html)
    """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.beta_rec = float(hyperparameters["beta_rec"])
        self.beta_kl = float(hyperparameters["beta_kl"])
        self.beta_neg = float(hyperparameters["beta_neg"])
        self.gamma_r = float(hyperparameters["gamma_r"])

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        enc_parameters = self.enc_block.parameters()
        dec_parameters = self.dec_block.parameters()
        
        return {
            "enc": torch.optim.Adam(enc_parameters, lr=self.lr, betas=(0.9, 0.999)),
            "dec": torch.optim.Adam(dec_parameters, lr=self.lr, betas=(0.9, 0.999)),
        }

    def loss_both(self, x, z, mean, logvar, x2):

        loss_rec = self.reconstruction_term(x, x2).mean()
        loss_kl  = kl_gaussian(mean, logvar).mean()
        loss_both = loss_rec * self.beta_rec + loss_kl * self.beta_kl

        return loss_both, detach_dict({
            "loss_both": loss_both,
            "loss_both_loss_rec": loss_rec,
            "loss_both_loss_kl": loss_kl,
        })

    def loss_enc(self, x, z, mean, logvar, x2):

        s = 1 / (IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)

        zf = torch.randn_like(z)
        xf = self.decode(zf)
        mean2f, logvar2f = self.encode_gauss(x2.detach())
        meanff, logvarff = self.encode_gauss(xf.detach())
        z2f = self.reparameterize(mean2f, logvar2f)
        zff = self.reparameterize(meanff, logvarff)
        x2f = self.decode(z2f)
        xff = self.decode(zff)

        elbo   = -s * (self.beta_rec * self.reconstruction_term(x , x2 ) + self.beta_kl  * kl_gaussian(mean  , logvar  ))
        elbo_r = -s * (self.beta_rec * self.reconstruction_term(x2, x2f) + self.beta_neg * kl_gaussian(mean2f, logvar2f))
        elbo_f = -s * (self.beta_rec * self.reconstruction_term(xf, xff) + self.beta_neg * kl_gaussian(meanff, logvarff))

        exp_elbo_r = 0.5 * torch.exp(2 * elbo_r)
        exp_elbo_f = 0.5 * torch.exp(2 * elbo_f)

        loss_enc = -torch.mean(elbo - 0.5 * (exp_elbo_f + exp_elbo_r))

        return loss_enc, detach_dict({
            "loss_enc": loss_enc,
            "loss_enc_elbo": elbo.mean(),
            "loss_enc_elbo_r": elbo_r.mean(),
            "loss_enc_elbo_f": elbo_f.mean(),
            "loss_enc_exp_elbo_r": exp_elbo_r.mean(),
            "loss_enc_exp_elbo_f": exp_elbo_f.mean(),
        })

    def loss_dec(self, x, z, mean, logvar, x2):

        s = 1 / (IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH)

        zf = torch.randn_like(z)
        xf = self.decode(zf)
        mean2f, logvar2f = self.encode_gauss(x2.detach())
        meanff, logvarff = self.encode_gauss(xf.detach())
        z2f = self.reparameterize(mean2f, logvar2f)
        zff = self.reparameterize(meanff, logvarff)
        x2f = self.decode(z2f.detach())
        xff = self.decode(zff.detach())

        elbo   = -s * self.beta_rec * self.reconstruction_term(x, x2)
        elbo_r = -s * (self.gamma_r * self.beta_rec * self.reconstruction_term(x2, x2f) + self.beta_kl * kl_gaussian(mean2f, logvar2f))
        elbo_f = -s * (self.gamma_r * self.beta_rec * self.reconstruction_term(xf, xff) + self.beta_kl * kl_gaussian(meanff, logvarff))

        loss_dec = -torch.mean(elbo + 0.5 * (elbo_r + elbo_f))

        return loss_dec, detach_dict({
            "loss_dec": loss_dec,
            "loss_dec_elbo": elbo.mean(),
            "loss_dec_elbo_r": elbo_r.mean(),
            "loss_dec_elbo_f": elbo_f.mean(),
        })


    def step_batch(self, batch, optimizers = None, training: bool = False, progress = None):

        x, t = batch

        x = x.to(self.device)

        # both training
        if training:
            self.zero_grad()
        args = self(x)
        loss, loss_both_dict = self.loss_both(x, *args)
        if training:
            loss.backward()
            optimizers["enc"].step()
            optimizers["dec"].step()

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
            **loss_both_dict,
            **loss_enc_dict,
            **loss_dec_dict,
        }

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):

        return self.step_batch(batch, optimizers, training=True)

    def eval_batch(self, batch):

        return self.step_batch(batch, training=False)