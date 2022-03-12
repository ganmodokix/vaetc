from typing import Optional, Dict
import itertools

import torch
from torchvision.models.vgg import vgg19

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

PRETRAINED_VGG19 = None
def get_pretrained_vgg19():
    global PRETRAINED_VGG19
    if PRETRAINED_VGG19 is None:
        PRETRAINED_VGG19 = vgg19(pretrained=True).cuda()
    return PRETRAINED_VGG19

class DFCVAE(VAE):
    """ Deep Feature Consistent VAE
    [Hou+, 2017 (https://ieeexplore.ieee.org/document/7926714)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.beta = float(hyperparameters["beta"])

    def features(self, x: torch.Tensor):
        
        extractor = get_pretrained_vgg19()

        # features
        features = [x]
        h = x
        for (name, layer) in extractor.features._modules.items():
            h = layer(h)
            if name in ["14", "24", "34", "43"]:
                features.append(h)

        return features

    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):

        # Feature Conceptual Loss
        phi  = self.features(x)
        phi2 = self.features(x2)
        losses_ae = []
        for i in range(len(phi)):
            loss_ae_l = 0.5 * torch.mean((phi[i] - phi2[i]) ** 2, dim=list(range(1, len(phi[i].shape))))
            losses_ae.append(loss_ae_l)

        # Losses
        x_size = x.shape[1] * x.shape[2] * x.shape[3]
        loss_ae  = torch.mean(torch.stack(losses_ae, dim=1).mean(dim=1) * x_size)
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        # Total loss
        loss = loss_ae + loss_reg * self.beta

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
        })