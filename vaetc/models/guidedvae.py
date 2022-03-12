from typing import Optional, Tuple

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from vaetc.data.utils import IMAGE_SHAPE

from .utils import detach_dict
from vaetc.network.reparam import reparameterize
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class DeformablePCA(nn.Module):

    def __init__(self, z_def_dim: int, z_cont_dim: int, out_shape: Tuple[int, int, int]):
        super().__init__()

        self.z_def_dim = int(z_def_dim)
        assert self.z_def_dim > 0

        self.z_cont_dim = int(z_cont_dim)
        assert self.z_cont_dim > 0

        self.out_shape = out_shape
        self.out_channels, self.out_height, self.out_width = self.out_shape
        assert self.out_channels > 0
        assert self.out_height > 0
        assert self.out_width > 0

        self.out_size = self.out_channels * self.out_height * self.out_width
        self.bases = nn.Parameter(torch.randn(size=(self.z_cont_dim + 1, self.out_size)) * 0.01, requires_grad=True)

        self.deform = nn.Sequential(
            nn.Linear(self.z_def_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 6)
        )

    def forward(self, z_def: torch.Tensor, z_cont: torch.Tensor):

        batch_size = z_def.shape[0]

        theta = F.affine_grid(self.deform(z_def).view(batch_size, 2, 3), size=(batch_size, *IMAGE_SHAPE), align_corners=False)

        x = (torch.cat([z_cont, torch.ones(size=(batch_size, 1), device=z_cont.device)], dim=1) @ self.bases).view(batch_size, *self.out_shape)
        x = F.grid_sample(x, theta, mode="bilinear", padding_mode="border", align_corners=False)

        return x

class GuidedVAE(VAE):
    """ GuidedVAE
    [Ding+, 2020 (https://openaccess.thecvf.com/content_CVPR_2020/html/Ding_Guided_Variational_Autoencoder_for_Disentanglement_Learning_CVPR_2020_paper.html)]"""

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.z_def_dim = int(hyperparameters["z_def_dim"])
        self.z_cont_dim = self.z_dim - self.z_def_dim
        assert self.z_cont_dim > 0

        self.dec_sub = DeformablePCA(self.z_def_dim, self.z_cont_dim, IMAGE_SHAPE)
        
    def loss(self, x, z, mean, logvar, x2, progress: Optional[float] = None):
        
        z_def = z[:,:self.z_def_dim]
        z_cont = z[:,self.z_def_dim:]

        x_sub = self.dec_sub(z_def, z_cont)
        loss_dpca = torch.mean(torch.sum((x - x_sub) ** 2, dim=[1, 2, 3]))

        # Losses
        loss_ae  = torch.mean(neglogpxz_gaussian(x, x2))
        loss_reg = torch.mean(kl_gaussian(mean, logvar))

        ip = (self.dec_sub.bases @ self.dec_sub.bases.t()) ** 2
        loss_pca = ip.sum() - torch.diagonal(ip).sum()

        # Total loss
        loss = loss_ae + loss_reg + loss_pca + loss_dpca

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "loss_dpca": loss_dpca,
            "loss_pca": loss_pca,
        })