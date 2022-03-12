from typing import Dict, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from vaetc.data.utils import IMAGE_HEIGHT, IMAGE_WIDTH

from vaetc.network.cnn import ConvGaussianEncoder, ConvDecoder
from vaetc.network.reparam import reparameterize

from .utils import detach_dict

from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from .vae import VAE

class VITAE(VAE):
    """ U-VITAE
    [Detlefsen and Hauberg, 2019 (https://proceedings.neurips.cc/paper/2019/hash/3493894fa4ea036cfc6433c3e2ee63b0-Abstract.html)] """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        # hyperparameters
        self.zp_dim = int(hyperparameters["zp_dim"])
        self.za_dim = self.z_dim - self.zp_dim
        assert 0 < self.zp_dim < self.z_dim, "# of the dimensions of z_p must be a positive integer"

        # todo: it still remains a magic number
        self.num_sampling = 4

        # network layers
        self.enc_block = ConvGaussianEncoder(z_dim=self.z_dim)
        self.dec_appearance  = ConvDecoder(z_dim=self.za_dim)
        self.dec_perspective = nn.Sequential(
            nn.Linear(self.zp_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 2 * 5 * 5),
            nn.Tanh(),
            nn.Unflatten(1, (2, 5, 5)),
            nn.UpsamplingBilinear2d(size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
        )

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr)
        }

    def encode_gauss(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.enc_block(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:

        za = z[:,self.zp_dim:]
        a = self.dec_appearance(za)

        pos = torch.linspace(-1, 1, 64).to(self.device)
        grid_x, grid_y = torch.meshgrid(pos, pos, indexing="ij")

        zp = z[:,:self.zp_dim]
        p = self.dec_perspective(zp)
        p = p.permute(0, 2, 3, 1)
        p[:,:,:,0] += grid_x[None,:,:]
        p[:,:,:,1] += grid_y[None,:,:]

        x2 = F.grid_sample(a, p, padding_mode="border", align_corners=False)

        return x2

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        mean, logvar = self.encode_gauss(x)

        zs = []
        x2s = []
        for i in range(self.num_sampling):
            z = reparameterize(mean, logvar)
            x2 = self.decode(z)
            zs.append(z)
            x2s.append(x2)
        
        zs  = torch.stack(zs , dim=1)
        x2s = torch.stack(x2s, dim=1)

        return zs, mean, logvar, x2s

    def loss(self, x, zs, mean, logvar, x2s, progress: Optional[float] = None):

        # losses
        batch_size = x.shape[0]
        xs = x[:,None,...].tile(1, self.num_sampling, 1, 1)
        xsf = xs.view(batch_size, self.num_sampling, -1)
        x2sf = x2s.view(batch_size, self.num_sampling, -1)
        loss_ae  = torch.mean(neglogpxz_gaussian(xsf, x2sf))
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
        zs, mean, logvar, x2s = self(x)
        loss, loss_dict = self.loss(x, zs, mean, logvar, x2s, progress)
        loss.backward()
        optimizers["main"].step()

        return loss_dict

    def eval_batch(self, batch):

        x, t = batch

        x = x.to(self.device)
        zs, mean, logvar, x2s = self(x)
        loss, loss_dict = self.loss(x, zs, mean, logvar, x2s)

        return loss_dict