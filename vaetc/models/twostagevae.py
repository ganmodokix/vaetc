import itertools
import math

import torch
from torch import nn

from vaetc.models.utils import detach_dict
from .vae import VAE
from vaetc.network.losses import neglogpxz_gaussian, kl_gaussian
from vaetc.network.cnn import ConvDecoder, ConvGaussianEncoder
from vaetc.data.utils import IMAGE_CHANNELS

class kSimpleImageEncoder(nn.Module):

    def __init__(self, z_dim: int) -> None:
        super().__init__()

        self.enc_block = ConvGaussianEncoder(z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        mean_z, logvar_z = self.enc_block(x)

        return mean_z, logvar_z

class kSimpleLatentEncoder(nn.Module):

    def __init__(self, z_dim: int) -> None:
        super().__init__()

        hidden_dim = 256

        self.enc_block_second = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(True),
        )
        self.enc_block_second_mean = nn.Linear(hidden_dim + z_dim, z_dim)
        self.enc_block_second_logvar = nn.Linear(hidden_dim + z_dim, z_dim)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        h = self.enc_block_second(z)
        hz = torch.cat([h, z], dim=1)

        mean_u = self.enc_block_second_mean(hz)
        logvar_u = self.enc_block_second_logvar(hz)

        return mean_u, logvar_u

class kSimpleLatentDecoder(nn.Module):

    def __init__(self, z_dim: int) -> None:
        super().__init__()

        hidden_dim = 256

        self.dec_block_second = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.SiLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(True),
        )

        self.dec_block_z = nn.Linear(hidden_dim + z_dim, z_dim)
        self.dec_block_gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, u: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        h = self.dec_block_second(u)
        hu = torch.cat([h, u], dim=1)

        z2 = self.dec_block_z(hu)

        return z2, self.dec_block_gamma

class kSimpleImageDecoder(nn.Module):

    def __init__(self, z_dim: int) -> None:
        super().__init__()

        self.dec_block = ConvDecoder(z_dim)
        self.dec_block_gamma = nn.Parameter(torch.tensor(0.0), requires_grad=True)

    def forward(self, z: torch.Tensor):

        x2 = self.dec_block(z)
        return x2, self.dec_block_gamma

class TwoStageVAE(VAE):
    """ TwoStageVAE [Dai and Wipf, 2019]
    (https://openreview.net/forum?id=B1e0X3C9tQ)
    """

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)
        
        del self.enc_block
        del self.dec_block

        self.enc_stage1 = kSimpleImageEncoder(self.z_dim)
        self.enc_stage2 = kSimpleLatentEncoder(self.z_dim)
        self.dec_stage2 = kSimpleLatentDecoder(self.z_dim)
        self.dec_stage1 = kSimpleImageDecoder(self.z_dim)

        self.running_progress = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:

        params_stage1 = itertools.chain(
            self.enc_stage1.parameters(),
            self.dec_stage1.parameters(),
        )

        params_stage2 = itertools.chain(
            self.enc_stage2.parameters(),
            self.dec_stage2.parameters(),
        )

        return {
            "stage1": torch.optim.Adam(params_stage1, lr=self.lr),
            "stage2": torch.optim.Adam(params_stage2, lr=self.lr),
        }

    def reparameterize_dec(self, x2: torch.Tensor, loggamma: torch.Tensor) -> torch.Tensor:

        eps = torch.randn_like(x2)
        gamma = loggamma.exp()

        return x2 + eps * (gamma[:,None] if x2.ndim == 2 else gamma[:,None,None,None])

    def forward(self, x: torch.Tensor):

        mean_z, logvar_z = self.enc_stage1(x)
        z = self.reparameterize(mean_z, logvar_z)

        mean_u, logvar_u = self.enc_stage2(z)
        u = self.reparameterize(mean_u, logvar_u)

        z2, loggamma_z = self.dec_stage2(u)
        x2, loggamma_x = self.dec_stage1(self.reparameterize_dec(z2, loggamma_z))

        return self.reparameterize_dec(x2, loggamma_x)

    def encode_gauss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        stage = 1 if self.running_progress.item() < 1/2 else 2
        
        mean_z, logvar_z = self.enc_stage1(x)

        if stage == 1:
            
            return mean_z, logvar_z

        elif stage == 2:

            z = self.reparameterize(mean_z, logvar_z)
            return self.enc_stage2(z)

        else:

            raise RuntimeError(f"Invalid stage {stage}, neither 1 nor 2")

    def decode(self, zu: torch.Tensor) -> torch.Tensor:

        stage = 1 if self.running_progress.item() < 1/2 else 2

        if stage == 1:

            z = zu

            x2, loggamma_x = self.dec_stage1(z)
            return x2

        elif stage == 2:

            u = zu
            
            z2, loggamma_z = self.dec_stage2(u)
            # z2 = self.reparameterize_dec(z2, loggamma_z)

            x2, loggamma_x = self.dec_stage1(z2)
            
            return x2

        else:

            raise RuntimeError(f"Invalid stage {stage}, neither 1 nor 2")

    def sample_prior(self, batch_size: int) -> torch.Tensor:

        return torch.randn(size=[batch_size, self.z_dim], device="cuda")

    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        x = x.cuda()
        batch_size = x.shape[0]

        if progress is not None:
            self.running_progress.fill_(progress)
        
        stage = 1 if self.running_progress.item() < 1/2 else 2

        mean_z, logvar_z = self.enc_stage1(x)
        z = self.reparameterize(mean_z, logvar_z)

        log2pi = math.log(math.pi * 2)

        if stage == 1:

            x2, loggamma_x = self.dec_stage1(z)
            invgamma_x = loggamma_x.neg().exp()

            loss_ae = (0.5 * (((x - x2) ** 2).view(batch_size, -1) * invgamma_x + loggamma_x + log2pi).sum(dim=1)).mean()
            loss_reg = kl_gaussian(mean_z, logvar_z).mean()

            loss = loss_ae + loss_reg
            
            if training:
                self.zero_grad()
                loss.backward()
                optimizers["stage1"].step()
            
        elif stage == 2:

            z = z.detach()

            mean_u, logvar_u = self.enc_stage2(z)
            u = self.reparameterize(mean_u, logvar_u)

            z2, loggamma_z = self.dec_stage2(u)
            invgamma_z = loggamma_z.neg().exp()

            loss_ae = (0.5 * ((z - z2) ** 2 * invgamma_z + loggamma_z + log2pi).sum(dim=1)).mean()
            loss_reg = kl_gaussian(mean_u, logvar_u).mean()

            loss = loss_ae + loss_reg
            
            if training:
                self.zero_grad()
                loss.backward()
                optimizers["stage2"].step()

        else:

            raise RuntimeError(f"Invalid stage {stage}, neither 1 nor 2")
        
        return detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_reg": loss_reg,
            "stage": stage,
        })

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)
        
    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)

