import math
import torch
from torch import nn
from torch.nn import functional as F
from vaetc.models.abstract import AutoEncoderRLModel
from vaetc.network.cnn import ConvEncoder, ConvDecoder
from vaetc.models.utils import detach_dict

class DAGMM(AutoEncoderRLModel):
    """ DAGMM [Zong+, ICLR 2018]
    (https://openreview.net/forum?id=BJJLHbb0-)
    """

    def __init__(self, hyperparameters: dict):
        super().__init__(hyperparameters)

        self.num_components = int(hyperparameters["num_components"])
        assert self.num_components >= 2

        self.lr = float(hyperparameters["lr"])
        self.z_dim = int(hyperparameters["z_dim"])

        self.coef_energy = float(hyperparameters["coef_energy"])
        self.coef_prior = float(hyperparameters["coef_prior"])

        self.rec_features_dim = 2

        self.enc_block = ConvEncoder(self.z_dim)
        self.dec_block = ConvDecoder(self.z_dim)
        self.estimation_block = nn.Sequential(
            nn.Linear(self.z_dim + self.rec_features_dim, 256),
            nn.SiLU(True),
            nn.Linear(256, self.num_components),
            nn.Softmax(dim=1),
        )

        self.momentum = 0.1
        self.running_mean = nn.Parameter(torch.zeros(size=[self.z_dim, self.num_components]), requires_grad=False)
        self.running_sigma = nn.Parameter(torch.eye(self.z_dim).unsqueeze(2).tile(1, 1, self.num_components), requires_grad=False)
        self.running_phi = nn.Parameter(torch.ones(self.num_components, ) / self.num_components, requires_grad=False) # mixing ratio
        self.num_running = nn.Parameter(torch.tensor(0), requires_grad=False)

    def build_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), self.lr, betas=[0.9, 0.999], eps=1e-7)
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_block(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_block(z)

    def reconstruction_features(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:

        batch_size = x.shape[0]

        xf = x.view(batch_size, -1)
        x2f = x2.view(batch_size, -1)

        xf_norm = (xf ** 2).sum(dim=1) ** 0.5
        x2f_norm = (x2f ** 2).sum(dim=1) ** 0.5

        EPS = 1e-5
        rel_euc = ((xf - x2f) ** 2).sum(dim=1) ** 0.5 / (xf_norm + EPS)
        cos_sim = (xf * x2f).sum(dim=1) / (xf_norm * x2f_norm + EPS)
        return torch.stack([rel_euc, cos_sim], dim=1)

    def energy(self, z: torch.Tensor, mean: torch.Tensor, sigma: torch.Tensor, phi: torch.Tensor):
        
        inv_sigma = torch.linalg.inv(sigma.permute(2, 0, 1)).permute(1, 2, 0) # (L, L, K)

        diff_z = z[:,:,None] - mean[None,:,:] # (B, L, K)
        mahalanobis_dist = (diff_z[:,:,None,:] * diff_z[:,None,:,:] * inv_sigma[None,:,:,:]).sum(dim=[1,2]) # (B, K)
        logdet_sigma = torch.logdet(sigma.permute(2, 0, 1) * math.log(math.pi * 2)) # (K, )

        gauss_density = -0.5 * (torch.relu(mahalanobis_dist) + logdet_sigma[None,:]) # (B, K)
        energy = -torch.logsumexp(gauss_density + phi.log()[None,:], dim=1) # (B, )

        return energy

    def forward(self, x: torch.Tensor):

        EPS = 1e-5

        z = self.encode(x) # (B, L)
        x2 = self.decode(z)

        zr = self.reconstruction_features(x, x2)
        zcr = torch.cat([z, zr], dim=1)

        if self.training:
            
            gamma_hat: torch.Tensor = self.estimation_block(zcr) # (B, K)
            phi_hat = gamma_hat.mean(dim=0) # (K, )

            mean_hat = (gamma_hat[:,None,:] * z[:,:,None]).sum(dim=0) / (gamma_hat.sum(dim=0) + EPS)[None,:] # (L, K)

            diff_z = z[:,:,None] - mean_hat[None,:,:] # (B, L, K)
            sigma_hat = (gamma_hat[:,None,None,:] * diff_z[:,:,None,:] * diff_z[:,None,:,:]).sum(dim=0) / (gamma_hat.sum(dim=0) + EPS)[None,None,:] # (L, L, K)
            sigma_hat = sigma_hat + torch.eye(self.z_dim, device=sigma_hat.device)[:,:,None] * 1e-6

            if self.num_running.item() == 0:
                self.running_mean.copy_(mean_hat.detach())
                self.running_sigma.copy_(sigma_hat.detach())
                self.running_phi.copy_(phi_hat.detach())
            else:
                self.running_mean.copy_(mean_hat.detach() * self.momentum + self.running_mean * (1 - self.momentum))
                self.running_sigma.copy_(sigma_hat.detach() * self.momentum + self.running_sigma * (1 - self.momentum))
                self.running_phi.copy_(phi_hat.detach() * self.momentum + self.running_phi * (1 - self.momentum))
            self.num_running += 1

        else:
            
            mean_hat = self.running_mean
            sigma_hat = self.running_sigma
            phi_hat = self.running_phi
        
        energy = self.energy(z, mean_hat, sigma_hat, phi_hat)

        return z, x2, zr, zcr, mean_hat, sigma_hat, phi_hat, energy

    def loss(self, x, z, x2, zr, zcr, mean_hat, sigma_hat, phi_hat, energy):

        EPS = 1e-5

        batch_size = x.shape[0]
        
        loss_ae = ((x - x2) ** 2).view(batch_size, -1).sum(dim=1).mean()
        loss_energy = energy.mean()

        sigma_hat_diag = sigma_hat[range(self.z_dim),range(self.z_dim)]
        loss_prior = (1 / (sigma_hat_diag.abs() + EPS)).sum()

        loss = loss_ae + loss_energy * self.coef_energy + loss_prior * self.coef_prior

        return loss, detach_dict({
            "loss": loss,
            "loss_ae": loss_ae,
            "loss_energy": loss_energy,
            "loss_prior": loss_prior,
        })

    def step_batch(self, batch, optimizers=None, progress=None, training=False):

        x, t = batch
        x = x.cuda()

        z, x2, zr, zcr, mean_hat, sigma_hat, phi_hat, energy = self(x)
        loss, loss_dict = self.loss(x, z, x2, zr, zcr, mean_hat, sigma_hat, phi_hat, energy)

        if training:
            self.zero_grad()
            loss.backward()
            optimizers["main"].step()

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float):
        return self.step_batch(batch, optimizers, progress, training=True)

    def eval_batch(self, batch):
        return self.step_batch(batch, training=False)
