from typing import Dict, List, Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F

from .abstract import RLModel
from .utils import detach_dict
from vaetc.network.cnn import ConvEncoder

class CNNClassifier(RLModel):
    """ CNN-based classifier as a representation learning method. """

    def __init__(self, hyperparameters: dict):

        super().__init__(hyperparameters)

        self.z_dim = int(hyperparameters["z_dim"])
        assert self.z_dim > 0
        self.t_dim = int(hyperparameters["t_dim"])
        assert self.t_dim > 0
        self.lr = float(hyperparameters["lr"])

        self.features = ConvEncoder(z_dim=self.z_dim)
        self.classifier = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(self.z_dim, self.t_dim),
            nn.Sigmoid(),
        )

    def build_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return {
            "main": torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.999))
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.features(x)

    def forward(self, x: torch.Tensor):
        
        z = self.encode(x)
        t2 = self.classifier(z)

        return z, t2

    def loss(self, x, t, z, t2, progress = None):

        # loss = F.binary_cross_entropy(t2, t, reduction="none")
        loss = (t - t2) ** 2
        loss = loss.mean()

        return loss, detach_dict({
            "loss": loss
        })

    def step_batch(self, batch, optimizers = None, training: bool = False):

        x, t = batch
        x = x.to(self.device)
        t = t.to(self.device)

        self.zero_grad()
        z, t2 = self(x)
        loss, loss_dict = self.loss(x, t, z, t2)
        if training:
            loss.backward()
            optimizers["main"].step()

        return loss_dict

    def train_batch(self, batch, optimizers, progress: float) -> Dict[str, List[float]]:
        
        return self.step_batch(batch, optimizers, True)

    def eval_batch(self, batch) -> Dict[str, List[float]]:
        
        return self.step_batch(batch)