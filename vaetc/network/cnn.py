import torch
from torch import nn
from torch.nn import functional as F
from .blocks import SigmoidInverse, ResBlock

class ConvEncoder(nn.Module):
    """ A convolutional encoder, the same as :class:`ConvGaussianEncoder` without logvar """

    def __init__(self,
        z_dim: int, in_features: int = 3,
        batchnorm: bool = True, batchnorm_momentum: float = 0.1,
        inplace: bool = True, resblock: bool = False
    ):

        super().__init__()

        self.z_dim = int(z_dim)
        self.in_features = int(in_features)

        padding_mode = "zeros" if torch.are_deterministic_algorithms_enabled() else "replicate"

        layers = [
            SigmoidInverse(),
            
            nn.Conv2d(self.in_features, 32, 4, stride=2, padding=1, padding_mode=padding_mode),
            nn.SiLU(inplace),
            nn.BatchNorm2d(32, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(32, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(32, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,

            nn.Conv2d(32, 64, 4, stride=2, padding=1, padding_mode=padding_mode),
            nn.SiLU(inplace),
            nn.BatchNorm2d(64, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(64, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(64, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            
            nn.Conv2d(64, 128, 4, stride=2, padding=1, padding_mode=padding_mode),
            nn.SiLU(inplace),
            nn.BatchNorm2d(128, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(128, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(128, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            
            nn.Conv2d(128, 256, 4, stride=2, padding=1, padding_mode=padding_mode),
            nn.SiLU(inplace),
            nn.BatchNorm2d(256, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(256, batchnorm=batchnorm, momentum=batchnorm_momentum) if resblock else None,
            ResBlock(256, batchnorm=batchnorm, momentum=batchnorm_momentum) if resblock else None,
            
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.SiLU(inplace),
        ]
        layers = [layer for layer in layers if layer is not None]

        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(256, self.z_dim)

    def forward(self, x: torch.Tensor):

        h = self.net(x)
        return self.fc(h)

class ConvGaussianEncoder(ConvEncoder):
    """ A convolutional encoder mainly based on the original VAE paper's settings
    [Higgins+, 2016 (https://openreview.net/forum?id=Sy2fzU9gl)] """

    def __init__(self,
        z_dim: int, in_features: int = 3,
        batchnorm: bool = True, batchnorm_momentum: float = 0.1,
        inplace: bool = True, resblock: bool = False
    ):

        super().__init__(z_dim, in_features, batchnorm, batchnorm_momentum, inplace, resblock)

        if batchnorm:
            self.fc = nn.Sequential(
                nn.Linear(256, self.z_dim),
                nn.BatchNorm1d(self.z_dim, momentum=batchnorm_momentum),
            )
            self.fc_logvar = nn.Sequential(
                nn.Linear(256, self.z_dim),
                nn.BatchNorm1d(self.z_dim, momentum=batchnorm_momentum),
            )
        else:
            self.fc_logvar = nn.Linear(256, self.z_dim)

    def forward(self, x: torch.Tensor):

        h = self.net(x)

        mean   = self.fc(h)
        logvar = self.fc_logvar(h)

        return mean, logvar

class ConvDecoder(nn.Module):
    """ A convolutional decoder mainly based on the original VAE paper's settings
    [Higgins+, 2016 (https://openreview.net/forum?id=Sy2fzU9gl)] """

    def __init__(self,
        z_dim: int, out_features: int = 3,
        batchnorm: bool = True, batchnorm_momentum: float = 0.1,
        inplace: bool = True, resblock: bool = False
    ):
        super().__init__()

        self.z_dim = z_dim
        self.out_features = int(out_features)

        layers = [
            nn.Linear(self.z_dim, 256),
            nn.SiLU(inplace),
            nn.Linear(256, 256 * 4 * 4),
            nn.SiLU(inplace),
            nn.Unflatten(1, [256, 4, 4]),

            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, padding_mode="zeros"),
            nn.SiLU(inplace),
            nn.BatchNorm2d(128, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(128, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(128, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, padding_mode="zeros"),
            nn.SiLU(inplace),
            nn.BatchNorm2d(64, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(64, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(64, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, padding_mode="zeros"),
            nn.SiLU(inplace),
            nn.BatchNorm2d(32, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(32, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(32, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,

            nn.ConvTranspose2d(32, self.out_features, 4, stride=2, padding=1, padding_mode="zeros"),
            ResBlock(self.out_features, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(self.out_features, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,

            nn.Sigmoid(),
        ]
        layers = [layer for layer in layers if layer is not None]

        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:

        return self.net(z)