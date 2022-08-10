import torch
from torch import nn
from torch.nn import functional as F
from .blocks import SigmoidInverse, ResBlock

class ConvEncoder(nn.Module):
    """ A convolutional encoder, the same as :class:`ConvGaussianEncoder` without logvar """

    def __init__(self,
        z_dim: int, in_features: int = 3,
        batchnorm: bool = True, batchnorm_momentum: float = 0.1,
        inplace: bool = True, resblock: bool = False,
        hidden_filters: list[int] = [32, 64, 128, 256]
    ):

        super().__init__()

        self.z_dim = int(z_dim)
        self.in_features = int(in_features)

        assert len(hidden_filters) == 4

        padding_mode = "zeros"

        layers_conv = []
        layers_conv += [
            nn.Conv2d(self.in_features, hidden_filters[0], 4, stride=2, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(0.2, inplace),
            nn.BatchNorm2d(hidden_filters[0], momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(hidden_filters[0], batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(hidden_filters[0], batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
        ]
        for in_filters, out_filters in zip(hidden_filters[:-1], hidden_filters[1:]):
            layers_conv += [
                nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1, padding_mode=padding_mode),
                nn.LeakyReLU(0.2, inplace),
                nn.BatchNorm2d(out_filters, momentum=batchnorm_momentum) if batchnorm else None,
                ResBlock(out_filters, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
                ResBlock(out_filters, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ]
            
        layers_fc = [
            nn.Flatten(),
            nn.Linear(hidden_filters[-1] * 4 * 4, 256),
            nn.LeakyReLU(0.2, inplace),
        ]

        layers = [SigmoidInverse()] + layers_conv + layers_fc
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
        inplace: bool = True, resblock: bool = False,
        hidden_filters: list[int] = [32, 64, 128, 256]
    ):

        super().__init__(z_dim, in_features, batchnorm, batchnorm_momentum, inplace, resblock, hidden_filters)

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
        inplace: bool = True, resblock: bool = False,
        hidden_filters: list[int] = [256, 128, 64, 32]
    ):
        super().__init__()

        self.z_dim = z_dim
        self.out_features = int(out_features)

        assert len(hidden_filters) == 4

        layers_fc = [
            nn.Linear(self.z_dim, 256),
            nn.LeakyReLU(0.2, inplace),
            nn.Linear(256, hidden_filters[0] * 4 * 4),
            nn.LeakyReLU(0.2, inplace),
            nn.Unflatten(1, [hidden_filters[0], 4, 4]),
        ]

        layers_conv = []
        for in_filters, out_filters in zip(hidden_filters[:-1], hidden_filters[1:]):
            
            unit = [
                nn.ConvTranspose2d(in_filters, out_filters, 4, stride=2, padding=1, padding_mode="zeros"),
                nn.LeakyReLU(0.2, inplace),
                nn.BatchNorm2d(out_filters, momentum=batchnorm_momentum) if batchnorm else None,
                ResBlock(out_filters, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
                ResBlock(out_filters, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ]

            layers_conv += unit
            
        last_filters = hidden_filters[-1]
        layers_conv += [
            nn.ConvTranspose2d(last_filters, last_filters, 4, stride=2, padding=1, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace),
            nn.BatchNorm2d(last_filters, momentum=batchnorm_momentum) if batchnorm else None,
            ResBlock(last_filters, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            ResBlock(last_filters, batchnorm=batchnorm, batchnorm_momentum=batchnorm_momentum) if resblock else None,
            nn.Conv2d(last_filters, self.out_features, 5, 1, 2, padding_mode="zeros"),
        ]

        layers = layers_fc + layers_conv + [nn.Sigmoid()]
        layers = [layer for layer in layers if layer is not None]

        self.net = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:

        return self.net(z)