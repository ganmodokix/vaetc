
import torch
from torch import nn

class SigmoidInverse(nn.Module):
    
    def __init__(self, eps: float = 1e-4) -> None:
        super().__init__()

        self.eps = float(eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, min=self.eps, max=1 - self.eps)
        xc = torch.clamp(1. - x, min=self.eps, max=1 - self.eps)
        return x.log() - xc.log()

class SEBlock(nn.Module):
    """ Squeeze-and-Excitation Block by [Hu+, 2019 (https://arxiv.org/abs/1709.01507)] """

    def __init__(self, num_features, reduction: int = 16):
        
        super().__init__()

        self.num_features = int(num_features)
        self.reduction = int(reduction)
        self.num_reduced = (self.num_features + self.reduction - 1) // self.reduction
        assert self.num_features > 0
        assert self.reduction > 0

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.num_features, self.num_reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_reduced, self.num_features, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        h = self.pool(x).view(-1, self.num_features)
        h = self.fc(h).view(-1, self.num_features, 1, 1)
        return x * h

class ResBlock(nn.Module):
    """ Plain Residual block
    (https://arxiv.org/abs/1409.1556)
    (https://arxiv.org/abs/1603.05027)
    """

    def __init__(self, num_channels: int, inplace: bool = True, batchnorm: bool = False, batchnorm_momentum: float = 0.1) -> None:
        super().__init__()

        padding_mode = "zeros" if torch.are_deterministic_algorithms_enabled() else "replicate"

        layers = [
            nn.Conv2d(num_channels, num_channels, 1, 1, 0, padding_mode=padding_mode),
            nn.SiLU(inplace),
            nn.BatchNorm2d(num_channels, momentum=batchnorm_momentum) if batchnorm else None,

            nn.Conv2d(num_channels, num_channels, 1, 1, 0, padding_mode=padding_mode),
            nn.SiLU(inplace),
            nn.BatchNorm2d(num_channels, momentum=batchnorm_momentum) if batchnorm else None,
        ]
        layers = [layer for layer in layers if layer is not None]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return x + self.net(x)

class NVAEGenerativeResidualCell(nn.Module):
    """ Residual cell of the Generative model in NVAE [Vahdat and Kautz, 2020 (https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html)] """

    def __init__(self, num_features: int, e_features: int, bn_momentum: float = 0.1):

        super().__init__()

        self.num_features = int(num_features)
        assert self.num_features > 0
        self.e_features = int(e_features)
        assert self.e_features > 0

        self.bn_momentum = float(bn_momentum)

        c = self.num_features
        ec = self.num_features * self.e_features

        padding_mode = "zeros" if torch.are_deterministic_algorithms_enabled() else "replicate"
        self.residual_module = nn.Sequential(
            nn.BatchNorm2d(c, momentum=self.bn_momentum),
            nn.Conv2d(c, ec, kernel_size=1, stride=1, padding=0, padding_mode=padding_mode),
            nn.BatchNorm2d(ec, momentum=self.bn_momentum),
            nn.SiLU(inplace=True),
            nn.Conv2d(ec, ec, kernel_size=5, stride=1, padding=2, groups=ec, padding_mode=padding_mode),
            nn.BatchNorm2d(ec, momentum=self.bn_momentum),
            nn.SiLU(inplace=True),
            nn.Conv2d(ec, c, kernel_size=1, stride=1, padding=0, padding_mode=padding_mode),
            nn.BatchNorm2d(c, momentum=self.bn_momentum),
            SEBlock(c),
        )

    def forward(self, x):

        h = self.residual_module(x)

        return x + h

class NVAEInferenceResidualCell(nn.Module):
    """ Residual cell of the Inference model in NVAE [Vahdat and Kautz, 2020 (https://proceedings.neurips.cc/paper/2020/hash/e3b21256183cf7c2c7a66be163579d37-Abstract.html)] """

    def __init__(self, num_features: int, bn_momentum: float = 0.1):

        super().__init__()

        self.num_features = int(num_features)
        assert self.num_features > 0

        self.bn_momentum = float(bn_momentum)

        c = self.num_features
        padding_mode = "zeros" if torch.are_deterministic_algorithms_enabled() else "replicate"
        self.residual_module = nn.Sequential(
            nn.BatchNorm2d(c, momentum=self.bn_momentum),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            nn.BatchNorm2d(c, momentum=self.bn_momentum),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode),
            SEBlock(c),
        )

    def forward(self, x):

        h = self.residual_module(x)
        
        return x + h