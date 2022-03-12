import torch
from torch import nn
from torch.nn import functional as F
from blocks import NVAEGenerativeResidualCell, NVAEInferenceResidualCell

class NVAEConvGaussianEncoder(nn.Module):
    """ A large convolutional encoder for VAE """

    def __init__(self, z_dim: int, in_features: int = 3):

        super().__init__()

        self.z_dim = z_dim
        self.in_features = int(in_features)

        channels = [64, 64, 64, 32]

        self.net = nn.Sequential(

            nn.Conv2d(self.in_features, channels[0], kernel_size=3, stride=1, padding=1),
            NVAEInferenceResidualCell(channels[0]),
            nn.AdaptiveAvgPool2d(32),

            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            NVAEInferenceResidualCell(channels[1]),
            nn.AdaptiveAvgPool2d(16),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            NVAEInferenceResidualCell(channels[2]),
            nn.AdaptiveAvgPool2d(8),

            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            NVAEInferenceResidualCell(channels[3]),
            nn.AdaptiveAvgPool2d(4),

            nn.Flatten(),
            nn.Linear(channels[3] * 4 * 4, 256, bias=False),
            nn.SiLU(inplace=True),

        )

        self.fc_mean   = nn.Linear(256, self.z_dim)
        self.fc_logvar = nn.Linear(256, self.z_dim)

    def forward(self, x: torch.Tensor):

        h = self.net(x)

        mean   = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar

class NVAEConvDecoder(nn.Module):

    def __init__(self, z_dim: int, out_features: int = 3):
        super().__init__()

        self.z_dim = z_dim
        self.out_features = int(out_features)

        channels = [32, 64, 64, 64]

        self.net = nn.Sequential(
            
            nn.Linear(self.z_dim, 256),
            nn.SiLU(inplace=True),
            nn.Linear(256, channels[0] * 4 * 4),
            nn.SiLU(inplace=True),
            nn.Unflatten(1, [channels[0], 4, 4]),

            NVAEGenerativeResidualCell(channels[0], 4),
            nn.AdaptiveAvgPool2d(8),
            
            nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            NVAEGenerativeResidualCell(channels[1], 4),
            nn.AdaptiveAvgPool2d(16),
            
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            NVAEGenerativeResidualCell(channels[2], 4),
            nn.AdaptiveAvgPool2d(32),
            
            nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            NVAEGenerativeResidualCell(channels[3], 4),
            nn.AdaptiveAvgPool2d(64),
            
            nn.Conv2d(channels[3], 3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),

        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.net(x)