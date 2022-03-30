from typing import Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self,
        num_features, style_dim, noise_input=True,
        eps=1e-5,
        # momentum=0.1, track_running_stats=True,
        device=None, dtype=None
    ) -> None:
        
        super().__init__()
        
        # self.instance_norm = nn.InstanceNorm2d(
        #     num_features, eps, momentum, affine=False,
        #     track_running_stats=track_running_stats,
        #     device=device, dtype=dtype
        # )
        self.eps = float(eps)

        if noise_input:
            self.noise_scale_logit = nn.Parameter(torch.zeros(size=[num_features, ], device=device, dtype=dtype), requires_grad=True)
        else:
            self.noise_scale_logit = None

        self.gamma = nn.Sequential(
            nn.Linear(style_dim, num_features),
        )

        self.beta = nn.Sequential(
            nn.Linear(style_dim, num_features),
        )
        

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:

        if self.noise_scale_logit is not None:
            content = content + self.noise_scale_logit.exp()[None,:,None,None] * torch.randn_like(content)

        # content_normed = self.instance_norm(content)
        content_normed = (content - content.mean(dim=[2,3], keepdim=True)) / (content.var(dim=[2,3], keepdim=True, unbiased=False) + self.eps) ** 0.5
        content_adained = content_normed * self.gamma(style)[:,:,None,None] + self.beta(style)[:,:,None,None]

        return content_adained

class AdainDecoder(nn.Module):
    
    def __init__(self, z_dim: int, mapping_network=True) -> None:
        super().__init__()

        self.z_dim = int(z_dim)
        assert self.z_dim > 0

        style_dim = 256 if mapping_network else self.z_dim

        self.dec_tensor = nn.Parameter(torch.randn(size=[256, 4, 4]), requires_grad=True)
        self.dec_adains = nn.ModuleList([
            AdaptiveInstanceNorm2d(256, style_dim, noise_input=True),
            AdaptiveInstanceNorm2d(128, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d(128, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d( 64, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d( 64, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d( 32, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d( 32, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d( 16, style_dim, noise_input=False),
            AdaptiveInstanceNorm2d( 16, style_dim, noise_input=False),
        ])
        self.dec_convs = nn.ModuleList([
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(64, 128, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(128, 128, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(32, 64, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(64, 64, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(16, 32, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(32, 32, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(8, 16, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(16, 16, 3, 1, 1, bias=False),
                nn.SiLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(16, 3, 3, 1, 1, bias=False),
                nn.Sigmoid(),
            ),
        ])

        if mapping_network:
            self.style_net = nn.Sequential(
                nn.Linear(z_dim, 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
                nn.Linear(256, 256),
                nn.SiLU(True),
                nn.Linear(256, style_dim),
            )
        else:
            self.style_net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        s = self.style_net(x)

        h = self.dec_tensor[None,...].tile(x.shape[0], 1, 1, 1)
        for i in range(len(self.dec_adains)):
            h = self.dec_adains[i](h, s)
            h = self.dec_convs[i](h)
        return h