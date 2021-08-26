import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=16):
        super().__init__()
        self.layers = nn.Sequential(
            spectral_norm(nn.Conv2d(nc, ndf, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf),
            nn.GELU(),
            spectral_norm(nn.Conv2d(ndf * 1, ndf * 2, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 2),
            nn.GELU(),
            spectral_norm(nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 4),
            nn.GELU(),
            spectral_norm(nn.Conv2d(ndf * 4, ndf * 8, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 8),
            nn.GELU(),
            spectral_norm(nn.Conv2d(ndf * 8, ndf * 16, 3, 2, 1, bias=False)),
            nn.BatchNorm2d(ndf * 16),
            nn.GELU(),
            spectral_norm(nn.Conv2d(ndf * 16, 1, 3, 2, 1, bias=False)),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.flatten(1)


if __name__ == "__main__":
    print(Discriminator()(torch.randn(3, 3, 128, 128)).shape)
