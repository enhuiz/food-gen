import torch.nn as nn

from .layers import Residual


class Encoder(nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent

    @property
    def device(self):
        return next(self.parameters()).device


class Block(Residual):
    def __init__(self, in_features, out_features):
        super().__init__(
            nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_features),
            nn.GELU(),
            nn.Conv2d(out_features, out_features, 3, padding=1),
            nn.BatchNorm2d(out_features),
            residual=nn.Conv2d(in_features, out_features, 1, 2),
            activation=nn.GELU(),
        )


class VanillaEncoder(Encoder):
    def __init__(self, dim_latent=128):
        super().__init__(dim_latent)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3),
            nn.BatchNorm2d(32),
            nn.GELU(),
            Block(32, 64),
            Block(64, 64),
            Block(64, 128),
            Block(128, dim_latent),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    from torchsummary import summary

    encoder = VanillaEncoder()
    print(encoder)
    summary(encoder, (3, 112, 112), device="cpu")
