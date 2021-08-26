import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, dim_latent):
        super().__init__()
        self.dim_latent = dim_latent

    def forward(self, n):
        raise NotImplementedError

    @property
    def device(self):
        return next(self.parameters()).device

    def create_log_dict(self, n=16, z=None, prefix=""):
        if z is not None:
            images = self(z=z)
        else:
            images = self(n=n)
        return {
            f"{prefix}fake": wandb.Image(
                images + 0.5,  # denormalize
            ),
        }


class VanillaGenerator(Generator):
    def __init__(self, nc=3, dim_latent=128, ngf=16):
        super().__init__(dim_latent)
        self.main = nn.Sequential(
            # state size. 4 x 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(dim_latent // 16, ngf * 8, 3, 1, 1),
            nn.BatchNorm2d(ngf * 8),
            nn.GELU(),
            # state size. 8 x 8
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 8, ngf * 4, 3, 1, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.GELU(),
            # state size. 16 x 16
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 4, ngf * 2, 3, 1, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.GELU(),
            # state size. 32 x 32
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1),
            nn.BatchNorm2d(ngf),
            nn.GELU(),
            # state size. 64 x 64
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf, nc, 3, 1, 1),
            # state size. 128 x 128
        )

    def forward(self, z=None, n=None):
        if z is None:
            z = torch.randn(n, self.dim_latent, device=self.device)
        return self.main(z.view(len(z), -1, 4, 4))


if __name__ == "__main__":
    generator = VanillaGenerator()
    print(generator(n=3).shape)
    print(generator.create_log_dict())
