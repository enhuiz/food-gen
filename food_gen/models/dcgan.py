import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class Generator(nn.Module):
    def __init__(self, nc=3, zdim=128, ngf=16):
        super().__init__()
        self.zdim = zdim
        self.main = nn.Sequential(
            # state size. 4 x 4
            nn.Upsample(scale_factor=2),
            nn.Conv2d(zdim // 16, ngf * 8, 3, 1, 1),
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

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, n):
        z = torch.randn(n, self.zdim, device=self.device)
        z = F.normalize(z, dim=-1)
        return self.main(z.view(len(z), -1, 4, 4))

    def create_log_dict(self, prefix=""):
        images = self(16) + 0.5
        return dict(fake=wandb.Image(images))


class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=16):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
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

    def forward(self, input):
        assert input.shape[-2:] == (128, 128)
        output = self.main(input)
        return output.flatten(1)


if __name__ == "__main__":
    generator = Generator()
    print(generator(n=3).shape)
    print(Discriminator()(torch.randn(3, 3, 128, 128)).shape)
    print(generator.create_log_dict())
