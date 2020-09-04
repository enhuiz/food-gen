import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from linear_attention_transformer import ImageLinearAttention


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Lambda(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Attention(nn.Sequential):
    def __init__(self, nc):
        super().__init__(
            Residual(
                Rezero(ImageLinearAttention(nc, key_dim=nc, value_dim=nc, heads=4))
            ),
            Residual(
                Rezero(
                    nn.Sequential(
                        nn.Conv2d(nc, nc * 2, 1),
                        nn.LeakyReLU(),
                        nn.Conv2d(nc * 2, nc, 1),
                    )
                )
            ),
        )

        class CCBN(nn.Module):
            zdim = 20

    cdim = 128

    def __init__(self, out_channels, getter, **kwargs):
        super().__init__()
        self.getter = getter
        self.gain = nn.Linear(self.zdim + self.cdim, out_channels)
        self.bias = nn.Linear(self.zdim + self.cdim, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels, affine=False, **kwargs)

    def forward(self, x):
        z = self.getter()
        gain = (1 + self.gain(z)).view(len(z), -1, 1, 1)
        bias = self.bias(z).view(len(z), -1, 1, 1)
        out = self.batch_norm(x)
        return out * gain + bias


class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return self.main(x) + self.shortcut(x)


class ResBlockUp(ResBlock):
    def __init__(self, in_channels, out_channels, getter):
        super().__init__()
        self.getter = getter
        self.main = nn.Sequential(
            CCBN(in_channels, getter),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            CCBN(out_channels, getter),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1),
        )


class ResBlockDown(ResBlock):
    def __init__(self, in_channels, out_channels, is_first=False):
        super().__init__()

        self.main = nn.Sequential(
            nn.Identity() if is_first else nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.AvgPool2d(2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.AvgPool2d(2),
        )


class Generator(nn.Module):
    factors = [16, 16, 8, 4, 2, 1]  # 128

    zdim = 120
    cdim = 128

    def __init__(self, ngf=16, nc=3):
        super().__init__()
        ncs = [factor * ngf for factor in self.factors] + [nc]
        self.main = nn.Sequential(
            nn.Linear(self.zdim // 6 + self.cdim, ncs[0] * 4 * 4),
            Lambda(lambda x: x.view(len(x), ncs[0], 4, 4)),
            *[
                ResBlockUp(ic, oc, lambda: self.z[i + 1])
                for i, (ic, oc) in enumerate(pairwise(ncs[1:-2]))
            ],
            Attention(ncs[-3]),
            ResBlockUp(ncs[-3], ncs[-2], lambda: self.z[-1]),
            nn.BatchNorm2d(ncs[-2]),
            nn.ReLU(),
            nn.Conv2d(ncs[-2], ncs[-1], 3, padding=1),
            nn.Tanh(),
        )

        def forward(self, z, c=None):
            if c is None:
                c = torch.zeros(len(z), self.cdim).to(z.device)

        self.z = [torch.cat([zi, c], dim=-1) for zi in z.chunk(6, dim=-1)]
        return self.main(self.z[0])


class Discriminator(nn.Module):
    # 128
    factors = [16, 16, 8, 4, 2, 1]

    def __init__(self, ndf=16, nc=3):
        super().__init__()
        ncs = [factor * ndf for factor in self.factors] + [nc]
        self.main = nn.Sequential(
            ResBlockDown(ncs[-1], ncs[-2], is_first=True),
            ResBlockDown(ncs[-2], ncs[-3]),
            Attention(ncs[-3]),
            *[ResBlockDown(ic, oc) for ic, oc in pairwise(ncs[1:-2])],
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
        )
        self.linear = nn.Linear(ncs[0], 1)

    def forward(self, x, c=0):
        h = self.main(x)
        return self.linear(h) + (h * c).sum(dim=1, keepdim=True)


if __name__ == "__main__":
    from torchsummary import summary

    summary(Generator(), (120,), device="cpu")
    summary(Discriminator(), (3, 128, 128), device="cpu")
