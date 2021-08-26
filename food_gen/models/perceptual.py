import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.models import vgg19


class Vgg19(nn.ModuleList):
    def __init__(self):
        super().__init__()
        vgg_layers = list(vgg19(pretrained=True).features)

        self.slices = nn.ModuleList(
            [
                nn.Sequential(*vgg_layers[s:e])
                for s, e in [
                    [0, 2],
                    [2, 7],
                    [7, 12],
                    [12, 21],
                    [21, 30],
                ]
            ]
        )

        self.mean = nn.Parameter(
            data=rearrange(torch.tensor([0.485, 0.456, 0.406]), "c -> 1 c 1 1"),
            requires_grad=False,
        )

        self.std = nn.Parameter(
            data=rearrange(torch.tensor([0.229, 0.224, 0.225]), "c -> 1 c 1 1"),
            requires_grad=False,
        )

        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = []
        for slice in self.slices:
            x = slice(x)
            out.append(x)
        return out


class PerceptualFeatures(nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25]):
        super().__init__()
        self.scales = scales
        self.vgg19 = Vgg19()

    def pyramide(self, x):
        for scale in self.scales:
            yield F.interpolate(x, scale_factor=scale, recompute_scale_factor=True)

    def forward(self, x):
        out = []
        x = self.pyramide(x)
        for xi in x:
            xi = self.vgg19(xi)
            for xil in xi:
                out.append(xil.flatten(1))
        return torch.cat(out, dim=1)
