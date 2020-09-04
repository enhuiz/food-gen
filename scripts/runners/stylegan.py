#!/usr/bin/env python3

import numpy as np
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchzq
from pathlib import Path
from torch.utils.data import ConcatDataset
from torchzq.parsing import listof
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torchsummary import summary

sys.path.append(".")

from scripts.models.stylegan import Generator
from scripts.models.dcgan import Discriminator
from scripts.runners.dcgan import AugWrapper, Runner as DCGANRunner


class Runner(DCGANRunner):
    def __init__(self, parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--lr-mul", type=float, default=0.1)
        parser.add_argument("--mixed-prob", type=float, default=0.9)
        parser.add_argument("--fmap-max", type=int, default=512)
        super().__init__(parser)

    def create_model(self):
        G = self.autofeed(
            Generator,
            mapping=dict(image_size="crop_size", latent_dim="zdim"),
        )
        D = AugWrapper(
            self.autofeed(
                Discriminator, dict(nc=3), mapping=dict(image_size="crop_size")
            ),
            self.args.crop_size,
            self.args.aug_prob,
        )
        model = nn.ModuleList([G, D])
        print(model)
        return model

    def noise(self, n):
        args = self.args
        return torch.randn(n, args.zdim).to(args.device)

    def noise_list(self, n, mixed):
        num_layers = self.G.num_layers
        if mixed:
            first = np.random.randint(1, num_layers - 1)
            second = num_layers - first
            z1 = (self.noise(n), first)
            z2 = (self.noise(n), second)
            ret = [z1, z2]
        else:
            z = (self.noise(n), num_layers)
            ret = [z]
        return ret

    def image_noise(self, n):
        size = self.args.crop_size
        device = self.args.device
        return torch.zeros(n, 1, size, size).uniform_(0.0, 1.0).to(device)

    def g_feed(self, x):
        args = self.args
        n = len(x)
        mixed = self.training and np.random.rand() < args.mixed_prob
        s = self.noise_list(n, mixed)
        z = self.image_noise(n)
        self.fake = self.G(s, z)
        return self.fake

    @staticmethod
    def save_image(images, *args, **kwargs):
        images = [image + 0.5 for image in images]
        save_image(images, *args, **kwargs)

    def update(self, batch):
        args = self.args
        super().update(batch)
        if not self.training:
            self.fakes += list(self.fake.detach().cpu())
            self.reals += list(batch.real.detach().cpu())
        elif self.step % args.vis_every == 0:
            path = Path(args.vis_dir, self.name, self.command, f"{self.step:06d}.png")
            path.parent.mkdir(exist_ok=True, parents=True)
            nrow = min(args.batch_size, 8)
            self.save_image(
                [*self.fake[:nrow], *self.get_real(batch)[:nrow]], path, nrow
            )

    def test(self):
        self.fakes = []
        self.reals = []
        super().test()
        args = self.args
        folder = Path(args.vis_dir, self.name, self.command)
        folder.mkdir(parents=True, exist_ok=True)
        self.save_image(torch.stack(self.fakes), Path(folder, "fakes.png"))
        self.save_image(torch.stack(self.reals), Path(folder, "reals.png"))


if __name__ == "__main__":
    Runner().run()
