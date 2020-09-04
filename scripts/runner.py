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

from scripts.dcgan import Generator, Discriminator


class Runner(torchzq.GANRunner):
    def __init__(self, parser=None):
        parser = parser or argparse.ArgumentParser()
        parser.add_argument("--root", type=str, default="data/processed")
        parser.add_argument("--capacity", type=int, default=16)
        parser.add_argument("--zdim", type=int, default=128)
        parser.add_argument("--ds-repeat", type=int, default=100)
        parser.add_argument("--base-size", type=int, default=144)
        parser.add_argument("--crop-size", type=int, default=128)
        parser.add_argument("--mixed-prob", type=float, default=0.9)
        parser.add_argument("--lr-mul", type=float, default=0.001)
        super().__init__(parser, name="foodgan", batch_size=8, save_every=1)

    def create_dataset(self):
        dataset = self.autofeed(
            ImageFolder,
            dict(
                transform=transforms.Compose(
                    [
                        transforms.Resize(self.args.base_size),
                        transforms.RandomCrop(self.args.crop_size),
                        transforms.ToTensor(),
                        transforms.Normalize(0.5, 1),
                    ]
                )
            ),
        )
        return ConcatDataset([dataset] * self.args.ds_repeat)

    def create_model(self):
        G = self.autofeed(
            Generator, dict(nc=3), dict(image_size="crop_size", latent_dim="zdim")
        )
        D = self.autofeed(Discriminator, dict(nc=3), dict(image_size="crop_size"))
        model = nn.ModuleList([G, D])
        print(model)
        return model

    def prepare_batch(self, batch):
        x, y = batch
        return x.to(self.args.device)

    def get_real(self, x):
        return x

    def g_feed(self, x):
        args = self.args
        n = len(x)
        mixed = self.training and np.random.rand() < args.mixed_prob
        s = self.noise_list(n, mixed)
        z = self.image_noise(n)
        self.fake = self.G(s, z)
        return self.fake

    def d_feed(self, x, _):
        return self.D(x)

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
