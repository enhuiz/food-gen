import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchzq

from .base import Runner as BaseRunner
from ..models.generators import VanillaGenerator
from ..models.gan.discriminator import Discriminator


class AugWrapper(nn.Module):
    def __init__(self, D, prob):
        super().__init__()
        self.D = D
        self.prob = prob

    @staticmethod
    def random_crop_and_resize(images, scale):
        size = images.shape[-1]

        new_size = int(size * scale)

        dsize = size - new_size

        h0 = int(np.random.random() * dsize)
        h1 = h0 + new_size

        w0 = int(np.random.random() * dsize)
        w1 = w0 + new_size

        cropped = images[..., h0:h1, w0:w1]
        cropped = cropped.clone()

        return F.interpolate(
            cropped,
            size=(size, size),
            mode="bilinear",
            align_corners=True,
        )

    @staticmethod
    def random_hflip(tensor, prob):
        if prob > np.random.random():
            return tensor
        return torch.flip(tensor, dims=(3,))

    def forward(self, images, detach=False):
        if np.random.random() < self.prob:
            random_scale = np.random.uniform(0.75, 0.95)
            images = self.random_hflip(images, prob=0.5)
            images = self.random_crop_and_resize(images, scale=random_scale)

        return self.D(images)


class Runner(BaseRunner):
    def __init__(
        self,
        d_lr: float = 1e-4,
        g_lr: float = 1e-4,
        aug_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def create_optimizers(self):
        args = self.args
        d_optimizer = torch.optim.Adam(self.discriminator.parameters(), args.d_lr)
        g_optimizer = torch.optim.Adam(self.generator.parameters(), args.g_lr)
        return [d_optimizer, g_optimizer]

    def create_model(self):
        self.generator = VanillaGenerator()
        self.discriminator = Discriminator()
        return nn.ModuleDict(
            dict(
                generator=self.generator,
                discriminator=self.discriminator,
            )
        )

    def clip_grad_norm(self, optimizer_idx):
        args = self.args
        return nn.utils.clip_grad_norm_(
            self.discriminator.parameters()
            if optimizer_idx == 0
            else self.generator.parameters(),
            args.grad_clip_thres or 1e9,
        )

    def training_step(self, real, optimizer_idx):
        args = self.args
        if optimizer_idx == 0:
            with torch.no_grad():
                fake = self.generator(n=len(real))

            loss_fake = F.relu(1 - self.discriminator(fake)).mean()
            loss_real = F.relu(1 + self.discriminator(real)).mean()

            loss = loss_fake + loss_real

            stat_dict = {
                "loss/d/fake": loss_fake.item(),
                "loss/d/real": loss_fake.item(),
            }

            if self.global_step % args.demo_every == 1:
                self.logger.log(self.generator.create_log_dict(), self.global_step)

        elif optimizer_idx == 1:
            fake = self.generator(n=len(real))

            params = [p for p in self.discriminator.parameters() if p.requires_grad]

            for p in params:
                p.requires_grad_(False)

            loss_fake = self.discriminator(fake).mean()

            for p in params:
                p.requires_grad_(True)

            loss = loss_fake

            stat_dict = {
                "loss/g/fake": loss_fake.item(),
            }
        else:
            assert False, "Too many optimizers!"

        return loss, stat_dict


def main():
    torchzq.start(Runner)
