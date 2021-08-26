import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchzq

from .base import Runner as BaseRunner
from ..models.encoders import VanillaEncoder
from ..models.vae.bottleneck import Bottleneck
from ..models.perceptual import PerceptualFeatures


class Runner(BaseRunner):
    def __init__(self, β: int = 1, perceptual: bool = False, **kwargs):
        super().__init__(**kwargs)

    def create_model(self):
        self.encoder = VanillaEncoder()
        self.bottleneck = Bottleneck()
        self.generator = self.create_generator()
        self.perceptual = PerceptualFeatures([0.5, 0.25])
        self.perceptual.to(self.args.device)
        return nn.Sequential(self.encoder, self.bottleneck, self.generator)

    def training_step(self, real, _):
        args = self.args

        h = self.encoder(real)
        z = self.bottleneck(h)
        fake = self.generator(z=z)

        if args.perceptual:
            p_fake = self.perceptual(fake)
            p_real = self.perceptual(real)
            numel = p_fake.shape[1]
            l1_loss = F.l1_loss(p_fake, p_real)
        else:
            numel = np.prod(fake.shape[1:])
            l1_loss = F.l1_loss(fake, real)

        kl_loss = args.β / numel * self.bottleneck.kl_prior.mean()

        loss = l1_loss + kl_loss

        stat_dict = {
            "loss/l1": l1_loss.item(),
            "loss/kl": kl_loss.item(),
        }

        if self.global_step % args.demo_every == 1:
            with torch.no_grad():
                z = self.bottleneck(self.encoder(real[:16]))
            self.logger.log(
                self.generator.create_log_dict(z=z),
                self.global_step,
            )

        return loss, stat_dict


def main():
    torchzq.start(Runner)
