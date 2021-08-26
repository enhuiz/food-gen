import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchzq

from .base import Runner as BaseRunner
from ..models.encoders import VanillaEncoder
from ..models.generators import VanillaGenerator
from ..models.vae.bottleneck import Bottleneck


class Runner(BaseRunner):
    def __init__(self, β=1, **kwargs):
        super().__init__(**kwargs)

    def create_model(self):
        self.encoder = VanillaEncoder()
        self.bottleneck = Bottleneck()
        self.generator = VanillaGenerator()
        return nn.Sequential(self.encoder, self.bottleneck, self.generator)

    def training_step(self, real, _):
        args = self.args

        h = self.encoder(real)
        z = self.bottleneck(h)
        fake = self.generator(z=z)

        l1_loss = F.l1_loss(fake, real)
        kl_loss = args.β / np.prod(fake.shape[1:]) * self.bottleneck.kl_prior.mean()

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
