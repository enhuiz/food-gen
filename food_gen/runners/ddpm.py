import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchzq
from einops import repeat

from .base import Runner as BaseRunner
from ..models.encoders import VanillaEncoder
from ..models.layers import PositionalEncoding


def compute_κ(t, β0, β1):
    # continous noise level (α_sqrt_bar)
    # from: https://openreview.net/pdf/ef0eadbe07115b0853e964f17aa09d811cd490f1.pdf
    return torch.exp(-0.25 * t ** 2 * (β1 - β0) - 0.5 * t * β0)


def _unsqueeze(t, *, dim=-1, n=1):
    if n <= 0:
        return t
    return _unsqueeze(t.unsqueeze(dim=dim), n=n - 1)


def unsqueeze_like(a, b, dim=-1):
    return _unsqueeze(a, dim=dim, n=b.dim() - a.dim())


class Runner(BaseRunner):
    def __init__(
        self,
        β0: float = 1e-3,
        β1: float = 10,
        dim_pe: int = 128,
        eval_steps: int = 100,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @property
    def eval_κ(self):
        """Discrete κ for evaluation"""
        args = self.args
        β = torch.linspace(args.β0, args.β1, args.eval_steps)
        return (1 - β * 1e-3).cumprod(0).sqrt()

    def create_model(self):
        args = self.args
        self.encoder = VanillaEncoder()
        self.generator = self.create_generator()
        self.κ_encoder = PositionalEncoding(args.dim_pe)
        self.proj = nn.Linear(256, 128)
        return nn.ModuleList(
            [
                self.encoder,
                self.κ_encoder,
                self.proj,
                self.generator,
            ]
        )

    def compute_κ(self, t):
        args = self.args
        return compute_κ(t, args.β0, args.β1)

    def forward(self, x, κ):
        κ = κ.flatten(1)
        z = self.encoder(x)
        z = self.proj(torch.cat([z, self.κ_encoder(κ)], dim=-1))
        return self.generator(z)

    def inverse(self, xi, κi, κim1, last=False):
        κi = unsqueeze_like(κi, xi)
        κim1 = unsqueeze_like(κim1, xi)
        βi = 1 - (κi / κim1).pow(2)

        κim1 = κi / (1 - βi).sqrt()
        σ2i = 1 - κi.pow(2)
        σ2im1 = 1 - κim1.pow(2)
        σi = σ2i.sqrt()

        ε_hat = self.forward(xi, κi)

        score = -ε_hat / σi
        xim1 = (xi + βi * score) / (1 - βi).sqrt()

        if not last:
            z = torch.randn_like(xim1)
            xim1 += (σ2im1 / σ2i * βi).sqrt() * z

        return xim1

    def training_step(self, real, _):
        args = self.args

        t = torch.rand(len(real), device=real.device)
        κ = unsqueeze_like(self.compute_κ(t), real)

        σ2 = 1 - κ.pow(2)
        σ = σ2.sqrt()

        ε = torch.randn_like(real, device=args.device)
        perturbed_real = κ * real + σ * ε

        ε_hat = self.forward(perturbed_real, κ)

        loss = F.l1_loss(ε_hat, ε)
        stat_dict = {"loss": loss.item()}

        del κ

        if self.global_step % args.demo_every == 1:
            κ = self.eval_κ.to(args.device)

            xT = torch.randn_like(real)[:16]

            xs = [xT]

            # expand such that every sample can have different κ
            κ = repeat(κ, "t -> t b", b=len(xT))

            x = xT
            for i in tqdm.tqdm(list(reversed(range(len(κ))))):
                κi = κ[i]
                κim1 = κ[i - 1] if i > 1 else torch.full_like(κ[i], 1)
                x = self.inverse(x, κi, κim1, last=i == 0)
                xs.append(x)

            log_dict = {"fake": self.logger.Image(xs[-1])}
            self.logger.log(log_dict, self.global_step)

        return loss, stat_dict


def main():
    torchzq.start(Runner)
