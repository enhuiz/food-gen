import torch
import torch.nn as nn
from operator import itemgetter


class Bottleneck(nn.Module):
    def __init__(self, dim_latent=128):
        super().__init__()
        self.linear = nn.Linear(dim_latent, dim_latent * 2)
        self._saved_for_later = {}

    def save_for_later(self, **kwargs):
        self._saved_for_later.update(kwargs)

    def forward(self, x, dim=-1):
        self._saved_for_later.clear()
        x = x.transpose(dim, -1)
        μ, logσ = self.linear(x).chunk(2, dim=-1)
        self.save_for_later(μ=μ, logσ=logσ)
        z = μ + logσ.exp() * torch.randn_like(x)
        z = z.transpose(dim, -1)
        return z

    @property
    def kl_prior(self):
        μ, logσ = itemgetter("μ", "logσ")(self._saved_for_later)
        return -0.5 * torch.sum(1 + 2 * logσ - μ.pow(2) - (2 * logσ).exp())
