import math
import torch
import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, *layers, residual, activation):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.residual = residual
        self.activation = activation

    def forward(self, x):
        return self.activation(self.layers(x) + self.residual(x))


class PositionalEncoding(nn.Module):
    def __init__(self, dim, linear_scale=5000):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        exponents = torch.arange(half_dim, dtype=torch.float32)
        exponents = exponents / half_dim
        ω = linear_scale * torch.exp(-math.log(1e4) * exponents)
        ω = ω.unsqueeze(0)
        self.register_buffer("ω", ω, False)

    def forward(self, t):
        """
        Args:
            t: (b 1) or (b)
        Returns:
            pe: (b d)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        assert t.dim() == 2, f"but got shape: {t.shape}"
        ωt = self.ω * t
        return torch.cat([ωt.sin(), ωt.cos()], dim=-1)
