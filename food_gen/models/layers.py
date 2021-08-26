import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, *layers, residual, activation):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.residual = residual
        self.activation = activation

    def forward(self, x):
        return self.activation(self.layers(x) + self.residual(x))
