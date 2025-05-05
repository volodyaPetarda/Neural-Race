from typing import List

from torch import nn


class QNetwork(nn.Module):
    def __init__(self, hidden_sizes: List[int]):
        super().__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            if i != 0:
                layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i]))
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
