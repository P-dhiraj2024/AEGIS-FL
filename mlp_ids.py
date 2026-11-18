"""MLP for tabular IoT intrusion-detection / sensor datasets (CICIoT2023, Edge-IIoTset, SensIT)."""
import torch.nn as nn


class MLPIDS(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden=(128, 64)):
        super().__init__()
        layers, d = [], in_dim
        for i, h in enumerate(hidden):
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
