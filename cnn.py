"""Small CNNs for 28x28 grey-scale (Fashion-MNIST, FEMNIST) and 32x32 colour (CIFAR-10) images.

No batch normalisation: client models are trained in parallel with torch.func.vmap and the
aggregated parameter vector must contain every model state (no running statistics)."""
import torch.nn as nn


class CNN28(nn.Module):
    """conv5(16)-pool-conv5(32)-pool-fc128-fc(C)  (LEAF FEMNIST-style CNN, narrower)."""

    def __init__(self, n_classes: int, in_ch: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 16, 5, padding=2)
        self.c2 = nn.Conv2d(16, 32, 5, padding=2)
        self.f1 = nn.Linear(32 * 7 * 7, 128)
        self.f2 = nn.Linear(128, n_classes)
        self.act, self.pool = nn.ReLU(), nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))
        x = self.pool(self.act(self.c2(x)))
        return self.f2(self.act(self.f1(x.flatten(-3))))


class CNN32(nn.Module):
    """3 x (conv3-relu-pool) with 32/64/64 channels, fc128, fc(C)."""

    def __init__(self, n_classes: int, in_ch: int = 3):
        super().__init__()
        self.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 64, 3, padding=1)
        self.f1 = nn.Linear(64 * 4 * 4, 128)
        self.f2 = nn.Linear(128, n_classes)
        self.act, self.pool = nn.ReLU(), nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(self.act(self.c1(x)))
        x = self.pool(self.act(self.c2(x)))
        x = self.pool(self.act(self.c3(x)))
        return self.f2(self.act(self.f1(x.flatten(-3))))
