"""Layer-wise random-projection sketches (Eq. sketch).

s_i = [R_1 dw_{i,1}/||dw_{i,1}||; ...; R_L dw_{i,L}/||dw_{i,L}||] in R^{kL},
R_l in R^{k x d_l} with i.i.d. N(0, 1/k) entries (Johnson-Lindenstrauss), drawn from a seed the
UAVs share and the clients do not know.  The seed is kept for `refresh_every` rounds (0 = the
whole deployment): a learned detector needs a projection that is fixed while it is in use,
because for a fresh Gaussian R the marginal law of R v/||v|| is N(0, I/k) for every v.
"""
from __future__ import annotations

import torch


class Sketcher:
    def __init__(self, layers: list[tuple[int, int]], k: int, seed: int, device):
        self.layers, self.k, self.device = layers, k, device
        self.set_seed(seed)

    def set_seed(self, seed: int):
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        self.R = [torch.randn(self.k, b - a, device=self.device, generator=g) / self.k ** 0.5
                  for a, b in self.layers]

    @property
    def dim(self) -> int:
        return self.k * len(self.layers)

    def __call__(self, U: torch.Tensor) -> torch.Tensor:
        """U [n, d] -> sketches [n, kL]."""
        out = []
        for (a, b), R in zip(self.layers, self.R):
            u = U[:, a:b]
            u = u / u.norm(dim=1, keepdim=True).clamp(min=1e-12)
            out.append(u @ R.T)
        return torch.cat(out, 1)
