"""Cloud tier: secure aggregation of the UAV-accepted updates + Renyi-DP Gaussian noise.

Each accepted update is clipped to C at its UAV, the UAVs forward masked fixed-point messages
(secure_agg.secure_sum) and each of the U UAVs adds Gaussian noise N(0, sigma^2 C^2 / U), so the
cloud receives sum_i clip(u_i) + N(0, sigma^2 C^2 I).  sigma is calibrated with the RDP
accountant of the sampled Gaussian mechanism (core.privacy.rdp) for (epsilon, delta) after T
rounds with participation rate q = K / N (user-level DP).
"""
from __future__ import annotations

import math

import torch

from core.privacy.rdp import calibrate_sigma, epsilon
from .secure_agg import secure_sum


class Cloud:
    def __init__(self, cfg: dict, q: float, rounds: int):
        pc = cfg["privacy"]
        self.secure = pc.get("secure_agg", "pairwise") == "pairwise"
        self.eps, self.delta = float(pc.get("dp_epsilon", math.inf)), float(pc.get("dp_delta", 1e-5))
        self.sigma = 0.0 if not math.isfinite(self.eps) else calibrate_sigma(self.eps, self.delta, q, rounds)
        self.q, self.T = q, rounds
        self.clip = None
        self.last_err = 0.0

    def eps_spent(self, t: int) -> float:
        return math.inf if self.sigma == 0 else epsilon(self.q, self.sigma, t, self.delta)

    def aggregate(self, U: torch.Tensor, clip: float, seed: int, n_uavs: int, gen: torch.Generator) -> torch.Tensor:
        """Mean of clipped accepted updates via secure sum (+ DP noise if epsilon < inf)."""
        if len(U) == 0:
            return torch.zeros(U.shape[1], device=U.device)
        Uc = U * torch.clamp(clip / U.norm(dim=1).clamp(min=1e-12), max=1.0)[:, None]
        if self.secure:
            s, self.last_err = secure_sum(Uc, seed)
        else:
            s = Uc.sum(0)
        if self.sigma > 0:
            noise = sum(torch.randn(s.shape, device=s.device, generator=gen) for _ in range(n_uavs))
            s = s + noise * (self.sigma * clip / math.sqrt(n_uavs))
        return s / len(U)
