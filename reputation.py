"""Decaying reputation (Eq. rep): rho_i^{t+1} = beta rho_i^t + (1 - beta) 1{S(s_i^t) <= tau^t}.

Clients with rho_i < rho_min are not scheduled, except with probability p_probe per round
(probation), so that an honest client hit by a run of false rejections can recover.
"""
from __future__ import annotations

import numpy as np


class Reputation:
    def __init__(self, n: int, beta: float, rho_min: float, rho0: float, p_probe: float, rho_trust: float):
        self.rho = np.full(n, rho0)
        self.beta, self.rho_min, self.p_probe, self.rho_trust = beta, rho_min, p_probe, rho_trust

    def update(self, clients: np.ndarray, accepted: np.ndarray):
        self.rho[clients] = self.beta * self.rho[clients] + (1 - self.beta) * accepted.astype(float)

    def eligible(self, rng: np.random.Generator) -> np.ndarray:
        return (self.rho >= self.rho_min) | (rng.random(len(self.rho)) < self.p_probe)

    def trusted(self, clients: np.ndarray) -> np.ndarray:
        return self.rho[clients] >= self.rho_trust
