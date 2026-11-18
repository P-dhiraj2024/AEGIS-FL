"""VAE on benign sketches (Eq. elbo) and the anomaly score (Eq. score).

S(s) = ||s - s_hat||^2 + omega (mu_z - mu_bar)^T Sigma^{-1} (mu_z - mu_bar),
s_hat = decoder(mu_z) (posterior mean, deterministic at scoring time); mu_bar and Sigma are the
mean and (ridge-regularised) covariance of the latent means of the training buffer.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, dim: int, latent: int = 16, hidden: int = 128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.dec = nn.Sequential(nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, dim))
        self.register_buffer("mu_bar", torch.zeros(latent))
        self.register_buffer("prec", torch.eye(latent))
        self.register_buffer("scale", torch.ones(dim))   # per-coordinate input standardisation
        self.register_buffer("shift", torch.zeros(dim))

    def encode(self, s):
        h = self.enc((s - self.shift) / self.scale)
        return self.mu(h), self.logvar(h).clamp(-8, 8)

    def neg_elbo(self, s: torch.Tensor) -> torch.Tensor:
        """-ELBO per sample (Gaussian decoder with unit variance, up to constants)."""
        mu, lv = self.encode(s)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        rec = (self.dec(z) - (s - self.shift) / self.scale).pow(2).sum(1)
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(1)
        return rec + kl

    def score(self, s: torch.Tensor, omega: float = 1.0) -> torch.Tensor:
        mu, _ = self.encode(s)
        rec = (self.dec(mu) - (s - self.shift) / self.scale).pow(2).sum(1)
        dz = mu - self.mu_bar
        maha = ((dz @ self.prec) * dz).sum(1)
        return rec + omega * maha

    @torch.no_grad()
    def fit_input_stats(self, s: torch.Tensor, per_coord: bool):
        """Global scale (std of all coordinates) or per-coordinate mean/std of the benign buffer."""
        if per_coord:
            self.shift.copy_(s.mean(0))
            self.scale.copy_(s.std(0).clamp(min=1e-3))
        else:
            self.shift.zero_()
            self.scale.fill_(float(s.std()) + 1e-8)

    @torch.no_grad()
    def fit_latent_stats(self, s: torch.Tensor, ridge: float = 1e-3):
        mu, _ = self.encode(s)
        mu = torch.nan_to_num(mu)
        self.mu_bar.copy_(mu.mean(0))
        c = torch.cov(mu.T) if len(mu) > 1 else torch.eye(mu.shape[1], device=mu.device)
        c = torch.nan_to_num(c) + ridge * torch.eye(len(c), device=c.device)
        inv, info = torch.linalg.inv_ex(c)
        if int(info) != 0 or not bool(torch.isfinite(inv).all()):   # singular (e.g. collapsed latent under DP noise)
            inv = torch.linalg.pinv(c, hermitian=True)
        self.prec.copy_(inv)
