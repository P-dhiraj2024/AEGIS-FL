"""Defence-aware (adaptive) attacker against AEGIS-FL.

Threat model (Sec. III-D): the colluding attackers know the aggregation rule and the detector
architecture (layer-wise sketch, the feature set -- per-layer log-norm and collusion features
when AEGIS-FL uses them --, VAE score, conformal threshold) and observe the round's benign
updates, but not the UAVs' projection seed or the detector weights.  They therefore build a
surrogate: their own projection (independent seed), the same features, their own VAE trained on
the benign updates they observe (centred on the sketch of the benign mean, their estimate of
the trusted reference), retrained every `retrain_every` rounds.

Crafting (mimicry + perturbation).  Attacker j starts from a synthetic benign-looking update
b_j = mu + xi_j, xi_j ~ N(0, diag(sigma^2)) with the observed benign per-coordinate spread (benign
norm, low similarity to any observed update, no duplicates among attackers) and sends m_j(lambda) = b_j + lambda p with p = -sign(mu) * sigma (the "inverse sign" direction of
Shejwalkar & Houmansadr, NDSS 2021).  lambda is the largest common value (bisection) for which
every m_j's surrogate score stays below the `target_quantile` of the surrogate scores of this
round's benign updates (and, when AEGIS-FL runs its alignment / collusion tests, whose alignment
with the benign mean and maximum similarity to other updates stay on the benign side of the
median), i.e. the most harmful perturbation the surrogate still accepts.
"""
from __future__ import annotations

import torch

from ..defense.gan_redteam import train_detector
from ..defense.sketch import Sketcher
from ..defense.vae_detector import VAE


class AdaptiveAttacker:
    def __init__(self, layers, k: int, latent: int, seed: int, device, cfg: dict, dcfg: dict | None = None):
        dcfg = dcfg or {}
        self.norm_feat = bool(dcfg.get("norm_features", False))
        self.dup_feat = bool(dcfg.get("dup_feature", False))
        self.per_coord = bool(dcfg.get("per_coord_scale", False))
        self.align_test = bool(dcfg.get("align_test", False))
        self.coll_test = bool(dcfg.get("collusion_test", False))
        self.norm_test = bool(dcfg.get("norm_test", False))
        self.dcfg = dcfg
        self.layers = layers
        self.sk = Sketcher(layers, k, seed=seed, device=device)
        dim = self.sk.dim + (len(layers) if self.norm_feat else 0) + (1 if self.dup_feat else 0)
        self.vae = VAE(dim, latent).to(device)
        self.opt = torch.optim.Adam(self.vae.parameters(), lr=1e-3)
        self.buf: list[torch.Tensor] = []
        self.cfg = cfg
        self.L = len(layers)
        self.trained_at = -10 ** 9
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed + 17)
        self.last_lambda = 0.0

    def _ln(self, X):
        return torch.stack([X[:, a:b].norm(dim=1).clamp(min=1e-12).log() for a, b in self.layers], 1)

    def _features(self, X, benign, mu, pool):
        """Surrogate detector input of rows X; `pool` = all sketches the UAVs would compare against."""
        sx = self.sk(X)
        f = [sx - self.sk(mu[None])]
        if self.norm_feat:
            f.append(self.dcfg.get("norm_scale", 1.0) * (self._ln(X) - self._ln(benign).median(0).values))
        if self.dup_feat:
            A = torch.nn.functional.normalize(sx, dim=1)
            P = torch.nn.functional.normalize(pool, dim=1)
            cc = A @ P.T
            cc[cc > 1 - 1e-6] = -1.0                       # ignore the row itself
            f.append(self.dcfg.get("dup_scale", 1.0) * cc.max(1).values[:, None])
        return torch.cat(f, 1)

    def craft(self, benign: torch.Tensor, n_m: int, t: int) -> torch.Tensor:
        mu, sd = benign.mean(0), benign.std(0, unbiased=False)
        s = self._features(benign, benign, mu, self.sk(benign))
        self.buf.append(s.detach())
        self.buf = self.buf[-self.cfg.get("buffer_rounds", 20):]
        if t - self.trained_at >= self.cfg.get("retrain_every", 10):
            data = torch.cat(self.buf)
            self.vae.fit_input_stats(data, self.per_coord)
            steps = 300 if self.trained_at < 0 else 100
            train_detector(self.vae, None, self.opt, None, data, None, self.L, steps, 0.0, 0.0, False, self.gen)
            with torch.no_grad():
                self.vae.fit_latent_stats(data)
            self.trained_at = t
        with torch.no_grad():
            thr = torch.quantile(self.vae.score(s), self.cfg.get("target_quantile", 0.5))
            p = -torch.sign(mu) * sd
            # synthetic benign-looking bases: benign mean + Gaussian noise with the benign per-coordinate
            # spread (benign norm, low similarity to any observed update, no duplicates)
            base = mu + torch.randn((n_m, len(mu)), device=benign.device, generator=self.gen) * sd

            def align(X):
                return torch.stack([torch.nn.functional.cosine_similarity(X[:, a:b], mu[None, a:b], dim=1)
                                    for a, b in self.layers], 1).mean(1)

            def nncos(X, pool):
                A = torch.nn.functional.normalize(self.sk(X), dim=1)
                P = torch.nn.functional.normalize(pool, dim=1)
                cc = A @ P.T
                cc[cc > 1 - 1e-6] = -1.0
                return cc.max(1).values

            sb = self.sk(benign)
            al_thr = torch.quantile(align(benign), 1 - self.cfg.get("target_quantile", 0.5))
            nn_thr = torch.quantile(nncos(benign, sb), self.cfg.get("target_quantile", 0.5))
            lnb = self._ln(benign)
            nrm_thr = torch.quantile((lnb - lnb.median(0).values).max(1).values, self.cfg.get("target_quantile", 0.5))

            def ok(lam):
                M = base + lam * p
                pool = torch.cat([sb, self.sk(M)])
                good = bool(self.vae.score(self._features(M, benign, mu, pool)).max() <= thr)
                if good and self.norm_test:           # no layer norm above a typical benign update
                    good = bool((self._ln(M) - self._ln(benign).median(0).values).max(1).values.max()
                                <= nrm_thr)
                if good and self.align_test:          # stay as aligned as a typical benign update
                    good = bool(align(M).min() >= al_thr)
                if good and self.coll_test:           # and no more similar to others than a typical one
                    good = bool(nncos(M, pool).max() <= nn_thr)
                return good

            lo, hi = 0.0, 1.0
            if not ok(0.0):
                self.last_lambda = 0.0
                return base.clone()
            while ok(hi) and hi < 1e4:
                lo, hi = hi, hi * 2
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ok(mid) else (lo, mid)
            self.last_lambda = lo
            return base + lo * p
