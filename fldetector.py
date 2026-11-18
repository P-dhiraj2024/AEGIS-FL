"""FLDetector (Zhang et al., KDD 2022), re-implemented (no official code vendored).

The cloud predicts each client's update from its previous one with an L-BFGS Hessian
approximation built from the last `window` global model / global update differences:
g_hat_i^t = g_i^{t'} + H^t (w^t - w^{t'}),  t' = client i's previous participation.
Suspicious score d_i^t = ||g_hat_i^t - g_i^t|| normalised by its sum over the round, averaged
over the client's last `window` participations.  A 1-D gap statistic decides whether the
round's scores form more than one cluster; if so, 2-means flags the high-score cluster and the
flagged clients are removed (excluded now and never scheduled again), as in the paper; the
remaining updates are averaged.  Adaptation to partial participation: t' is the client's own
previous round (the paper assumes full participation, t' = t - 1); a client without history is
not scored in that round.  g = -update is used as the gradient proxy.
"""
from __future__ import annotations

import numpy as np
import torch


def lbfgs_hvp(S: list, Y: list, v: torch.Tensor) -> torch.Tensor:
    """Compact L-BFGS approximation of H v (Byrd, Nocedal & Schnabel 1994), as in FLDetector; v: [d] or [d, n]."""
    Sk, Yk = torch.stack(S, 1), torch.stack(Y, 1)            # [d, m]
    StY, StS = Sk.T @ Yk, Sk.T @ Sk
    R = torch.triu(StY)
    L = StY - R
    sigma = (Y[-1] @ S[-1]) / (S[-1] @ S[-1]).clamp(min=1e-12)
    D = torch.diag(torch.diag(StY))
    M = torch.cat([torch.cat([sigma * StS, L], 1), torch.cat([L.T, -D], 1)], 0)
    p = torch.cat([Sk.T @ (sigma * v), Yk.T @ v])
    if not bool(torch.isfinite(M).all()) or not bool(torch.isfinite(p).all()):
        return torch.zeros_like(v)
    sol, info = torch.linalg.solve_ex(M, p)
    if int(info.max() if info.dim() else info) != 0 or not bool(torch.isfinite(sol).all()):
        sol = torch.linalg.pinv(M) @ p                     # singular history (e.g. a zero global update)
    return sigma * v - torch.cat([sigma * Sk, Yk], 1) @ sol


def kmeans1d(x: np.ndarray, k: int, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        lab = np.abs(x[:, None] - c[None]).argmin(1)
        new = np.array([x[lab == j].mean() if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return lab, c


def _wk(x, k):
    lab, c = kmeans1d(x, k)
    return max(float(((x - c[lab]) ** 2).sum()), 1e-12)


def gap_k(x: np.ndarray, rng: np.random.Generator, kmax: int = 4, B: int = 10) -> int:
    """Number of clusters by the gap statistic (Tibshirani et al., 2001)."""
    kmax = min(kmax, len(x) - 1)
    if kmax < 2 or not np.isfinite(x).all() or not np.isfinite(np.ptp(x)) or np.ptp(x) <= 1e-12:
        return 1
    gaps, sks = [], []
    for k in range(1, kmax + 1):
        ref = [np.log(_wk(rng.uniform(x.min(), x.max(), len(x)), k)) for _ in range(B)]
        gaps.append(np.mean(ref) - np.log(_wk(x, k)))
        sks.append(np.std(ref) * np.sqrt(1 + 1 / B))
    for k in range(kmax - 1):
        if gaps[k] >= gaps[k + 1] - sks[k + 1]:
            return k + 1
    return kmax


class FLDetector:
    needs_root = False

    def __init__(self, n_clients: int, d: int, window: int = 10, device="cuda"):
        self.N, self.win = n_clients, window
        self.prev_g = {}                  # client -> (round, fp16 gradient proxy)
        self.w_hist = {}                  # round -> global model
        self.S, self.Y = [], []
        self.last_w = self.last_gbar = None
        self.scores = {i: [] for i in range(n_clients)}
        self.banned = np.zeros(n_clients, bool)
        self.rng = np.random.default_rng(0)

    def eligible(self, rng):
        return ~self.banned

    def aggregate(self, U, ctx):
        t, clients, w = ctx["t"], ctx["clients"], ctx["w"]
        G = -U
        flagged = torch.zeros(len(U), dtype=torch.bool, device=U.device)
        if len(self.S) >= 2:
            d = torch.full((len(U),), float("nan"), device=U.device)
            ks = [k for k, c in enumerate(clients) if c in self.prev_g and self.prev_g[c][0] in self.w_hist]
            if ks:                                       # all clients' H (w - w_t') in one batched product
                V = torch.stack([w - self.w_hist[self.prev_g[clients[k]][0]] for k in ks], 1)
                P = torch.stack([self.prev_g[clients[k]][1].float() for k in ks], 1) + lbfgs_hvp(self.S, self.Y, V)
                d[ks] = (P - G[ks].T).norm(dim=0)
            has = ~torch.isnan(d)
            bad_num = has & ~torch.isfinite(d)          # non-finite prediction error: flag directly
            flagged |= bad_num
            ok = has & torch.isfinite(d)
            if int(ok.sum()) >= 3:
                dn = d[ok] / d[ok].sum()
                for c, v in zip(clients[ok.cpu().numpy()], dn.cpu().numpy()):
                    self.scores[c] = (self.scores[c] + [float(v) * int(ok.sum())])[-self.win:]
                cand = np.array([k for k, c in enumerate(clients) if self.scores[c]])
                if len(cand) >= 3:
                    sc = np.array([np.mean(self.scores[clients[k]]) for k in cand])
                    if gap_k(sc, self.rng) > 1:
                        lab, cen = kmeans1d(sc, 2)
                        bad = cand[lab == int(np.argmax(cen))]
                        flagged[torch.as_tensor(bad, device=U.device)] = True
            self.banned[clients[flagged.cpu().numpy()]] = True
        keep = ~flagged
        agg = U[keep].mean(0) if bool(keep.any()) else torch.zeros_like(U[0])
        # history for the next round
        for k, c in enumerate(clients):
            self.prev_g[c] = (t, G[k].half())
        self.w_hist[t] = w.clone()
        for r in [r for r in self.w_hist if r < t - 5 * self.win]:
            del self.w_hist[r]
        gbar = -agg
        if self.last_w is not None:
            self.S.append(w - self.last_w)
            self.Y.append(gbar - self.last_gbar)
            self.S, self.Y = self.S[-self.win:], self.Y[-self.win:]
        self.last_w, self.last_gbar = w.clone(), gbar.clone()
        return agg, flagged
