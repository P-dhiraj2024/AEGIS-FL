"""FreqFed (Fereidooni et al., NDSS 2024), re-implemented (no official code vendored).

Every weight tensor of a client update is reshaped to a matrix (out, rest), transformed with an
orthonormal 2-D DCT-II and its low-frequency block (first ceil(out/2) x ceil(rest/2)
coefficients) is kept; the concatenated low-frequency vectors are clustered with HDBSCAN on
cosine distance (min cluster size K/2 + 1) and the updates of the largest cluster are averaged.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def dct_matrix(n: int, device) -> torch.Tensor:
    k = torch.arange(n, device=device, dtype=torch.float32)
    D = torch.cos(math.pi * (2 * k[None] + 1) * k[:, None] / (2 * n)) * math.sqrt(2 / n)
    D[0] /= math.sqrt(2)
    return D


class FreqFed:
    needs_root = False

    def __init__(self, fm):
        self.mats = []
        off = 0
        for name, shape, n in zip(fm.names, fm.shapes, fm.numels):
            if len(shape) >= 2:
                rows, cols = shape[0], n // shape[0]
                self.mats.append((off, rows, cols))
            off += n
        self._D = {}

    def _dct(self, n, device):
        if (n, device) not in self._D:
            self._D[(n, device)] = dct_matrix(n, device)
        return self._D[(n, device)]

    def low_freq(self, U: torch.Tensor) -> torch.Tensor:
        out = []
        for off, r, c in self.mats:
            M = U[:, off:off + r * c].reshape(len(U), r, c)
            F = self._dct(r, U.device) @ M @ self._dct(c, U.device).T
            out.append(F[:, : math.ceil(r / 2), : math.ceil(c / 2)].reshape(len(U), -1))
        return torch.cat(out, 1)

    def aggregate(self, U, ctx):
        from sklearn.cluster import HDBSCAN

        K = len(U)
        L = torch.nn.functional.normalize(self.low_freq(U), dim=1)
        dist = (1 - L @ L.T).clamp(min=0).double().cpu().numpy()
        lab = HDBSCAN(min_cluster_size=K // 2 + 1, min_samples=1, metric="precomputed",
                      allow_single_cluster=True, copy=True).fit(dist).labels_
        if (lab >= 0).any():
            keep = torch.as_tensor(lab == np.bincount(lab[lab >= 0]).argmax(), device=U.device)
        else:
            keep = torch.ones(K, dtype=torch.bool, device=U.device)
        return U[keep].mean(0), ~keep
