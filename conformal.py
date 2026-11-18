"""Online conformal threshold (Eq. conformal) with adaptive alpha.

tau^t = Quantile({S_j}_{j=1}^n ; ceil((1 - alpha_t)(n + 1)) / n), reject i if S(s_i^t) > tau^t.
Adaptive conformal inference (Gibbs & Candes, NeurIPS 2021): alpha_{t+1} = alpha_t +
eps (alpha - err_t), err_t = fraction of this round's trusted scores above tau^t.
The calibration buffer is a FIFO of the most recent `size` trusted scores.
Several nonconformity scores are combined through their conformal p-values with a Bonferroni
correction: reject if min_j p_j <= alpha_t / J, so that P(benign rejected) <= alpha_t under
exchangeability (union bound); with J = 1 this is exactly the threshold rule above.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np


class OnlineConformal:
    def __init__(self, alpha: float, size: int, eps: float = 0.005, adaptive: bool = True):
        self.alpha, self.alpha_t, self.eps, self.adaptive = alpha, alpha, eps, adaptive
        self.buf: deque = deque(maxlen=size)

    def add(self, scores):
        self.buf.extend(float(s) for s in np.atleast_1d(scores))

    def threshold(self) -> float:
        n = len(self.buf)
        if n == 0:
            return math.inf
        a = min(max(self.alpha_t, 1e-6), 1.0)
        r = math.ceil((1 - a) * (n + 1))
        if r > n:
            return math.inf
        return float(np.sort(np.fromiter(self.buf, float))[r - 1])

    def pvalue(self, scores) -> np.ndarray:
        """Conformal p-value (1 + #{S_j >= s}) / (n + 1) of each score w.r.t. the buffer."""
        b = np.sort(np.fromiter(self.buf, float))
        s = np.atleast_1d(np.asarray(scores, float))
        n_ge = len(b) - np.searchsorted(b, s, side="left")
        return (1.0 + n_ge) / (len(b) + 1.0)

    def threshold_at(self, a: float) -> float:
        n = len(self.buf)
        r = math.ceil((1 - min(max(a, 1e-6), 1.0)) * (n + 1))
        return math.inf if n == 0 or r > n else float(np.sort(np.fromiter(self.buf, float))[r - 1])

    def update(self, err_t: float):
        if self.adaptive:
            self.alpha_t = self.alpha_t + self.eps * (self.alpha - err_t)
