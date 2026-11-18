"""FLAME (Nguyen et al., USENIX Security 2022), core.fl.aggregators.robust.flame
(HDBSCAN on cosine distances, median-norm clipping, Gaussian noise lambda * S)."""
from core.fl.aggregators.robust import flame


class Flame:
    needs_root = False

    def __init__(self, lam: float):
        self.lam = lam

    def aggregate(self, U, ctx):
        agg, keep = flame(U, noise_lambda=self.lam, generator=ctx["gen_cpu"], return_mask=True)
        return agg, ~keep
