"""Coordinate-wise trimmed mean (Yin et al., ICML 2018); trim fraction beta = f / K."""
from core.fl.aggregators.robust import trimmed_mean


class TrimmedMean:
    needs_root = False

    def aggregate(self, U, ctx):
        return trimmed_mean(U, beta=min(ctx["f"] / len(U), 0.45)), None
