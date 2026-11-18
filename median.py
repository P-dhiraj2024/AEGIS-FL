"""Coordinate-wise median (Yin et al., ICML 2018)."""
from core.fl.aggregators.robust import median


class Median:
    needs_root = False

    def aggregate(self, U, ctx):
        return median(U), None
