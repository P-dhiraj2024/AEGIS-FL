"""Krum (Blanchard et al., NeurIPS 2017), core.fl.aggregators.robust.krum."""
import torch

from core.fl.aggregators.robust import krum_scores


class Krum:
    needs_root = False

    def aggregate(self, U, ctx):
        i = int(krum_scores(U, ctx["f"]).argmin())
        flagged = torch.ones(len(U), dtype=torch.bool, device=U.device)
        flagged[i] = False
        return U[i].clone(), flagged
