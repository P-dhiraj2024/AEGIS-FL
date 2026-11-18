"""Neurotoxin (Zhang et al., ICML 2022): backdoor training with the full trigger, every local
gradient projected onto the coordinates outside the top-`ratio` |previous global update|
(the ones benign clients change most), so the backdoor lives in rarely-updated weights."""
from __future__ import annotations

import torch


def mask_from_global_update(g_prev: torch.Tensor, ratio: float) -> torch.Tensor:
    """True = coordinate may be updated (bottom (1 - ratio) of |g_prev|)."""
    k = max(1, int(ratio * g_prev.numel()))
    m = torch.ones_like(g_prev, dtype=torch.bool)
    m[g_prev.abs().topk(k).indices] = False
    return m.float()
