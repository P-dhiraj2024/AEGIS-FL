"""Pairwise-mask secure aggregation on the GPU (Bonawitz et al., CCS 2017, no dropout recovery).

Updates are fixed-point encoded (2^frac_bits) into Z_{2^48}; every pair (i, j), i < j, shares a
PRG stream (seeded per pair-row here; in deployment from a Diffie-Hellman key agreement), client
i adds the mask and client j subtracts it, so the masks cancel in the sum and the cloud only
sees masked rows.  The decoded sum equals the plain sum up to fixed-point rounding
(<= K 2^-(frac_bits+1) per coordinate), which `secure_sum` reports.
Same protocol as core.privacy.shamir.secure_sum (CPU, GF(2^31-1)), vectorised per client row.
"""
from __future__ import annotations

import torch

BITS = 48
MOD = 1 << BITS


def secure_sum(U: torch.Tensor, seed: int, frac_bits: int = 24) -> tuple[torch.Tensor, float]:
    """Sum of the rows of U through masked fixed-point messages. Returns (sum, max |error|)."""
    K, d = U.shape
    enc = torch.round(U.double() * (1 << frac_bits)).long() % MOD
    masked = enc.clone()
    g = torch.Generator(device=U.device)
    for i in range(K - 1):
        g.manual_seed(seed * 1_000_003 + i)
        M = torch.randint(0, MOD, (K - i - 1, d), device=U.device, generator=g, dtype=torch.long)
        masked[i] = (masked[i] + M.sum(0)) % MOD          # client i adds m_ij for all j > i
        masked[i + 1:] = (masked[i + 1:] - M) % MOD       # client j subtracts m_ij
    tot = masked.sum(0) % MOD
    tot = torch.where(tot >= MOD // 2, tot - MOD, tot).double() / (1 << frac_bits)
    err = float((tot - U.double().sum(0)).abs().max()) if K else 0.0
    return tot.to(U.dtype), err
