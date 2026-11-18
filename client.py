"""Batched local training of many heterogeneous clients at once (torch.func.vmap).

Every scheduled client k starts from the global model w^t and runs `steps[k]` mini-batch SGD
steps on its own data (IoT devices run fewer steps than mobiles, see uav_edge.DEVICE_TYPES).
All clients share one vmapped forward/backward per step; a client whose budget is used up is
masked out.  Data-poisoning behaviour (label flipping, DBA / Neurotoxin backdoor training) is
applied per client inside the same batch.

The same routine trains the server's root-data updates (FLTrust g0, AEGIS reference shards):
those "virtual clients" draw their mini-batches from an explicit index table into the root set.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.func import grad_and_value, vmap

from ..data import FedData, stamp_trigger


@dataclass
class Poison:
    """Per-client data-poisoning flags for one training call (all [K] tensors)."""
    label_flip: torch.Tensor
    backdoor: torch.Tensor
    part: torch.Tensor            # DBA local trigger part 0..3, or 4 = full trigger
    nt_mask: torch.Tensor | None = None   # Neurotoxin coordinate mask [d] (True = may be updated)
    nt_rows: torch.Tensor | None = None   # [K] clients projected with nt_mask
    frac: float = 0.5


class BatchedTrainer:
    def __init__(self, fm, fd: FedData, batch: int = 32, lr: float = 0.05, chunk: int = 64, amp: bool = False):
        self.fm, self.fd, self.B, self.lr, self.chunk = fm, fd, batch, lr, chunk
        self.amp = amp            # bf16 autocast for the CNNs (parameters and updates stay fp32)
        self._g = vmap(grad_and_value(fm.loss))

    def _batch(self, X, Y, table, counts, gen):
        K = len(table)
        r = torch.rand(K, self.B, device=table.device, generator=gen)
        pos = (r * counts[:, None]).long().clamp(max=table.shape[1] - 1)
        idx = table.gather(1, pos)
        return self.fd.to_input(X[idx]), Y[idx]

    def train(self, w: torch.Tensor, table: torch.Tensor, counts: torch.Tensor, steps: torch.Tensor,
              gen: torch.Generator, X: torch.Tensor | None = None, Y: torch.Tensor | None = None,
              poison: Poison | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns updates U [K, d] = w_k^local - w and the mean training loss [K]."""
        X = self.fd.x_train if X is None else X
        Y = self.fd.y_train if Y is None else Y
        Us, Ls = [], []
        for s in range(0, len(table), self.chunk):
            sl = slice(s, s + self.chunk)
            p = None
            if poison is not None:
                p = Poison(poison.label_flip[sl], poison.backdoor[sl], poison.part[sl], poison.nt_mask,
                           None if poison.nt_rows is None else poison.nt_rows[sl], poison.frac)
            U, L = self._train_chunk(w, table[sl], counts[sl], steps[sl], gen, X, Y, p)
            Us.append(U)
            Ls.append(L)
        return torch.cat(Us), torch.cat(Ls)

    def _train_chunk(self, w, table, counts, steps, gen, X, Y, poison):
        K = len(table)
        W = w.detach().unsqueeze(0).repeat(K, 1)
        loss_sum = torch.zeros(K, device=w.device)
        C = self.fd.n_classes
        for t in range(int(steps.max())):
            x, y = self._batch(X, Y, table, counts, gen)
            if poison is not None:
                y = torch.where(poison.label_flip[:, None], C - 1 - y, y)
                if bool(poison.backdoor.any()):
                    sel = poison.backdoor[:, None] & (torch.rand(y.shape, device=y.device, generator=gen)
                                                      < poison.frac)
                    part = poison.part[:, None].expand_as(y)
                    full = part == 4
                    x = torch.where(full.reshape(*y.shape, *[1] * len(self.fd.in_shape)),
                                    stamp_trigger(self.fd, x, None, sel), stamp_trigger(self.fd, x, part.clamp(max=3), sel))
                    y = torch.where(sel, torch.full_like(y, self.fd.target_class), y)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.amp):
                g, l = self._g(W, x, y)
            g, l = g.float(), l.float()
            active = (t < steps).float()
            if poison is not None and poison.nt_rows is not None and bool(poison.nt_rows.any()):
                g = torch.where(poison.nt_rows[:, None], g * poison.nt_mask, g)
            W -= self.lr * active[:, None] * g
            loss_sum += active * l.detach()
        return W - w[None], loss_sum / steps.float()


def root_tables(n_root: int, y_root: torch.Tensor, n_shards: int, shard_size: int, alpha: float,
                gen_np, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Index tables of `n_shards` non-IID virtual clients drawn from the root set (Dirichlet(alpha)
    label skew, sampled with replacement) plus one row covering the full root set (FLTrust g0)."""
    import numpy as np

    y = y_root.cpu().numpy()
    C = int(y.max()) + 1
    by_c = [np.flatnonzero(y == c) for c in range(C)]
    present = np.array([len(b) > 0 for b in by_c])
    rows = []
    for _ in range(n_shards):
        p = gen_np.dirichlet(alpha * np.ones(C)) * present
        p = p / p.sum()
        cls = gen_np.choice(C, shard_size, p=p)
        rows.append(np.array([gen_np.choice(by_c[c]) for c in cls]))
    width = max(shard_size, n_root)
    table = np.zeros((n_shards + 1, width), np.int64)
    counts = np.zeros(n_shards + 1, np.int64)
    for i, r in enumerate(rows):
        table[i, :len(r)] = r
        table[i, len(r):] = r[0]
        counts[i] = len(r)
    table[-1, :n_root] = np.arange(n_root)
    counts[-1] = n_root
    return torch.as_tensor(table, device=device), torch.as_tensor(counts, device=device)
