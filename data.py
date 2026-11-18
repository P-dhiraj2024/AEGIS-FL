"""Federated data for Paper B: six datasets, N-client partitions, train / validation / test.

Datasets (all real, see data/processed):
  ciciot2023  CICIoT2023 IoT IDS, 8 classes, 39 features (prepared by Paper A)
  edgeiiot    Edge-IIoTset IIoT IDS, 15 classes, 65 features (scripts/prepare_data.py)
  sensit      SensIT vehicle, acoustic + seismic sensor nodes, 3 classes, 100 features
  fmnist      Fashion-MNIST, 10 classes, 28x28
  femnist     FEMNIST (LEAF), 62 classes, natural writer split
  cifar10     CIFAR-10, 10 classes, 32x32x3

Splits.  Each client keeps 90% of its partition for local training; the other 10% are pooled
into a validation set (core.data.loaders.val_split).  The server's small clean root set
(FLTrust root data, also used by AEGIS-FL's reference updates) is drawn from that pool and
removed from it, so root, validation and client training samples are disjoint.  The test set
is the dataset's official test split (FEMNIST: test samples of the selected writers).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from core.data.loaders import load_dataset, load_partition, val_split
from core.data.partition import client_indices
from core.utils.paths import PROCESSED

TABULAR = ("ciciot2023", "edgeiiot", "sensit")
IMAGE = ("fmnist", "femnist", "cifar10")
DATASETS = TABULAR + IMAGE
# backdoor target class: IDS -> the benign class (attack traffic passed as benign); SensIT -> "noise"
BACKDOOR_TARGET = {"ciciot2023": "Benign", "edgeiiot": "Normal", "sensit": "noise"}


@dataclass
class FedData:
    name: str
    kind: str                     # "tabular" | "image"
    n_classes: int
    x_train: torch.Tensor         # on device; uint8 images or float32 features
    y_train: torch.Tensor
    idx_table: torch.Tensor       # [N, max_n] client -> train indices (padded)
    counts: torch.Tensor          # [N]
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    root_idx: torch.Tensor        # indices into x_train... (root samples live in the val pool)
    x_root: torch.Tensor
    y_root: torch.Tensor
    in_shape: tuple
    target_class: int
    trig_feat: torch.Tensor | None = None   # tabular backdoor: feature ids [4, 2]
    trig_val: torch.Tensor | None = None    # tabular backdoor: values [4, 2]

    def to_input(self, x: torch.Tensor) -> torch.Tensor:
        """Raw stored samples [..., *raw] -> model input [..., *in_shape] (float)."""
        if self.kind == "tabular":
            return x
        x = x.float() / 255.0
        if x.shape[-1] == 3:                                  # [...,H,W,C] -> [...,C,H,W]
            x = x.movedim(-1, -3)
        else:
            x = x.unsqueeze(-3)
        return (x - 0.5) / 0.25


def _tabular_npz(name: str):
    z = np.load(PROCESSED / f"{name}.npz", allow_pickle=False)
    return (z["x_train"], z["y_train"].astype(np.int64), z["x_test"], z["y_test"].astype(np.int64),
            [str(c) for c in z["class_names"]])


def load_fed(name: str, n_clients: int = 500, n_part: int = 500, alpha: float = 0.5, val_frac: float = 0.1,
             root_size: int = 500, val_max: int = 5000, device: str = "cuda", seed: int = 0,
             smoke: bool = False) -> FedData:
    rng = np.random.default_rng(1000 + seed)
    if name in TABULAR:
        xtr, ytr, xte, yte, names = _tabular_npz(name)
        C = len(names)
        target = names.index(BACKDOOR_TARGET[name])
        part = load_partition(name, f"alpha_{alpha}_n_{n_part}")
        kind = "tabular"
    else:
        ds = load_dataset(name)
        xtr, ytr, xte, yte, C = ds.x_train, ds.y_train, ds.x_test, ds.y_test, ds.n_classes
        target = 0
        kind = "image"
        if name == "femnist":
            part = load_partition(name, f"natural_n_{n_part}")
            wtr, wte = ds.meta["writer_train"], ds.meta["writer_test"]
            chosen = np.unique(wtr[part >= 0])
            keep = np.isin(wte, chosen)
            xte, yte = xte[keep], yte[keep]
        else:
            part = load_partition(name, f"alpha_{alpha}_n_{n_part}")
    if smoke:                                                   # first n_clients clients only
        part = np.where(part < n_clients, part, -1)
    idx = client_indices(part)[:n_clients]
    train_idx, val_pool = val_split(idx, val_frac, rng)
    val_pool = rng.permutation(val_pool)
    root, val = val_pool[:root_size], val_pool[root_size:root_size + val_max]
    max_n = max(len(i) for i in train_idx)
    table = np.zeros((len(train_idx), max_n), np.int64)
    for k, i in enumerate(train_idx):
        table[k, :len(i)] = i
        table[k, len(i):] = i[0] if len(i) else 0
    dev = torch.device(device)
    T = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)
    fd = FedData(name, kind, C, T(xtr), T(ytr), T(table), T([len(i) for i in train_idx]),
                 T(xtr[val]), T(ytr[val]), T(xte), T(yte), T(root), T(xtr[root]), T(ytr[root]),
                 in_shape=(), target_class=target)
    fd.in_shape = tuple(fd.to_input(fd.x_train[:1]).shape[1:])
    if kind == "tabular":
        F = xtr.shape[1]
        feats = np.random.default_rng(7).choice(F, 8, replace=False).reshape(4, 2)
        fd.trig_feat = T(feats)
        fd.trig_val = T(xtr.max(0)[feats].astype(np.float32))
    return fd


def stamp_trigger(fd: FedData, x: torch.Tensor, part: torch.Tensor | int | None, sel: torch.Tensor | None = None):
    """Stamp the DBA trigger on model inputs.

    x: [..., *in_shape] float inputs.  part: local trigger part 0..3 per leading row (tensor
    broadcastable to x's leading dims) or None for the full global trigger.  sel: bool mask of
    the leading dims selecting which samples are stamped (None = all).
    Image trigger = 4 bars of length 4 at the top-left (Xie et al., ICLR 2020), value = the
    brightest normalised pixel (2.0); tabular trigger = 4 pairs of features set to their
    training maximum.
    """
    lead = x.shape[: x.dim() - len(fd.in_shape)]
    if fd.kind == "image":
        H, W = fd.in_shape[-2:]
        masks = torch.zeros(5, H, W, dtype=torch.bool, device=x.device)
        for k in range(4):
            r, c = 1 + (k // 2) * 2, 1 + (k % 2) * 5
            masks[k, r, c:c + 4] = True
        masks[4] = masks[:4].any(0)
        if part is None:
            m = masks[4].expand(*lead, H, W)
        else:
            m = masks[torch.as_tensor(part, device=x.device).expand(lead)]
        m = m.unsqueeze(-3).expand(*lead, *fd.in_shape)
        val = torch.full_like(x, 2.0)
    else:
        F = fd.in_shape[0]
        masks = torch.zeros(5, F, dtype=torch.bool, device=x.device)
        vals = torch.zeros(F, device=x.device)
        for k in range(4):
            masks[k, fd.trig_feat[k]] = True
            vals[fd.trig_feat[k]] = fd.trig_val[k]
        masks[4] = masks[:4].any(0)
        m = masks[4].expand(*lead, F) if part is None else masks[torch.as_tensor(part, device=x.device).expand(lead)]
        val = vals.expand_as(x)
    if sel is not None:
        m = m & sel.reshape(*lead, *([1] * len(fd.in_shape)))
    return torch.where(m, val, x)
