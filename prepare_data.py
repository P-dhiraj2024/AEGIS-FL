"""Prepare the two extra IoT datasets used by Paper B and the N=500 Dirichlet partitions.

Outputs (repo data/):
  processed/edgeiiot.npz : Edge-IIoTset ML split (Ferrag et al., IEEE Access 2022), 15 classes
  processed/sensit.npz   : SensIT vehicle (Duarte & Hu, JPDC 2004), acoustic+seismic, 3 classes
  partitions/<ds>/alpha_0.5_n_500.npy for ciciot2023, edgeiiot, sensit (balanced Dirichlet, seed 0)

CICIoT2023 (processed/ciciot2023.npz) was prepared by Paper A's scripts/prepare_data.py.
Raw files are read from <workspace>/datasets_raw/{edgeiiot,sensit}.

    python scripts/prepare_data.py
"""
from __future__ import annotations

import bz2
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from core.data.partition import dirichlet_balanced, label_histograms  # noqa: E402

RAW = REPO.parent / "datasets_raw"
OUT = REPO / "data" / "processed"
PART = REPO / "data" / "partitions"

# identifier / free-text / leakage columns removed in the Edge-IIoTset reference pipeline
EDGE_DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
             "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
             "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport",
             "udp.port", "mqtt.msg", "Attack_label"]
EDGE_CAT = ["http.request.method", "http.referer", "http.request.version", "dns.qry.name.len",
            "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic"]


def _split_standardise(X, y, rng, test_frac=0.2, log=True):
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if log:
        X = np.sign(X) * np.log1p(np.abs(X))
    test = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        test[idx[: int(round(test_frac * len(idx)))]] = True
    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    keep = Xtr.std(0) > 1e-8                       # constant columns carry no information
    Xtr, Xte = ((Xtr - mu) / sd)[:, keep], ((Xte - mu) / sd)[:, keep]
    p = rng.permutation(len(ytr))
    return Xtr[p].astype(np.float32), ytr[p], Xte.astype(np.float32), yte, keep


def prep_edgeiiot(seed=0):
    import pandas as pd

    df = pd.read_csv(RAW / "edgeiiot" / "ML-EdgeIIoT-dataset.csv", low_memory=False)
    df = df.drop(columns=[c for c in EDGE_DROP if c in df.columns])
    df = df.dropna().drop_duplicates()
    names = sorted(df["Attack_type"].unique())
    y = df["Attack_type"].map({n: i for i, n in enumerate(names)}).to_numpy(np.int64)
    feats = df.drop(columns=["Attack_type"])
    feats = pd.get_dummies(feats, columns=[c for c in EDGE_CAT if c in feats.columns], dtype=float)
    X = feats.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte, keep = _split_standardise(X, y, rng)
    np.savez_compressed(OUT / "edgeiiot.npz", x_train=Xtr, y_train=ytr, x_test=Xte, y_test=yte,
                        class_names=np.asarray(names), feature_names=np.asarray(feats.columns)[keep])
    print("edgeiiot", Xtr.shape, Xte.shape, np.bincount(ytr))


def _svmlight(path, n_feat=100):
    rows, ys = [], []
    with bz2.open(path, "rt") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            ys.append(int(float(parts[0])) - 1)
            r = np.zeros(n_feat)
            for kv in parts[1:]:
                k, v = kv.split(":")
                r[int(k) - 1] = float(v)
            rows.append(r)
    return np.asarray(rows), np.asarray(ys, np.int64)


def prep_sensit(seed=0):
    """Official LIBSVM train/test split is kept; standardised with train statistics."""
    Xtr, ytr = _svmlight(RAW / "sensit" / "combined.bz2")
    Xte, yte = _svmlight(RAW / "sensit" / "combined.t.bz2")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(ytr))
    np.savez_compressed(OUT / "sensit.npz", x_train=((Xtr - mu) / sd)[p].astype(np.float32), y_train=ytr[p],
                        x_test=((Xte - mu) / sd).astype(np.float32), y_test=yte,
                        class_names=np.asarray(["AAV", "DW", "noise"]),
                        feature_names=np.asarray([f"f{i}" for i in range(100)]))
    print("sensit", Xtr.shape, Xte.shape, np.bincount(ytr))


def make_partition(name, n_clients=500, alpha=0.5, seed=0):
    y = np.load(OUT / f"{name}.npz")["y_train"]
    part = dirichlet_balanced(y, n_clients, alpha, np.random.default_rng(seed))
    d = PART / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / f"alpha_{alpha}_n_{n_clients}.npy", part)
    h = label_histograms(part, y)
    p = h / h.sum(1, keepdims=True)
    g = np.bincount(y) / len(y)
    sizes = h.sum(1)
    (d / f"alpha_{alpha}_n_{n_clients}.json").write_text(json.dumps(dict(
        n_clients=n_clients, n_samples_used=int((part >= 0).sum()), n_samples_total=len(y),
        size_min=int(sizes.min()), size_median=float(np.median(sizes)), size_max=int(sizes.max()),
        classes_per_client_mean=float((h > 0).sum(1).mean()),
        tv_to_global_mean=float(0.5 * np.abs(p - g).sum(1).mean()),
        scheme="dirichlet-balanced", alpha=alpha, seed=seed), indent=2))


if __name__ == "__main__":
    prep_edgeiiot()
    prep_sensit()
    for ds in ("ciciot2023", "edgeiiot", "sensit"):
        make_partition(ds)
        print("partition", ds, "ok")
