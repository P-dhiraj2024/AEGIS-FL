"""Write the job lists (results/jobs_<phase>.txt) for scripts/jobqueue.py.

Phases
  lr        FedAvg, no attack, 150 rounds, seed 0, per-dataset learning-rate grid (rule 4)
  main      6 datasets x 9 defences x 10 attack conditions (gamma = 0.3) x seeds
  ratio     gamma in {0.1, 0.2, 0.4, 0.5}, 4 attacks, 9 defences, CICIoT2023 + Fashion-MNIST
  conformal AEGIS-FL, alpha in {0.01, 0.02, 0.1, 0.2} (0.05 is in main), no attack and Min-Max
  ablation  AEGIS-FL variants x 6 datasets x {sign_flip, label_flip, min_max}
  adaptive  all defences vs. the adaptive attacker with target quantile 0.99 (stronger)
  privacy   AEGIS-FL with user-level RDP epsilon in {1, 4, 8, 16}, no attack and Min-Max
Learning rates come from configs/lr.yaml (written by scripts/pick_lr.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

CODE = Path(__file__).resolve().parents[1]
DATASETS = ["ciciot2023", "edgeiiot", "sensit", "fmnist", "femnist", "cifar10"]
DEFENSES = ["fedavg", "krum", "trimmed_mean", "median", "fltrust", "flame", "fldetector", "freqfed", "aegis"]
ATTACKS = ["none", "label_flip", "sign_flip", "gaussian", "lie", "min_max", "min_sum", "dba", "neurotoxin", "adaptive"]
ABLATIONS = ["no_norm", "no_normtest", "no_align", "no_collusion", "no_vae", "no_gan", "no_conformal", "no_reputation"]
LR_GRID = {"tabular": [0.03, 0.1, 0.3, 1.0], "image": [0.01, 0.03, 0.1, 0.3]}   # extended upward: 0.3 / 0.1 were best
PY = "../../../.venv/bin/python"


def job(exp, seed, sets):
    s = " ".join(f"{k}={v}" for k, v in sets.items())
    return f"{exp} {seed} {PY} -m src.run --config configs/default.yaml --seed {seed} --exp {exp} --set {s}"


def tag(attack, ratio):
    return "none" if attack == "none" else f"{attack}_g{ratio}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    out = []
    if a.phase == "lr":
        for ds in DATASETS:
            for lr in LR_GRID["tabular" if ds in DATASETS[:3] else "image"]:
                out.append(job(f"lr/{ds}/lr{lr}", 0, {"data.dataset": ds, "defense.name": "fedavg", "rounds": 150,
                                                      "train.lr": lr}))
    else:
        lrs = yaml.safe_load((CODE / "configs" / "lr.yaml").read_text())
        base = lambda ds: {"data.dataset": ds, "train.lr": lrs[ds]}
        if a.phase == "main":                      # seed-major order: complete seeds finish first
            for s in a.seeds:
                for att in ATTACKS:
                    for ds in DATASETS:
                        for d in DEFENSES:
                            out.append(job(f"{ds}/{d}/{tag(att, 0.3)}", s,
                                           dict(base(ds), **{"defense.name": d, "attack.type": att})))
        elif a.phase == "ratio":
            for ds in ["ciciot2023", "fmnist"]:
                for g in [0.1, 0.2, 0.4, 0.5]:
                    for att in ["label_flip", "min_max", "dba", "adaptive"]:
                        for d in DEFENSES:
                            for s in a.seeds:
                                out.append(job(f"{ds}/{d}/{tag(att, g)}", s, dict(base(ds), **{
                                    "defense.name": d, "attack.type": att, "attack.ratio": g})))
        elif a.phase == "conformal":
            for ds in DATASETS:
                for al in [0.01, 0.02, 0.1, 0.2]:
                    for att in ["none", "min_max"]:
                        for s in a.seeds:
                            t = tag(att, 0.3) + f"_a{al}"
                            out.append(job(f"{ds}/aegis/{t}", s, dict(base(ds), **{
                                "attack.type": att, "defense.conformal_alpha": al})))
        elif a.phase == "ablation":
            for ds in DATASETS:
                for v in ABLATIONS:
                    for att in ["sign_flip", "label_flip", "min_max"]:
                        for s in a.seeds:
                            out.append(job(f"{ds}/aegis-{v}/{tag(att, 0.3)}", s, dict(base(ds), **{
                                "defense.variant": v, "attack.type": att})))
        elif a.phase == "adaptive":                # stronger defence-aware attacker (target quantile 0.99)
            for s in a.seeds:
                for ds in DATASETS:
                    for d in DEFENSES:
                        out.append(job(f"{ds}/{d}/adaptive-q0.99_g0.3", s, dict(base(ds), **{
                            "defense.name": d, "attack.type": "adaptive", "attack.adaptive.target_quantile": 0.99})))
        elif a.phase == "privacy":
            for ds in DATASETS:
                for eps in [1.0, 4.0, 8.0, 16.0]:
                    for att in ["none", "min_max"]:
                        for s in a.seeds:
                            out.append(job(f"{ds}/aegis/{tag(att, 0.3)}_eps{eps}", s, dict(base(ds), **{
                                "attack.type": att, "privacy.dp_epsilon": eps})))
        else:
            raise SystemExit(a.phase)
    f = CODE / "results" / f"jobs_{a.phase}.txt"
    f.write_text("\n".join(out) + "\n")
    print(f, len(out), "jobs")


if __name__ == "__main__":
    main()
