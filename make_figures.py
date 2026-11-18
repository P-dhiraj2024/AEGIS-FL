"""Generate the figures of Paper B from results/runs.

    python scripts/make_figures.py --out ../figures

fig_ratio.pdf      accuracy (ASR for DBA) vs malicious ratio, 2 datasets x 4 attacks, 9 defences
fig_conformal.pdf  empirical benign rejection vs target alpha (no attack / Min-Max)
fig_curves.pdf     test accuracy vs round under Min-Max and the adaptive attack, 6 datasets
fig_privacy.pdf    AEGIS-FL accuracy vs RDP epsilon

Colours: the 8 baselines take the 8 validated categorical slots in fixed order (dataviz reference
palette, CVD-checked); AEGIS-FL is drawn in near-black and thicker; markers are the secondary
encoding (several light slots are below 3:1 contrast, so every series also has a marker and a
legend entry, and all values are in the tables).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from make_tables import AN, DS, DSN, METH  # noqa: E402
from results_io import load_runs  # noqa: E402

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COL = dict(zip(METH[:-1], SLOTS)) | {"aegis": "#0b0b0b"}
MRK = dict(zip(METH, ["o", "v", "s", "D", "^", "P", "X", "<", "*"]))
LBL = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "TrMean", "median": "Median", "fltrust": "FLTrust",
       "flame": "FLAME", "fldetector": "FLDetector", "freqfed": "FreqFed", "aegis": "AEGIS-FL"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "font.family": "serif"})


def style(m):
    a = m == "aegis"
    return dict(color=COL[m], marker=MRK[m], lw=2.0 if a else 1.2, ms=6 if a else 4, zorder=5 if a else 2,
                label=LBL[m])


def agg(df, ds, m, t, col):
    x = df[(df.dataset == ds) & (df.method == m) & (df.tag == t)][col].astype(float).dropna()
    return (x.mean(), x.std(ddof=1) if len(x) > 1 else 0.0) if len(x) else (math.nan, math.nan)


def panel(ax, i, dy=None):
    """Panel tag (a), (b), ... as a second x-label line; the panel description is in the LaTeX caption."""
    ax.set_xlabel(ax.get_xlabel() + f"\n({chr(97 + i)})")


def legend_below(fig, axes, ncol=9):
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=ncol, frameon=True, edgecolor=INK2, fancybox=False, framealpha=1.0, bbox_to_anchor=(0.5, -0.02), columnspacing=1.2, handlelength=1.8)


def fig_ratio(df, out):
    dss, atts = ["ciciot2023", "fmnist"], ["label_flip", "min_max", "dba", "adaptive"]
    gs = [0.1, 0.2, 0.3, 0.4, 0.5]
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.0))
    for i, ds in enumerate(dss):
        for j, a in enumerate(atts):
            ax = axes[i, j]
            col = "test_asr" if a == "dba" else "test_acc"
            for m in METH:
                y = [agg(df, ds, m, f"{a}_g{g}", col + "_last5")[0] * 100 for g in gs]
                ax.plot(gs, y, **style(m))
            ax.set_ylabel("ASR (%)" if a == "dba" else "Accuracy (%)")
            ax.set_xlabel(r"Malicious ratio $\gamma$")
            ax.set_xticks(gs)
            panel(ax, 4 * i + j, dy=-0.30)
    legend_below(fig, axes)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.6)
    fig.savefig(out / "fig_ratio.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_conformal(df, out):
    al = [0.01, 0.02, 0.05, 0.1, 0.2]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4), sharey=True)
    dcol = dict(zip(DS, SLOTS[:6]))
    dm = dict(zip(DS, ["o", "s", "D", "^", "v", "P"]))
    for pi, (ax, att) in enumerate(zip(axes, ["none", "min_max_g0.3"])):
        ax.plot([0, 0.21], [0, 0.21], color=INK2, lw=1, ls="--", label=r"target $\alpha$")
        for ds in DS:
            y = [agg(df, ds, "aegis", att + ("" if a == 0.05 else f"_a{a}"), "benign_reject")[0] for a in al]
            ax.plot(al, y, color=dcol[ds], marker=dm[ds], lw=1.2, ms=4, label=DSN[ds])
        ax.set_xlabel(r"Target $\alpha$")
        panel(ax, pi, dy=-0.5)
        ax.set_xlim(0, 0.21)
        ax.set_ylim(0, None)
    axes[0].set_ylabel("Benign rejection rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=True, edgecolor=INK2, fancybox=False, framealpha=1.0, bbox_to_anchor=(0.5, 0.0), fontsize=6)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out / "fig_conformal.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_curves(df, out):
    h = load_runs(with_history=True)
    for att, name in [("min_max_g0.3", "fig_curves_minmax.pdf"), ("adaptive_g0.3", "fig_curves_adaptive.pdf"),
                      ("label_flip_g0.3", "fig_curves_labelflip.pdf")]:
        fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.0))
        for pi, (ax, ds) in enumerate(zip(axes.flat, DS)):
            for m in METH:
                runs = h[(h.dataset == ds) & (h.method == m) & (h.tag == att)]
                if not len(runs):
                    continue
                curves = [{r["step"]: r["test_acc"] for r in hist} for hist in runs.history]
                steps = sorted(set.intersection(*[set(c) for c in curves]))
                y = np.array([[c[s] for s in steps] for c in curves]).mean(0) * 100
                st = style(m)
                st["marker"], st["ms"] = None, None
                ax.plot(steps, y, **{k: v for k, v in st.items() if v is not None})
            ax.set_xlabel("Round")
            ax.set_ylabel("Test accuracy (%)")
            panel(ax, pi, dy=-0.30)
        legend_below(fig, axes)
        fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.6)
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)


def fig_privacy(df, out):
    eps = [1.0, 4.0, 8.0, 16.0]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.4), sharey=True)
    dcol = dict(zip(DS, SLOTS[:6]))
    dm = dict(zip(DS, ["o", "s", "D", "^", "v", "P"]))
    for pi, (ax, att) in enumerate(zip(axes, ["none", "min_max_g0.3"])):
        for ds in DS:
            y = [agg(df, ds, "aegis", f"{att}_eps{e}", "test_acc_last5")[0] * 100 for e in eps]
            y_inf = agg(df, ds, "aegis", att, "test_acc_last5")[0] * 100
            ax.plot(eps + [32], y + [y_inf], color=dcol[ds], marker=dm[ds], lw=1.2, ms=4, label=DSN[ds])
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1", "4", "8", "16", r"$\infty$"])
        ax.set_xlabel(r"$\varepsilon$")
        panel(ax, pi, dy=-0.5)
    axes[0].set_ylabel("Test accuracy (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=True, edgecolor=INK2, fancybox=False, framealpha=1.0, bbox_to_anchor=(0.5, 0.0), fontsize=6)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out / "fig_privacy.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_heatmap(df, out):
    """Accuracy per attack (rows) x defence (columns), averaged over the six datasets; one sequential hue."""
    from matplotlib.colors import LinearSegmentedColormap
    atts = ["none", "label_flip", "sign_flip", "gaussian", "lie", "min_max", "min_sum", "dba", "neurotoxin", "adaptive"]
    M = np.full((len(atts), len(METH)), np.nan)
    for i, a in enumerate(atts):
        t = "none" if a == "none" else f"{a}_g0.3"
        for j, m in enumerate(METH):
            v = [agg(df, ds, m, t, "test_acc_last5")[0] for ds in DS]
            if all(x == x for x in v):
                M[i, j] = 100 * np.mean(v)
    cmap = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#9cc3ee", "#2a78d6", "#0d3d78"])
    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.grid(False)
    for i in range(M.shape[0]):
        best = np.nanmax(M[i]) if np.isfinite(M[i]).any() else np.nan
        for j in range(M.shape[1]):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if M[i, j] > 60 else INK, fontweight="bold" if M[i, j] == best else "normal")
    ax.set_xticks(range(len(METH)))
    ax.set_xticklabels([LBL[m] for m in METH], rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(atts)))
    ax.set_yticklabels([AN[a] for a in atts], fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("Test accuracy (%)", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_scores(df, out):
    """Distributions of the four AEGIS-FL scores for benign vs malicious updates (F-MNIST, seed 0, rounds 101-300)
    from the diagnostic runs results/runs/diag/fmnist/<attack>/0/scores.pkl; dashed line = median threshold."""
    import pickle
    from make_tables import RUNS_DIR
    atts = ["sign_flip", "label_flip", "min_max", "neurotoxin"]
    tests = [("vae", "VAE score (ratio)"), ("norm", "Norm excess"), ("align", "Misalignment"),
             ("collusion", "Collusion")]
    fig, axes = plt.subplots(len(atts), len(tests), figsize=(7.16, 5.2))
    k = 0
    for i, a in enumerate(atts):
        f = RUNS_DIR / "diag" / "fmnist" / a / "0" / "scores.pkl"
        D = [r for r in pickle.loads(f.read_bytes()) if r["t"] > 100] if f.exists() else []
        for j, (key, lab) in enumerate(tests):
            ax = axes[i, j]
            if D:
                b = np.concatenate([r[key][~r["mal"]] for r in D])
                m = np.concatenate([r[key][r["mal"]] for r in D])
                thr = np.median([r["thr"][key] for r in D])
                lo, hi = np.percentile(np.concatenate([b, m]), [0.5, 99.5])
                bins = np.linspace(lo, max(hi, thr) + 1e-9, 40)
                ax.hist(b, bins=bins, density=True, color=SLOTS[0], alpha=0.75, label="benign")
                ax.hist(m, bins=bins, density=True, color=SLOTS[1], alpha=0.75, label="malicious")
                ax.axvline(thr, color=INK, ls="--", lw=1, label="threshold")
            ax.set_yticks([])
            if i == len(atts) - 1:
                ax.set_xlabel(lab)
            if j == 0:
                ax.set_ylabel(AN[a])
            ax.text(0.97, 0.93, f"({chr(97 + k)})", transform=ax.transAxes, ha="right", va="top", color=INK)
            k += 1
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=True, edgecolor=INK2, fancybox=False, framealpha=1.0, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "fig_scores.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../figures")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    for fn in (fig_heatmap, fig_scores, fig_ratio, fig_conformal, fig_curves, fig_privacy):
        try:
            fn(df, out)
            print("ok", fn.__name__)
        except Exception as e:      # a figure whose runs are not finished yet
            print("skip", fn.__name__, type(e).__name__, e)


if __name__ == "__main__":
    main()
