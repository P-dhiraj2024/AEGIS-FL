"""Generate every LaTeX table of Paper B from results/runs (CLAUDE.md rule 1: no hand-typed numbers).

    python scripts/make_tables.py --out ../tables

Writes tab_*.tex and numbers.tex (macros used in the text / abstract) plus numbers_provenance.md
(which runs every macro comes from).  Cells with missing runs print "--"; the seed count used
for each cell is checked and reported in numbers_provenance.md.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from results_io import RUNS as RUNS_DIR, load_runs

DS = ["ciciot2023", "edgeiiot", "sensit", "fmnist", "femnist", "cifar10"]
DSN = {"ciciot2023": "CICIoT2023", "edgeiiot": "Edge-IIoTset", "sensit": "SensIT", "fmnist": "F-MNIST",
       "femnist": "FEMNIST", "cifar10": "CIFAR-10"}
DSA = {"ciciot2023": "CIC", "edgeiiot": "EIoT", "sensit": "SIT", "fmnist": "FM", "femnist": "FEM",
       "cifar10": "C10"}                       # abbreviations (defined in Table tab:data)
METH = ["fedavg", "krum", "trimmed_mean", "median", "fltrust", "flame", "fldetector", "freqfed", "aegis"]
MN = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "TrMean", "median": "Median", "fltrust": "FLTrust",
      "flame": "FLAME", "fldetector": "FLDetector", "freqfed": "FreqFed", "aegis": r"\textbf{AEGIS-FL}"}
ATT = ["label_flip", "sign_flip", "gaussian", "lie", "min_max", "min_sum", "dba", "neurotoxin", "adaptive"]
AN = {"none": "No attack", "label_flip": "Label flip", "sign_flip": "Sign flip", "gaussian": "Gaussian",
      "lie": "LIE", "min_max": "Min-Max", "min_sum": "Min-Sum", "dba": "DBA", "neurotoxin": "Neurotoxin",
      "adaptive": "Adaptive"}
VAR = ["aegis", "aegis-no_norm", "aegis-no_normtest", "aegis-no_align", "aegis-no_collusion", "aegis-no_vae",
       "aegis-no_gan", "aegis-no_conformal", "aegis-no_reputation"]
VN = {"aegis": "AEGIS-FL (full)", "aegis-no_norm": r"$-$norm features (Eq.~\eqref{eq:sketch} only)", "aegis-no_normtest": r"$-$norm test", "aegis-no_align": r"$-$alignment test",
      "aegis-no_collusion": r"$-$collusion test", "aegis-no_vae": r"$-$VAE (distance score)", "aegis-no_gan": r"$-$GAN red team",
      "aegis-no_conformal": r"$-$online conformal", "aegis-no_reputation": r"$-$reputation"}
DETECT = ["krum", "fltrust", "flame", "fldetector", "freqfed", "aegis"]
prov: list[str] = []
prov_cells: set = set()
EXTRA: dict = {}


def tag(att, g=0.3, extra=""):
    return ("none" if att == "none" else f"{att}_g{g}") + extra


def cell(df, ds, method, t, col="test_acc_last5"):
    x = df[(df.dataset == ds) & (df.method == method) & (df.tag == t)][col].astype(float)
    x = x[~x.isna()]
    if len(x):
        prov_cells.add((ds, method, t, len(x)))
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, len(x)) if len(x) else (math.nan, math.nan, 0)


def f(m, s=None, pct=True, d=1):
    if m != m:
        return "--"
    k = 100 if pct else 1
    return f"{m * k:.{d}f}" if s is None else f"{m * k:.{d}f}$\\pm${s * k:.{d}f}"


def bold_best(vals, higher=True):
    ok = [v for v in vals if v == v]
    if not ok:
        return set()
    b = max(ok) if higher else min(ok)
    return {i for i, v in enumerate(vals) if v == v and abs(v - b) < 1e-9}


def _shrink(block: str) -> str:
    """Shrink-only scaling of every tabular to the column (table) or text (table*) width."""
    w = r"\textwidth" if r"\begin{table*}" in block else r"\columnwidth"
    return (block.replace(r"\begin{tabular}", rf"\begin{{adjustbox}}{{max width={w}}}" + "\n" + r"\begin{tabular}")
                 .replace(r"\end{tabular}", r"\end{tabular}" + "\n" + r"\end{adjustbox}"))


def write(out, name, body):
    parts = body.split(r"\end{table")
    body = r"\end{table".join(_shrink(p) for p in parts)
    (out / name).write_text(body)
    print("wrote", name)


def table_summary(df, out):
    """Clean / mean-under-attack / worst-case accuracy per dataset."""
    rows, cols = {}, []
    for ds in DS:
        for kind in ("clean", "avg", "worst"):
            vals = []
            for m in METH:
                if kind == "clean":
                    v = cell(df, ds, m, "none")[0]
                else:
                    per = [cell(df, ds, m, tag(a))[0] for a in ATT]
                    per = [p for p in per if p == p]
                    v = (np.mean(per) if kind == "avg" else np.min(per)) if len(per) == len(ATT) else math.nan
                vals.append(v)
            cols.append(vals)
    best = [bold_best(c) for c in cols]
    lines = [r"\begin{table*}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\caption{Test accuracy (\%, mean over seeds of the average of the evaluations at rounds 260--300; "
             r"300 rounds, $\gamma=0.3$): without attack (Cl.), "
             r"averaged over the nine attacks (Avg.) and under the worst attack (Wst.). Bold: best per column.}",
             r"\label{tab:summary}",
             r"\begin{tabular}{l" + "ccc" * len(DS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{DSN[d]}}}" for d in DS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(DS))),
             "Method & " + " & ".join(["Cl.", "Avg.", "Wst."] * len(DS)) + r" \\", r"\midrule"]
    for i, m in enumerate(METH):
        cs = [(rf"\textbf{{{f(c[i])}}}" if i in best[j] else f(c[i])) for j, c in enumerate(cols)]
        lines.append(MN[m] + " & " + " & ".join(cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write(out, "tab_summary.tex", "\n".join(lines))
    return cols


def table_full(df, out):
    """Per-dataset attack x method accuracy (mean +- std), for the appendix."""
    body = []
    for ds in DS:
        lines = [r"\begin{table*}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
                 rf"\caption{{{DSN[ds]}: test accuracy (\%, rounds 260--300 average, mean$\pm$std over seeds) per attack ($\gamma=0.3$).}}",
                 rf"\label{{tab:full_{ds}}}", r"\begin{tabular}{l" + "c" * len(METH) + "}", r"\toprule",
                 "Attack & " + " & ".join(MN[m] for m in METH) + r" \\", r"\midrule"]
        for a in ["none"] + ATT:
            cs = [cell(df, ds, m, tag(a)) for m in METH]
            b = bold_best([c[0] for c in cs])
            lines.append(AN[a] + " & " + " & ".join(
                (rf"\textbf{{{f(c[0], c[1])}}}" if i in b else f(c[0], c[1])) for i, c in enumerate(cs)) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
        body.append("\n".join(lines))
    write(out, "tab_full.tex", "\n".join(body))


def table_sig(df, out):
    """AEGIS-FL minus each baseline (pp), paired over (dataset, seed); dagger: one-sided Wilcoxon p < 0.05."""
    from scipy.stats import wilcoxon
    short = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "TrM", "median": "Med", "fltrust": "FLTr",
             "flame": "FLAME", "fldetector": "FLD", "freqfed": "FrqF"}
    base = METH[:-1]
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{Accuracy gain of AEGIS-FL over each baseline (percentage points, mean over the six datasets "
             r"and seeds). $^\dagger$: AEGIS-FL better with $p<0.05$ (one-sided Wilcoxon signed-rank test, paired over "
             r"dataset$\times$seed).}", r"\label{tab:sig}",
             r"\begin{tabular}{l" + "c" * len(base) + "}", r"\toprule",
             "Attack & " + " & ".join(short[m] for m in base) + r" \\", r"\midrule"]
    n_sig = n_all = 0
    for a in ["none"] + ATT:
        x = df[df.tag == tag(a)]
        ae = x[x.method == "aegis"].set_index(["dataset", "seed"]).test_acc_last5.astype(float)
        cs = []
        for m in base:
            b = x[x.method == m].set_index(["dataset", "seed"]).test_acc_last5.astype(float)
            j = ae.index.intersection(b.index)
            d = (ae[j] - b[j]).dropna().values
            if len(d) < 3:
                cs.append("--")
                continue
            pv = wilcoxon(d, alternative="greater").pvalue if np.any(d != 0) else 1.0
            n_all += 1
            n_sig += int(pv < 0.05)
            cs.append(f"{100 * d.mean():+.1f}" + (r"$^\dagger$" if pv < 0.05 else ""))
        lines.append(AN[a] + " & " + " & ".join(cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_sig.tex", "\n".join(lines))
    prov.append(f"- sigWins / sigAll: {n_sig} of {n_all} (attack, baseline) pairs significant in tab_sig")
    return n_sig, n_all


def table_asr(df, out):
    cols = [(ds, a) for ds in DS for a in ("dba", "neurotoxin")]
    vals = [[cell(df, ds, m, tag(a), "test_asr_last5") for m in METH] for ds, a in cols]
    best = [bold_best([v[0] for v in c], higher=False) for c in vals]
    lines = [r"\begin{table*}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.6pt}",
             r"\caption{Backdoor attack success rate (\%, lower is better, average of rounds 260--300, mean$\pm$std over seeds), "
             r"DBA and Neurotoxin, $\gamma=0.3$. Bold: lowest per column.}", r"\label{tab:asr}",
             r"\begin{tabular}{l" + "cc" * len(DS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in DS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
             "Method & " + " & ".join(["DBA", "NT"] * len(DS)) + r" \\", r"\midrule"]
    for i, m in enumerate(METH):
        cs = [(rf"\textbf{{{f(c[i][0], c[i][1])}}}" if i in best[j] else f(c[i][0], c[i][1]))
              for j, c in enumerate(vals)]
        lines.append(MN[m] + " & " + " & ".join(cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write(out, "tab_asr.tex", "\n".join(lines))
    return vals


def table_detect(df, out):
    """Single column: TPR / FPR (%) per method and dataset."""
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Detection of malicious updates (\%), TPR/FPR per dataset: TPR averaged over the nine "
             r"attacks, FPR (benign updates flagged) over all ten conditions, $\gamma=0.3$. FedAvg, trimmed mean "
             r"and median do not flag clients.}", r"\label{tab:detect}",
             r"\begin{tabular}{l" + "c" * len(DS) + "}", r"\toprule",
             "Method & " + " & ".join(DSA[d] for d in DS) + r" \\", r"\midrule"]
    for m in DETECT:
        cs = []
        for ds in DS:
            tp = [cell(df, ds, m, tag(a), "tpr")[0] for a in ATT]
            fp = [cell(df, ds, m, tag(a), "fpr")[0] for a in ["none"] + ATT]
            ok_t, ok_f = any(v == v for v in tp), any(v == v for v in fp)
            cs.append((f(np.nanmean(tp), d=0) if ok_t else "--") + "/" + (f(np.nanmean(fp), d=0) if ok_f else "--"))
        lines.append(MN[m] + " & " + " & ".join(cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_detect.tex", "\n".join(lines))


def table_conformal(df, out):
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2]
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Empirical benign rejection rate of AEGIS-FL (\%, mean$\pm$std over seeds) vs.\ target "
             r"$\alpha$, without attack (top) and under Min-Max with $\gamma=0.3$ (bottom).}",
             r"\label{tab:conformal}", r"\begin{tabular}{c" + "c" * len(DS) + "}", r"\toprule",
             r"$\alpha$ & " + " & ".join(DSA[d] for d in DS) + r" \\", r"\midrule"]
    for att in ("none", "min_max"):
        if att == "min_max":
            lines.append(r"\midrule")
        for al in alphas:
            ex = "" if al == 0.05 else f"_a{al}"
            cs = [cell(df, ds, "aegis", tag(att, extra=ex), "benign_reject") for ds in DS]
            lines.append(f"{al:g} & " + " & ".join(f(c[0], c[1]) for c in cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_conformal.tex", "\n".join(lines))


def table_ablation(df, out):
    atts = ["sign_flip", "label_flip", "min_max"]
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.2pt}",
             r"\caption{Ablation: test accuracy (\%) averaged over sign flip, label flip and Min-Max "
             r"($\gamma=0.3$), and benign rejection rate (\%, same runs).}", r"\label{tab:ablation}",
             r"\begin{tabular}{l" + "c" * len(DS) + "c}", r"\toprule",
             "Variant & " + " & ".join(DSA[d] for d in DS) + r" & Rej. \\", r"\midrule"]
    for v in VAR:
        cs, rej = [], []
        for ds in DS:
            per = [cell(df, ds, v, tag(a))[0] for a in atts]
            cs.append(f(np.mean(per)) if all(p == p for p in per) else "--")
            rej += [cell(df, ds, v, tag(a), "benign_reject")[0] for a in atts]
        rj = [r for r in rej if r == r]
        lines.append(VN[v] + " & " + " & ".join(cs) + " & " + (f(np.mean(rj)) if rj else "--") + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_ablation.tex", "\n".join(lines))


def table_overhead(df, out):
    lines = [r"\begin{table*}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Overhead per round: defence computation (ms, measured on one shared RTX A4000, all "
             r"UAVs together, averaged over all runs of the main grid) and UAV$\to$cloud backhaul (MB) under "
             r"Min-Max ($\gamma=0.3$). For AEGIS-FL the backhaul carries only accepted updates plus sketches; "
             r"its time includes sketching, scoring, detector retraining and the pairwise-mask secure sum.}",
             r"\label{tab:overhead}", r"\begin{tabular}{l" + "cc" * len(DS) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in DS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
             "Method & " + " & ".join(["ms", "MB"] * len(DS)) + r" \\", r"\midrule"]
    main = df[df.tag.isin(["none"] + [tag(a) for a in ATT])]
    for m in METH:
        cs = []
        for ds in DS:
            x = main[(main.dataset == ds) & (main.method == m)]
            ms = float(x.ms_defense_per_round.mean()) if len(x) else math.nan
            mm = cell(df, ds, m, tag("min_max"), "mb_backhaul_per_round")[0]
            sk = cell(df, ds, m, tag("min_max"), "mb_sketch_per_round")[0]
            cs += [f(ms, pct=False, d=1), f(mm + (sk if sk == sk else 0), pct=False, d=2)]
        lines.append(MN[m] + " & " + " & ".join(cs) + r" \\")
    lines += [r"\midrule", "Model size (MB) & " + " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{f(float(df[df.dataset == ds].d.iloc[0]) * 4 / 1e6, pct=False, d=3)}}}"
        if (df.dataset == ds).any() else r"\multicolumn{2}{c}{--}" for ds in DS) + r" \\"]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    write(out, "tab_overhead.tex", "\n".join(lines))


def table_energy(df, out):
    """Effective attacker share of accepted updates and UAV energy per round (sign flip, label flip, Min-Max)."""
    ms = ["fedavg", "fltrust", "flame", "aegis-no_reputation", "aegis"]
    nm = {"fedavg": "FedAvg", "fltrust": "FLTrust", "flame": "FLAME", "aegis-no_reputation": r"AEGIS-FL $-$rep.",
          "aegis": r"\textbf{AEGIS-FL}"}
    atts = ["sign_flip", "label_flip", "min_max"]
    x = df[df.tag.isin([tag(a) for a in atts]) & df.method.isin(ms)].copy()
    m, tp, fp = x.mal_frac_received.astype(float), x.tpr.astype(float).fillna(0), x.fpr.astype(float).fillna(0)
    x["geff"] = m * (1 - tp) / (m * (1 - tp) + (1 - m) * (1 - fp))
    x["eround"] = x.energy_uav_kJ.astype(float) / x.rounds_run.astype(float)
    g = x.groupby(["dataset", "method"])[["geff", "eround"]].mean()
    ge = g.geff.unstack()
    EXTRA.update(geffFedAvgMin=f(ge["fedavg"].min()), geffFedAvgMax=f(ge["fedavg"].max()),
                 geffBaseMin=f(ge[["fltrust", "flame"]].min().min()), geffBaseMax=f(ge[["fltrust", "flame"]].max().max()),
                 geffAEGISMin=f(ge["aegis"].min()), geffAEGISMax=f(ge["aegis"].max()))
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.4pt}",
             r"\caption{Edge cost under sign flip, label flip and Min-Max ($\gamma=0.3$): attackers' share of the "
             r"accepted updates $\gamma^{\mathrm{eff}}$ (\%, Proposition~\ref{prop:conv}; estimated from the received "
             r"share, TPR and FPR) and UAV energy per round (kJ, hovering + backhaul, 10 UAVs).}", r"\label{tab:energy}",
             r"\begin{tabular}{l" + "cc" * 3 + "}", r"\toprule"]

    def block(dss):
        rows = [" & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{DSN[d]}}}" for d in dss) + r" \\",
                " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(3)),
                r"Method & " + " & ".join([r"$\gamma^{\mathrm{eff}}$", "kJ"] * 3) + r" \\", r"\midrule"]
        for mm in ms:
            cs = []
            for ds in dss:
                if (ds, mm) in g.index:
                    cs += [f(g.loc[(ds, mm), "geff"]), f(g.loc[(ds, mm), "eround"], pct=False, d=2)]
                else:
                    cs += ["--", "--"]
            rows.append(nm[mm] + " & " + " & ".join(cs) + r" \\")
        return rows
    lines += block(DS[:3]) + [r"\midrule"] + block(DS[3:])
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_energy.tex", "\n".join(lines))


def table_privacy(df, out):
    eps = [math.inf, 16.0, 8.0, 4.0, 1.0]
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Privacy--utility trade-off of AEGIS-FL with user-level RDP noise at the cloud "
             r"($\delta=10^{-5}$, 300 rounds, $q=0.1$): test accuracy (\%) without attack / under Min-Max "
             r"($\gamma=0.3$).}", r"\label{tab:privacy}", r"\begin{tabular}{l" + "c" * len(DS) + "}", r"\toprule",
             r"$\varepsilon$ & " + " & ".join(DSA[d] for d in DS) + r" \\", r"\midrule"]
    for e in eps:
        ex = "" if e == math.inf else f"_eps{e}"
        cs = []
        for ds in DS:
            a = cell(df, ds, "aegis", tag("none", extra=ex))[0]
            b = cell(df, ds, "aegis", tag("min_max", extra=ex))[0]
            cs.append(f"{f(a)} / {f(b)}")
        lines.append((r"$\infty$" if e == math.inf else f"{e:g}") + " & " + " & ".join(cs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_privacy.tex", "\n".join(lines))


def table_hetero(df, out):
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Benign rejection rate of AEGIS-FL by device type (\%, $\alpha=0.05$, mean over seeds and "
             r"all ten attack conditions): IoT nodes run 5 local steps, smartphones 10.}", r"\label{tab:hetero}",
             r"\begin{tabular}{l" + "c" * len(DS) + "}", r"\toprule",
             "Device & " + " & ".join(DSA[d] for d in DS) + r" \\", r"\midrule"]
    a = df[(df.method == "aegis") & df.tag.isin(["none"] + [tag(x) for x in ATT])]
    for col, nm in [("benign_reject_iot", "IoT sensor"), ("benign_reject_mobile", "Smartphone"),
                    ("benign_reject", "All")]:
        lines.append(nm + " & " + " & ".join(f(float(a[a.dataset == ds][col].mean())) if (a.dataset == ds).any()
                                             else "--" for ds in DS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_hetero.tex", "\n".join(lines))


def table_data(df, out):
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data import load_fed
    lines = [r"\begin{table}[!t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
             r"\caption{Datasets, federated splits ($N=500$ clients) and models. Val.\ = pooled 10\% "
             r"client hold-out minus the 500-sample root set.}", r"\label{tab:data}",
             r"\begin{tabular}{l l c c r r r c r}", r"\toprule",
             r"Dataset & Abbr. & Feat. & $C$ & Train & Val. & Test & Split & $d$ \\", r"\midrule"]
    for ds in DS:
        fd = load_fed(ds, device="cpu")
        feat = "x".join(str(s) for s in fd.in_shape)
        split = "writers" if ds == "femnist" else r"Dir(0.5)"
        dd = int(df[df.dataset == ds].d.iloc[0]) if (df.dataset == ds).any() else 0
        lines.append(f"{DSN[ds]} & {DSA[ds]} & {feat} & {fd.n_classes} & {int(fd.counts.sum()):,} & {len(fd.y_val):,} & "
                     f"{len(fd.y_test):,} & {split} & {dd:,}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    write(out, "tab_data.tex", "\n".join(lines).replace(",", "{,}"))


def numbers(df, out, summary_cols, asr, sig=(0, 0)):
    """Macros for the text: \\aegisAvgGap etc.  Each macro's provenance goes to numbers_provenance.md."""
    mac = {}
    for j, ds in enumerate(DS):
        clean_c, avg_c, worst_c = summary_cols[3 * j], summary_cols[3 * j + 1], summary_cols[3 * j + 2]
        ia = METH.index("aegis")
        key = {"ciciot2023": "CICIoT", "edgeiiot": "EdgeIIoT", "sensit": "SensIT", "fmnist": "FMNIST",
               "femnist": "FEMNIST", "cifar10": "CIFAR"}[ds]
        mac[f"clean{key}"] = f(clean_c[ia])
        mac[f"avg{key}"] = f(avg_c[ia])
        mac[f"worst{key}"] = f(worst_c[ia])
        others = [(avg_c[i], METH[i]) for i in range(len(METH)) if i != ia and avg_c[i] == avg_c[i]]
        if others:
            b = max(others)
            mac[f"bestBaseAvg{key}"] = f(b[0])
            mac[f"bestBaseName{key}"] = MN[b[1]]
        prov.append(f"- clean/avg/worst{key}: tab_summary column {DSN[ds]} (runs {ds}/<method>/<tag>/seed)")
    # averaged across datasets
    ia = METH.index("aegis")
    avg_all = np.nanmean([summary_cols[3 * j + 1][ia] for j in range(len(DS))])
    clean_all = np.nanmean([summary_cols[3 * j][ia] for j in range(len(DS))])
    mac["aegisAvgAll"], mac["aegisCleanAll"] = f(avg_all), f(clean_all)
    mac["aegisGapAll"] = f(clean_all - avg_all)
    bd = [v[ia][0] for v in asr if v[ia][0] == v[ia][0]]
    fa = [v[0][0] for v in asr if v[0][0] == v[0][0]]
    mac["aegisASRmean"] = f(np.mean(bd)) if bd else "--"
    mac["fedavgASRmean"] = f(np.mean(fa)) if fa else "--"
    b = df[(df.method == "aegis") & (df.tag == "none")].benign_reject
    mac["aegisBenignRejNone"] = f(float(b.mean())) if len(b) else "--"
    # per attack, averaged over the six datasets: \\acc<Attack><Method>, \\tpr<Attack>AEGIS
    an = {"none": "None", "label_flip": "LabelFlip", "sign_flip": "SignFlip", "gaussian": "Gaussian", "lie": "LIE",
          "min_max": "MinMax", "min_sum": "MinSum", "dba": "DBA", "neurotoxin": "Neurotoxin", "adaptive": "Adaptive"}
    mn = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "TrMean", "median": "Median", "fltrust": "FLTrust",
          "flame": "FLAME", "fldetector": "FLDetector", "freqfed": "FreqFed", "aegis": "AEGIS"}
    for a in ["none"] + ATT:
        for m in METH:
            v = [cell(df, ds, m, tag(a))[0] for ds in DS]
            mac[f"acc{an[a]}{mn[m]}"] = f(np.mean(v)) if all(x == x for x in v) else "--"
            if a in ("dba", "neurotoxin"):
                v = [cell(df, ds, m, tag(a), "test_asr_last5")[0] for ds in DS]
                mac[f"asr{an[a]}{mn[m]}"] = f(np.mean(v)) if all(x == x for x in v) else "--"
        v = [cell(df, ds, "aegis", tag(a), "tpr")[0] for ds in DS]
        if a != "none":
            mac[f"tpr{an[a]}AEGIS"] = f(np.mean(v)) if all(x == x for x in v) else "--"
    prov.append("- acc<Attack><Method>, asr<Attack><Method>, tpr<Attack>AEGIS: mean over the six datasets of the "
                "per-dataset seed means (tab_full / tab_asr / summary tpr)")
    mac["sigWins"], mac["sigAll"] = str(sig[0]), str(sig[1])
    mac.update(EXTRA)
    # stronger adaptive attacker (target quantile 0.99): dataset-averaged accuracy per method
    for m in METH:
        v = [cell(df, ds, m, "adaptive-q0.99_g0.3")[0] for ds in DS]
        mac[f"accAdaptStrong{mn[m]}"] = f(np.mean(v)) if all(x == x for x in v) else "--"
    v = [cell(df, ds, "aegis", "adaptive-q0.99_g0.3", "tpr")[0] for ds in DS]
    mac["tprAdaptStrongAEGIS"] = f(np.mean(v)) if all(x == x for x in v) else "--"
    main = df[df.tag.isin(["none"] + [tag(a) for a in ATT]) & df.method.isin(METH)]
    # ablation macros (sign flip, label flip, Min-Max; averaged as in tab_ablation)
    abl_atts = ["sign_flip", "label_flip", "min_max"]
    def abl(v, ds=None, col="test_acc_last5"):
        dss = [ds] if ds else DS
        vals = [cell(df, d, v, tag(a), col)[0] for d in dss for a in abl_atts]
        vals = [x for x in vals if x == x]
        return float(np.mean(vals)) if vals else math.nan
    for v, k in [("aegis", "Full"), ("aegis-no_conformal", "NoConf"), ("aegis-no_reputation", "NoRep")]:
        mac[f"ablRej{k}"] = f(abl(v, col="benign_reject"), d=0)
    for v, k in [("aegis-no_norm", "NoNorm"), ("aegis-no_collusion", "NoColl")]:
        mac[f"ablDropCIFAR{k}"] = f(abl("aegis", "cifar10") - abl(v, "cifar10"))
    x = df[df.tag.isin([tag(a) for a in abl_atts]) & (df.method == "aegis-no_reputation")]
    mac["ablMalNoRep"] = f(float(x.mal_frac_received_last100.astype(float).mean()), d=0)
    mac["nSeedsMain"] = str(int(main.groupby(["dataset", "method", "tag"]).seed.nunique().min()))
    mac["nPairs"] = str(int(main.groupby(["dataset", "method", "tag"]).seed.nunique().min()) * len(DS))
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    write(out, "numbers.tex", "\n".join(lines) + "\n")
    prov.append("- aegisAvgAll / aegisCleanAll / aegisGapAll: mean over the six datasets of tab_summary AEGIS-FL")
    prov.append("- aegisASRmean / fedavgASRmean: mean of tab_asr AEGIS-FL / FedAvg cells (12 columns)")
    prov.append("- aegisBenignRejNone: mean benign_reject of <ds>/aegis/none runs")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../tables")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    print(len(df), "finished runs")
    df = df[df.rounds_run.astype(float) > 0]
    main_seeds = df[df.tag.isin(["none"] + [tag(x) for x in ATT]) & df.method.isin(METH)].groupby(
        ["dataset", "method", "tag"]).seed.nunique()
    prov.append(f"Main-grid cells: {len(main_seeds)} (expected {len(DS) * len(METH) * 10}); "
                f"seeds per cell min {main_seeds.min() if len(main_seeds) else 0}, max "
                f"{main_seeds.max() if len(main_seeds) else 0}.")
    prov.append(f"Diverged runs: {int(df.diverged.astype(bool).sum())} of {len(df)} "
                "(a diverged run reports the accuracy of its non-finite model, i.e. chance level).")
    cols = table_summary(df, out)
    table_full(df, out)
    asr = table_asr(df, out)
    sig = table_sig(df, out)
    table_detect(df, out)
    table_conformal(df, out)
    table_ablation(df, out)
    table_overhead(df, out)
    table_energy(df, out)
    table_privacy(df, out)
    table_hetero(df, out)
    table_data(df, out)
    numbers(df, out, cols, asr, sig)
    (out / "numbers_provenance.md").write_text("# Provenance of generated numbers\n\n" + "\n".join(prov) + "\n")


if __name__ == "__main__":
    main()
