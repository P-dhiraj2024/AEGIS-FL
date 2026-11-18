"""Load every finished run under results/runs into one table (one row per run)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CODE = Path(__file__).resolve().parents[1]
RUNS = CODE / "results" / "runs"


def load_runs(with_history: bool = False) -> pd.DataFrame:
    rows = []
    for s in RUNS.glob("*/*/*/*/summary.json"):
        meta = s.parent / "meta.json"
        if not meta.exists() or '"FINISHED"' not in meta.read_text():
            continue
        exp = s.parent.parent.relative_to(RUNS).as_posix()
        if exp.split("/")[0] in ("lr", "dbg", "smoke", "diag"):
            continue
        d = json.loads(s.read_text())
        d["exp"], d["tag"] = exp, exp.split("/")[2]
        d["method"] = exp.split("/")[1]
        d["git_commit"] = json.loads(meta.read_text()).get("git_commit")
        h = json.loads((s.parent / "history.json").read_text())
        if "test_acc_last5" not in d and h:          # runs from before the summary had it
            d["test_acc_last5"] = sum(r["test_acc"] for r in h[-5:]) / len(h[-5:])
            d["test_f1_last5"] = sum(r["test_f1"] for r in h[-5:]) / len(h[-5:])
            if "test_asr" in h[-1]:
                d["test_asr_last5"] = sum(r["test_asr"] for r in h[-5:]) / len(h[-5:])
        if with_history:
            d["history"] = h
        rows.append(d)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_runs()
    print(len(df), "runs")
    if len(df):
        cols = ["dataset", "method", "tag", "seed", "test_acc", "test_f1", "test_asr", "tpr", "fpr", "benign_reject",
                "wall_s"]
        print(df[cols].sort_values(["dataset", "tag", "method", "seed"]).to_string(float_format=lambda v: f"{v:.3f}"))
