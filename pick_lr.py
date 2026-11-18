"""Pick the learning rate per dataset from the FedAvg no-attack runs (results/runs/lr/<ds>/lr<lr>/0):
score = mean validation accuracy of the last 3 evaluations (diverged = -1); the selected LR is
the smallest one within TOL = 1 pp of the best score (prefers stability at equal accuracy).
Writes configs/lr.yaml and results/lr_selection.txt."""
import json
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parents[1]
TOL = 0.01
out, report = {}, []
for d in sorted((CODE / "results" / "runs" / "lr").iterdir()):
    cand = []
    for r in sorted(d.iterdir()):
        h = json.loads((r / "0" / "history.json").read_text())
        s = json.loads((r / "0" / "summary.json").read_text())
        v = float(np.mean([x["val_acc"] for x in h[-3:]])) if not s.get("diverged") else -1
        lr = float(r.name[2:])
        report.append(f"{d.name:12s} lr={lr:<6} val(last3)={v:.4f} test={s['test_acc']:.4f}")
        cand.append((lr, v))
    top = max(v for _, v in cand)
    out[d.name] = min(lr for lr, v in cand if v >= top - TOL)
(CODE / "configs" / "lr.yaml").write_text(yaml.safe_dump(out))
(CODE / "results" / "lr_selection.txt").write_text("\n".join(report) + "\n\nselected: " + json.dumps(out) + "\n")
print("\n".join(report)); print(out)
