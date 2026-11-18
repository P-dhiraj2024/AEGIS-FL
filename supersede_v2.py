"""v2: runs whose history is empty or shorter than expected because rounds with no finite update
skipped the evaluation (fixed in src/run.py); moved to results/runs_superseded/v2 (re-run by the
pipeline's retry pass)."""
import json
import shutil
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
RUNS = CODE / "results" / "runs"
dst = CODE / "results" / "runs_superseded" / "v2"
moved = []
for s in sorted(RUNS.glob("*/*/*/*/summary.json")):
    d = s.parent
    if d.parts[-4] in ("lr", "dbg", "smoke"):
        continue
    summ = json.loads(s.read_text())
    h = json.loads((d / "history.json").read_text())
    if summ.get("test_acc") is None or (not summ.get("diverged") and len(h) < 30):
        t = dst / d.relative_to(RUNS)
        t.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(t))
        moved.append(str(d.relative_to(RUNS)) + f" (history {len(h)})")
with open(dst / "MOVED.txt", "a") as fh:
    fh.write("\n".join(moved) + ("\n" if moved else ""))
print(len(moved), "moved"); print("\n".join(moved))
