"""Move runs made with superseded code to results/runs_superseded/<version>/ (never deleted, rule 7).

v1 (2026-11-03): AEGIS-FL without norm features, adaptive attacker without feature mirroring,
and runs that diverged before non-finite client updates were discarded.
    python scripts/supersede.py v1
"""
import json
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
RUNS = CODE / "results" / "runs"
ver = sys.argv[1]
dst_root = CODE / "results" / "runs_superseded" / ver
moved = []
for seed_dir in sorted(RUNS.glob("*/*/*/*")):
    exp = seed_dir.parent.relative_to(RUNS)
    ds, method, tag = exp.parts
    if ds in ("lr", "dbg", "smoke"):
        continue
    s = seed_dir / "summary.json"
    div = s.exists() and json.loads(s.read_text()).get("diverged")
    unfinished = not (seed_dir / "meta.json").exists() or '"FINISHED"' not in (seed_dir / "meta.json").read_text()
    if method.startswith("aegis") or tag.startswith("adaptive") or div or unfinished:
        d = dst_root / exp / seed_dir.name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(seed_dir), str(d))
        moved.append(f"{exp}/{seed_dir.name}" + (" (diverged)" if div else "") + (" (unfinished)" if unfinished else ""))
(dst_root / "MOVED.txt").write_text("\n".join(moved) + "\n")
print(len(moved), "runs moved to", dst_root)
