"""Minimal job queue: run shell commands from a file with N parallel workers, skipping jobs whose
results/runs/<exp>/<seed>/summary.json already exists (resume-safe).

    python scripts/jobqueue.py jobs.txt --workers 3
Each line: <exp> <seed> <command...>
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


def run(line):
    exp, seed, cmd = line.split(" ", 2)
    done = CODE / "results" / "runs" / exp / seed / "summary.json"
    meta = CODE / "results" / "runs" / exp / seed / "meta.json"
    if done.exists() and meta.exists() and '"FINISHED"' in meta.read_text():
        return f"skip {exp} {seed}"
    log = CODE / "results" / "logs" / (exp.replace("/", "__") + f"__s{seed}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for attempt in range(4):                     # retry transient CUDA out-of-memory failures (shared GPU)
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, shell=True, cwd=CODE, stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0 or "out of memory" not in log.read_text(errors="ignore").lower():
            break
        time.sleep(60 * (attempt + 1))
    return f"{'ok' if rc == 0 else 'FAIL(' + str(rc) + ')'} {exp} {seed} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    lines = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    with ThreadPoolExecutor(a.workers) as ex:
        for msg in ex.map(run, lines):
            print(msg, flush=True)
