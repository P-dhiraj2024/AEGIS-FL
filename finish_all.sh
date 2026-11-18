#!/usr/bin/env bash
# Full Paper B pipeline (resume-safe: finished runs are skipped).
#   [wait for a running queue] -> ablation (0,1) -> privacy (0,1) -> conformal (0-2) -> adaptive q0.99 (0-2) -> ratio (0,1)
#   -> main seeds 0-4 -> retry pass -> MLflow sync -> tables -> figures
set -uo pipefail
cd "$(dirname "$0")/.."
PY=../../../.venv/bin/python
W=${WORKERS:-7}
log() { echo "[$(date '+%F %T')] $*"; }
if [ -f results/logs/queue_main.pid ]; then
  P=$(cat results/logs/queue_main.pid)
  while kill -0 "$P" 2>/dev/null; do sleep 60; done
  log "running main queue finished"
fi
run_phase() {   # phase, seeds...
  local ph=$1; shift
  $PY scripts/make_jobs.py "$ph" --seeds "$@" > /dev/null
  cp "results/jobs_${ph}.txt" "results/jobs_${ph}_$(echo "$@" | tr ' ' '_').txt"
  log "phase $ph seeds $*: $(wc -l < results/jobs_${ph}.txt) jobs"
  $PY scripts/jobqueue.py "results/jobs_${ph}.txt" --workers "$W" >> "results/logs/queue_${ph}.log" 2>&1
}
tables() {
  $PY scripts/make_tables.py --out ../tables > results/logs/make_tables.log 2>&1
  $PY scripts/make_figures.py --out ../figures > results/logs/make_figures.log 2>&1
  log "tables/figures refreshed"
}
# (after the running ratio queue) retry runs lost to GPU OOM, refresh tables, then 5 seeds
run_phase privacy 0 1
run_phase ablation 0 1
run_phase conformal 0 1 2
run_phase adaptive 0 1 2
run_phase ratio 0 1
run_phase main 0 1 2
tables
run_phase main 0 1 2 3 4
# retry pass for anything that failed (e.g. transient CUDA OOM)
run_phase conformal 0 1 2
run_phase ablation 0 1
run_phase privacy 0 1
run_phase adaptive 0 1 2
run_phase ratio 0 1
run_phase main 0 1 2 3 4
log "runs done; failures: $(grep -c FAIL results/logs/queue_*.log | tr '\n' ' ')"
$PY scripts/sync_mlflow.py > results/logs/sync_mlflow.log 2>&1
tables
log "all done"
