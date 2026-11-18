"""Import every finished run (results/runs/<exp>/<seed>) into MLflow (results/mlruns/mlflow.db).

Runs are executed by up to 7 parallel workers; logging them to one SQLite MLflow store from
all workers at once causes lock contention, so runs write plain files (config, seed, git
commit, wall time, history, summary; core.utils.tracking.RunLogger) and this script imports
them afterwards in one process.  Already imported runs (tag source_dir) are skipped.
"""
import json
import os
from pathlib import Path

import yaml

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
RES = CODE / "results"


def flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


def main():
    db = RES / "mlruns"
    db.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{db / 'mlflow.db'}")
    n = 0
    for s in sorted(RES.glob("runs/*/*/*/*/summary.json")):
        run_dir = s.parent
        meta = json.loads((run_dir / "meta.json").read_text())
        if meta.get("status") != "FINISHED":
            continue
        exp = "paperB/" + run_dir.parent.relative_to(RES / "runs").parts[0]
        if mlflow.get_experiment_by_name(exp) is None:
            mlflow.create_experiment(exp, artifact_location=(db / "artifacts" / exp.replace("/", "_")).as_uri())
        e = mlflow.get_experiment_by_name(exp)
        src = str(run_dir.relative_to(RES))
        if mlflow.search_runs([e.experiment_id], filter_string=f"tags.source_dir = '{src}'").shape[0]:
            continue
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
        with mlflow.start_run(experiment_id=e.experiment_id, run_name=src):
            mlflow.set_tags(dict(source_dir=src, git_commit=meta.get("git_commit"), host=meta.get("host"),
                                 start_time=meta.get("start_time")))
            params = {k: str(v)[:500] for k, v in flat(cfg).items()}
            for i in range(0, len(params), 90):
                mlflow.log_params(dict(list(params.items())[i:i + 90]))
            for row in json.loads((run_dir / "history.json").read_text()):
                st = int(row.pop("step"))
                mlflow.log_metrics({k: float(v) for k, v in row.items() if v == v}, step=st)
            summ = json.loads(s.read_text())
            mlflow.log_metrics({f"final.{k}": float(v) for k, v in summ.items()
                                if isinstance(v, (int, float)) and not isinstance(v, bool) and v == v
                                and abs(v) != float("inf")})
            mlflow.log_metric("wall_time_s", float(meta.get("wall_time_s", 0)))
        n += 1
    print("imported", n, "runs")


if __name__ == "__main__":
    main()
