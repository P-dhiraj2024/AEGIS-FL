#!/usr/bin/env python3
"""
upload_to_github.py
-------------------
Upload the entire contents of a folder (code + LaTeX + figures) to a
GitHub repository in one command.

Works with either:
  * a GitHub personal access token  (recommended, no extra tool needed), or
  * the `gh` CLI, if it is already installed and authenticated.

Install the one Python dependency:
    pip install requests

Create a token (one-time):
    https://github.com/settings/tokens
    Scope needed: "repo" (full control of private repositories).
    Export it so the script can read it:
        export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    (You can also pass it on the command line with --token.)

Usage from the folder you want to publish:

    # Brand-new repository, private by default, push everything in CWD:
    python upload_to_github.py --repo AEGIS-FL

    # Public release after acceptance:
    python upload_to_github.py --repo AEGIS-FL --visibility public --update

    # Point at a folder other than CWD:
    python upload_to_github.py --repo AEGIS-FL \
        --folder "/mnt/.../papers/B_AEGIS_TNSM"

    # Preview every command without executing anything:
    python upload_to_github.py --repo AEGIS-FL --dry-run

The script:
  1. ensures a sensible `.gitignore` and `README.md` exist,
  2. runs `git init` if the folder is not yet a repo,
  3. stages every file the `.gitignore` allows,
  4. makes a commit (skipped if nothing changed),
  5. creates the GitHub repo (via API or `gh`) if it does not exist yet,
  6. adds / resets `origin` and pushes the branch.

Datasets, result dumps, virtual envs, and LaTeX build junk are excluded
by default so you never ship a multi-gigabyte accident.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

GITHUB_API = "https://api.github.com"


# ---------------------------------------------------------------------------
# Files we ship with every upload
# ---------------------------------------------------------------------------

GITIGNORE = """\
# ---- byte-compiled / cache
__pycache__/
*.py[cod]
*.egg-info/
.ipynb_checkpoints/
.pytest_cache/

# ---- virtual envs
.venv/
venv/
env/

# ---- data (download separately, too large for GitHub)
data/raw/
data/processed/
data/partitions/

# ---- experiment results (regenerate locally)
results/
mlruns/
runs_superseded/

# ---- LaTeX build artefacts
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
*.fls
*.fdb_latexmk
*.nav
*.snm
*.vrb

# ---- OS / editor
.DS_Store
Thumbs.db
*.swp
.idea/
.vscode/

# ---- backups kept by the authoring workflow
*.bak
*.v?bak

# ---- secrets
.env
*.pem
*.key
"""

README_TEMPLATE = """\
# {repo_name}

Source for the paper

> **AEGIS-FL: Conformally Calibrated Multi-Test Update Filtering for
>  Byzantine-Robust Federated Learning in UAV-Assisted Edge Networks.**
> Dhiraj Pandey and Om Jee Pandey. IEEE TNSM, 2027 (under review).

## Repository layout

| Path                         | What it holds                              |
|------------------------------|--------------------------------------------|
| `main.tex`, `references.bib` | paper source                               |
| `cover.tex`                  | cover letter sent to the journal           |
| `figures/`, `tables/`        | generated figures / tables                 |
| `code/src/`                  | simulator, defense, attacks, baselines     |
| `code/configs/`              | experiment configurations (YAML)           |
| `code/scripts/`              | run grids and figure / table generators    |
| `code/requirements.txt`      | pinned Python dependencies                 |

## Quick start

```bash
cd code
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.run --config configs/default.yaml --seed 0 --smoke
```

Full reproduction:

```bash
bash scripts/finish_all.sh
python scripts/make_tables.py  --out ../tables
python scripts/make_figures.py --out ../figures
```

## Datasets

Place each dataset under `data/raw/<name>/` before running:
CICIoT2023, Edge-IIoTset, SensIT, Fashion-MNIST, FEMNIST, CIFAR-10.
Partition scripts live in `data/scripts/`.

## Citation

```bibtex
@article{{pandey2027aegisfl,
  author  = {{Pandey, Dhiraj and Pandey, Om Jee}},
  title   = {{AEGIS-FL}: Conformally Calibrated Multi-Test Update Filtering
             for Byzantine-Robust Federated Learning in UAV-Assisted
             Edge Networks},
  journal = {{IEEE Transactions on Network and Service Management}},
  year    = {{2027}},
  note    = {{under review}}
}}
```
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def shell(cmd: list[str], *, cwd: Path, dry: bool, check: bool = True,
          capture: bool = False) -> subprocess.CompletedProcess | None:
    """Run (or print) a shell command inside `cwd`."""
    printable = " ".join(shlex.quote(c) for c in cmd)
    print(f"  $ {printable}")
    if dry:
        return None
    return subprocess.run(
        cmd, cwd=cwd, check=check, text=True,
        capture_output=capture,
    )


def have(binary: str) -> bool:
    return shutil.which(binary) is not None


def ensure_file(path: Path, content: str, dry: bool) -> None:
    if path.exists():
        print(f"  = {path.name} already exists, keeping it")
        return
    print(f"  + writing {path.name}")
    if not dry:
        path.write_text(content)


# ---------------------------------------------------------------------------
# GitHub repo creation (REST API, fallback to gh)
# ---------------------------------------------------------------------------

def api_repo_exists(owner: str, name: str, token: str) -> bool:
    r = requests.get(
        f"{GITHUB_API}/repos/{owner}/{name}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    return r.status_code == 200


def api_user(token: str) -> str:
    r = requests.get(
        f"{GITHUB_API}/user",
        headers={"Authorization": f"Bearer {token}"},
        timeout=20,
    )
    r.raise_for_status()
    return r.json()["login"]


def api_create_repo(name: str, visibility: str, token: str,
                    description: str, owner: str | None) -> str:
    """Create a repository and return its clone URL (SSH form)."""
    private = visibility == "private"
    payload = {
        "name": name,
        "description": description,
        "private": private,
        "has_issues": True,
        "has_wiki": False,
        "auto_init": False,
    }
    if owner and owner != api_user(token):
        url = f"{GITHUB_API}/orgs/{owner}/repos"
    else:
        url = f"{GITHUB_API}/user/repos"
    r = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        },
        data=json.dumps(payload),
        timeout=30,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GitHub API refused repo creation "
                           f"({r.status_code}): {r.text}")
    return r.json()["ssh_url"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Push a folder (paper + code) to a GitHub repository."
    )
    p.add_argument("--repo", required=True,
                   help="Repository name, or OWNER/NAME to push under "
                        "an organization.")
    p.add_argument("--folder", default=".",
                   help="Folder to publish (default: current directory).")
    p.add_argument("--visibility",
                   choices=["public", "private", "internal"],
                   default="private",
                   help="Visibility used only on first creation. "
                        "Default: private.")
    p.add_argument("--branch", default="main",
                   help="Branch to push. Default: main.")
    p.add_argument("--message",
                   default="Initial release of AEGIS-FL",
                   help="Commit message for staged changes.")
    p.add_argument("--token",
                   help="GitHub personal access token with 'repo' scope. "
                        "Falls back to env var GITHUB_TOKEN. "
                        "If both are empty, the script will try the gh CLI.")
    p.add_argument("--update", action="store_true",
                   help="Do not create the repo; assume it already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print every command without executing.")
    args = p.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"[error] folder does not exist: {folder}", file=sys.stderr)
        return 1

    # Parse OWNER/NAME
    if "/" in args.repo:
        owner, name = args.repo.split("/", 1)
    else:
        owner, name = None, args.repo

    for b in ("git",):
        if not have(b):
            print(f"[error] {b!r} not found on PATH", file=sys.stderr)
            return 1

    token = args.token or os.environ.get("GITHUB_TOKEN")
    use_gh = (token is None) and have("gh")
    if token is None and not use_gh:
        print("[error] no --token/GITHUB_TOKEN and `gh` is not installed. "
              "Pick one and retry.", file=sys.stderr)
        return 1
    if token and requests is None:
        print("[error] requests is not installed. "
              "Run: pip install requests", file=sys.stderr)
        return 1

    print(f"[info] folder     : {folder}")
    print(f"[info] repo       : {args.repo}")
    print(f"[info] visibility : {args.visibility}")
    print(f"[info] branch     : {args.branch}")
    print(f"[info] auth mode  : {'token (API)' if token else 'gh CLI'}")
    print(f"[info] dry-run    : {args.dry_run}")
    dry = args.dry_run

    # 1. init if needed
    if not (folder / ".git").exists():
        print("[step] git init")
        shell(["git", "init", "-b", args.branch], cwd=folder, dry=dry)
    else:
        print("[step] git repo already initialised")

    # 2. ship README / .gitignore
    print("[step] ensuring .gitignore and README.md")
    ensure_file(folder / ".gitignore", GITIGNORE, dry)
    ensure_file(folder / "README.md",
                README_TEMPLATE.format(repo_name=name), dry)

    # 3. stage + commit
    print("[step] staging files")
    shell(["git", "add", "-A"], cwd=folder, dry=dry)

    print("[step] committing (skip if nothing to commit)")
    if dry:
        shell(["git", "commit", "-m", args.message], cwd=folder,
              dry=True, check=False)
    else:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=folder,
            text=True, capture_output=True, check=True)
        if status.stdout.strip():
            shell(["git", "commit", "-m", args.message], cwd=folder, dry=dry)
        else:
            print("  = nothing to commit, tree clean")

    # 4. create the GitHub repo (unless --update)
    description = ("AEGIS-FL: Conformally Calibrated Multi-Test Update "
                   "Filtering for Byzantine-Robust Federated Learning in "
                   "UAV-Assisted Edge Networks (IEEE TNSM).")
    clone_url: str | None = None

    if args.update:
        print("[step] skipping repo creation (--update)")
    else:
        print("[step] creating GitHub repo (if not already present)")
        if token:
            # resolve owner for API create
            api_owner = owner or (None if dry else api_user(token))
            if dry:
                print(f"  (dry) would create {api_owner or '<me>'}/{name}")
            else:
                full_owner = api_owner or api_user(token)
                if api_repo_exists(full_owner, name, token):
                    print(f"  = {full_owner}/{name} already exists, reusing")
                    clone_url = (
                        f"git@github.com:{full_owner}/{name}.git"
                    )
                else:
                    clone_url = api_create_repo(
                        name, args.visibility, token, description, owner)
                    print(f"  + created {full_owner}/{name}")
        else:
            # gh CLI path
            flag = f"--{args.visibility}"
            shell(["gh", "repo", "create", args.repo, flag,
                   "--source", str(folder),
                   "--remote", "origin",
                   "--push",
                   "--description", description],
                  cwd=folder, dry=dry)
            print("[done] created and pushed via gh")
            return 0

    # 5. make sure 'origin' points at the right URL, then push
    print("[step] configuring remote `origin`")
    if not dry:
        remotes = subprocess.run(
            ["git", "remote"], cwd=folder,
            text=True, capture_output=True, check=True).stdout.split()
        if "origin" in remotes:
            if clone_url:
                shell(["git", "remote", "set-url", "origin", clone_url],
                      cwd=folder, dry=dry)
        elif clone_url:
            shell(["git", "remote", "add", "origin", clone_url],
                  cwd=folder, dry=dry)
        elif args.update:
            print("  [error] --update was passed but no 'origin' remote "
                  "is configured.", file=sys.stderr)
            return 1
    else:
        print("  (dry) would ensure `origin` is set")

    print("[step] pushing")
    shell(["git", "push", "-u", "origin", args.branch], cwd=folder, dry=dry)

    full_name = f"{owner}/{name}" if owner else name
    print(f"[done] https://github.com/{full_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
