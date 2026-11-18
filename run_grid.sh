#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 0 1 2 3 4; do python -m src.run --config configs/attacks.yaml --seed $s; done
