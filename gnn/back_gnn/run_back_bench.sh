#!/usr/bin/env bash
set -euo pipefail

# BackGNN benchmark-ready commands. Wrap externally with RAPL, e.g.:
#   ./measure_rapl.sh ./back_gnn/run_back_bench.sh

PYTHON=${PYTHON:-python3}

# Train benchmark: includes backloaded materialization plus checkpoint commits.
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m back_gnn.cli train --epochs 30 --commit-every 10

# Inference benchmark: commit/prepared artifact replay over many hot iterations.
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m back_gnn.cli infer --graph ethanol --seed 42 --iterations 100000

# Lifecycle benchmark: materialize-only followed by a short inference replay.
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m back_gnn.cli materialize --graph ethanol --projection default
PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m back_gnn.cli infer --graph ethanol --seed 42 --iterations 1000
