#!/usr/bin/env bash
set -euo pipefail

# AWS-friendly end-to-end demo runner for the Functor GNN reproducibility workflow.
#
# Produces a timestamped artifact directory with:
#   - full console log
#   - per-step outputs
#   - a short summary file
#
# Examples:
#   ./run_all.sh
#   DATASET=synthetic GRAPH=cycle_00 ./run_all.sh
#   EPOCHS=50 COMMIT_EVERY=5 SEED=7 ./run_all.sh
#   OUTPUT_ROOT=demo_runs ./run_all.sh
#   PYTHON_BIN=/home/ubuntu/model-api/venv/bin/python3 ./run_all.sh

DATASET="${DATASET:-chemistry}"
if [[ "${DATASET}" == "chemistry" ]]; then
  GRAPH="${GRAPH:-ethanol}"
else
  GRAPH="${GRAPH:-cycle_00}"
fi

EPOCHS="${EPOCHS:-30}"
COMMIT_EVERY="${COMMIT_EVERY:-10}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-demo_runs}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}_${DATASET}_${GRAPH}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
LOG_FILE="${RUN_DIR}/run.log"
SUMMARY_FILE="${RUN_DIR}/summary.txt"

mkdir -p "${RUN_DIR}"

# Capture all output to both console and log file.
exec > >(tee -a "${LOG_FILE}")
exec 2>&1

run_step() {
  local step_num="$1"
  local step_name="$2"
  local step_file="$3"
  shift 3

  echo
  echo "=== STEP ${step_num}: ${step_name} ==="
  echo "[command] $*"
  "$@" | tee "${step_file}"
}

cat > "${SUMMARY_FILE}" <<EOF
Functor GNN reproducibility demo run
====================================
Run ID: ${RUN_ID}
Dataset: ${DATASET}
Graph: ${GRAPH}
Epochs: ${EPOCHS}
Commit every: ${COMMIT_EVERY}
Seed: ${SEED}
Output directory: ${RUN_DIR}

Expected reviewer flow
----------------------
1. Read run.log for the full execution trace.
2. Inspect 03_commits.txt for versioned commit history.
3. Inspect 04_infer.txt for baseline commit-indexed inference output.
4. Inspect 05_equiv_demo.txt for deterministic replay under fixed inputs
   using identity/default projection.
5. Inspect 06_rollback_demo.txt for reproducible rollback behavior.
6. Inspect 07_projection_equiv_demo.txt for non-identity projection equivalence
   with a negative control.

Scope note
----------
This run demonstrates commit-indexed reproducibility, deterministic replay, rollback,
and projection equivalence for a non-identity toy projection. The toy
contains_hetero_atom and cycle_detection tasks are reproducibility fixtures, not
benchmarks or chemistry/mQSAR validation; any fixture accuracy printed by low-level
training logs must not be read as a performance result.
EOF

echo "[config] DATASET=${DATASET} GRAPH=${GRAPH}"
echo "[config] EPOCHS=${EPOCHS} COMMIT_EVERY=${COMMIT_EVERY} SEED=${SEED}"
echo "[python] PYTHON_BIN=${PYTHON_BIN}"
"${PYTHON_BIN}" - <<'PYENV'
import sys
print(f"[python] executable={sys.executable}")
print(f"[python] version={sys.version.split()[0]}")
try:
    import torch
    print(f"[torch] version={torch.__version__}")
except Exception as exc:
    print(f"[torch] import_error={exc.__class__.__name__}: {exc}")
    raise
PYENV
echo "[artifacts] RUN_DIR=${RUN_DIR}"
echo "[artifacts] LOG_FILE=${LOG_FILE}"
echo "[artifacts] SUMMARY_FILE=${SUMMARY_FILE}"

run_step 1 "GENERATE GRAPH" "${RUN_DIR}/01_generate.txt" \
  "${PYTHON_BIN}" cli.py generate --dataset "${DATASET}"

run_step 2 "TRAIN MODEL (VERSIONED COMMITS)" "${RUN_DIR}/02_train.txt" \
  "${PYTHON_BIN}" cli.py train --epochs "${EPOCHS}" --commit-every "${COMMIT_EVERY}"

run_step 3 "SHOW COMMITS" "${RUN_DIR}/03_commits.txt" \
  "${PYTHON_BIN}" cli.py commits

run_step 4 "INFERENCE" "${RUN_DIR}/04_infer.txt" \
  "${PYTHON_BIN}" cli.py infer --graph "${GRAPH}" --seed "${SEED}"

run_step 5 "DETERMINISTIC REPLAY DEMO" "${RUN_DIR}/05_equiv_demo.txt" \
  "${PYTHON_BIN}" cli.py equiv-demo --graph "${GRAPH}" --seed "${SEED}"

run_step 6 "ROLLBACK DEMO" "${RUN_DIR}/06_rollback_demo.txt" \
  "${PYTHON_BIN}" cli.py rollback-demo --graph "${GRAPH}" --seed "${SEED}"

run_step 7 "PROJECTION-EQUIVALENCE DEMO" "${RUN_DIR}/07_projection_equiv_demo.txt" \
  "${PYTHON_BIN}" cli.py projection-equiv-demo --graph "${GRAPH}" --seed "${SEED}"

echo
echo "[done] Demo artifacts available in ${RUN_DIR}"
echo "[done] Start with ${SUMMARY_FILE} and ${LOG_FILE}"
