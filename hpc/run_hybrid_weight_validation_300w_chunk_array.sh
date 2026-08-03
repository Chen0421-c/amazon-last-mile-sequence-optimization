#!/bin/bash
#SBATCH --job-name=hybrid_w300_chunk
#SBATCH --output=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/hybrid_w300_chunk_%A_%a.out
#SBATCH --error=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/hybrid_w300_chunk_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-21%4

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH

PY=/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python
REPO=/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
DATA=/home/user/chenziliang/dissertation/amazon_last_mile
ROOT=${DATA}/final_experiment_outputs/hybrid_weight_search_validation_full_300w_chunks
CHUNK_DIR=${ROOT}/route_chunks

CHUNK_ID=$(printf "%03d" "${SLURM_ARRAY_TASK_ID}")
ROUTE_IDS=${CHUNK_DIR}/validation_chunk_${CHUNK_ID}.csv
OUT=${ROOT}/chunk_${CHUNK_ID}

mkdir -p "${DATA}/final_experiment_outputs/logs" "$OUT"

cd "$REPO"

echo "Start hybrid validation 300w chunk ${CHUNK_ID}"
date
hostname
echo "Route IDs: $ROUTE_IDS"
echo "Output: $OUT"
$PY -s -u --version

if [[ ! -f "$ROUTE_IDS" ]]; then
  echo "ERROR: route chunk file not found: $ROUTE_IDS" >&2
  exit 1
fi

$PY -s -u - <<'PYTEST'
import sys, site
import sklearn, numpy, pandas, joblib, catboost, xgboost, lightgbm, pyarrow
print("python:", sys.executable)
print("user site enabled:", site.ENABLE_USER_SITE)
print("sklearn:", sklearn.__version__)
print("numpy:", numpy.__file__)
print("pandas:", pandas.__file__)
print("catboost ok")
print("xgboost ok")
print("lightgbm ok")
print("pyarrow ok")
PYTEST

$PY -s -u 12_optimize_hybrid_weights.py \
  --config config/config_dicc.yaml \
  --model-output-dir "${DATA}/final_experiment_outputs/model_outputs_full_top3" \
  --output-dir "$OUT" \
  --split validation \
  --route-ids "$ROUTE_IDS" \
  --weight-step 0.05 \
  --max-weight-combinations 300 \
  --seed 42 \
  --save-route-level \
  --overwrite

echo "Finished hybrid validation 300w chunk ${CHUNK_ID}"
date
ls -lh "$OUT"
cat "$OUT/hybrid_weight_search_run_summary.csv"
