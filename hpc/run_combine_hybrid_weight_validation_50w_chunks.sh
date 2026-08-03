#!/bin/bash
#SBATCH --job-name=combine_w50
#SBATCH --output=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/combine_w50_%j.out
#SBATCH --error=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/combine_w50_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH

PY=/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python
REPO=/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
DATA=/home/user/chenziliang/dissertation/amazon_last_mile
ROOT=${DATA}/final_experiment_outputs/hybrid_weight_search_validation_full_50w_chunks
OUT=${DATA}/final_experiment_outputs/hybrid_weight_search_validation_full_50w

mkdir -p "${DATA}/final_experiment_outputs/logs"

cd "$REPO"

echo "Start combining hybrid validation 50w chunks"
date
hostname
$PY -s -u --version

$PY -s -u hpc/combine_hybrid_weight_chunks.py \
  --chunk-root "$ROOT" \
  --output-dir "$OUT" \
  --expected-chunks 15 \
  --overwrite

echo "Finished combining hybrid validation 50w chunks"
date
ls -lh "$OUT"
cat "$OUT/best_weight_summary.csv"
cat "$OUT/baseline_method_summary.csv"
cat "$OUT/hybrid_weight_search_run_summary.csv"
