#!/bin/bash
#SBATCH --job-name=route_gen_best
#SBATCH --output=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/route_gen_best_%j.out
#SBATCH --error=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/route_gen_best_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH

PY=/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python
REPO=/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
DATA=/home/user/chenziliang/dissertation/amazon_last_mile

cd "$REPO"
mkdir -p "${DATA}/final_experiment_outputs/logs"

echo "Start best-weight route generation"
date
hostname
$PY -s --version

$PY -s -u 13_route_generation_best_weights.py \
  --config config/config_dicc.yaml \
  --model-output-dir "${DATA}/final_experiment_outputs/model_outputs_full_top3" \
  --weight-search-dir "${DATA}/final_experiment_outputs/hybrid_weight_search_validation_full_300w" \
  --output-dir "${DATA}/final_experiment_outputs/route_generation_best_weights" \
  --splits validation test \
  --overwrite \
  --verbose

echo "Finished best-weight route generation"
date

OUT=${DATA}/final_experiment_outputs/route_generation_best_weights
ls -lh "$OUT"
cat "$OUT/method_summary_validation.csv"
cat "$OUT/method_summary_test.csv"
cat "$OUT/route_generation_run_summary.csv"
