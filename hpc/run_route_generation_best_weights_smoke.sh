#!/bin/bash
#SBATCH --job-name=route_gen_smoke
#SBATCH --output=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/route_gen_smoke_%j.out
#SBATCH --error=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/route_gen_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH

PY=/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python
REPO=/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
DATA=/home/user/chenziliang/dissertation/amazon_last_mile

cd "$REPO"
mkdir -p "${DATA}/final_experiment_outputs/logs"

echo "Start best-weight route generation smoke test"
date
hostname
$PY -s --version

$PY -s -u 13_route_generation_best_weights.py \
  --config config/config_dicc.yaml \
  --model-output-dir "${DATA}/final_experiment_outputs/model_outputs_full_top3" \
  --weight-search-dir "${DATA}/final_experiment_outputs/hybrid_weight_search_validation_full_300w" \
  --output-dir "${DATA}/final_experiment_outputs/route_generation_best_weights_smoke" \
  --splits validation \
  --max-routes 10 \
  --overwrite \
  --verbose

echo "Finished best-weight route generation smoke test"
date

OUT=${DATA}/final_experiment_outputs/route_generation_best_weights_smoke
ls -lh "$OUT"
cat "$OUT/method_summary_validation.csv"
cat "$OUT/route_generation_run_summary.csv"
