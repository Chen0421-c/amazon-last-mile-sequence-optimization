#!/bin/bash
#SBATCH --job-name=hybrid_w_50w
#SBATCH --output=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/hybrid_w_50w_%j.out
#SBATCH --error=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/hybrid_w_50w_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

export PYTHONNOUSERSITE=1
unset PYTHONPATH

PY=/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python

cd /home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization

mkdir -p /home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs

echo "Start hybrid validation full 50w weight search"
date
hostname
$PY -s --version

$PY -s - <<'PYTEST'
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

$PY -s 12_optimize_hybrid_weights.py \
  --config config/config_dicc.yaml \
  --model-output-dir "/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/model_outputs_full_top3" \
  --output-dir "/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/hybrid_weight_search_validation_full_50w" \
  --split validation \
  --weight-step 0.10 \
  --max-weight-combinations 50 \
  --seed 42 \
  --overwrite \
  --verbose

echo "Finished hybrid validation full 50w weight search"
date

OUT=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/hybrid_weight_search_validation_full_50w

ls -lh "$OUT"
cat "$OUT/best_weight_summary.csv"
cat "$OUT/baseline_method_summary.csv"
cat "$OUT/hybrid_weight_search_run_summary.csv"
