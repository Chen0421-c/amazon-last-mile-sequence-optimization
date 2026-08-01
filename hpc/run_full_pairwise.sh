#!/bin/bash
#SBATCH --job-name=full_pairwise
#SBATCH --output=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/full_pairwise_%j.out
#SBATCH --error=/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs/full_pairwise_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

REPO_DIR=/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
DATA_DIR=/home/user/chenziliang/dissertation/amazon_last_mile
LOG_DIR=${DATA_DIR}/final_experiment_outputs/logs

JOB_SUFFIX=${JOB_SUFFIX:-full}
if [[ "$JOB_SUFFIX" == "full" ]]; then
  DEFAULT_OUTPUT_DIR=${DATA_DIR}/final_experiment_outputs/pairwise_samples_full
else
  DEFAULT_OUTPUT_DIR=${DATA_DIR}/final_experiment_outputs/pairwise_samples_${JOB_SUFFIX}
fi

OUTPUT_DIR=${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}
MAX_ROUTES_PER_SPLIT=${MAX_ROUTES_PER_SPLIT:-}
SEED=${SEED:-42}
NEGATIVE_SAMPLES_PER_POSITIVE=${NEGATIVE_SAMPLES_PER_POSITIVE:-5}
OUTPUT_FORMAT=${OUTPUT_FORMAT:-parquet}
SPLITS=${SPLITS:-"train validation test"}

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

date
hostname
pwd
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Slurm job id: $SLURM_JOB_ID"
fi
echo "Job suffix: $JOB_SUFFIX"
echo "Output dir: $OUTPUT_DIR"
echo "Splits: $SPLITS"
echo "Max routes per split: ${MAX_ROUTES_PER_SPLIT:-none}"
echo "Seed: $SEED"
echo "Negative samples per positive: $NEGATIVE_SAMPLES_PER_POSITIVE"
echo "Output format: $OUTPUT_FORMAT"

module load miniconda/24.11.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lastmile

which python
python --version

cd "$REPO_DIR"

python scripts/00_check_final_inputs.py --config config/config_dicc.yaml

MAX_ROUTE_ARGS=()
if [[ -n "$MAX_ROUTES_PER_SPLIT" ]]; then
  MAX_ROUTE_ARGS+=(--max-routes-per-split "$MAX_ROUTES_PER_SPLIT")
fi

python 09_create_full_pairwise_samples.py \
  --config config/config_dicc.yaml \
  --splits $SPLITS \
  --seed "$SEED" \
  --negative-samples-per-positive "$NEGATIVE_SAMPLES_PER_POSITIVE" \
  --output-format "$OUTPUT_FORMAT" \
  --output-dir "$OUTPUT_DIR" \
  "${MAX_ROUTE_ARGS[@]}"

date
ls -lh "$OUTPUT_DIR"
if [[ -f "$OUTPUT_DIR/pairwise_sample_summary.csv" ]]; then
  cat "$OUTPUT_DIR/pairwise_sample_summary.csv"
fi
if [[ -f "$OUTPUT_DIR/pairwise_quality_report.csv" ]]; then
  cat "$OUTPUT_DIR/pairwise_quality_report.csv"
fi
