# Final Experiment Runbook

This runbook records the final reproducible experiment sequence for the dissertation project:

**Road Freight Delivery Sequence Optimization Using Machine Learning**

It is intended for final submission, dissertation writing, and defence preparation. It does not store raw data or large output artifacts in GitHub. The paths below refer to Google Drive or DICC storage.

## 1. Final research objective

The final experiment learns driver-like next-stop preference from historical Amazon last-mile driver sequences and integrates the learned preference signal into a hybrid route-generation framework.

The final goal is to evaluate a trade-off:

- **Operational efficiency:** travel-time ratio to actual route.
- **Driver-like routing behaviour:** LCS similarity, position match ratio, same-zone continuity, and zone-change reduction.

The final result should be interpreted as a trade-off. The hybrid method does not beat the travel-time nearest-neighbour baseline in pure travel-time efficiency, but it significantly improves driver-like route structure.

## 2. Storage paths

### Google Drive / Colab

```text
/content/drive/MyDrive/dissertation/amazon_last_mile
/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs
/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs/final_cleaned
/content/drive/MyDrive/dissertation/amazon_last_mile/final_experiment_outputs
```

### DICC

```text
/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
/home/user/chenziliang/dissertation/amazon_last_mile
/home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs
```

## 3. Environment

### Colab

Install final dependencies:

```bash
pip install -r requirements.txt
```

CatBoost, XGBoost, LightGBM, PyArrow, SciPy, and scikit-learn are required for final model loading, prediction, and statistical analysis.

### DICC

The stable final DICC Python path was:

```text
/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python -s
```

Use:

```bash
export PYTHONNOUSERSITE=1
unset PYTHONPATH
PY=/home/user/chenziliang/.conda/envs/lastmile_clean/bin/python
```

The `-s` flag is important because it prevents Python from loading user-site packages from `~/.local`, which previously caused GLIBC and package-version conflicts.

The final model was trained with `scikit-learn==1.6.1`. Loading the saved `best_model.joblib` under newer scikit-learn versions may fail.

Check DICC environment:

```bash
$PY -s - <<'PY'
import sys, site
import sklearn, numpy, pandas, joblib, catboost, xgboost, lightgbm, pyarrow
print("python:", sys.executable)
print("user site enabled:", site.ENABLE_USER_SITE)
print("sklearn:", sklearn.__version__)
print("numpy:", numpy.__file__)
print("pandas:", pandas.__file__)
print("all packages ok")
PY
```

Expected:

```text
user site enabled: False
sklearn: 1.6.1
```

## 4. Final experiment pipeline

### Step 09: Full pairwise sample generation

Script:

```text
09_create_full_pairwise_samples.py
```

Final output:

```text
final_experiment_outputs/pairwise_samples_full
```

Main outputs:

```text
train_pairwise_samples.parquet
validation_pairwise_samples.parquet
test_pairwise_samples.parquet
pairwise_sample_summary.csv
pairwise_quality_report.csv
pairwise_full_consolidated_summary.csv
```

Final sample counts:

```text
train:      4,240,455 rows, 5,009 routes
validation: 899,583 rows, 1,073 routes
test:       914,796 rows, 1,073 routes
```

Each decision context contains exactly one positive sample.

### Step 10: Full preference model training

Script:

```text
10_train_full_preference_models.py
```

Final output:

```text
final_experiment_outputs/model_outputs_full_top3
```

Final top-3 models:

```text
XGBoost
LightGBM
CatBoost
```

Selected model:

```text
CatBoost
```

Final model files:

```text
model_outputs_full_top3/models/best_model.joblib
model_outputs_full_top3/feature_columns.json
```

### Step 11: Subgroup analysis

Script:

```text
11_model_subgroup_analysis.py
```

Final output:

```text
final_experiment_outputs/model_subgroup_analysis_full_top3
```

Key subgroup analyses:

```text
route_score
number_of_stops_bin
route_progress_bin
remaining_stop_count_bin
transition_zone_type
candidate_time_window_group
candidate_package_count_bin
candidate_service_time_bin
```

### Step 12: Hybrid weight optimization

Script:

```text
12_optimize_hybrid_weights.py
```

Final output:

```text
final_experiment_outputs/hybrid_weight_search_validation_full_300w
```

DICC chunked execution output:

```text
final_experiment_outputs/hybrid_weight_search_validation_full_300w_chunks
```

Final selected weights:

```text
weight_id          = w000091
travel_weight      = 0.70
preference_weight  = 0.15
zone_weight        = 0.15
time_window_weight = 0.00
workload_weight    = 0.00
```

Important outputs:

```text
best_weight_summary.csv
baseline_method_summary.csv
weight_grid_results.csv
hybrid_weight_search_run_summary.csv
best_weight_route_metrics.csv
baseline_route_metrics.csv
```

### Step 13: Final route generation

Script:

```text
13_route_generation_best_weights.py
```

Final output:

```text
final_experiment_outputs/route_generation_best_weights
```

Compared methods:

```text
travel_time_nearest_neighbor
preference_greedy
hybrid_greedy_best_weight
```

Important outputs:

```text
best_weight_used.csv
route_generation_run_summary.csv
route_metrics_validation.csv
route_metrics_test.csv
route_metrics_all_splits.csv
method_summary_validation.csv
method_summary_test.csv
method_summary_all_splits.csv
generated_routes_validation.csv
generated_routes_test.csv
```

Final route counts:

```text
validation: 1073 processed routes
test:       1072 processed routes
```

One test route was skipped due to invalid sequence, but valid route rate for generated routes is 1.0.

### Step 15: Statistical tests

Script:

```text
15_statistical_tests.py
```

Final output:

```text
final_experiment_outputs/route_generation_best_weights/statistical_tests
```

Compared methods:

```text
hybrid_greedy_best_weight vs travel_time_nearest_neighbor
hybrid_greedy_best_weight vs preference_greedy
```

Metrics:

```text
generated_total_travel_time
travel_time_ratio_to_actual
lcs_similarity
position_match_ratio
generated_same_zone_ratio
zone_change_count
```

Outputs:

```text
descriptive_stats_by_method.csv
paired_statistical_tests.csv
pairwise_comparison_summary.csv
statistical_tests_run_summary.csv
statistical_tests_run_summary.json
```

The final interpretation is:

- Hybrid is slightly worse than travel-time nearest neighbour for pure travel-time ratio.
- Hybrid significantly improves LCS similarity, position match ratio, same-zone ratio, and zone-change count compared with travel-time nearest neighbour.
- Hybrid significantly improves travel-time efficiency and route-similarity metrics compared with preference-only greedy.

### Step 16: Ablation and sensitivity analysis

Script:

```text
16_ablation_study.py
```

Final output:

```text
final_experiment_outputs/ablation_study
```

Outputs:

```text
ablation_weight_candidates.csv
component_family_summary.csv
component_weight_correlation.csv
route_generation_method_tradeoff_summary.csv
ablation_interpretation_summary.csv
ablation_study_run_summary.csv
ablation_study_run_summary.json
```

Key ablation findings:

- The validation-selected best configuration uses travel time, ML preference, and zone continuity.
- The P1 preliminary all-component setting remains competitive, ranking 9th among 300 tested combinations.
- Time-window and workload proxy terms receive zero weight in the final selected configuration because they do not add route-level benefit under the current proxy definitions and validation objective.
- This does not mean time-window and workload are theoretically irrelevant.

### Step 17: Final tables and figures

Script:

```text
17_generate_final_tables_figures.py
```

Final output:

```text
final_experiment_outputs/final_tables_figures
```

Run command:

```bash
python 17_generate_final_tables_figures.py \
  --base-dir "/content/drive/MyDrive/dissertation/amazon_last_mile/final_experiment_outputs" \
  --output-dir "/content/drive/MyDrive/dissertation/amazon_last_mile/final_experiment_outputs/final_tables_figures" \
  --overwrite
```

Main output folders:

```text
artifact_manifest.csv
tables_csv/
tables_md/
figures/
narrative/
```

Important final tables:

```text
tables_csv/table_01_model_comparison.csv
tables_csv/table_02_best_hybrid_weights.csv
tables_csv/table_03_route_generation_summary.csv
tables_csv/table_04_statistical_tests_summary.csv
tables_csv/table_05_ablation_weight_candidates.csv
tables_csv/table_06_component_family_summary.csv
tables_csv/table_07_ablation_interpretation.csv
```

Important final figures:

```text
figures/fig_01_model_top1.png
figures/fig_02_best_weights.png
figures/fig_03_route_tradeoff_test.png
figures/fig_04_zone_changes_test.png
figures/fig_05_ablation_family.png
figures/fig_06_subgroup_route_progress.png
```

Narrative summary:

```text
narrative/final_experiment_summary.md
```

## 5. Final validation checks

After pulling the final repository:

```bash
python -m py_compile 09_create_full_pairwise_samples.py
python -m py_compile 10_train_full_preference_models.py
python -m py_compile 11_model_subgroup_analysis.py
python -m py_compile 12_optimize_hybrid_weights.py
python -m py_compile 13_route_generation_best_weights.py
python -m py_compile 15_statistical_tests.py
python -m py_compile 16_ablation_study.py
python -m py_compile 17_generate_final_tables_figures.py
```

Check final result directories:

```bash
BASE=/content/drive/MyDrive/dissertation/amazon_last_mile/final_experiment_outputs

ls "$BASE/pairwise_samples_full"
ls "$BASE/model_outputs_full_top3"
ls "$BASE/model_subgroup_analysis_full_top3"
ls "$BASE/hybrid_weight_search_validation_full_300w"
ls "$BASE/route_generation_best_weights"
ls "$BASE/route_generation_best_weights/statistical_tests"
ls "$BASE/ablation_study"
ls "$BASE/final_tables_figures"
```

## 6. Completed and not completed items

Completed:

```text
Full-scale pairwise generation
Full-scale preference model training
CatBoost final preference model selection
Subgroup analysis
Validation-based hybrid weight optimization
Full validation/test route generation
Route-level paired statistical testing
Ablation and sensitivity analysis
Final tables and figures
```

Not completed, to be reported as future work:

```text
Beam Search route generation
OR-Tools baseline
TabPFN comparison
MCTS rollout
multi-seed repeated trials
Hugging Face Space / Streamlit UAT demo
```

## 7. Final dissertation conclusion framing

The final conclusion should be:

```text
The hybrid method is not the shortest-travel-time method. Travel-time nearest neighbour remains stronger for pure travel-time efficiency. However, the hybrid method achieves a better balance between operational efficiency and driver-like route structure. It keeps travel-time ratio close to the travel-time baseline while significantly improving LCS similarity, position match ratio, same-zone continuity, and zone-change reduction.
```

This is the safest and most accurate final defence framing.
