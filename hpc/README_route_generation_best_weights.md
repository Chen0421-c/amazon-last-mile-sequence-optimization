# Best-weight route generation

This helper runs `13_route_generation_best_weights.py` after the hybrid weight search in step 12.

## Inputs

Expected DICC paths:

- Repository: `/home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization`
- Data root: `/home/user/chenziliang/dissertation/amazon_last_mile`
- Best preference model: `final_experiment_outputs/model_outputs_full_top3/models/best_model.joblib`
- Feature columns: `final_experiment_outputs/model_outputs_full_top3/feature_columns.json`
- Best weight search output: `final_experiment_outputs/hybrid_weight_search_validation_full_300w/best_weight_summary.csv`

## Smoke test

```bash
cd /home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
git pull origin main
bash -n hpc/run_route_generation_best_weights_smoke.sh
sbatch hpc/run_route_generation_best_weights_smoke.sh
```

Check logs:

```bash
squeue -u chenziliang
ls -lt /home/user/chenziliang/dissertation/amazon_last_mile/final_experiment_outputs/logs | head
```

## Full validation/test generation

```bash
cd /home/user/chenziliang/dissertation/amazon-last-mile-sequence-optimization
git pull origin main
bash -n hpc/run_route_generation_best_weights.sh
sbatch hpc/run_route_generation_best_weights.sh
```

Expected outputs:

- `generated_routes_validation.csv`
- `generated_routes_test.csv`
- `route_metrics_validation.csv`
- `route_metrics_test.csv`
- `method_summary_validation.csv`
- `method_summary_test.csv`
- `route_metrics_all_splits.csv`
- `method_summary_all_splits.csv`
- `route_generation_run_summary.csv`
- `route_generation_run_summary.json`
