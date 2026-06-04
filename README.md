# Road Freight Delivery Sequence Optimization using Machine Learning

This repository supports a master's dissertation project using the **Amazon Last Mile Routing Research Challenge** dataset.

The current project stage is **memory-safe data cleaning and EDA preparation only**. No machine learning model, including XGBoost, is trained in this step.

## Repository rules

- Do **not** commit or copy raw Amazon JSON files into this GitHub repository.
- Do **not** commit large generated CSV files into this GitHub repository.
- Keep generated data outputs in Google Drive.
- The repository should contain only code, notebooks, README documentation, `requirements.txt`, and `.gitignore`.

## Google Drive data locations

Raw data root:

```text
/content/drive/MyDrive/dissertation/amazon_last_mile
```

Generated CSV output directory:

```text
/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs/
```

The pipeline reads the official challenge folder structure:

```text
almrrc2021-data-training/model_build_inputs/
  route_data.json
  package_data.json
  travel_times.json
  actual_sequences.json
  invalid_sequence_scores.json

almrrc2021-data-training/model_apply_inputs/
  new_route_data.json
  new_package_data.json
  new_travel_times.json

almrrc2021-data-training/model_score_inputs/
  new_actual_sequences.json
  new_invalid_sequence_scores.json

almrrc2021-data-evaluation/model_apply_inputs/
  eval_route_data.json
  eval_package_data.json
  eval_travel_times.json

almrrc2021-data-evaluation/model_score_inputs/
  eval_actual_sequences.json
  eval_invalid_sequence_scores.json
```

## Memory-safe cleaning approach

The JSON files are large top-level objects keyed by `route_id`. The cleaning code does **not** use `json.load()` on full source files. Instead, it streams each top-level route entry one at a time.

The Amazon JSON files can contain non-standard values such as:

- `"zone_id": NaN`
- `"start_time_utc": NaN`
- `"end_time_utc": NaN`

The streaming decoder converts `NaN`, `Infinity`, and `-Infinity` to `None` during parsing. The original raw files are never overwritten or modified.

## Generated outputs

Running `01_data_cleaning_pipeline.py` writes these CSV files under `processed_outputs/`:

1. `routes_summary.csv`
   - Route-level metadata and stop counts.
   - Includes `missing_zone_count` and `missing_zone_ratio`.

2. `stops_base_features.csv`
   - One row per stop.
   - Includes `zone_missing`, `is_station`, and `is_dropoff` flags.

3. `actual_transitions.csv`
   - Positive consecutive stop-to-stop transitions from `actual_sequences.json`-style files.
   - Each row has `label = 1`.

4. `stop_package_features.csv`
   - Stop-level aggregates from package-level JSON.
   - Includes package counts, planned service time, time-window flags, volume, and scan-status counts.

5. `data_quality_report.csv`
   - Route-level checks for sequence availability, stop-set consistency, package stop consistency, station count, and `can_use_for_training`.

6. `missing_value_summary.csv`
   - Missing-value summary for `zone_id`, package time windows, planned service time, package dimensions, and scan status.

## How to run in Google Colab

1. Mount Google Drive in Colab.
2. Clone this repository or upload/open it in the Colab runtime.
3. From the repository root, run:

```bash
python 01_data_cleaning_pipeline.py
```

You can also pass explicit paths:

```bash
python 01_data_cleaning_pipeline.py \
  --data-root /content/drive/MyDrive/dissertation/amazon_last_mile \
  --output-dir /content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs
```

The same commands are available in `01_data_cleaning_pipeline.ipynb`.

## Progress and final summary

The script prints progress every 500 routes for each source file. After all outputs are generated, it prints:

- number of routes processed
- number of usable routes
- number of rows in each output CSV

## Notes for later dissertation stages

This repository currently prepares clean CSV files for EDA and future feature engineering. Model training and route sequence optimization should be added in later tasks only after the cleaned outputs have been inspected.
