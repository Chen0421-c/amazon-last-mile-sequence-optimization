# Road Freight Delivery Sequence Optimization using Machine Learning

This repository supports a master's dissertation project using the **Amazon Last Mile Routing Research Challenge** dataset. The current stage is data cleaning only: no model training is implemented in this first task.

## Data location

Raw Amazon JSON data is expected in Google Drive at:

```text
/content/drive/MyDrive/dissertation/amazon_last_mile/
```

Cleaned CSV outputs are written to:

```text
/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs/
```

Raw JSON files and generated CSV outputs are intentionally excluded from Git by `.gitignore`.

## Cleaning pipeline

The cleaning pipeline is designed to be memory-safe for large JSON files. It streams each top-level route entry one at a time instead of loading an entire Amazon JSON file into memory. It also safely converts non-standard JSON constants (`NaN`, `Infinity`, and `-Infinity`) and floating-point NaN/Infinity values to missing CSV cells.

Expected source files, when available:

- `route_data.json`
- `package_data.json`
- `travel_times.json`
- `actual_sequences.json`
- `invalid_sequence_scores.json`

Generated output files:

- `cleaned_routes.csv`
- `cleaned_stops.csv`
- `cleaned_packages.csv`
- `cleaned_travel_times.csv`
- `cleaned_actual_sequences.csv`
- `cleaned_invalid_sequence_scores.csv`

## Run in Google Colab

1. Mount Google Drive.
2. Clone or open this repository.
3. Run the cleaning script:

```bash
python scripts/clean_amazon_data.py
```

Optional explicit paths:

```bash
python scripts/clean_amazon_data.py \
  --data-root /content/drive/MyDrive/dissertation/amazon_last_mile \
  --output-dir /content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs
```

The script skips missing optional source files and prints row counts for each generated output.

## Project scope for this task

- Create code and documentation for data cleaning.
- Do not commit raw JSON files or generated large CSV files.
- Do not train a machine learning model yet.
