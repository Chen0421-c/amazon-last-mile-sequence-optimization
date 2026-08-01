#!/usr/bin/env python3
"""Analyze trained preference model performance by subgroup.

This script reads pairwise samples and trained model outputs from
10_train_full_preference_models.py, scores candidate next stops, and produces
dissertation-ready overall and subgroup performance tables.

It does not train models or generate routes. It evaluates an already trained
preference model, usually models/best_model.joblib.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import warnings
from datetime import datetime, timezone
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Sequence


VALID_SPLITS = ("validation", "test", "train")
INPUT_FORMATS = ("auto", "parquet", "csv")

DEFAULT_CONFIG_PATH = Path("config/config_final.yaml")
DEFAULT_INPUT_DIR = Path(
    "/content/drive/MyDrive/dissertation/amazon_last_mile/"
    "final_experiment_outputs/pairwise_samples_full"
)
DEFAULT_MODEL_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/dissertation/amazon_last_mile/"
    "final_experiment_outputs/model_outputs_full"
)

CONTEXT_COLUMNS = ["route_id", "position"]
TARGET_COLUMN = "label"
PROBABILITY_COLUMN = "predicted_probability"

ID_COLUMNS = [
    "route_id",
    "position",
    "label",
    "current_stop",
    "candidate_stop",
    "actual_next_stop",
]

DEFAULT_FEATURE_COLUMNS = [
    "travel_time_ij",
    "same_zone",
    "zone_changed",
    "zone_missing_in_pair",
    "number_of_stops",
    "route_progress",
    "remaining_stop_count",
    "current_is_station",
    "current_is_dropoff",
    "candidate_is_station",
    "candidate_is_dropoff",
    "candidate_package_count",
    "candidate_total_planned_service_time",
    "candidate_has_time_window",
    "candidate_time_window_package_count",
    "candidate_total_package_volume_cm3",
    "candidate_delivered_count",
    "candidate_attempted_count",
    "candidate_rejected_count",
    "candidate_unknown_status_count",
]

SUBGROUP_SOURCE_COLUMNS = [
    "route_score",
    "number_of_stops",
    "route_progress",
    "remaining_stop_count",
    "same_zone",
    "zone_changed",
    "zone_missing_in_pair",
    "candidate_has_time_window",
    "candidate_package_count",
    "candidate_total_planned_service_time",
    "candidate_time_window_package_count",
]

SUBGROUP_TABLES = [
    "route_score",
    "number_of_stops_bin",
    "route_progress_bin",
    "remaining_stop_count_bin",
    "transition_zone_type",
    "candidate_time_window_group",
    "candidate_package_count_bin",
    "candidate_service_time_bin",
]

PREDICTION_OUTPUT_COLUMNS = [
    "route_id",
    "position",
    "current_stop",
    "candidate_stop",
    "actual_next_stop",
    "label",
    "predicted_probability",
    "route_score",
    "number_of_stops",
    "route_progress",
    "remaining_stop_count",
    "same_zone",
    "zone_changed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained full preference model overall and by route, "
            "decision, and candidate subgroups."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML. Default: config/config_final.yaml.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing split pairwise samples. Defaults to "
            "outputs.pairwise_full_dir from config."
        ),
    )
    parser.add_argument(
        "--model-output-dir",
        type=Path,
        default=None,
        help=(
            "Directory created by 10_train_full_preference_models.py. Defaults "
            "to outputs.model_full_dir from config."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help=(
            "Path to a trained model. Defaults to "
            "{model_output_dir}/models/best_model.joblib."
        ),
    )
    parser.add_argument(
        "--feature-columns",
        type=Path,
        default=None,
        help=(
            "Optional path to feature_columns.json. Defaults to "
            "{model_output_dir}/feature_columns.json."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for subgroup outputs. Defaults to "
            "{model_output_dir}/subgroup_analysis."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=["validation", "test"],
        help="Data splits to analyze. Default: validation test.",
    )
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default="auto",
        help="Input file format. Auto prefers parquet, then CSV. Default: auto.",
    )
    parser.add_argument(
        "--max-contexts-per-split",
        type=int,
        default=None,
        help="Optional maximum complete route_id + position contexts per split.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save row-level prediction files. Default: false.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting generated outputs in an existing output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for context sampling. Default: 42.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional diagnostics.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        warnings.warn(f"Config file not found: {path}. Falling back to defaults.")
        return {}

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Please install pyyaml: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")

    return config


def resolve_path(path: Any) -> Path:
    if path is None:
        raise ValueError("Cannot resolve a null path.")
    resolved = Path(str(path)).expanduser()
    if resolved.is_absolute():
        return resolved
    return resolved.resolve()


def resolve_runtime_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path]:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs"), dict) else {}
    input_dir = resolve_path(args.input_dir or outputs.get("pairwise_full_dir", DEFAULT_INPUT_DIR))
    model_output_dir = resolve_path(
        args.model_output_dir or outputs.get("model_full_dir", DEFAULT_MODEL_OUTPUT_DIR)
    )
    model_path = resolve_path(args.model_path or model_output_dir / "models" / "best_model.joblib")
    feature_columns_path = resolve_path(
        args.feature_columns or model_output_dir / "feature_columns.json"
    )
    output_dir = resolve_path(args.output_dir or model_output_dir / "subgroup_analysis")
    return {
        "input_dir": input_dir,
        "model_output_dir": model_output_dir,
        "model_path": model_path,
        "feature_columns_path": feature_columns_path,
        "output_dir": output_dir,
    }


def find_pairwise_file(input_dir: Path, split: str, input_format: str) -> Path:
    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    parquet_path = input_dir / f"{split}_pairwise_samples.parquet"
    csv_path = input_dir / f"{split}_pairwise_samples.csv"

    if input_format == "auto":
        if parquet_path.exists():
            return parquet_path
        if csv_path.exists():
            return csv_path
        raise FileNotFoundError(
            f"Missing {split} pairwise samples. Checked: {parquet_path}, {csv_path}"
        )

    if input_format == "parquet":
        if parquet_path.exists():
            return parquet_path
        raise FileNotFoundError(f"Missing requested parquet input: {parquet_path}")

    if input_format == "csv":
        if csv_path.exists():
            return csv_path
        raise FileNotFoundError(f"Missing requested CSV input: {csv_path}")

    raise ValueError(f"Unsupported input format: {input_format}")


def parquet_engine_available() -> bool:
    return find_spec("pyarrow") is not None or find_spec("fastparquet") is not None


def list_parquet_columns(path: Path) -> list[str]:
    if find_spec("pyarrow") is not None:
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    if find_spec("fastparquet") is not None:
        import fastparquet

        return list(fastparquet.ParquetFile(path).columns)
    raise RuntimeError(
        "Parquet input requires pyarrow or fastparquet. "
        "Use --input-format csv or install a parquet engine."
    )


def list_input_columns(path: Path) -> list[str]:
    import pandas as pd

    if path.suffix.lower() == ".parquet":
        return list_parquet_columns(path)
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Unsupported input extension: {path}")


def load_feature_columns(
    model_output_dir: Path,
    feature_columns_path: Path | None,
    df_columns: Sequence[str] | None = None,
) -> tuple[list[str], str, list[str]]:
    path = feature_columns_path or model_output_dir / "feature_columns.json"
    source = "default_feature_list"

    if path.exists():
        payload = load_json(path)
        if isinstance(payload, dict):
            raw_columns = payload.get("feature_columns") or payload.get("feature_columns_used")
            if isinstance(raw_columns, str):
                feature_columns = [column for column in raw_columns.split(",") if column]
            elif isinstance(raw_columns, list):
                feature_columns = [str(column) for column in raw_columns]
            else:
                raise ValueError(f"No feature column list found in {path}")
        elif isinstance(payload, list):
            feature_columns = [str(column) for column in payload]
        else:
            raise ValueError(f"Unsupported feature column payload in {path}")
        source = str(path)
    else:
        warnings.warn(
            f"Feature column file not found: {path}. Falling back to default feature list."
        )
        feature_columns = list(DEFAULT_FEATURE_COLUMNS)

    feature_columns = list(dict.fromkeys(feature_columns))
    missing = []
    if df_columns is not None:
        available = set(df_columns)
        missing = [column for column in feature_columns if column not in available]
    return feature_columns, source, missing


def read_split_data(
    input_dir: Path,
    split: str,
    input_format: str,
    feature_columns: Sequence[str],
) -> tuple[Any, list[str], Path]:
    import pandas as pd

    path = find_pairwise_file(input_dir, split, input_format)
    available_columns = list_input_columns(path)
    available = set(available_columns)

    required = list(dict.fromkeys([*ID_COLUMNS, *feature_columns]))
    subgroup_columns = [column for column in SUBGROUP_SOURCE_COLUMNS if column in available]
    needed_columns = list(dict.fromkeys([*required, *subgroup_columns]))

    missing_required = [column for column in required if column not in available]
    if missing_required:
        raise ValueError(f"{split}: missing required ID/label/feature columns: {missing_required}")

    missing_subgroups = [
        column for column in SUBGROUP_SOURCE_COLUMNS if column not in available
    ]
    if missing_subgroups:
        warnings.warn(f"{split}: missing subgroup source columns: {missing_subgroups}")

    print(f"Reading {split}: {path}")
    print(f"{split}: reading {len(needed_columns)} of {len(available_columns)} columns")

    if path.suffix.lower() == ".parquet":
        if not parquet_engine_available():
            raise RuntimeError(
                "Parquet input requires pyarrow or fastparquet. "
                "Use --input-format csv or install a parquet engine."
            )
        df = pd.read_parquet(path, columns=needed_columns)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=needed_columns)
    else:
        raise ValueError(f"Unsupported input extension: {path}")

    return df, missing_subgroups, path


def sample_decision_contexts(df: Any, max_contexts: int | None, seed: int) -> Any:
    if max_contexts is None or max_contexts <= 0:
        return df

    missing = [column for column in CONTEXT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot sample decision contexts; missing columns: {missing}")

    contexts = df.loc[:, CONTEXT_COLUMNS].drop_duplicates()
    if len(contexts) <= max_contexts:
        return df

    sampled_contexts = contexts.sample(n=max_contexts, random_state=seed)
    sampled = df.merge(sampled_contexts, on=CONTEXT_COLUMNS, how="inner")
    sampled = sampled.reset_index(drop=True)
    print(
        f"Sampled {len(sampled_contexts):,} complete contexts; "
        f"rows kept: {len(sampled):,} of {len(df):,}"
    )
    return sampled


def numeric_series(df: Any, column: str) -> Any:
    import pandas as pd

    return pd.to_numeric(df[column], errors="coerce")


def create_subgroup_columns(df: Any) -> tuple[Any, list[str]]:
    import numpy as np
    import pandas as pd

    result = df.copy()
    skipped = []

    if "route_score" in result.columns:
        result["route_score"] = (
            result["route_score"].fillna("Missing").astype(str).replace({"": "Missing", "nan": "Missing"})
        )
    else:
        skipped.append("route_score")

    if "number_of_stops" in result.columns:
        values = numeric_series(result, "number_of_stops")
        bins = [-np.inf, 50, 100, 150, 200, np.inf]
        labels = ["<=50", "51-100", "101-150", "151-200", ">200"]
        result["number_of_stops_bin"] = pd.cut(values, bins=bins, labels=labels, right=True)
        result["number_of_stops_bin"] = result["number_of_stops_bin"].astype("object").fillna("Missing")
    else:
        skipped.append("number_of_stops_bin")

    if "route_progress" in result.columns:
        values = numeric_series(result, "route_progress")
        bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        result["route_progress_bin"] = pd.cut(values, bins=bins, labels=labels, right=True)
        result["route_progress_bin"] = result["route_progress_bin"].astype("object").fillna("Missing")
    else:
        skipped.append("route_progress_bin")

    if "remaining_stop_count" in result.columns:
        values = numeric_series(result, "remaining_stop_count")
        bins = [-np.inf, 10, 25, 50, 100, np.inf]
        labels = ["<=10", "11-25", "26-50", "51-100", ">100"]
        result["remaining_stop_count_bin"] = pd.cut(values, bins=bins, labels=labels, right=True)
        result["remaining_stop_count_bin"] = result["remaining_stop_count_bin"].astype("object").fillna("Missing")
    else:
        skipped.append("remaining_stop_count_bin")

    if "same_zone" in result.columns and "zone_changed" in result.columns:
        same_zone = numeric_series(result, "same_zone").fillna(0).astype(int)
        zone_changed = numeric_series(result, "zone_changed").fillna(0).astype(int)
        result["transition_zone_type"] = np.select(
            [same_zone == 1, zone_changed == 1],
            ["same_zone", "zone_changed"],
            default="zone_missing_or_unknown",
        )
    else:
        skipped.append("transition_zone_type")

    if "candidate_has_time_window" in result.columns:
        has_time_window = numeric_series(result, "candidate_has_time_window").fillna(0).astype(int)
        result["candidate_time_window_group"] = np.where(
            has_time_window == 1, "has_time_window", "no_time_window"
        )
    else:
        skipped.append("candidate_time_window_group")

    if "candidate_package_count" in result.columns:
        values = numeric_series(result, "candidate_package_count")
        bins = [-np.inf, 0, 1, 2, 5, np.inf]
        labels = ["0", "1", "2", "3-5", ">5"]
        result["candidate_package_count_bin"] = pd.cut(values, bins=bins, labels=labels, right=True)
        result["candidate_package_count_bin"] = (
            result["candidate_package_count_bin"].astype("object").fillna("Missing")
        )
    else:
        skipped.append("candidate_package_count_bin")

    if "candidate_total_planned_service_time" in result.columns:
        values = numeric_series(result, "candidate_total_planned_service_time")
        bins = [-np.inf, 0, 60, 120, 300, np.inf]
        labels = ["0", "1-60", "61-120", "121-300", ">300"]
        result["candidate_service_time_bin"] = pd.cut(values, bins=bins, labels=labels, right=True)
        result["candidate_service_time_bin"] = (
            result["candidate_service_time_bin"].astype("object").fillna("Missing")
        )
    else:
        skipped.append("candidate_service_time_bin")

    return result, skipped


def prepare_features(df: Any, feature_columns: Sequence[str]) -> Any:
    import pandas as pd

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns for prediction: {missing}")

    features = df.loc[:, list(feature_columns)].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.astype("float32")


def predict_probabilities(model: Any, X: Any) -> Any:
    import numpy as np

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(X), dtype=float)
        if probabilities.ndim != 2:
            return np.clip(probabilities.reshape(-1), 1e-15, 1.0 - 1e-15)

        classes = getattr(model, "classes_", None)
        if classes is not None and 1 in list(classes):
            positive_index = list(classes).index(1)
        else:
            positive_index = min(1, probabilities.shape[1] - 1)
        return np.clip(probabilities[:, positive_index], 1e-15, 1.0 - 1e-15)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float).reshape(-1)
        scores = np.clip(scores, -700, 700)
        return np.clip(1.0 / (1.0 + np.exp(-scores)), 1e-15, 1.0 - 1e-15)

    raise TypeError(
        "Model does not provide predict_proba or decision_function; "
        "cannot compute preference probabilities."
    )


def safe_metric(name: str, func: Any) -> float:
    try:
        return float(func())
    except Exception as exc:
        warnings.warn(f"Could not compute {name}: {exc}")
        return float("nan")


def compute_binary_metrics(y_true: Any, y_prob: Any) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)
    rows = int(len(y_true))
    positive_rows = int((y_true == 1).sum())
    negative_rows = int((y_true == 0).sum())

    return {
        "rows": rows,
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_rate": float(positive_rows / rows) if rows else float("nan"),
        "accuracy": safe_metric("accuracy", lambda: accuracy_score(y_true, y_pred)),
        "precision": safe_metric(
            "precision", lambda: precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": safe_metric("recall", lambda: recall_score(y_true, y_pred, zero_division=0)),
        "f1": safe_metric("f1", lambda: f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_metric("roc_auc", lambda: roc_auc_score(y_true, y_prob)),
        "average_precision": safe_metric(
            "average_precision", lambda: average_precision_score(y_true, y_prob)
        ),
        "log_loss": safe_metric("log_loss", lambda: log_loss(y_true, y_prob, labels=[0, 1])),
    }


def compute_ranking_metrics(
    df: Any,
    prob_col: str = PROBABILITY_COLUMN,
) -> dict[str, Any]:
    import numpy as np

    required = [*CONTEXT_COLUMNS, TARGET_COLUMN, prob_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for ranking metrics: {missing}")

    if df.empty:
        return {
            "sampled_top1_accuracy": float("nan"),
            "sampled_top3_accuracy": float("nan"),
            "sampled_top5_accuracy": float("nan"),
            "evaluated_contexts": 0,
            "skipped_contexts_no_positive": 0,
            "contexts_multiple_positive": 0,
            "mean_actual_positive_rank": float("nan"),
            "median_actual_positive_rank": float("nan"),
            "mrr": float("nan"),
        }

    temp = df.loc[:, required].copy()
    temp[TARGET_COLUMN] = temp[TARGET_COLUMN].astype("int64")

    positive_counts = temp.groupby(CONTEXT_COLUMNS, dropna=False)[TARGET_COLUMN].sum()
    skipped_no_positive = int((positive_counts == 0).sum())
    multiple_positive = int((positive_counts > 1).sum())

    ranked = temp.sort_values(
        by=[*CONTEXT_COLUMNS, prob_col],
        ascending=[True, True, False],
        kind="mergesort",
    )
    ranked["_rank"] = ranked.groupby(CONTEXT_COLUMNS, dropna=False).cumcount() + 1

    positive_ranks = (
        ranked.loc[ranked[TARGET_COLUMN] == 1]
        .groupby(CONTEXT_COLUMNS, dropna=False)["_rank"]
        .min()
    )
    evaluated_contexts = int(len(positive_ranks))

    if evaluated_contexts == 0:
        return {
            "sampled_top1_accuracy": float("nan"),
            "sampled_top3_accuracy": float("nan"),
            "sampled_top5_accuracy": float("nan"),
            "evaluated_contexts": 0,
            "skipped_contexts_no_positive": skipped_no_positive,
            "contexts_multiple_positive": multiple_positive,
            "mean_actual_positive_rank": float("nan"),
            "median_actual_positive_rank": float("nan"),
            "mrr": float("nan"),
        }

    ranks = positive_ranks.astype(float)
    return {
        "sampled_top1_accuracy": float((ranks <= 1).mean()),
        "sampled_top3_accuracy": float((ranks <= 3).mean()),
        "sampled_top5_accuracy": float((ranks <= 5).mean()),
        "evaluated_contexts": evaluated_contexts,
        "skipped_contexts_no_positive": skipped_no_positive,
        "contexts_multiple_positive": multiple_positive,
        "mean_actual_positive_rank": float(ranks.mean()),
        "median_actual_positive_rank": float(ranks.median()),
        "mrr": float((1.0 / ranks).mean()),
    }


def compute_metrics_for_dataframe(
    df: Any,
    split: str,
    group_name: str,
    group_value: str,
) -> dict[str, Any]:
    if df.empty:
        row = {
            "split": split,
            "group_name": group_name,
            "group_value": group_value,
        }
        row.update(compute_binary_metrics([], []))
        row.update(compute_ranking_metrics(df))
        return row

    y_true = df[TARGET_COLUMN].astype("int64").to_numpy()
    y_prob = df[PROBABILITY_COLUMN].astype(float).to_numpy()
    row = {
        "split": split,
        "group_name": group_name,
        "group_value": str(group_value),
    }
    row.update(compute_binary_metrics(y_true, y_prob))
    row.update(compute_ranking_metrics(df))
    return row


def ordered_group_values(df: Any, group_column: str) -> list[Any]:
    values = df[group_column].dropna().unique().tolist()
    return sorted(values, key=lambda value: str(value))


def analyze_subgroup(df: Any, split: str, group_column: str) -> list[dict[str, Any]]:
    if group_column not in df.columns:
        warnings.warn(f"{split}: skipping subgroup table; missing {group_column}")
        return []

    rows = []
    for value in ordered_group_values(df, group_column):
        group_df = df.loc[df[group_column] == value]
        rows.append(
            compute_metrics_for_dataframe(
                group_df,
                split=split,
                group_name=group_column,
                group_value=value,
            )
        )
    return rows


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    generated = {
        "overall_metrics.csv",
        "subgroup_route_score.csv",
        "subgroup_number_of_stops_bin.csv",
        "subgroup_route_progress_bin.csv",
        "subgroup_remaining_stop_count_bin.csv",
        "subgroup_transition_zone_type.csv",
        "subgroup_candidate_time_window_group.csv",
        "subgroup_candidate_package_count_bin.csv",
        "subgroup_candidate_service_time_bin.csv",
        "subgroup_metrics_all.csv",
        "subgroup_analysis_run_summary.csv",
        "subgroup_analysis_run_summary.json",
    }

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory exists and is not empty: {output_dir}. "
            "Use --overwrite to replace generated subgroup outputs."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return

    for name in generated:
        path = output_dir / name
        if path.exists():
            path.unlink()
    for prediction_path in output_dir.glob("predictions_*.csv"):
        prediction_path.unlink()
    predictions_dir = output_dir / "predictions"
    if predictions_dir.exists():
        shutil.rmtree(predictions_dir)


def json_clean(value: Any) -> Any:
    import numpy as np

    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def save_prediction_file(df: Any, output_path: Path) -> None:
    columns = [column for column in PREDICTION_OUTPUT_COLUMNS if column in df.columns]
    missing = [column for column in PREDICTION_OUTPUT_COLUMNS if column not in df.columns]
    if missing:
        warnings.warn(f"Prediction file missing optional output columns: {missing}")
    df.loc[:, columns].to_csv(output_path, index=False)


def save_outputs(
    output_dir: Path,
    overall_rows: list[dict[str, Any]],
    subgroup_rows_by_table: dict[str, list[dict[str, Any]]],
    run_summary: dict[str, Any],
) -> None:
    import pandas as pd

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(output_dir / "overall_metrics.csv", index=False)

    all_subgroup_rows = []
    for table_name, rows in subgroup_rows_by_table.items():
        table_path = output_dir / f"subgroup_{table_name}.csv"
        table_df = pd.DataFrame(rows)
        table_df.to_csv(table_path, index=False)
        all_subgroup_rows.extend(rows)

    pd.DataFrame(all_subgroup_rows).to_csv(
        output_dir / "subgroup_metrics_all.csv", index=False
    )
    pd.DataFrame([run_summary]).to_csv(
        output_dir / "subgroup_analysis_run_summary.csv", index=False
    )
    with (output_dir / "subgroup_analysis_run_summary.json").open(
        "w", encoding="utf-8"
    ) as file_obj:
        json.dump(json_clean(run_summary), file_obj, indent=2)
        file_obj.write("\n")


def load_model(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        import joblib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit("Please install joblib: pip install joblib") from exc
    return joblib.load(path)


def analyze(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    paths = resolve_runtime_paths(args, config)
    input_dir = paths["input_dir"]
    model_output_dir = paths["model_output_dir"]
    model_path = paths["model_path"]
    feature_columns_path = paths["feature_columns_path"]
    output_dir = paths["output_dir"]

    print(f"Input directory: {input_dir}")
    print(f"Model output directory: {model_output_dir}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {args.splits}")

    feature_columns, feature_source, _ = load_feature_columns(
        model_output_dir,
        feature_columns_path,
        df_columns=None,
    )
    print(f"Feature columns source: {feature_source}")
    print(f"Feature columns used ({len(feature_columns)}): {feature_columns}")

    prepare_output_dir(output_dir, args.overwrite)
    model = load_model(model_path)

    overall_rows = []
    subgroup_rows_by_table = {table: [] for table in SUBGROUP_TABLES}
    rows_loaded_by_split = {}
    contexts_loaded_by_split = {}
    split_input_files = {}
    missing_subgroup_columns_by_split = {}
    skipped_subgroup_tables = set()
    missing_feature_columns = set()

    for split in args.splits:
        df, missing_subgroups, input_path = read_split_data(
            input_dir,
            split,
            args.input_format,
            feature_columns,
        )
        df = sample_decision_contexts(df, args.max_contexts_per_split, args.seed)

        missing_features = [column for column in feature_columns if column not in df.columns]
        if missing_features:
            missing_feature_columns.update(missing_features)
            raise ValueError(f"{split}: missing feature columns: {missing_features}")

        X = prepare_features(df, feature_columns)
        df[PROBABILITY_COLUMN] = predict_probabilities(model, X)
        df, skipped_for_split = create_subgroup_columns(df)

        rows_loaded_by_split[split] = int(len(df))
        contexts_loaded_by_split[split] = int(df.loc[:, CONTEXT_COLUMNS].drop_duplicates().shape[0])
        split_input_files[split] = str(input_path)
        missing_subgroup_columns_by_split[split] = missing_subgroups
        skipped_subgroup_tables.update(skipped_for_split)

        overall_rows.append(
            compute_metrics_for_dataframe(
                df,
                split=split,
                group_name="overall",
                group_value="all",
            )
        )

        for table in SUBGROUP_TABLES:
            if table in skipped_for_split:
                continue
            subgroup_rows_by_table[table].extend(analyze_subgroup(df, split, table))

        if args.save_predictions:
            save_prediction_file(df, output_dir / f"predictions_{split}.csv")

        if args.verbose:
            print(
                f"{split}: rows={rows_loaded_by_split[split]:,}, "
                f"contexts={contexts_loaded_by_split[split]:,}"
            )

    for table_name, rows in subgroup_rows_by_table.items():
        if not rows:
            skipped_subgroup_tables.add(table_name)

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "model_output_dir": str(model_output_dir),
        "model_path": str(model_path),
        "feature_columns_source": feature_source,
        "output_dir": str(output_dir),
        "splits": ",".join(args.splits),
        "max_contexts_per_split": args.max_contexts_per_split,
        "seed": args.seed,
        "rows_loaded_by_split": rows_loaded_by_split,
        "contexts_loaded_by_split": contexts_loaded_by_split,
        "split_input_files": split_input_files,
        "feature_columns_used": ",".join(feature_columns),
        "missing_feature_columns": sorted(missing_feature_columns),
        "missing_subgroup_columns_by_split": missing_subgroup_columns_by_split,
        "skipped_subgroup_tables": sorted(skipped_subgroup_tables),
        "save_predictions": bool(args.save_predictions),
    }

    save_outputs(output_dir, overall_rows, subgroup_rows_by_table, run_summary)

    print("\nModel subgroup analysis complete.")
    print(f"Overall rows written: {len(overall_rows)}")
    print(
        "Subgroup tables written: "
        f"{', '.join(table for table, rows in subgroup_rows_by_table.items() if rows)}"
    )
    if skipped_subgroup_tables:
        print(f"Skipped subgroup tables: {sorted(skipped_subgroup_tables)}")
    print(f"Output directory: {output_dir}")


def main() -> int:
    args = parse_args()
    try:
        analyze(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
