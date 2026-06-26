#!/usr/bin/env python3
"""Train preference learning models on 500-route pairwise samples.

This script trains and compares machine learning models for the next-stop
preference learning task.

Task:
Given current stop i and candidate stop j, predict whether candidate stop j is
the actual next stop selected in the driver-executed route sequence.

label = 1: candidate_stop is the actual next stop.
label = 0: candidate_stop is a sampled negative candidate.

The predicted probability for label 1 is:
ML_preference_probability(i, j)
"""

from __future__ import annotations

import argparse
import math
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_DIR = Path(
    "/content/drive/MyDrive/dissertation/amazon_last_mile/"
    "processed_outputs/final_cleaned/pairwise_samples_500routes"
)
DEFAULT_OUTPUT_DIR = Path(
    "/content/drive/MyDrive/dissertation/amazon_last_mile/"
    "processed_outputs/final_cleaned/model_outputs_500routes"
)

TRAIN_FILE = "train_pairwise_samples.csv"
VALIDATION_FILE = "validation_pairwise_samples.csv"
TEST_FILE = "test_pairwise_samples.csv"
TARGET_COLUMN = "label"

GROUP_COLUMNS = ["route_id", "position"]

DEFAULT_MODELS = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "lightgbm",
    "catboost",
]

CANDIDATE_FEATURE_COLUMNS = [
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

EXCLUDED_FEATURE_COLUMNS = {
    "route_id",
    "split",
    "position",
    "current_stop",
    "candidate_stop",
    "actual_next_stop",
    "label",
    "route_score",
    "current_zone",
    "candidate_zone",
    "current_type",
    "candidate_type",
    "negative_sampling_seed",
    "negative_sample_rank",
}

PREDICTION_OUTPUT_COLUMNS = [
    "route_id",
    "position",
    "current_stop",
    "candidate_stop",
    "actual_next_stop",
    "label",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Train and compare preference learning models on pairwise samples."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing pairwise sample CSV files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for model outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=DEFAULT_MODELS,
        help="Models to train.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional maximum train rows after decision-context sampling.",
    )
    parser.add_argument(
        "--max-val-rows",
        type=int,
        default=None,
        help="Optional maximum validation rows after decision-context sampling.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=None,
        help="Optional maximum test rows after decision-context sampling.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save validation/test predictions for the selected best model.",
    )
    return parser.parse_args()


def require_input_files(input_dir: Path) -> Dict[str, Path]:
    """Validate required input CSV files."""

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = {
        "train": input_dir / TRAIN_FILE,
        "validation": input_dir / VALIDATION_FILE,
        "test": input_dir / TEST_FILE,
    }

    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input file(s): " + ", ".join(missing))

    return files


def read_pairwise_csv(path: Path, split_name: str) -> pd.DataFrame:
    """Read a pairwise sample CSV file."""

    print(f"Reading {split_name} data: {path}")
    return pd.read_csv(path)


def label_counts(df: pd.DataFrame) -> Dict[Any, int]:
    """Return label counts for a dataframe."""

    if TARGET_COLUMN not in df.columns:
        return {}
    counts = df[TARGET_COLUMN].value_counts(dropna=False).sort_index()
    return {label: int(count) for label, count in counts.items()}


def print_data_summary(split_name: str, df: pd.DataFrame) -> None:
    """Print shape and label distribution."""

    print(f"{split_name} shape: {df.shape}")
    print(f"{split_name} label distribution: {label_counts(df)}")


def validate_label_exists(df: pd.DataFrame, split_name: str) -> None:
    """Check that the target label column exists."""

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing '{TARGET_COLUMN}' column in {split_name} data.")


def validate_binary_labels(df: pd.DataFrame, split_name: str) -> None:
    """Check that both positive and negative labels exist."""

    validate_label_exists(df, split_name)
    labels = set(pd.to_numeric(df[TARGET_COLUMN], errors="coerce").dropna().astype(int))
    if not {0, 1}.issubset(labels):
        raise ValueError(
            f"{split_name} data must contain both label 0 and label 1. "
            f"Observed labels: {sorted(labels)}"
        )


def validate_group_columns(df: pd.DataFrame, split_name: str) -> None:
    """Check grouped top-k columns."""

    missing = [column for column in GROUP_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(
            f"{split_name} data must contain {GROUP_COLUMNS} for grouped top-k metrics. "
            f"Missing: {missing}"
        )


def select_feature_columns(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> List[str]:
    """Select feature columns that exist in all splits."""

    common_columns = set(train_df.columns) & set(validation_df.columns) & set(test_df.columns)
    feature_columns = [
        column
        for column in CANDIDATE_FEATURE_COLUMNS
        if column in common_columns and column not in EXCLUDED_FEATURE_COLUMNS
    ]

    if not feature_columns:
        raise ValueError("No candidate feature columns are available in all input files.")

    return feature_columns


def sample_groups_to_max_rows(
    df: pd.DataFrame,
    max_rows: Optional[int],
    seed: int,
    split_name: str,
) -> pd.DataFrame:
    """Sample complete route_id-position decision contexts up to a row budget."""

    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df

    if all(column in df.columns for column in GROUP_COLUMNS):
        group_sizes = (
            df.groupby(GROUP_COLUMNS, dropna=False)
            .size()
            .reset_index(name="_group_size")
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
        group_sizes["_cumulative_rows"] = group_sizes["_group_size"].cumsum()
        selected_groups = group_sizes[group_sizes["_cumulative_rows"] <= max_rows]

        if selected_groups.empty:
            selected_groups = group_sizes.head(1)

        sampled = df.merge(selected_groups[GROUP_COLUMNS], on=GROUP_COLUMNS, how="inner")
        sampled = sampled.reset_index(drop=True)

        print(
            f"{split_name}: sampled {len(sampled):,} rows from {len(df):,} rows "
            f"using {len(selected_groups):,} complete decision-context groups."
        )
        return sampled

    sampled = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    print(
        f"{split_name}: sampled {len(sampled):,} rows from {len(df):,} rows "
        "without group preservation because route_id/position are unavailable."
    )
    return sampled


def prepare_feature_matrix(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    """Convert selected feature columns to numeric and fill missing values."""

    features = df.loc[:, feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features.fillna(0)


def prepare_labels(df: pd.DataFrame) -> pd.Series:
    """Prepare binary labels."""

    return pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0).astype(int)


def safe_metric(
    metric_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Compute a metric safely and return NaN on failure."""

    try:
        if metric_name == "accuracy":
            return float(accuracy_score(y_true, y_pred))
        if metric_name == "precision":
            return float(precision_score(y_true, y_pred, zero_division=0))
        if metric_name == "recall":
            return float(recall_score(y_true, y_pred, zero_division=0))
        if metric_name == "f1":
            return float(f1_score(y_true, y_pred, zero_division=0))
        if metric_name == "roc_auc":
            return float(roc_auc_score(y_true, y_prob))
        if metric_name == "average_precision":
            return float(average_precision_score(y_true, y_prob))
        if metric_name == "log_loss":
            return float(log_loss(y_true, y_prob, labels=[0, 1]))
    except Exception:
        return float("nan")

    return float("nan")


def positive_class_probability(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Return the predicted probability for label 1."""

    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)

        if probabilities.ndim == 2 and probabilities.shape[1] >= 2:
            return probabilities[:, 1]

        if probabilities.ndim == 2 and probabilities.shape[1] == 1:
            return probabilities[:, 0]

        return probabilities.reshape(-1)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float).reshape(-1)
        return 1.0 / (1.0 + np.exp(-scores))

    return np.asarray(model.predict(features), dtype=float).reshape(-1)


def grouped_topk_metrics(
    df: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute sampled-candidate top-k accuracy by route_id-position context."""

    if any(column not in df.columns for column in GROUP_COLUMNS):
        return {
            "sampled_top1_accuracy": float("nan"),
            "sampled_top3_accuracy": float("nan"),
            "sampled_top5_accuracy": float("nan"),
            "topk_group_count": 0,
        }

    ranking_df = df.loc[:, GROUP_COLUMNS].copy()
    ranking_df[TARGET_COLUMN] = np.asarray(y_true, dtype=int)
    ranking_df["predicted_probability"] = np.asarray(y_prob, dtype=float)

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    valid_group_count = 0

    for _key, group in ranking_df.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        if len(group) < 2:
            continue
        if int((group[TARGET_COLUMN] == 1).sum()) < 1:
            continue

        ranked = group.sort_values(
            "predicted_probability",
            ascending=False,
            kind="mergesort",
        ).reset_index(drop=True)

        positive_ranks = np.flatnonzero(ranked[TARGET_COLUMN].to_numpy() == 1)
        if positive_ranks.size == 0:
            continue

        best_positive_rank = int(positive_ranks[0]) + 1
        valid_group_count += 1

        if best_positive_rank <= 1:
            top1_correct += 1
        if best_positive_rank <= 3:
            top3_correct += 1
        if best_positive_rank <= 5:
            top5_correct += 1

    if valid_group_count == 0:
        return {
            "sampled_top1_accuracy": float("nan"),
            "sampled_top3_accuracy": float("nan"),
            "sampled_top5_accuracy": float("nan"),
            "topk_group_count": 0,
        }

    return {
        "sampled_top1_accuracy": top1_correct / valid_group_count,
        "sampled_top3_accuracy": top3_correct / valid_group_count,
        "sampled_top5_accuracy": top5_correct / valid_group_count,
        "topk_group_count": valid_group_count,
    }


def evaluate_model(
    model_name: str,
    model: Any,
    df: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.Series,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Evaluate one model on a split."""

    y_prob = positive_class_probability(model, features)
    y_prob = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "model_name": model_name,
        "accuracy": safe_metric("accuracy", labels, y_pred, y_prob),
        "precision": safe_metric("precision", labels, y_pred, y_prob),
        "recall": safe_metric("recall", labels, y_pred, y_prob),
        "f1": safe_metric("f1", labels, y_pred, y_prob),
        "roc_auc": safe_metric("roc_auc", labels, y_pred, y_prob),
        "average_precision": safe_metric("average_precision", labels, y_pred, y_prob),
        "log_loss": safe_metric("log_loss", labels, y_pred, y_prob),
    }
    metrics.update(grouped_topk_metrics(df, labels, y_prob))

    return metrics, y_prob


def build_logistic_regression(seed: int) -> Pipeline:
    """Build a Logistic Regression pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_random_forest(seed: int) -> RandomForestClassifier:
    """Build a Random Forest classifier."""

    return RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=seed,
    )


def build_xgboost(seed: int, train_labels: pd.Series) -> Any:
    """Build an XGBoost classifier if xgboost is installed."""

    from xgboost import XGBClassifier

    positive_count = int((train_labels == 1).sum())
    negative_count = int((train_labels == 0).sum())
    scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1.0

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
        scale_pos_weight=scale_pos_weight,
    )


def build_lightgbm(seed: int) -> Any:
    """Build a LightGBM classifier if lightgbm is installed."""

    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
        verbose=-1,
    )


def build_catboost(seed: int) -> Any:
    """Build a CatBoost classifier if catboost is installed."""

    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        random_seed=seed,
        auto_class_weights="Balanced",
    )


def optional_package_available(model_name: str) -> bool:
    """Check if an optional package is available."""

    package_names = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
    }

    package_name = package_names.get(model_name)
    if package_name is None:
        return True

    return find_spec(package_name) is not None


def build_model(model_name: str, seed: int, train_labels: pd.Series) -> Any:
    """Build one model by name."""

    if model_name == "logistic_regression":
        return build_logistic_regression(seed)
    if model_name == "random_forest":
        return build_random_forest(seed)
    if model_name == "xgboost":
        return build_xgboost(seed, train_labels)
    if model_name == "lightgbm":
        return build_lightgbm(seed)
    if model_name == "catboost":
        return build_catboost(seed)

    raise ValueError(f"Unsupported model: {model_name}")


def save_trained_model(model: Any, model_name: str, models_dir: Path) -> Path:
    """Save a trained model."""

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib_path = models_dir / f"{model_name}.joblib"

    try:
        joblib.dump(model, joblib_path)
        return joblib_path
    except Exception:
        if model_name == "catboost" and hasattr(model, "save_model"):
            catboost_path = models_dir / f"{model_name}.cbm"
            model.save_model(str(catboost_path))
            return catboost_path
        raise


def extract_feature_importance(
    model_name: str,
    model: Any,
    feature_columns: Sequence[str],
) -> List[Dict[str, Any]]:
    """Extract feature importance or coefficients when available."""

    importances: Optional[np.ndarray] = None
    signed_importances: Optional[np.ndarray] = None

    try:
        if model_name == "logistic_regression":
            classifier = model.named_steps["model"] if isinstance(model, Pipeline) else model
            if hasattr(classifier, "coef_"):
                signed_importances = np.asarray(classifier.coef_).reshape(-1)
                importances = np.abs(signed_importances)

        elif hasattr(model, "feature_importances_"):
            importances = np.asarray(model.feature_importances_).reshape(-1)
            signed_importances = importances

        elif model_name == "catboost" and hasattr(model, "get_feature_importance"):
            importances = np.asarray(model.get_feature_importance()).reshape(-1)
            signed_importances = importances

    except Exception:
        importances = None
        signed_importances = None

    if importances is None or len(importances) != len(feature_columns):
        return []

    rows = []
    for index, feature in enumerate(feature_columns):
        rows.append(
            {
                "model_name": model_name,
                "feature": feature,
                "importance": float(importances[index]),
                "signed_importance": (
                    float(signed_importances[index])
                    if signed_importances is not None
                    else float(importances[index])
                ),
            }
        )

    return rows


def metric_sort_value(value: Any) -> float:
    """Convert metric values to sortable floats, treating NaN as negative infinity."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return -math.inf

    if math.isnan(numeric_value):
        return -math.inf

    return numeric_value


def select_best_model(validation_results: pd.DataFrame) -> str:
    """Select best model using validation sampled_top1_accuracy and tie-breakers."""

    if validation_results.empty:
        raise ValueError("No validation results are available for model selection.")

    sort_columns = [
        "sampled_top1_accuracy",
        "roc_auc",
        "f1",
        "average_precision",
    ]

    sortable = validation_results.copy()
    for column in sort_columns:
        sortable[f"_{column}_sort"] = sortable[column].apply(metric_sort_value)

    sortable = sortable.sort_values(
        by=[f"_{column}_sort" for column in sort_columns],
        ascending=[False, False, False, False],
        kind="mergesort",
    )

    return str(sortable.iloc[0]["model_name"])


def build_best_model_summary(
    best_model_name: str,
    validation_results: pd.DataFrame,
    test_results: pd.DataFrame,
    input_dir: Path,
    output_dir: Path,
    seed: int,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Create best model summary table."""

    validation_row = validation_results[validation_results["model_name"] == best_model_name].iloc[0]
    test_row = test_results[test_results["model_name"] == best_model_name].iloc[0]

    return pd.DataFrame(
        [
            {
                "best_model_name": best_model_name,
                "selection_metric": "validation_sampled_top1_accuracy",
                "validation_sampled_top1_accuracy": validation_row.get(
                    "sampled_top1_accuracy", np.nan
                ),
                "validation_roc_auc": validation_row.get("roc_auc", np.nan),
                "validation_f1": validation_row.get("f1", np.nan),
                "validation_average_precision": validation_row.get(
                    "average_precision", np.nan
                ),
                "test_sampled_top1_accuracy": test_row.get(
                    "sampled_top1_accuracy", np.nan
                ),
                "test_roc_auc": test_row.get("roc_auc", np.nan),
                "test_f1": test_row.get("f1", np.nan),
                "test_average_precision": test_row.get("average_precision", np.nan),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "seed": seed,
                "feature_count": len(feature_columns),
                "feature_columns": ",".join(feature_columns),
            }
        ]
    )


def build_training_data_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: Sequence[str],
    input_dir: Path,
    seed: int,
) -> pd.DataFrame:
    """Create training data summary."""

    rows = []

    for split_name, df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        labels = prepare_labels(df)
        rows.append(
            {
                "split": split_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "positive_count": int((labels == 1).sum()),
                "negative_count": int((labels == 0).sum()),
                "positive_rate": float((labels == 1).mean()) if len(labels) else np.nan,
                "feature_count": len(feature_columns),
                "input_dir": str(input_dir),
                "seed": seed,
            }
        )

    return pd.DataFrame(rows)


def write_feature_columns(feature_columns: Sequence[str], output_path: Path) -> None:
    """Write selected feature columns to text file."""

    with output_path.open("w", encoding="utf-8") as output_file:
        for column in feature_columns:
            output_file.write(f"{column}\n")


def save_predictions(
    df: pd.DataFrame,
    probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save model predictions."""

    output_columns = [
        column for column in PREDICTION_OUTPUT_COLUMNS if column in df.columns
    ]
    predictions = df.loc[:, output_columns].copy()
    predictions["predicted_probability"] = probabilities
    predictions.to_csv(output_path, index=False)


def train_and_evaluate(args: argparse.Namespace) -> None:
    """Run the complete training and evaluation pipeline."""

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"

    input_files = require_input_files(input_dir)

    train_df = read_pairwise_csv(input_files["train"], "train")
    validation_df = read_pairwise_csv(input_files["validation"], "validation")
    test_df = read_pairwise_csv(input_files["test"], "test")

    validate_label_exists(train_df, "train")
    validate_label_exists(validation_df, "validation")
    validate_label_exists(test_df, "test")

    train_df = sample_groups_to_max_rows(
        train_df, args.max_train_rows, args.seed, "train"
    )
    validation_df = sample_groups_to_max_rows(
        validation_df, args.max_val_rows, args.seed, "validation"
    )
    test_df = sample_groups_to_max_rows(
        test_df, args.max_test_rows, args.seed, "test"
    )

    validate_binary_labels(train_df, "train")
    validate_binary_labels(validation_df, "validation")
    validate_group_columns(validation_df, "validation")
    validate_group_columns(test_df, "test")

    print_data_summary("train", train_df)
    print_data_summary("validation", validation_df)
    print_data_summary("test", test_df)

    feature_columns = select_feature_columns(train_df, validation_df, test_df)
    print(f"Feature columns used ({len(feature_columns)}): {feature_columns}")

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_features = prepare_feature_matrix(train_df, feature_columns)
    validation_features = prepare_feature_matrix(validation_df, feature_columns)
    test_features = prepare_feature_matrix(test_df, feature_columns)

    train_labels = prepare_labels(train_df)
    validation_labels = prepare_labels(validation_df)
    test_labels = prepare_labels(test_df)

    validation_results: List[Dict[str, Any]] = []
    test_results: List[Dict[str, Any]] = []
    skipped_models: List[Dict[str, str]] = []
    feature_importance_rows: List[Dict[str, Any]] = []
    trained_models: Dict[str, Any] = {}
    validation_predictions: Dict[str, np.ndarray] = {}
    test_predictions: Dict[str, np.ndarray] = {}

    requested_models = list(dict.fromkeys(args.models))

    for model_name in requested_models:
        print(f"\nTraining model: {model_name}")

        if not optional_package_available(model_name):
            reason = f"Optional package for {model_name} is not installed."
            print(f"Skipping {model_name}: {reason}")
            skipped_models.append({"model_name": model_name, "reason": reason})
            continue

        try:
            model = build_model(model_name, args.seed, train_labels)
            model.fit(train_features, train_labels)

            validation_metrics, validation_probabilities = evaluate_model(
                model_name,
                model,
                validation_df,
                validation_features,
                validation_labels,
            )
            test_metrics, test_probabilities = evaluate_model(
                model_name,
                model,
                test_df,
                test_features,
                test_labels,
            )

            validation_results.append(validation_metrics)
            test_results.append(test_metrics)

            saved_model_path = save_trained_model(model, model_name, models_dir)
            print(f"Saved model to: {saved_model_path}")

            trained_models[model_name] = model
            validation_predictions[model_name] = validation_probabilities
            test_predictions[model_name] = test_probabilities

            feature_importance_rows.extend(
                extract_feature_importance(model_name, model, feature_columns)
            )

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"Skipping {model_name}: {reason}")
            skipped_models.append({"model_name": model_name, "reason": reason})

    validation_results_df = pd.DataFrame(validation_results)
    test_results_df = pd.DataFrame(test_results)
    skipped_models_df = pd.DataFrame(skipped_models, columns=["model_name", "reason"])

    validation_results_df.to_csv(
        output_dir / "model_comparison_validation.csv",
        index=False,
    )
    test_results_df.to_csv(
        output_dir / "model_comparison_test.csv",
        index=False,
    )
    skipped_models_df.to_csv(
        output_dir / "skipped_models.csv",
        index=False,
    )

    write_feature_columns(feature_columns, output_dir / "feature_columns_used.txt")

    training_summary = build_training_data_summary(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        feature_columns=feature_columns,
        input_dir=input_dir,
        seed=args.seed,
    )
    training_summary.to_csv(
        output_dir / "training_data_summary.csv",
        index=False,
    )

    if feature_importance_rows:
        pd.DataFrame(feature_importance_rows).to_csv(
            output_dir / "feature_importance.csv",
            index=False,
        )

    if validation_results_df.empty:
        raise RuntimeError(
            "No models were trained successfully. Check skipped_models.csv for details."
        )

    best_model_name = select_best_model(validation_results_df)
    best_summary = build_best_model_summary(
        best_model_name=best_model_name,
        validation_results=validation_results_df,
        test_results=test_results_df,
        input_dir=input_dir,
        output_dir=output_dir,
        seed=args.seed,
        feature_columns=feature_columns,
    )
    best_summary.to_csv(output_dir / "best_model_summary.csv", index=False)

    if args.save_predictions:
        save_predictions(
            validation_df,
            validation_predictions[best_model_name],
            output_dir / "validation_predictions_best_model.csv",
        )
        save_predictions(
            test_df,
            test_predictions[best_model_name],
            output_dir / "test_predictions_best_model.csv",
        )

    validation_best_row = validation_results_df[
        validation_results_df["model_name"] == best_model_name
    ].iloc[0]
    test_best_row = test_results_df[
        test_results_df["model_name"] == best_model_name
    ].iloc[0]

    trained_model_names = list(trained_models.keys())
    skipped_model_names = skipped_models_df["model_name"].tolist()

    print("\nPreference model training complete.")
    print(
        "Models trained: "
        f"{', '.join(trained_model_names) if trained_model_names else 'None'}"
    )
    print(
        "Models skipped: "
        f"{', '.join(skipped_model_names) if skipped_model_names else 'None'}"
    )
    print(f"Best model: {best_model_name}")
    print(
        "Validation sampled_top1_accuracy: "
        f"{validation_best_row.get('sampled_top1_accuracy', np.nan)}"
    )
    print(
        "Test sampled_top1_accuracy: "
        f"{test_best_row.get('sampled_top1_accuracy', np.nan)}"
    )
    print(f"Validation roc_auc: {validation_best_row.get('roc_auc', np.nan)}")
    print(f"Test roc_auc: {test_best_row.get('roc_auc', np.nan)}")
    print(f"Output directory: {output_dir}")


def main() -> int:
    """Program entry point."""

    args = parse_args()

    try:
        train_and_evaluate(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
