#!/usr/bin/env python3
"""Generate small route-construction demos using a trained preference model.

This script is intentionally designed as a Colab-friendly prototype. It uses the
500-route pairwise sample outputs and a saved preference model to demonstrate how
ML_preference_probability(i, j) can be used in route generation.

Implemented route construction methods:
1. travel_time_nearest_neighbor: choose the reachable unvisited stop with the
   lowest travel time from the current stop.
2. preference_greedy: choose the unvisited stop with the highest predicted
   driver-like next-stop probability.
3. hybrid_greedy: choose the unvisited stop with the lowest preliminary hybrid
   cost using travel time, ML preference, zone change, time-window priority and
   workload terms.

This is a prototype/demo script, not the final full experimental evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    from last_mile_cleaning.clean_pipeline import is_missing, stream_top_level_object
except Exception:  # pragma: no cover - fallback for non-repo environments
    is_missing = None
    stream_top_level_object = None

DEFAULT_BASE = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_PROCESSED_DIR = DEFAULT_BASE / "processed_outputs"
DEFAULT_FINAL_CLEANED_DIR = DEFAULT_PROCESSED_DIR / "final_cleaned"
DEFAULT_PAIRWISE_DIR = DEFAULT_FINAL_CLEANED_DIR / "pairwise_samples_500routes"
DEFAULT_MODEL_DIR = DEFAULT_FINAL_CLEANED_DIR / "model_outputs_500routes_catboost_only"
DEFAULT_OUTPUT_DIR = DEFAULT_FINAL_CLEANED_DIR / "route_generation_demo_500routes"
DEFAULT_TRAVEL_TIME_OUTPUT_DIR = DEFAULT_PROCESSED_DIR / "travel_time_multisource_outputs"

TRAVEL_TIME_SOURCE_PATHS = {
    "training_build": Path("almrrc2021-data-training/model_build_inputs/travel_times.json"),
    "training_apply": Path("almrrc2021-data-training/model_apply_inputs/new_travel_times.json"),
    "evaluation_apply": Path("almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json"),
}

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

SUPPORTED_METHODS = [
    "travel_time_nearest_neighbor",
    "preference_greedy",
    "hybrid_greedy",
]

STOP_FEATURE_DEFAULTS: Dict[str, Any] = {
    "zone": "UNKNOWN_ZONE",
    "type": "UNKNOWN_TYPE",
    "is_station": 0,
    "is_dropoff": 0,
    "package_count": 0.0,
    "total_planned_service_time": 0.0,
    "has_time_window": 0,
    "time_window_package_count": 0.0,
    "total_package_volume_cm3": 0.0,
    "delivered_count": 0.0,
    "attempted_count": 0.0,
    "rejected_count": 0.0,
    "unknown_status_count": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate small route construction demos using a trained preference model."
    )
    parser.add_argument("--pairwise-dir", type=Path, default=DEFAULT_PAIRWISE_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-columns-path", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--final-cleaned-dir", type=Path, default=DEFAULT_FINAL_CLEANED_DIR)
    parser.add_argument("--travel-time-output-dir", type=Path, default=DEFAULT_TRAVEL_TIME_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--max-routes", type=int, default=10)
    parser.add_argument("--route-ids", nargs="*", default=None)
    parser.add_argument("--methods", nargs="+", choices=SUPPORTED_METHODS, default=SUPPORTED_METHODS)
    parser.add_argument("--hybrid-weights", default="travel=0.35,preference=0.40,zone=0.15,time_window=0.05,workload=0.05")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-step-details", action="store_true")
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def split_file_name(split: str) -> str:
    if split == "validation":
        return "validation_pairwise_samples.csv"
    return f"{split}_pairwise_samples.csv"


def load_pairwise_split(pairwise_dir: Path, split: str) -> pd.DataFrame:
    path = require_file(pairwise_dir / split_file_name(split), f"{split} pairwise samples")
    print(f"Reading {split} pairwise samples: {path}")
    return pd.read_csv(path)


def infer_model_path(model_dir: Path, explicit_model_path: Optional[Path]) -> Path:
    if explicit_model_path is not None:
        return require_file(explicit_model_path, "model file")

    candidates = [
        model_dir / "models" / "catboost.joblib",
        model_dir / "models" / "xgboost.joblib",
        model_dir / "models" / "lightgbm.joblib",
        model_dir / "models" / "random_forest.joblib",
        model_dir / "models" / "logistic_regression.joblib",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not infer model path. Provide --model-path or ensure model_dir/models contains a saved model."
    )


def load_feature_columns(model_dir: Path, explicit_path: Optional[Path]) -> List[str]:
    path = explicit_path if explicit_path is not None else model_dir / "feature_columns_used.txt"
    if path.exists():
        columns = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if columns:
            return columns
    return list(DEFAULT_FEATURE_COLUMNS)


def parse_hybrid_weights(text: str) -> Dict[str, float]:
    default = {"travel": 0.35, "preference": 0.40, "zone": 0.15, "time_window": 0.05, "workload": 0.05}
    if not text:
        return default

    weights: Dict[str, float] = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        try:
            weights[key] = float(value)
        except ValueError:
            raise ValueError(f"Invalid hybrid weight value: {part}")

    for key, value in default.items():
        weights.setdefault(key, value)

    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        raise ValueError("Hybrid weights must have positive total weight.")
    return {key: max(0.0, value) / total for key, value in weights.items()}


def choose_route_ids(df: pd.DataFrame, explicit_route_ids: Optional[Sequence[str]], max_routes: int) -> List[str]:
    if "route_id" not in df.columns:
        raise ValueError("pairwise samples must contain route_id")
    available = list(dict.fromkeys(df["route_id"].astype(str).tolist()))
    available_set = set(available)

    if explicit_route_ids:
        selected = [route_id for route_id in explicit_route_ids if route_id in available_set]
        missing = [route_id for route_id in explicit_route_ids if route_id not in available_set]
        if missing:
            print(f"Warning: {len(missing)} requested route IDs not found in split and will be ignored.")
        return selected[:max_routes] if max_routes and max_routes > 0 else selected

    if max_routes is None or max_routes <= 0:
        return available
    return available[:max_routes]


def reconstruct_actual_sequence(route_df: pd.DataFrame) -> List[str]:
    positive_rows = route_df[route_df["label"].astype(int) == 1].copy()
    if positive_rows.empty:
        return []
    positive_rows = positive_rows.sort_values("position", kind="mergesort")
    first_current = str(positive_rows.iloc[0]["current_stop"])
    next_stops = positive_rows["actual_next_stop"].astype(str).tolist()
    sequence = [first_current] + next_stops
    return sequence


def first_non_missing(values: Iterable[Any], default: Any) -> Any:
    for value in values:
        if pd.isna(value):
            continue
        value_str = str(value).strip()
        if value_str == "" or value_str.lower() in {"nan", "none", "null"}:
            continue
        return value
    return default


def numeric_first(values: Iterable[Any], default: float = 0.0) -> float:
    value = first_non_missing(values, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def build_stop_feature_lookup(route_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}

    def ensure_stop(stop_id: str) -> Dict[str, Any]:
        if stop_id not in lookup:
            lookup[stop_id] = dict(STOP_FEATURE_DEFAULTS)
        return lookup[stop_id]

    if "current_stop" in route_df.columns:
        for stop_id, group in route_df.groupby("current_stop", dropna=False):
            stop = ensure_stop(str(stop_id))
            if "current_zone" in group.columns:
                stop["zone"] = str(first_non_missing(group["current_zone"], stop["zone"]))
            if "current_type" in group.columns:
                stop["type"] = str(first_non_missing(group["current_type"], stop["type"]))
            if "current_is_station" in group.columns:
                stop["is_station"] = int(round(numeric_first(group["current_is_station"], stop["is_station"])))
            if "current_is_dropoff" in group.columns:
                stop["is_dropoff"] = int(round(numeric_first(group["current_is_dropoff"], stop["is_dropoff"])))

    if "candidate_stop" in route_df.columns:
        for stop_id, group in route_df.groupby("candidate_stop", dropna=False):
            stop = ensure_stop(str(stop_id))
            if "candidate_zone" in group.columns:
                stop["zone"] = str(first_non_missing(group["candidate_zone"], stop["zone"]))
            if "candidate_type" in group.columns:
                stop["type"] = str(first_non_missing(group["candidate_type"], stop["type"]))
            if "candidate_is_station" in group.columns:
                stop["is_station"] = int(round(numeric_first(group["candidate_is_station"], stop["is_station"])))
            if "candidate_is_dropoff" in group.columns:
                stop["is_dropoff"] = int(round(numeric_first(group["candidate_is_dropoff"], stop["is_dropoff"])))

            mapping = {
                "candidate_package_count": "package_count",
                "candidate_total_planned_service_time": "total_planned_service_time",
                "candidate_has_time_window": "has_time_window",
                "candidate_time_window_package_count": "time_window_package_count",
                "candidate_total_package_volume_cm3": "total_package_volume_cm3",
                "candidate_delivered_count": "delivered_count",
                "candidate_attempted_count": "attempted_count",
                "candidate_rejected_count": "rejected_count",
                "candidate_unknown_status_count": "unknown_status_count",
            }
            for source_col, target_key in mapping.items():
                if source_col in group.columns:
                    value = numeric_first(group[source_col], stop[target_key])
                    stop[target_key] = value

    return lookup


def normalize_zone(value: Any) -> str:
    if pd.isna(value):
        return "UNKNOWN_ZONE"
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return "UNKNOWN_ZONE"
    return text


def read_source_lookup(path: Path) -> Dict[str, str]:
    if not path.exists():
        print(f"Warning: travel-time source lookup not found: {path}")
        return {}
    df = pd.read_csv(path)
    if "route_id" not in df.columns:
        print(f"Warning: route_id missing from source lookup: {path}")
        return {}

    source_col = None
    for candidate in ["travel_time_source", "source", "source_label", "matched_source"]:
        if candidate in df.columns:
            source_col = candidate
            break
    if source_col is None:
        for column in df.columns:
            if "source" in column.lower():
                source_col = column
                break
    if source_col is None:
        print(f"Warning: no source column found in source lookup: {path}")
        return {}

    lookup = {}
    for _, row in df.iterrows():
        route_id = str(row["route_id"])
        source = str(row[source_col])
        lookup[route_id] = source
    return lookup


def source_candidates_for_route(route_id: str, lookup: Dict[str, str]) -> List[str]:
    source = lookup.get(route_id, "")
    if source in TRAVEL_TIME_SOURCE_PATHS:
        return [source]
    if "training_build" in source:
        return ["training_build", "training_apply", "evaluation_apply"]
    if "training_apply" in source:
        return ["training_apply", "training_build", "evaluation_apply"]
    if "evaluation_apply" in source:
        return ["evaluation_apply", "training_build", "training_apply"]
    return ["training_build", "training_apply", "evaluation_apply"]


def missing_value(value: Any) -> bool:
    if is_missing is not None:
        try:
            return bool(is_missing(value))
        except Exception:
            pass
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def get_travel_time(matrix: Dict[str, Any], from_stop: str, to_stop: str) -> Optional[float]:
    if from_stop not in matrix:
        return None
    row = matrix[from_stop]
    if not isinstance(row, dict) or to_stop not in row:
        return None
    value = row[to_stop]
    if missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def stream_routes_from_json(path: Path, target_route_ids: set[str]) -> Dict[str, Dict[str, Any]]:
    if stream_top_level_object is None:
        raise RuntimeError(
            "stream_top_level_object is unavailable. Run this script from the repository root with PYTHONPATH=src."
        )
    if not path.exists():
        print(f"Warning: travel-time JSON file not found: {path}")
        return {}

    found: Dict[str, Dict[str, Any]] = {}
    remaining = set(target_route_ids)
    for route_id, matrix in stream_top_level_object(path):
        route_id_str = str(route_id)
        if route_id_str in remaining:
            found[route_id_str] = matrix
            remaining.remove(route_id_str)
            if not remaining:
                break
    return found


def load_travel_matrices(route_ids: Sequence[str], data_root: Path, source_lookup: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    route_ids_set = set(route_ids)
    matrices: Dict[str, Dict[str, Any]] = {}

    for source_name in ["training_build", "training_apply", "evaluation_apply"]:
        needed = {
            route_id
            for route_id in route_ids_set
            if route_id not in matrices and source_name in source_candidates_for_route(route_id, source_lookup)
        }
        if not needed:
            continue
        path = data_root / TRAVEL_TIME_SOURCE_PATHS[source_name]
        print(f"Streaming {len(needed)} route travel-time matrices from {source_name}: {path}")
        matrices.update(stream_routes_from_json(path, needed))

    missing_routes = route_ids_set - set(matrices.keys())
    if missing_routes:
        print(f"Warning: {len(missing_routes)} route travel-time matrices were not found.")
    return matrices


def build_feature_row(
    current_stop: str,
    candidate_stop: str,
    travel_time: float,
    position: int,
    number_of_stops: int,
    remaining_stop_count: int,
    stop_lookup: Dict[str, Dict[str, Any]],
    feature_columns: Sequence[str],
) -> Dict[str, float]:
    current = stop_lookup.get(current_stop, STOP_FEATURE_DEFAULTS)
    candidate = stop_lookup.get(candidate_stop, STOP_FEATURE_DEFAULTS)

    current_zone = normalize_zone(current.get("zone", "UNKNOWN_ZONE"))
    candidate_zone = normalize_zone(candidate.get("zone", "UNKNOWN_ZONE"))
    zone_missing = int(current_zone == "UNKNOWN_ZONE" or candidate_zone == "UNKNOWN_ZONE")
    same_zone = int(zone_missing == 0 and current_zone == candidate_zone)
    zone_changed = int(zone_missing == 0 and current_zone != candidate_zone)
    route_progress = position / (number_of_stops - 1) if number_of_stops > 1 else 0.0

    row = {
        "travel_time_ij": travel_time,
        "same_zone": same_zone,
        "zone_changed": zone_changed,
        "zone_missing_in_pair": zone_missing,
        "number_of_stops": number_of_stops,
        "route_progress": route_progress,
        "remaining_stop_count": remaining_stop_count,
        "current_is_station": current.get("is_station", 0),
        "current_is_dropoff": current.get("is_dropoff", 0),
        "candidate_is_station": candidate.get("is_station", 0),
        "candidate_is_dropoff": candidate.get("is_dropoff", 0),
        "candidate_package_count": candidate.get("package_count", 0.0),
        "candidate_total_planned_service_time": candidate.get("total_planned_service_time", 0.0),
        "candidate_has_time_window": candidate.get("has_time_window", 0),
        "candidate_time_window_package_count": candidate.get("time_window_package_count", 0.0),
        "candidate_total_package_volume_cm3": candidate.get("total_package_volume_cm3", 0.0),
        "candidate_delivered_count": candidate.get("delivered_count", 0.0),
        "candidate_attempted_count": candidate.get("attempted_count", 0.0),
        "candidate_rejected_count": candidate.get("rejected_count", 0.0),
        "candidate_unknown_status_count": candidate.get("unknown_status_count", 0.0),
    }
    return {feature: float(row.get(feature, 0.0) or 0.0) for feature in feature_columns}


def predict_probabilities(model: Any, feature_rows: List[Dict[str, float]], feature_columns: Sequence[str]) -> np.ndarray:
    if not feature_rows:
        return np.array([], dtype=float)
    features = pd.DataFrame(feature_rows)
    for column in feature_columns:
        if column not in features.columns:
            features[column] = 0.0
    features = features.loc[:, feature_columns]
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    features = features.fillna(0.0)

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


def minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if math.isclose(max_value, min_value):
        return np.zeros_like(values, dtype=float)
    return (values - min_value) / (max_value - min_value)


def choose_next_stop(
    method: str,
    candidates: List[str],
    travel_times: np.ndarray,
    probabilities: np.ndarray,
    feature_rows: List[Dict[str, float]],
    hybrid_weights: Dict[str, float],
) -> Tuple[str, Dict[str, float]]:
    if method == "travel_time_nearest_neighbor":
        index = int(np.argmin(travel_times))
        return candidates[index], {"selected_probability": float(probabilities[index]) if probabilities.size else np.nan, "selected_cost": float(travel_times[index])}

    if method == "preference_greedy":
        index = int(np.argmax(probabilities))
        return candidates[index], {"selected_probability": float(probabilities[index]), "selected_cost": float(1.0 - probabilities[index])}

    if method == "hybrid_greedy":
        travel_component = minmax(travel_times.astype(float))
        preference_component = 1.0 - probabilities.astype(float)
        zone_component = np.asarray([row.get("zone_changed", 0.0) for row in feature_rows], dtype=float)
        time_window_component = np.asarray([1.0 - row.get("candidate_has_time_window", 0.0) for row in feature_rows], dtype=float)
        workload_raw = np.asarray(
            [
                row.get("candidate_package_count", 0.0)
                + row.get("candidate_total_planned_service_time", 0.0) / 300.0
                + row.get("candidate_total_package_volume_cm3", 0.0) / 50000.0
                for row in feature_rows
            ],
            dtype=float,
        )
        workload_component = minmax(workload_raw)

        cost = (
            hybrid_weights.get("travel", 0.0) * travel_component
            + hybrid_weights.get("preference", 0.0) * preference_component
            + hybrid_weights.get("zone", 0.0) * zone_component
            + hybrid_weights.get("time_window", 0.0) * time_window_component
            + hybrid_weights.get("workload", 0.0) * workload_component
        )
        index = int(np.argmin(cost))
        return candidates[index], {"selected_probability": float(probabilities[index]), "selected_cost": float(cost[index])}

    raise ValueError(f"Unsupported route generation method: {method}")


def generate_route(
    method: str,
    actual_sequence: List[str],
    matrix: Dict[str, Any],
    stop_lookup: Dict[str, Dict[str, Any]],
    model: Any,
    feature_columns: Sequence[str],
    hybrid_weights: Dict[str, float],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    if len(actual_sequence) <= 1:
        return actual_sequence, []

    current_stop = actual_sequence[0]
    unvisited = list(actual_sequence[1:])
    generated = [current_stop]
    step_rows: List[Dict[str, Any]] = []
    number_of_stops = len(actual_sequence)
    position = 0

    while unvisited:
        candidate_stops: List[str] = []
        candidate_travel_times: List[float] = []
        candidate_feature_rows: List[Dict[str, float]] = []

        for candidate in unvisited:
            travel_time = get_travel_time(matrix, current_stop, candidate)
            if travel_time is None:
                continue
            feature_row = build_feature_row(
                current_stop=current_stop,
                candidate_stop=candidate,
                travel_time=travel_time,
                position=position,
                number_of_stops=number_of_stops,
                remaining_stop_count=len(unvisited),
                stop_lookup=stop_lookup,
                feature_columns=feature_columns,
            )
            candidate_stops.append(candidate)
            candidate_travel_times.append(travel_time)
            candidate_feature_rows.append(feature_row)

        if not candidate_stops:
            print(f"Warning: no reachable unvisited candidates from {current_stop}; appending remaining stops in actual order.")
            generated.extend(unvisited)
            break

        travel_times_array = np.asarray(candidate_travel_times, dtype=float)
        probabilities = predict_probabilities(model, candidate_feature_rows, feature_columns)
        selected_stop, detail = choose_next_stop(
            method=method,
            candidates=candidate_stops,
            travel_times=travel_times_array,
            probabilities=probabilities,
            feature_rows=candidate_feature_rows,
            hybrid_weights=hybrid_weights,
        )

        selected_travel_time = get_travel_time(matrix, current_stop, selected_stop)
        step_rows.append(
            {
                "position": position,
                "current_stop": current_stop,
                "selected_stop": selected_stop,
                "selected_travel_time": selected_travel_time,
                "selected_probability": detail.get("selected_probability", np.nan),
                "selected_cost": detail.get("selected_cost", np.nan),
                "candidate_count": len(candidate_stops),
            }
        )

        generated.append(selected_stop)
        unvisited.remove(selected_stop)
        current_stop = selected_stop
        position += 1

    return generated, step_rows


def total_route_travel_time(sequence: Sequence[str], matrix: Dict[str, Any]) -> Optional[float]:
    total = 0.0
    for from_stop, to_stop in zip(sequence[:-1], sequence[1:]):
        travel_time = get_travel_time(matrix, from_stop, to_stop)
        if travel_time is None:
            return None
        total += travel_time
    return total


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for item_a in a:
        current = [0]
        for index_b, item_b in enumerate(b, start=1):
            if item_a == item_b:
                current.append(previous[index_b - 1] + 1)
            else:
                current.append(max(previous[index_b], current[-1]))
        previous = current
    return previous[-1]


def position_match_ratio(actual: Sequence[str], generated: Sequence[str]) -> float:
    actual_body = list(actual[1:])
    generated_body = list(generated[1:])
    if not actual_body:
        return float("nan")
    matches = sum(1 for a, g in zip(actual_body, generated_body) if a == g)
    return matches / len(actual_body)


def lcs_similarity(actual: Sequence[str], generated: Sequence[str]) -> float:
    actual_body = list(actual[1:])
    generated_body = list(generated[1:])
    if not actual_body:
        return float("nan")
    return lcs_length(actual_body, generated_body) / len(actual_body)


def same_zone_ratio(sequence: Sequence[str], stop_lookup: Dict[str, Dict[str, Any]]) -> float:
    if len(sequence) <= 1:
        return float("nan")
    valid = 0
    same = 0
    for from_stop, to_stop in zip(sequence[:-1], sequence[1:]):
        from_zone = normalize_zone(stop_lookup.get(from_stop, STOP_FEATURE_DEFAULTS).get("zone", "UNKNOWN_ZONE"))
        to_zone = normalize_zone(stop_lookup.get(to_stop, STOP_FEATURE_DEFAULTS).get("zone", "UNKNOWN_ZONE"))
        if from_zone == "UNKNOWN_ZONE" or to_zone == "UNKNOWN_ZONE":
            continue
        valid += 1
        if from_zone == to_zone:
            same += 1
    return same / valid if valid else float("nan")


def evaluate_generated_route(
    route_id: str,
    method: str,
    actual_sequence: Sequence[str],
    generated_sequence: Sequence[str],
    matrix: Dict[str, Any],
    stop_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    actual_time = total_route_travel_time(actual_sequence, matrix)
    generated_time = total_route_travel_time(generated_sequence, matrix)
    travel_time_ratio = (
        generated_time / actual_time
        if actual_time is not None and generated_time is not None and actual_time > 0
        else np.nan
    )
    return {
        "route_id": route_id,
        "method": method,
        "stop_count": len(actual_sequence),
        "actual_total_travel_time": actual_time,
        "generated_total_travel_time": generated_time,
        "travel_time_ratio_to_actual": travel_time_ratio,
        "position_match_ratio": position_match_ratio(actual_sequence, generated_sequence),
        "lcs_similarity": lcs_similarity(actual_sequence, generated_sequence),
        "actual_same_zone_ratio": same_zone_ratio(actual_sequence, stop_lookup),
        "generated_same_zone_ratio": same_zone_ratio(generated_sequence, stop_lookup),
    }


def write_sequences(path: Path, route_id: str, method: str, sequence: Sequence[str]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["route_id", "method", "order", "stop_id"])
        if not file_exists:
            writer.writeheader()
        for order, stop_id in enumerate(sequence):
            writer.writerow({"route_id": route_id, "method": method, "order": order, "stop_id": stop_id})


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)

    pairwise_df = load_pairwise_split(args.pairwise_dir, args.split)
    route_ids = choose_route_ids(pairwise_df, args.route_ids, args.max_routes)
    if not route_ids:
        raise ValueError("No route IDs selected for route generation demo.")

    model_path = infer_model_path(args.model_dir, args.model_path)
    feature_columns = load_feature_columns(args.model_dir, args.feature_columns_path)
    hybrid_weights = parse_hybrid_weights(args.hybrid_weights)

    print(f"Selected {len(route_ids)} route(s) for demo.")
    print(f"Loading model: {model_path}")
    print(f"Using feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Hybrid weights: {hybrid_weights}")
    model = joblib.load(model_path)

    source_lookup = read_source_lookup(args.travel_time_output_dir / "route_travel_time_source_lookup.csv")
    travel_matrices = load_travel_matrices(route_ids, args.data_root, source_lookup)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: List[Dict[str, Any]] = []
    step_detail_rows: List[Dict[str, Any]] = []
    sequence_path = args.output_dir / "route_generation_sequences.csv"
    if sequence_path.exists():
        sequence_path.unlink()

    for route_id in route_ids:
        route_df = pairwise_df[pairwise_df["route_id"].astype(str) == route_id].copy()
        actual_sequence = reconstruct_actual_sequence(route_df)
        if len(actual_sequence) <= 1:
            print(f"Skipping {route_id}: could not reconstruct actual sequence.")
            continue
        if route_id not in travel_matrices:
            print(f"Skipping {route_id}: missing travel-time matrix.")
            continue

        matrix = travel_matrices[route_id]
        stop_lookup = build_stop_feature_lookup(route_df)
        write_sequences(sequence_path, route_id, "actual", actual_sequence)

        for method in args.methods:
            generated_sequence, step_rows = generate_route(
                method=method,
                actual_sequence=actual_sequence,
                matrix=matrix,
                stop_lookup=stop_lookup,
                model=model,
                feature_columns=feature_columns,
                hybrid_weights=hybrid_weights,
            )
            write_sequences(sequence_path, route_id, method, generated_sequence)
            summary_rows.append(
                evaluate_generated_route(
                    route_id=route_id,
                    method=method,
                    actual_sequence=actual_sequence,
                    generated_sequence=generated_sequence,
                    matrix=matrix,
                    stop_lookup=stop_lookup,
                )
            )
            if args.save_step_details:
                for row in step_rows:
                    row = dict(row)
                    row["route_id"] = route_id
                    row["method"] = method
                    step_detail_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.output_dir / "route_generation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    if step_detail_rows:
        pd.DataFrame(step_detail_rows).to_csv(args.output_dir / "route_generation_step_details.csv", index=False)

    if not summary_df.empty:
        aggregate = (
            summary_df.groupby("method", dropna=False)
            .agg(
                route_count=("route_id", "count"),
                avg_lcs_similarity=("lcs_similarity", "mean"),
                avg_position_match_ratio=("position_match_ratio", "mean"),
                avg_travel_time_ratio_to_actual=("travel_time_ratio_to_actual", "mean"),
                avg_generated_same_zone_ratio=("generated_same_zone_ratio", "mean"),
            )
            .reset_index()
        )
        aggregate.to_csv(args.output_dir / "route_generation_method_summary.csv", index=False)
        print("\nRoute generation method summary:")
        print(aggregate)

    print("\nRoute generation demo complete.")
    print(f"Routes selected: {len(route_ids)}")
    print(f"Methods: {', '.join(args.methods)}")
    print(f"Summary: {summary_path}")
    print(f"Sequences: {sequence_path}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
