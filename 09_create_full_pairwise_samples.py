#!/usr/bin/env python3
"""Create full-scale pairwise next-stop preference samples for final experiments.

This script upgrades the P1 pairwise-sample prototype to a config-driven,
full-scale generator. It keeps the same core learning formulation:

    current stop i + candidate stop j -> label

where label=1 means the candidate is the actual next stop and label=0 means
the candidate was available later in the same route but was not selected next.

Negative candidate travel times are looked up from the original route travel
time matrix JSON files, not from the actual-transition table, because unchosen
candidate edges are not necessarily actual driver transitions.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("Please install pyyaml: pip install pyyaml") from exc

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    from last_mile_cleaning.clean_pipeline import is_missing, stream_top_level_object
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Could not import last_mile_cleaning.clean_pipeline. "
        "Run from the repository root or set PYTHONPATH=src."
    ) from exc


VALID_SPLITS = ("train", "validation", "test")
SOURCE_PRIORITY = ("training_build", "training_apply", "evaluation_apply")
UNKNOWN_ZONE = "UNKNOWN_ZONE"

OUTPUT_COLUMNS = [
    "route_id",
    "split",
    "position",
    "current_stop",
    "candidate_stop",
    "actual_next_stop",
    "label",
    "route_score",
    "number_of_stops",
    "route_progress",
    "remaining_stop_count",
    "current_zone",
    "current_type",
    "current_is_station",
    "current_is_dropoff",
    "candidate_zone",
    "candidate_type",
    "candidate_is_station",
    "candidate_is_dropoff",
    "travel_time_ij",
    "same_zone",
    "zone_changed",
    "zone_missing_in_pair",
    "candidate_package_count",
    "candidate_total_planned_service_time",
    "candidate_has_time_window",
    "candidate_time_window_package_count",
    "candidate_total_package_volume_cm3",
    "candidate_delivered_count",
    "candidate_attempted_count",
    "candidate_rejected_count",
    "candidate_unknown_status_count",
    "negative_sampling_seed",
    "negative_sample_rank",
    "travel_time_source",
    "generation_timestamp",
]

NUMERIC_COLUMNS = {
    "position": "int",
    "label": "int",
    "number_of_stops": "int",
    "route_progress": "float",
    "remaining_stop_count": "int",
    "current_is_station": "int",
    "current_is_dropoff": "int",
    "candidate_is_station": "int",
    "candidate_is_dropoff": "int",
    "travel_time_ij": "float",
    "same_zone": "int",
    "zone_changed": "int",
    "zone_missing_in_pair": "int",
    "candidate_package_count": "float",
    "candidate_total_planned_service_time": "float",
    "candidate_has_time_window": "int",
    "candidate_time_window_package_count": "float",
    "candidate_total_package_volume_cm3": "float",
    "candidate_delivered_count": "float",
    "candidate_attempted_count": "float",
    "candidate_rejected_count": "float",
    "candidate_unknown_status_count": "float",
    "negative_sampling_seed": "int",
    "negative_sample_rank": "int",
}

SUMMARY_COLUMNS = [
    "split",
    "routes_requested",
    "routes_processed",
    "routes_skipped_missing_source",
    "routes_skipped_missing_transitions",
    "positive_samples_written",
    "negative_samples_written",
    "total_samples_written",
    "skipped_positive_missing_travel_time",
    "skipped_negative_missing_travel_time",
    "average_negatives_per_positive",
    "negative_samples_per_positive_requested",
    "seed",
    "output_format",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create full-scale pairwise positive/negative next-stop samples."
    )
    parser.add_argument("--config", type=Path, default=Path("config/config_final.yaml"))
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=list(VALID_SPLITS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative-samples-per-positive", type=int, default=None)
    parser.add_argument("--max-routes-per-split", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-format", choices=("parquet", "csv"), default=None)
    parser.add_argument("--compression", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def expand_path(value: Any) -> Path:
    text = str(value)
    if text.startswith("/content/"):
        return Path(text)
    return Path(text).expanduser().resolve()


def resolve_path_entry(entry: Any, *, required: bool = True, label: str = "path") -> Path | None:
    candidates: list[Any]
    if isinstance(entry, dict) and "candidates" in entry:
        candidates = entry["candidates"]
    elif isinstance(entry, (list, tuple)):
        candidates = list(entry)
    else:
        candidates = [entry]
    for raw in candidates:
        if raw is None:
            continue
        path = expand_path(raw)
        if path.exists():
            return path
    if required:
        checked = ", ".join(str(expand_path(p)) for p in candidates if p is not None)
        raise FileNotFoundError(f"Required {label} not found. Checked: {checked}")
    return None


def first_present(row: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        if name in row and not is_missing(row.get(name)):
            return row.get(name)
    return default


def to_int(value: Any, default: int = 0) -> int:
    if is_missing(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip().lower()
        if text in {"true", "yes", "y"}:
            return 1
        if text in {"false", "no", "n"}:
            return 0
        return default


def to_float(value: Any, default: float | None = 0.0) -> float | None:
    if is_missing(value):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_zone(value: Any) -> str:
    if is_missing(value):
        return UNKNOWN_ZONE
    text = str(value).strip()
    if text == "" or text.upper() in {"UNKNOWN", "UNKNOWN_ZONE", "NAN", "NONE", "NULL"}:
        return UNKNOWN_ZONE
    return text


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def load_route_ids(path: Path, max_routes: int | None = None) -> list[str]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    field = "route_id" if "route_id" in rows[0] else next(iter(rows[0]))
    route_ids = [str(row.get(field, "")).strip() for row in rows if not is_missing(row.get(field))]
    route_ids = list(dict.fromkeys(route_ids))
    return route_ids[:max_routes] if max_routes is not None else route_ids


def make_summary_row(
    seed: int,
    output_format: str,
    split: str,
    data: dict[str, Any],
    negatives_per_positive: int,
) -> dict[str, Any]:
    positives = data["positive_samples_written"]
    negatives = data["negative_samples_written"]
    total = positives + negatives
    return {
        "split": split,
        **data,
        "total_samples_written": total,
        "average_negatives_per_positive": 0 if positives == 0 else negatives / positives,
        "negative_samples_per_positive_requested": negatives_per_positive,
        "seed": seed,
        "output_format": output_format,
    }


def load_route_features(path: Path) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        route_id = str(first_present(row, ("route_id",))).strip()
        if not route_id:
            continue
        features[route_id] = {
            "route_score": first_present(row, ("route_score",), "Missing"),
            "number_of_stops": to_int(first_present(row, ("number_of_stops", "num_stops", "stop_count"), 0)),
        }
    return features


def load_stop_features(path: Path, route_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    features: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            route_id = str(first_present(row, ("route_id",))).strip()
            if route_id not in route_ids:
                continue
            stop_id = str(first_present(row, ("stop_id",))).strip()
            if not stop_id:
                continue
            features[(route_id, stop_id)] = {
                "zone": normalize_zone(first_present(row, ("zone_id", "zone", "zone_label"), UNKNOWN_ZONE)),
                "type": first_present(row, ("type", "stop_type"), ""),
                "is_station": to_int(first_present(row, ("is_station",), 0)),
                "is_dropoff": to_int(first_present(row, ("is_dropoff",), 0)),
            }
    return features


def load_package_features(path: Path, route_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    features: dict[tuple[str, str], dict[str, Any]] = {}
    aliases = {
        "package_count": ("package_count", "packages_count", "num_packages"),
        "total_planned_service_time": ("total_planned_service_time", "planned_service_time_sum"),
        "has_time_window": ("has_time_window",),
        "time_window_package_count": ("time_window_package_count",),
        "total_package_volume_cm3": ("total_package_volume_cm3", "package_volume_cm3"),
        "delivered_count": ("delivered_count",),
        "attempted_count": ("attempted_count",),
        "rejected_count": ("rejected_count",),
        "unknown_status_count": ("unknown_status_count",),
    }
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            route_id = str(first_present(row, ("route_id",))).strip()
            if route_id not in route_ids:
                continue
            stop_id = str(first_present(row, ("stop_id",))).strip()
            if not stop_id:
                continue
            pkg: dict[str, Any] = {}
            for name, names in aliases.items():
                value = first_present(row, names, 0)
                pkg[name] = to_int(value) if name == "has_time_window" else to_float(value)
            features[(route_id, stop_id)] = pkg
    return features


def load_transitions(path: Path, route_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            route_id = str(first_present(row, ("route_id",))).strip()
            if route_id in route_ids:
                grouped[route_id].append(row)
    return grouped


def build_sequence(rows: list[dict[str, Any]]) -> list[str]:
    def pos(row: dict[str, Any]) -> int:
        return to_int(first_present(row, ("position", "transition_position", "actual_position"), 0))

    ordered = sorted(rows, key=pos)
    if not ordered:
        return []
    first_from = str(first_present(ordered[0], ("from_stop", "from_stop_id"))).strip()
    if not first_from:
        return []
    sequence = [first_from]
    for row in ordered:
        to_stop = str(first_present(row, ("to_stop", "to_stop_id"))).strip()
        if not to_stop:
            return []
        sequence.append(to_stop)
    return sequence


def choose_source(row: dict[str, str]) -> str | None:
    raw = str(first_present(row, ("travel_time_source", "source", "source_label", "selected_source"), "")).strip()
    if raw in SOURCE_PRIORITY:
        return raw
    if raw == "multiple_sources":
        return SOURCE_PRIORITY[0]
    if raw == "missing_source":
        return None
    for source in SOURCE_PRIORITY:
        marker = str(row.get(source, row.get(f"has_{source}", ""))).strip().lower()
        if marker in {"1", "true", "yes", source}:
            return source
    return None


def load_source_lookup(path: Path, route_ids: set[str]) -> dict[str, str | None]:
    lookup: dict[str, str | None] = {}
    for row in read_csv_rows(path):
        route_id = str(first_present(row, ("route_id",))).strip()
        if route_id in route_ids:
            lookup[route_id] = choose_source(row)
    return lookup


def get_travel_time(matrix: Any, from_stop: str, to_stop: str) -> float | None:
    if not isinstance(matrix, dict):
        return None
    row = matrix.get(from_stop)
    if isinstance(row, dict):
        value = row.get(to_stop)
        if is_missing(value):
            return None
        return to_float(value, default=None)
    return None


def pair_flags(current_zone: str, candidate_zone: str) -> tuple[int, int, int]:
    missing = current_zone == UNKNOWN_ZONE or candidate_zone == UNKNOWN_ZONE
    same = (not missing) and current_zone == candidate_zone
    changed = (not missing) and current_zone != candidate_zone
    return int(same), int(changed), int(missing)


def make_output_row(
    *,
    route_id: str,
    split: str,
    position: int,
    current_stop: str,
    candidate_stop: str,
    actual_next_stop: str,
    label: int,
    travel_time: float,
    route_features: dict[str, Any],
    stop_features: dict[tuple[str, str], dict[str, Any]],
    package_features: dict[tuple[str, str], dict[str, Any]],
    sequence_length: int,
    seed: int,
    negative_rank: int,
    travel_time_source: str,
    generation_timestamp: str,
) -> dict[str, Any]:
    current = stop_features.get((route_id, current_stop), {})
    candidate = stop_features.get((route_id, candidate_stop), {})
    current_zone = normalize_zone(current.get("zone", UNKNOWN_ZONE))
    candidate_zone = normalize_zone(candidate.get("zone", UNKNOWN_ZONE))
    same_zone, zone_changed, zone_missing = pair_flags(current_zone, candidate_zone)
    packages = package_features.get((route_id, candidate_stop), {})
    denominator = sequence_length - 1
    return {
        "route_id": route_id,
        "split": split,
        "position": position,
        "current_stop": current_stop,
        "candidate_stop": candidate_stop,
        "actual_next_stop": actual_next_stop,
        "label": label,
        "route_score": route_features.get("route_score", "Missing"),
        "number_of_stops": route_features.get("number_of_stops") or sequence_length,
        "route_progress": 0 if denominator <= 0 else position / denominator,
        "remaining_stop_count": sequence_length - position - 1,
        "current_zone": current_zone,
        "current_type": current.get("type", ""),
        "current_is_station": to_int(current.get("is_station", 0)),
        "current_is_dropoff": to_int(current.get("is_dropoff", 0)),
        "candidate_zone": candidate_zone,
        "candidate_type": candidate.get("type", ""),
        "candidate_is_station": to_int(candidate.get("is_station", 0)),
        "candidate_is_dropoff": to_int(candidate.get("is_dropoff", 0)),
        "travel_time_ij": travel_time,
        "same_zone": same_zone,
        "zone_changed": zone_changed,
        "zone_missing_in_pair": zone_missing,
        "candidate_package_count": packages.get("package_count", 0),
        "candidate_total_planned_service_time": packages.get("total_planned_service_time", 0),
        "candidate_has_time_window": to_int(packages.get("has_time_window", 0)),
        "candidate_time_window_package_count": packages.get("time_window_package_count", 0),
        "candidate_total_package_volume_cm3": packages.get("total_package_volume_cm3", 0),
        "candidate_delivered_count": packages.get("delivered_count", 0),
        "candidate_attempted_count": packages.get("attempted_count", 0),
        "candidate_rejected_count": packages.get("rejected_count", 0),
        "candidate_unknown_status_count": packages.get("unknown_status_count", 0),
        "negative_sampling_seed": seed,
        "negative_sample_rank": negative_rank,
        "travel_time_source": travel_time_source,
        "generation_timestamp": generation_timestamp,
    }


class SplitWriter:
    def __init__(self, path: Path, output_format: str, compression: str = "snappy", batch_size: int = 50_000) -> None:
        self.path = path
        self.output_format = output_format
        self.compression = compression
        self.batch_size = batch_size
        self.rows: list[dict[str, Any]] = []
        self.csv_file = None
        self.csv_writer = None
        self.parquet_writer = None
        self.pa = None
        self.pq = None
        if output_format == "csv":
            self.csv_file = path.open("w", encoding="utf-8", newline="")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=OUTPUT_COLUMNS)
            self.csv_writer.writeheader()
        elif output_format == "parquet":
            try:
                import pyarrow as pa  # type: ignore
                import pyarrow.parquet as pq  # type: ignore
            except ImportError as exc:
                raise SystemExit(
                    "Parquet output requires pyarrow or fastparquet. "
                    "Use --output-format csv or install pyarrow."
                ) from exc
            self.pa = pa
            self.pq = pq
            self.schema = self._build_schema(pa)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _build_schema(self, pa: Any) -> Any:
        fields = []
        for name in OUTPUT_COLUMNS:
            kind = NUMERIC_COLUMNS.get(name)
            if kind == "int":
                fields.append((name, pa.int64()))
            elif kind == "float":
                fields.append((name, pa.float64()))
            else:
                fields.append((name, pa.string()))
        return pa.schema(fields)

    def write(self, row: dict[str, Any]) -> None:
        clean = {name: row.get(name, "") for name in OUTPUT_COLUMNS}
        if self.output_format == "csv":
            assert self.csv_writer is not None
            self.csv_writer.writerow(clean)
            return
        self.rows.append(clean)
        if len(self.rows) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if self.output_format != "parquet" or not self.rows:
            return
        table = self.pa.Table.from_pylist(self.rows, schema=self.schema)
        if self.parquet_writer is None:
            self.parquet_writer = self.pq.ParquetWriter(self.path, self.schema, compression=self.compression)
        self.parquet_writer.write_table(table)
        self.rows.clear()

    def close(self) -> None:
        if self.output_format == "parquet":
            self.flush()
            if self.parquet_writer is not None:
                self.parquet_writer.close()
        if self.csv_file is not None:
            self.csv_file.close()


def process_route(
    *,
    route_id: str,
    split: str,
    matrix: Any,
    writer: SplitWriter,
    transitions: dict[str, list[dict[str, Any]]],
    route_features_all: dict[str, dict[str, Any]],
    stop_features: dict[tuple[str, str], dict[str, Any]],
    package_features: dict[tuple[str, str], dict[str, Any]],
    rng: random.Random,
    seed: int,
    negatives_per_positive: int,
    summary: dict[str, dict[str, Any]],
    travel_time_source: str,
    generation_timestamp: str,
) -> None:
    sequence = build_sequence(transitions.get(route_id, []))
    if len(sequence) < 2:
        summary[split]["routes_skipped_missing_transitions"] += 1
        return
    route_features = route_features_all.get(route_id, {})
    summary[split]["routes_processed"] += 1
    for position in range(len(sequence) - 1):
        current_stop = sequence[position]
        actual_next_stop = sequence[position + 1]
        positive_time = get_travel_time(matrix, current_stop, actual_next_stop)
        if positive_time is None:
            summary[split]["skipped_positive_missing_travel_time"] += 1
        else:
            writer.write(
                make_output_row(
                    route_id=route_id,
                    split=split,
                    position=position,
                    current_stop=current_stop,
                    candidate_stop=actual_next_stop,
                    actual_next_stop=actual_next_stop,
                    label=1,
                    travel_time=positive_time,
                    route_features=route_features,
                    stop_features=stop_features,
                    package_features=package_features,
                    sequence_length=len(sequence),
                    seed=seed,
                    negative_rank=0,
                    travel_time_source=travel_time_source,
                    generation_timestamp=generation_timestamp,
                )
            )
            summary[split]["positive_samples_written"] += 1
        negatives = [stop for stop in sequence[position + 2:] if stop != actual_next_stop]
        sampled = rng.sample(negatives, min(negatives_per_positive, len(negatives))) if negatives else []
        for rank, candidate_stop in enumerate(sampled, start=1):
            negative_time = get_travel_time(matrix, current_stop, candidate_stop)
            if negative_time is None:
                summary[split]["skipped_negative_missing_travel_time"] += 1
                continue
            writer.write(
                make_output_row(
                    route_id=route_id,
                    split=split,
                    position=position,
                    current_stop=current_stop,
                    candidate_stop=candidate_stop,
                    actual_next_stop=actual_next_stop,
                    label=0,
                    travel_time=negative_time,
                    route_features=route_features,
                    stop_features=stop_features,
                    package_features=package_features,
                    sequence_length=len(sequence),
                    seed=seed,
                    negative_rank=rank,
                    travel_time_source=travel_time_source,
                    generation_timestamp=generation_timestamp,
                )
            )
            summary[split]["negative_samples_written"] += 1


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def resolve_all_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths_cfg = config.get("paths", {})
    outputs_cfg = config.get("outputs", {})
    data_root = resolve_path_entry(
        paths_cfg.get("data_root", "/content/drive/MyDrive/dissertation/amazon_last_mile"), required=False
    )
    processed_dir = resolve_path_entry(
        paths_cfg.get("processed_dir", "/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs"),
        required=False,
    )
    default_data_root = data_root or expand_path("/content/drive/MyDrive/dissertation/amazon_last_mile")
    result = {
        "route_splits": resolve_path_entry(paths_cfg.get("route_splits"), label="route_splits"),
        "train_route_ids": resolve_path_entry(paths_cfg.get("train_route_ids"), label="train_route_ids"),
        "validation_route_ids": resolve_path_entry(paths_cfg.get("validation_route_ids"), label="validation_route_ids"),
        "test_route_ids": resolve_path_entry(paths_cfg.get("test_route_ids"), label="test_route_ids"),
        "routes_summary": resolve_path_entry(
            paths_cfg.get("routes_summary", (processed_dir / "routes_summary.csv") if processed_dir else None),
            label="routes_summary",
        ),
        "stops_base_features": resolve_path_entry(paths_cfg.get("stops_base_features"), label="stops_base_features"),
        "stop_package_features": resolve_path_entry(paths_cfg.get("stop_package_features"), label="stop_package_features"),
        "transitions": resolve_path_entry(paths_cfg.get("actual_transitions_with_travel_time"), label="actual transitions"),
        "source_lookup": resolve_path_entry(paths_cfg.get("route_travel_time_source_lookup"), label="route_travel_time_source_lookup"),
        "pairwise_full_dir": expand_path(
            outputs_cfg.get(
                "pairwise_full_dir",
                "/content/drive/MyDrive/dissertation/amazon_last_mile/final_experiment_outputs/pairwise_samples_full",
            )
        ),
    }

    matrix_cfg = paths_cfg.get("travel_time_matrices", {})
    result["matrix_training_build"] = resolve_path_entry(
        matrix_cfg.get("training_build", default_data_root / "almrrc2021-data-training/model_build_inputs/travel_times.json"),
        label="training_build travel-time matrix",
    )
    result["matrix_training_apply"] = resolve_path_entry(
        matrix_cfg.get("training_apply", default_data_root / "almrrc2021-data-training/model_apply_inputs/new_travel_times.json"),
        label="training_apply travel-time matrix",
    )
    result["matrix_evaluation_apply"] = resolve_path_entry(
        matrix_cfg.get("evaluation_apply", default_data_root / "almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json"),
        label="evaluation_apply travel-time matrix",
    )
    return result


def check_route_overlap(split_routes: dict[str, list[str]]) -> None:
    splits = list(split_routes)
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            overlap = set(split_routes.get(left, [])) & set(split_routes.get(right, []))
            if overlap:
                raise ValueError(f"Route IDs overlap between {left} and {right}: {len(overlap)} routes")


def write_quality_report(output_dir: Path, split_paths: dict[str, Path], summary_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []

    def add(check_name: str, status: str, value: Any, details: str = "") -> None:
        rows.append({"check_name": check_name, "status": status, "value": value, "details": details})

    add("output_files_exist", "PASS" if all(path.exists() for path in split_paths.values()) else "FAIL", len(split_paths))
    for row in summary_rows:
        split = row["split"]
        add(f"{split}_positive_samples", "PASS" if int(row["positive_samples_written"]) > 0 else "FAIL", row["positive_samples_written"])
        add(f"{split}_negative_samples", "PASS" if int(row["negative_samples_written"]) > 0 else "FAIL", row["negative_samples_written"])
        path = split_paths[split]
        add(f"{split}_output_exists", "PASS" if path.exists() else "FAIL", path)
        if path.exists():
            add(f"{split}_output_size_mb", "INFO", round(path.stat().st_size / (1024 * 1024), 3))

    add("label_values_expected", "INFO", "0/1", "Full label scan is deferred to downstream model validation.")
    write_csv(output_dir / "pairwise_quality_report.csv", rows, ["check_name", "status", "value", "details"])


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = resolve_all_paths(config)

    pairwise_cfg = config.get("pairwise", {})
    seed = args.seed if args.seed is not None else int(pairwise_cfg.get("default_seed", 42))
    negatives_per_positive = (
        args.negative_samples_per_positive
        if args.negative_samples_per_positive is not None
        else int(pairwise_cfg.get("negative_samples_per_positive", 5))
    )
    if negatives_per_positive < 0:
        raise ValueError("--negative-samples-per-positive must be non-negative.")

    output_format = args.output_format or str(pairwise_cfg.get("output_format", "parquet"))
    compression = args.compression or str(pairwise_cfg.get("parquet_compression", "snappy"))
    output_dir = args.output_dir or paths["pairwise_full_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    split_file_map = {
        "train": paths["train_route_ids"],
        "validation": paths["validation_route_ids"],
        "test": paths["test_route_ids"],
    }
    split_routes = {split: load_route_ids(split_file_map[split], args.max_routes_per_split) for split in args.splits}
    check_route_overlap(split_routes)

    route_to_split = {route_id: split for split, ids in split_routes.items() for route_id in ids}
    requested_route_ids = set(route_to_split)

    if args.verbose:
        for split, ids in split_routes.items():
            print(f"{split}: requested {len(ids)} routes")

    route_features = load_route_features(paths["routes_summary"])
    stop_features = load_stop_features(paths["stops_base_features"], requested_route_ids)
    package_features = load_package_features(paths["stop_package_features"], requested_route_ids)
    transitions = load_transitions(paths["transitions"], requested_route_ids)
    source_lookup = load_source_lookup(paths["source_lookup"], requested_route_ids)

    summary = {
        split: {
            "routes_requested": len(ids),
            "routes_processed": 0,
            "routes_skipped_missing_source": 0,
            "routes_skipped_missing_transitions": 0,
            "positive_samples_written": 0,
            "negative_samples_written": 0,
            "skipped_positive_missing_travel_time": 0,
            "skipped_negative_missing_travel_time": 0,
        }
        for split, ids in split_routes.items()
    }

    generation_timestamp = datetime.now(timezone.utc).isoformat()
    suffix = "parquet" if output_format == "parquet" else "csv"
    split_paths = {split: output_dir / f"{split}_pairwise_samples.{suffix}" for split in args.splits}
    writers = {split: SplitWriter(split_paths[split], output_format=output_format, compression=compression) for split in args.splits}

    rng = random.Random(seed)
    by_source: dict[str, set[str]] = defaultdict(set)
    for route_id, split in route_to_split.items():
        source = source_lookup.get(route_id)
        if source is None:
            summary[split]["routes_skipped_missing_source"] += 1
        else:
            by_source[source].add(route_id)

    source_paths = {
        "training_build": paths["matrix_training_build"],
        "training_apply": paths["matrix_training_apply"],
        "evaluation_apply": paths["matrix_evaluation_apply"],
    }

    try:
        for source in SOURCE_PRIORITY:
            needed = by_source.get(source, set())
            if not needed:
                continue
            if args.verbose:
                print(f"Streaming {len(needed)} routes from {source}: {source_paths[source]}")
            for route_id, matrix in stream_top_level_object(source_paths[source]):
                if route_id not in needed:
                    continue
                split = route_to_split[route_id]
                process_route(
                    route_id=route_id,
                    split=split,
                    matrix=matrix,
                    writer=writers[split],
                    transitions=transitions,
                    route_features_all=route_features,
                    stop_features=stop_features,
                    package_features=package_features,
                    rng=rng,
                    seed=seed,
                    negatives_per_positive=negatives_per_positive,
                    summary=summary,
                    travel_time_source=source,
                    generation_timestamp=generation_timestamp,
                )
    finally:
        for writer in writers.values():
            writer.close()

    summary_rows = [make_summary_row(seed, output_format, split, summary[split], negatives_per_positive) for split in args.splits]
    write_csv(output_dir / "pairwise_sample_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_quality_report(output_dir, split_paths, summary_rows)

    print("Full pairwise sample generation complete.")
    for row in summary_rows:
        split = row["split"]
        print(
            f"{split}: routes_processed={row['routes_processed']}, "
            f"positives={row['positive_samples_written']}, "
            f"negatives={row['negative_samples_written']}, "
            f"total={row['total_samples_written']}"
        )
        print(f"Output: {split_paths[split]}")
    print(f"Summary: {output_dir / 'pairwise_sample_summary.csv'}")
    print(f"Quality report: {output_dir / 'pairwise_quality_report.csv'}")


if __name__ == "__main__":
    main()
