#!/usr/bin/env python3
"""Create pairwise next-stop preference samples for Amazon Last Mile routes."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from last_mile_cleaning.clean_pipeline import is_missing, stream_top_level_object

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_ROOT / "processed_outputs"
DEFAULT_FINAL_CLEANED_DIR = DEFAULT_PROCESSED_DIR / "final_cleaned"
DEFAULT_TRAVEL_TIME_OUTPUT_DIR = DEFAULT_PROCESSED_DIR / "travel_time_multisource_outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_FINAL_CLEANED_DIR / "pairwise_samples"
VALID_SPLITS = ("train", "validation", "test")
SOURCE_PRIORITY = ("training_build", "training_apply", "evaluation_apply")
UNKNOWN_ZONE = "UNKNOWN_ZONE"

OUTPUT_COLUMNS = [
    "route_id", "split", "position", "current_stop", "candidate_stop", "actual_next_stop", "label",
    "route_score", "number_of_stops", "route_progress", "remaining_stop_count",
    "current_zone", "current_type", "current_is_station", "current_is_dropoff",
    "candidate_zone", "candidate_type", "candidate_is_station", "candidate_is_dropoff",
    "travel_time_ij", "same_zone", "zone_changed", "zone_missing_in_pair",
    "candidate_package_count", "candidate_total_planned_service_time", "candidate_has_time_window",
    "candidate_time_window_package_count", "candidate_total_package_volume_cm3",
    "candidate_delivered_count", "candidate_attempted_count", "candidate_rejected_count",
    "candidate_unknown_status_count", "negative_sampling_seed", "negative_sample_rank",
]

SUMMARY_COLUMNS = [
    "split", "routes_requested", "routes_processed", "routes_skipped_missing_source",
    "routes_skipped_missing_transitions", "positive_samples_written", "negative_samples_written",
    "total_samples_written", "skipped_positive_missing_travel_time",
    "skipped_negative_missing_travel_time", "average_negatives_per_positive",
    "negative_samples_per_positive_requested", "seed",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create pairwise positive/negative next-stop samples.")
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--final-cleaned-dir", type=Path, default=DEFAULT_FINAL_CLEANED_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--travel-time-output-dir", type=Path, default=DEFAULT_TRAVEL_TIME_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--negative-samples-per-positive", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-routes-per-split", type=int, default=None)
    parser.add_argument("--splits", nargs="+", default=list(VALID_SPLITS), choices=VALID_SPLITS)
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def first_present(row: dict[str, Any], names: tuple[str, ...], default: Any = "") -> Any:
    for name in names:
        if name in row and not is_missing(row[name]):
            return row[name]
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


def to_float_or_text(value: Any, default: Any = 0) -> Any:
    if is_missing(value):
        return default
    return value


def normalize_zone(value: Any) -> str:
    if is_missing(value):
        return UNKNOWN_ZONE
    text = str(value).strip()
    if text == "" or text.upper() in {"UNKNOWN", "UNKNOWN_ZONE", "NAN", "NONE", "NULL"}:
        return UNKNOWN_ZONE
    return text


def load_route_ids(path: Path, max_routes: int | None = None) -> list[str]:
    rows = read_csv_rows(path)
    if not rows:
        return []
    field = "route_id" if "route_id" in rows[0] else next(iter(rows[0]))
    route_ids = [str(row[field]).strip() for row in rows if not is_missing(row.get(field))]
    route_ids = list(dict.fromkeys(route_ids))
    return route_ids[:max_routes] if max_routes is not None else route_ids


def validate_inputs(args: argparse.Namespace) -> dict[str, Path]:
    paths = {
        "route_splits": require_file(args.final_cleaned_dir / "route_splits.csv"),
        "train_ids": require_file(args.final_cleaned_dir / "train_route_ids.csv"),
        "validation_ids": require_file(args.final_cleaned_dir / "validation_route_ids.csv"),
        "test_ids": require_file(args.final_cleaned_dir / "test_route_ids.csv"),
        "transitions": require_file(args.final_cleaned_dir / "actual_transition_travel_time_complete_routes.csv"),
        "routes": require_file(args.processed_dir / "routes_summary.csv"),
        "stops": require_file(args.processed_dir / "stops_base_features.csv"),
        "packages": require_file(args.processed_dir / "stop_package_features.csv"),
        "source_lookup": require_file(args.travel_time_output_dir / "route_travel_time_source_lookup.csv"),
    }
    split_rows = read_csv_rows(paths["route_splits"])
    seen: dict[str, str] = {}
    for row in split_rows:
        route_id = str(first_present(row, ("route_id",))).strip()
        split = str(first_present(row, ("split", "split_label", "set"))).strip()
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split label in route_splits.csv: {split}")
        if route_id in seen:
            raise ValueError(f"Route {route_id} appears more than once in route_splits.csv.")
        seen[route_id] = split
    split_sets = {
        split: set(load_route_ids(paths[f"{split if split != 'validation' else 'validation'}_ids"]))
        for split in VALID_SPLITS
    }
    for left in VALID_SPLITS:
        for right in VALID_SPLITS:
            if left < right and split_sets[left] & split_sets[right]:
                raise ValueError(f"Route IDs overlap between {left} and {right}.")
    return paths


def load_route_features(path: Path) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {}
    for row in read_csv_rows(path):
        route_id = str(first_present(row, ("route_id",))).strip()
        if route_id:
            features[route_id] = {
                "route_score": first_present(row, ("route_score",), "Missing"),
                "number_of_stops": to_int(first_present(row, ("number_of_stops", "num_stops", "stop_count"), 0)),
            }
    return features


def load_stop_features(path: Path, route_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    features: dict[tuple[str, str], dict[str, Any]] = {}
    for row in read_csv_rows(path):
        route_id = str(first_present(row, ("route_id",))).strip()
        if route_id not in route_ids:
            continue
        stop_id = str(first_present(row, ("stop_id",))).strip()
        features[(route_id, stop_id)] = {
            "zone": normalize_zone(first_present(row, ("zone_id", "zone", "zone_label"), UNKNOWN_ZONE)),
            "type": first_present(row, ("type", "stop_type"), ""),
            "is_station": to_int(first_present(row, ("is_station",), 0)),
            "is_dropoff": to_int(first_present(row, ("is_dropoff",), 0)),
        }
    return features


def load_package_features(path: Path, route_ids: set[str]) -> dict[tuple[str, str], dict[str, Any]]:
    features: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
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
    for row in read_csv_rows(path):
        route_id = str(first_present(row, ("route_id",))).strip()
        if route_id not in route_ids:
            continue
        stop_id = str(first_present(row, ("stop_id",))).strip()
        features[(route_id, stop_id)] = {name: to_float_or_text(first_present(row, cols, 0), 0) for name, cols in aliases.items()}
        features[(route_id, stop_id)]["has_time_window"] = to_int(features[(route_id, stop_id)]["has_time_window"])
    return features


def load_transitions(path: Path, route_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            route_id = str(first_present(row, ("route_id",))).strip()
            if route_id in route_ids:
                grouped[route_id].append(row)
    return grouped


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


def get_travel_time(matrix: Any, from_stop: str, to_stop: str) -> Any:
    if not isinstance(matrix, dict):
        return None
    row = matrix.get(from_stop)
    if isinstance(row, dict):
        value = row.get(to_stop)
        return None if is_missing(value) else value
    return None


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


def pair_flags(current_zone: str, candidate_zone: str) -> tuple[int, int, int]:
    missing = current_zone == UNKNOWN_ZONE or candidate_zone == UNKNOWN_ZONE
    same = (not missing) and current_zone == candidate_zone
    changed = (not missing) and current_zone != candidate_zone
    return int(same), int(changed), int(missing)


def make_row(route_id: str, split: str, position: int, current_stop: str, candidate_stop: str,
             actual_next_stop: str, label: int, travel_time: Any, route_features: dict[str, Any],
             stop_features: dict[tuple[str, str], dict[str, Any]], package_features: dict[tuple[str, str], dict[str, Any]],
             sequence_length: int, seed: int, negative_rank: int) -> dict[str, Any]:
    current = stop_features.get((route_id, current_stop), {})
    candidate = stop_features.get((route_id, candidate_stop), {})
    current_zone = normalize_zone(current.get("zone", UNKNOWN_ZONE))
    candidate_zone = normalize_zone(candidate.get("zone", UNKNOWN_ZONE))
    same_zone, zone_changed, zone_missing = pair_flags(current_zone, candidate_zone)
    packages = package_features.get((route_id, candidate_stop), {})
    denominator = sequence_length - 1
    return {
        "route_id": route_id, "split": split, "position": position, "current_stop": current_stop,
        "candidate_stop": candidate_stop, "actual_next_stop": actual_next_stop, "label": label,
        "route_score": route_features.get("route_score", "Missing"),
        "number_of_stops": route_features.get("number_of_stops") or sequence_length,
        "route_progress": 0 if denominator <= 0 else position / denominator,
        "remaining_stop_count": sequence_length - position - 1,
        "current_zone": current_zone, "current_type": current.get("type", ""),
        "current_is_station": to_int(current.get("is_station", 0)), "current_is_dropoff": to_int(current.get("is_dropoff", 0)),
        "candidate_zone": candidate_zone, "candidate_type": candidate.get("type", ""),
        "candidate_is_station": to_int(candidate.get("is_station", 0)), "candidate_is_dropoff": to_int(candidate.get("is_dropoff", 0)),
        "travel_time_ij": travel_time, "same_zone": same_zone, "zone_changed": zone_changed,
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
        "negative_sampling_seed": seed, "negative_sample_rank": negative_rank,
    }


def process_route(route_id: str, split: str, matrix: Any, writer: csv.DictWriter,
                  transitions: dict[str, list[dict[str, Any]]], route_features_all: dict[str, dict[str, Any]],
                  stop_features: dict[tuple[str, str], dict[str, Any]], package_features: dict[tuple[str, str], dict[str, Any]],
                  rng: random.Random, args: argparse.Namespace, summary: dict[str, dict[str, Any]]) -> None:
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
            writer.writerow(make_row(route_id, split, position, current_stop, actual_next_stop, actual_next_stop, 1,
                                     positive_time, route_features, stop_features, package_features, len(sequence), args.seed, 0))
            summary[split]["positive_samples_written"] += 1
        negatives = [stop for stop in sequence[position + 2:] if stop != actual_next_stop]
        sampled = rng.sample(negatives, min(args.negative_samples_per_positive, len(negatives))) if negatives else []
        for rank, candidate_stop in enumerate(sampled, start=1):
            negative_time = get_travel_time(matrix, current_stop, candidate_stop)
            if negative_time is None:
                summary[split]["skipped_negative_missing_travel_time"] += 1
                continue
            writer.writerow(make_row(route_id, split, position, current_stop, candidate_stop, actual_next_stop, 0,
                                     negative_time, route_features, stop_features, package_features, len(sequence), args.seed, rank))
            summary[split]["negative_samples_written"] += 1


def main() -> None:
    args = parse_args()
    if args.negative_samples_per_positive < 0:
        raise ValueError("--negative-samples-per-positive must be non-negative.")
    paths = validate_inputs(args)
    split_routes = {split: load_route_ids(paths[f"{split if split != 'validation' else 'validation'}_ids"], args.max_routes_per_split) for split in args.splits}
    route_to_split = {route_id: split for split, ids in split_routes.items() for route_id in ids}
    requested_route_ids = set(route_to_split)
    route_features = load_route_features(paths["routes"])
    stop_features = load_stop_features(paths["stops"], requested_route_ids)
    package_features = load_package_features(paths["packages"], requested_route_ids)
    transitions = load_transitions(paths["transitions"], requested_route_ids)
    source_lookup = load_source_lookup(paths["source_lookup"], requested_route_ids)

    summary = {split: {column: 0 for column in SUMMARY_COLUMNS if column not in {"split", "average_negatives_per_positive", "negative_samples_per_positive_requested", "seed"}} for split in args.splits}
    for split, ids in split_routes.items():
        summary[split]["routes_requested"] = len(ids)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    files = {}
    for split in args.splits:
        file_obj = (args.output_dir / f"{split}_pairwise_samples.csv").open("w", encoding="utf-8", newline="")
        files[split] = file_obj
        writers[split] = csv.DictWriter(file_obj, fieldnames=OUTPUT_COLUMNS)
        writers[split].writeheader()

    rng = random.Random(args.seed)
    by_source: dict[str, set[str]] = defaultdict(set)
    for route_id, split in route_to_split.items():
        source = source_lookup.get(route_id)
        if source is None:
            summary[split]["routes_skipped_missing_source"] += 1
        else:
            by_source[source].add(route_id)

    source_paths = {
        "training_build": args.data_root / "almrrc2021-data-training/model_build_inputs/travel_times.json",
        "training_apply": args.data_root / "almrrc2021-data-training/model_apply_inputs/new_travel_times.json",
        "evaluation_apply": args.data_root / "almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json",
    }
    try:
        for source in SOURCE_PRIORITY:
            needed = by_source.get(source, set())
            if not needed:
                continue
            for route_id, matrix in stream_top_level_object(require_file(source_paths[source])):
                if route_id not in needed:
                    continue
                split = route_to_split[route_id]
                process_route(route_id, split, matrix, writers[split], transitions, route_features,
                              stop_features, package_features, rng, args, summary)
    finally:
        for file_obj in files.values():
            file_obj.close()

    summary_path = args.output_dir / "pairwise_sample_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for split in args.splits:
            row = {"split": split, **summary[split]}
            row["total_samples_written"] = row["positive_samples_written"] + row["negative_samples_written"]
            positives = row["positive_samples_written"]
            row["average_negatives_per_positive"] = 0 if positives == 0 else row["negative_samples_written"] / positives
            row["negative_samples_per_positive_requested"] = args.negative_samples_per_positive
            row["seed"] = args.seed
            writer.writerow(row)

    print("Pairwise sample generation complete.")
    for split in args.splits:
        row = summary[split]
        total = row["positive_samples_written"] + row["negative_samples_written"]
        print(f"{split}: routes_processed={row['routes_processed']}, positives={row['positive_samples_written']}, negatives={row['negative_samples_written']}, total={total}, skipped_positive_missing_travel_time={row['skipped_positive_missing_travel_time']}, skipped_negative_missing_travel_time={row['skipped_negative_missing_travel_time']}")
        print(f"Output: {args.output_dir / f'{split}_pairwise_samples.csv'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
