#!/usr/bin/env python3
"""Memory-safe multi-source travel-time verification for actual transitions.

This standalone script checks actual transition travel times across all available
Amazon Last Mile travel-time JSON sources. It streams each large JSON file
route-by-route and never uses json.load() on a complete travel-time file.

Inputs:
- processed_outputs/actual_transitions.csv
- raw travel-time JSON files from training build, training apply, and evaluation apply

Outputs:
- route_travel_time_source_lookup.csv
- actual_transition_travel_time_check_multisource.csv
- route_travel_time_completeness_multisource.csv
- travel_time_integrity_summary_multisource.csv
- travel_time_source_transition_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from last_mile_cleaning.clean_pipeline import is_missing, stream_top_level_object


DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_INPUT_DIR = DEFAULT_DATA_ROOT / "processed_outputs"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "travel_time_multisource_outputs"
PROGRESS_EVERY = 500

TRAVEL_TIME_SOURCES = [
    (
        "training_build",
        Path("almrrc2021-data-training/model_build_inputs/travel_times.json"),
    ),
    (
        "training_apply",
        Path("almrrc2021-data-training/model_apply_inputs/new_travel_times.json"),
    ),
    (
        "evaluation_apply",
        Path("almrrc2021-data-evaluation/model_apply_inputs/eval_travel_times.json"),
    ),
]

SOURCE_PRIORITY = [label for label, _path in TRAVEL_TIME_SOURCES]
ALL_SOURCE_LABELS = SOURCE_PRIORITY + ["multiple_sources", "missing_source"]


# ---------------------------------------------------------------------------
# CLI and generic helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Check actual-transition travel times across all available "
            "Amazon Last Mile travel-time JSON sources."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing actual_transitions.csv. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Amazon Last Mile raw dataset root. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for multi-source travel-time outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Optional number of route IDs to check for quick testing.",
    )
    return parser.parse_args()


def safe_is_missing(value: Any) -> bool:
    """Safely check whether a value should be treated as missing."""

    try:
        return bool(is_missing(value))
    except Exception:
        if value is None:
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False


def csv_value(value: Any) -> Any:
    """Return a CSV-safe value."""

    return "" if safe_is_missing(value) else value


def load_actual_transitions(input_dir: Path, max_routes: int | None) -> pd.DataFrame:
    """Load actual transition rows and optionally limit to the first N routes."""

    path = input_dir / "actual_transitions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")

    transitions = pd.read_csv(path)
    required_columns = {"route_id", "from_stop", "to_stop", "position"}
    missing_columns = required_columns - set(transitions.columns)
    if missing_columns:
        raise ValueError(
            f"actual_transitions.csv is missing required columns: {sorted(missing_columns)}"
        )

    transitions["route_id"] = transitions["route_id"].astype(str)
    transitions["from_stop"] = transitions["from_stop"].astype(str)
    transitions["to_stop"] = transitions["to_stop"].astype(str)

    if max_routes is not None:
        route_ids = transitions["route_id"].drop_duplicates().head(max_routes)
        transitions = transitions[transitions["route_id"].isin(route_ids)].copy()

    return transitions


def grouped_transitions(transitions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Group transition rows by route_id."""

    return {
        route_id: group.copy()
        for route_id, group in transitions.groupby("route_id", sort=False)
    }


def stream_source_routes(
    data_root: Path, source_label: str, relative_path: Path
) -> Iterator[tuple[str, Any]]:
    """Stream one travel-time source route-by-route."""

    path = data_root / relative_path
    if not path.exists():
        print(f"WARNING: travel-time source not found for {source_label}: {path}")
        return

    scanned = 0
    print(f"Streaming {source_label}: {path}")

    for route_id, matrix in stream_top_level_object(path):
        scanned += 1
        if scanned % PROGRESS_EVERY == 0:
            print(f"  {source_label}: streamed {scanned} routes")
        yield str(route_id), matrix


# ---------------------------------------------------------------------------
# Source lookup pass
# ---------------------------------------------------------------------------


def discover_route_sources(data_root: Path, route_ids: set[str]) -> dict[str, list[str]]:
    """Find which travel-time files contain each target route_id."""

    found_sources: dict[str, list[str]] = {route_id: [] for route_id in route_ids}
    unresolved = set(route_ids)

    for source_label, relative_path in TRAVEL_TIME_SOURCES:
        matched_in_source = 0

        for route_id, _matrix in stream_source_routes(data_root, source_label, relative_path):
            if route_id in route_ids:
                found_sources[route_id].append(source_label)
                matched_in_source += 1
                unresolved.discard(route_id)

        print(f"  {source_label}: matched {matched_in_source} target routes")

    print(f"Routes not found in any source after lookup pass: {len(unresolved)}")
    return found_sources


def choose_source_for_processing(source_labels: list[str]) -> str | None:
    """Choose the source to process if a route appears in one or more sources."""

    if not source_labels:
        return None

    for label in SOURCE_PRIORITY:
        if label in source_labels:
            return label

    return source_labels[0]


def source_status(source_labels: list[str]) -> tuple[str, str]:
    """Return display source and source status."""

    if not source_labels:
        return "missing_source", "route_missing_from_all_travel_time_sources"

    if len(source_labels) > 1:
        return "multiple_sources", "duplicate_route_in_multiple_sources"

    return source_labels[0], "ok_single_source"


def write_source_lookup(
    found_sources: dict[str, list[str]], output_dir: Path
) -> pd.DataFrame:
    """Write route-level source lookup table."""

    rows: list[dict[str, Any]] = []

    for route_id in sorted(found_sources):
        labels = found_sources[route_id]
        travel_time_source, status = source_status(labels)

        rows.append(
            {
                "route_id": route_id,
                "found_in_training_build": "training_build" in labels,
                "found_in_training_apply": "training_apply" in labels,
                "found_in_evaluation_apply": "evaluation_apply" in labels,
                "travel_time_source": travel_time_source,
                "source_count": len(labels),
                "source_status": status,
            }
        )

    lookup = pd.DataFrame(rows)
    lookup.to_csv(output_dir / "route_travel_time_source_lookup.csv", index=False)
    return lookup


# ---------------------------------------------------------------------------
# Transition travel-time check pass
# ---------------------------------------------------------------------------


def extract_travel_time(matrix: Any, from_stop: str, to_stop: str) -> Any:
    """Extract travel time for one directed edge from a route travel-time matrix."""

    if not isinstance(matrix, dict):
        return None

    from_row = matrix.get(from_stop)
    if not isinstance(from_row, dict):
        return None

    return from_row.get(to_stop)


def write_transition_rows(
    writer: csv.DictWriter,
    route_id: str,
    group: pd.DataFrame,
    matrix: Any,
    output_source_label: str,
) -> dict[str, Any]:
    """Write per-transition travel-time rows for one route."""

    transition_count = 0
    missing_count = 0
    values: list[float] = []

    for _index, transition in group.iterrows():
        transition_count += 1

        travel_time = extract_travel_time(
            matrix,
            str(transition["from_stop"]),
            str(transition["to_stop"]),
        )
        missing = safe_is_missing(travel_time)

        if missing:
            missing_count += 1
        else:
            try:
                values.append(float(travel_time))
            except (TypeError, ValueError):
                missing = True
                missing_count += 1
                travel_time = None

        writer.writerow(
            {
                "route_id": route_id,
                "from_stop": transition["from_stop"],
                "to_stop": transition["to_stop"],
                "position": transition["position"],
                "travel_time_ij": csv_value(travel_time),
                "travel_time_missing": 1 if missing else 0,
                "travel_time_source": output_source_label,
            }
        )

    return {
        "route_id": route_id,
        "travel_time_source": output_source_label,
        "transition_count": transition_count,
        "transitions_with_travel_time": transition_count - missing_count,
        "transitions_missing_travel_time": missing_count,
        "missing_travel_time_ratio": missing_count / transition_count
        if transition_count
        else 0.0,
        "has_complete_actual_transition_travel_time": missing_count == 0,
        "travel_time_values": values,
    }


def write_missing_source_rows(
    writer: csv.DictWriter, route_id: str, group: pd.DataFrame
) -> dict[str, Any]:
    """Write rows for a route that is not present in any travel-time source."""

    for _index, transition in group.iterrows():
        writer.writerow(
            {
                "route_id": route_id,
                "from_stop": transition["from_stop"],
                "to_stop": transition["to_stop"],
                "position": transition["position"],
                "travel_time_ij": "",
                "travel_time_missing": 1,
                "travel_time_source": "missing_source",
            }
        )

    transition_count = len(group)

    return {
        "route_id": route_id,
        "travel_time_source": "missing_source",
        "transition_count": transition_count,
        "transitions_with_travel_time": 0,
        "transitions_missing_travel_time": transition_count,
        "missing_travel_time_ratio": 1.0 if transition_count else 0.0,
        "has_complete_actual_transition_travel_time": False,
        "travel_time_values": [],
    }


def check_transition_travel_times(
    data_root: Path,
    output_dir: Path,
    transitions_by_route: dict[str, pd.DataFrame],
    found_sources: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[float]]:
    """Write per-transition checks using the selected source for each route."""

    selected_source_by_route = {
        route_id: choose_source_for_processing(labels)
        for route_id, labels in found_sources.items()
    }

    routes_by_selected_source: dict[str, set[str]] = {
        label: set() for label in SOURCE_PRIORITY
    }

    for route_id, selected_source in selected_source_by_route.items():
        if selected_source is not None:
            routes_by_selected_source[selected_source].add(route_id)

    completed_routes: set[str] = set()
    completeness_rows: list[dict[str, Any]] = []
    all_values: list[float] = []

    detail_path = output_dir / "actual_transition_travel_time_check_multisource.csv"
    fieldnames = [
        "route_id",
        "from_stop",
        "to_stop",
        "position",
        "travel_time_ij",
        "travel_time_missing",
        "travel_time_source",
    ]

    with detail_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()

        for source_label, relative_path in TRAVEL_TIME_SOURCES:
            needed_routes = routes_by_selected_source[source_label] - completed_routes
            if not needed_routes:
                continue

            print(f"Checking transitions using source {source_label}: {len(needed_routes)} routes")

            for route_id, matrix in stream_source_routes(
                data_root, source_label, relative_path
            ):
                if route_id not in needed_routes:
                    continue

                labels = found_sources.get(route_id, [])
                output_source_label = (
                    "multiple_sources" if len(labels) > 1 else source_label
                )

                result = write_transition_rows(
                    writer=writer,
                    route_id=route_id,
                    group=transitions_by_route[route_id],
                    matrix=matrix,
                    output_source_label=output_source_label,
                )

                all_values.extend(result.pop("travel_time_values"))
                completeness_rows.append(result)
                completed_routes.add(route_id)

                if completed_routes.issuperset(routes_by_selected_source[source_label]):
                    break

        for route_id in sorted(set(transitions_by_route) - completed_routes):
            result = write_missing_source_rows(
                writer, route_id, transitions_by_route[route_id]
            )
            result.pop("travel_time_values")
            completeness_rows.append(result)

    completeness_columns = [
        "route_id",
        "travel_time_source",
        "transition_count",
        "transitions_with_travel_time",
        "transitions_missing_travel_time",
        "missing_travel_time_ratio",
        "has_complete_actual_transition_travel_time",
    ]

    completeness = pd.DataFrame(completeness_rows, columns=completeness_columns)
    completeness.to_csv(
        output_dir / "route_travel_time_completeness_multisource.csv",
        index=False,
    )

    return completeness, all_values


# ---------------------------------------------------------------------------
# Summary outputs
# ---------------------------------------------------------------------------


def write_integrity_summary(
    lookup: pd.DataFrame,
    completeness: pd.DataFrame,
    travel_time_values: list[float],
    output_dir: Path,
) -> pd.DataFrame:
    """Write a single-row travel-time integrity summary."""

    values = pd.Series(travel_time_values, dtype="float64")

    summary = {
        "total_routes_checked": len(lookup),
        "routes_found_in_training_build": int(lookup["found_in_training_build"].sum())
        if len(lookup)
        else 0,
        "routes_found_in_training_apply": int(lookup["found_in_training_apply"].sum())
        if len(lookup)
        else 0,
        "routes_found_in_evaluation_apply": int(lookup["found_in_evaluation_apply"].sum())
        if len(lookup)
        else 0,
        "routes_missing_from_all_sources": int(
            (lookup["travel_time_source"] == "missing_source").sum()
        )
        if len(lookup)
        else 0,
        "routes_in_multiple_sources": int(
            (lookup["travel_time_source"] == "multiple_sources").sum()
        )
        if len(lookup)
        else 0,
        "total_actual_transitions_checked": int(completeness["transition_count"].sum())
        if len(completeness)
        else 0,
        "transitions_with_travel_time": int(
            completeness["transitions_with_travel_time"].sum()
        )
        if len(completeness)
        else 0,
        "transitions_missing_travel_time": int(
            completeness["transitions_missing_travel_time"].sum()
        )
        if len(completeness)
        else 0,
        "routes_with_complete_actual_transition_travel_time": int(
            completeness["has_complete_actual_transition_travel_time"].sum()
        )
        if len(completeness)
        else 0,
        "routes_with_any_missing_travel_time": int(
            (completeness["transitions_missing_travel_time"] > 0).sum()
        )
        if len(completeness)
        else 0,
        "average_actual_transition_travel_time": float(values.mean())
        if len(values)
        else 0.0,
        "median_actual_transition_travel_time": float(values.median())
        if len(values)
        else 0.0,
        "p95_actual_transition_travel_time": float(values.quantile(0.95))
        if len(values)
        else 0.0,
        "max_actual_transition_travel_time": float(values.max())
        if len(values)
        else 0.0,
    }

    total_checked = summary["total_actual_transitions_checked"]
    missing = summary["transitions_missing_travel_time"]
    summary["missing_travel_time_ratio"] = (
        missing / total_checked if total_checked else 0.0
    )

    ordered_fields = [
        "total_routes_checked",
        "routes_found_in_training_build",
        "routes_found_in_training_apply",
        "routes_found_in_evaluation_apply",
        "routes_missing_from_all_sources",
        "routes_in_multiple_sources",
        "total_actual_transitions_checked",
        "transitions_with_travel_time",
        "transitions_missing_travel_time",
        "missing_travel_time_ratio",
        "routes_with_complete_actual_transition_travel_time",
        "routes_with_any_missing_travel_time",
        "average_actual_transition_travel_time",
        "median_actual_transition_travel_time",
        "p95_actual_transition_travel_time",
        "max_actual_transition_travel_time",
    ]

    summary_frame = pd.DataFrame([{field: summary[field] for field in ordered_fields}])
    summary_frame.to_csv(
        output_dir / "travel_time_integrity_summary_multisource.csv",
        index=False,
    )

    return summary_frame


def write_source_transition_summary(
    completeness: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    """Write source-level transition completeness summary."""

    if completeness.empty:
        summary = pd.DataFrame(
            columns=[
                "travel_time_source",
                "route_count",
                "transition_count",
                "transitions_with_travel_time",
                "transitions_missing_travel_time",
                "missing_travel_time_ratio",
            ]
        )
    else:
        summary = (
            completeness.groupby("travel_time_source", dropna=False)
            .agg(
                route_count=("route_id", "count"),
                transition_count=("transition_count", "sum"),
                transitions_with_travel_time=("transitions_with_travel_time", "sum"),
                transitions_missing_travel_time=(
                    "transitions_missing_travel_time",
                    "sum",
                ),
            )
            .reset_index()
        )
        summary["missing_travel_time_ratio"] = (
            summary["transitions_missing_travel_time"]
            / summary["transition_count"].replace(0, pd.NA)
        )
        summary["missing_travel_time_ratio"] = summary[
            "missing_travel_time_ratio"
        ].fillna(0.0)

    # Ensure all source labels are represented, even if route_count is zero.
    existing_labels = set(summary["travel_time_source"]) if len(summary) else set()
    missing_rows = []
    for label in ALL_SOURCE_LABELS:
        if label not in existing_labels:
            missing_rows.append(
                {
                    "travel_time_source": label,
                    "route_count": 0,
                    "transition_count": 0,
                    "transitions_with_travel_time": 0,
                    "transitions_missing_travel_time": 0,
                    "missing_travel_time_ratio": 0.0,
                }
            )

    if missing_rows:
        summary = pd.concat([summary, pd.DataFrame(missing_rows)], ignore_index=True)

    summary["travel_time_source"] = pd.Categorical(
        summary["travel_time_source"],
        categories=ALL_SOURCE_LABELS,
        ordered=True,
    )
    summary = summary.sort_values("travel_time_source").reset_index(drop=True)
    summary["travel_time_source"] = summary["travel_time_source"].astype(str)

    summary.to_csv(output_dir / "travel_time_source_transition_summary.csv", index=False)
    return summary


def print_final_summary(
    lookup: pd.DataFrame, completeness: pd.DataFrame, summary: pd.DataFrame
) -> None:
    """Print final console summary."""

    row = summary.iloc[0]

    print("\nMulti-source travel-time verification complete.")
    print(f"Total routes checked: {int(row['total_routes_checked'])}")
    print("Routes by travel time source:")

    for label in ALL_SOURCE_LABELS:
        count = int((lookup["travel_time_source"] == label).sum()) if len(lookup) else 0
        print(f"  {label}: {count}")

    print(f"Total transitions checked: {int(row['total_actual_transitions_checked'])}")
    print(
        f"Corrected travel time missing ratio: "
        f"{float(row['missing_travel_time_ratio']):.4f}"
    )
    print(
        f"Routes missing from all sources: "
        f"{int(row['routes_missing_from_all_sources'])}"
    )
    print(f"Routes in multiple sources: {int(row['routes_in_multiple_sources'])}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_check(
    input_dir: Path = DEFAULT_INPUT_DIR,
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_routes: int | None = None,
) -> None:
    """Run the full multi-source travel-time verification workflow."""

    output_dir.mkdir(parents=True, exist_ok=True)

    transitions = load_actual_transitions(input_dir, max_routes)
    transitions_by_route = grouped_transitions(transitions)
    route_ids = set(transitions_by_route)

    print(f"Loaded {len(transitions)} transitions across {len(route_ids)} routes from {input_dir}")
    print(f"Writing outputs to: {output_dir}")
    if max_routes is not None:
        print(f"Limited to first {max_routes} route IDs for this run")

    found_sources = discover_route_sources(data_root, route_ids)
    lookup = write_source_lookup(found_sources, output_dir)

    completeness, travel_time_values = check_transition_travel_times(
        data_root=data_root,
        output_dir=output_dir,
        transitions_by_route=transitions_by_route,
        found_sources=found_sources,
    )

    summary = write_integrity_summary(
        lookup=lookup,
        completeness=completeness,
        travel_time_values=travel_time_values,
        output_dir=output_dir,
    )

    write_source_transition_summary(completeness, output_dir)
    print_final_summary(lookup, completeness, summary)


def main() -> None:
    args = parse_args()
    run_check(
        input_dir=args.input_dir,
        data_root=args.data_root,
        output_dir=args.output_dir,
        max_routes=args.max_routes,
    )


if __name__ == "__main__":
    main()
