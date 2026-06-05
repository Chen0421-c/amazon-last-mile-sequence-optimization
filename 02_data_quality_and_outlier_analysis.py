#!/usr/bin/env python3
"""Second-round data quality and outlier analysis for dissertation data.

This script validates the first-round cleaned Amazon Last Mile CSV files before
future EDA/modeling work. It writes quality reports and plots only; it does not
train any machine learning model and does not modify raw JSON files.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from last_mile_cleaning.clean_pipeline import is_missing, stream_top_level_object

DATA_ROOT = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
PROCESSED_OUTPUTS = DATA_ROOT / "processed_outputs"
QUALITY_OUTPUTS = PROCESSED_OUTPUTS / "quality_outputs"
TRAVEL_TIMES_RELATIVE_PATH = Path("almrrc2021-data-training/model_build_inputs/travel_times.json")

ROUTE_OUTLIER_COLUMNS = ["number_of_stops", "number_of_dropoff_stops", "executor_capacity_cm3", "missing_zone_ratio"]
PACKAGE_OUTLIER_COLUMNS = [
    "package_count",
    "total_planned_service_time",
    "total_package_volume_cm3",
    "time_window_package_count",
    "delivered_count",
    "attempted_count",
    "rejected_count",
    "unknown_status_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run second-round data quality, outlier, and travel-time integrity checks."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_OUTPUTS,
        help=f"Directory containing first-round cleaned CSV files. Default: {PROCESSED_OUTPUTS}",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Raw dataset root used only for streaming travel_times.json. Default: {DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QUALITY_OUTPUTS,
        help=f"Directory for generated quality CSV/PNG outputs. Default: {QUALITY_OUTPUTS}",
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Optional number of routes to check in travel_times.json for quick Colab testing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# General utility helpers
# ---------------------------------------------------------------------------


def require_csv(input_dir: Path, filename: str) -> Path:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required cleaned input CSV not found: {path}")
    return path


def read_cleaned_inputs(input_dir: Path) -> dict[str, pd.DataFrame]:
    """Read the first-round cleaned CSV files used by all quality checks."""

    return {
        "routes": pd.read_csv(require_csv(input_dir, "routes_summary.csv")),
        "stops": pd.read_csv(require_csv(input_dir, "stops_base_features.csv")),
        "transitions": pd.read_csv(require_csv(input_dir, "actual_transitions.csv")),
        "packages": pd.read_csv(require_csv(input_dir, "stop_package_features.csv")),
        "quality": pd.read_csv(require_csv(input_dir, "data_quality_report.csv")),
        "missing": pd.read_csv(require_csv(input_dir, "missing_value_summary.csv")),
    }


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(path, index=False)


def missing_lookup(missing: pd.DataFrame, field: str) -> tuple[int, int, float]:
    match = missing[missing["field"].astype(str) == field]
    if match.empty:
        return 0, 0, 0.0
    row = match.iloc[0]
    return int(row.get("missing_count", 0)), int(row.get("total_count", 0)), float(row.get("missing_ratio", 0.0))


def csv_safe(value: Any) -> Any:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else value


# ---------------------------------------------------------------------------
# 1. Basic dataset consistency summary
# ---------------------------------------------------------------------------


def dataset_consistency_summary(data: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, Any]:
    """Check whether cleaned route, stop, package, transition, and quality CSVs align."""

    routes = data["routes"]
    stops = data["stops"]
    transitions = data["transitions"]
    packages = data["packages"]
    quality = data["quality"]

    route_ids = set(routes["route_id"].astype(str))
    quality_ids = set(quality["route_id"].astype(str))

    stop_keys = stops[["route_id", "stop_id"]].astype(str).drop_duplicates()
    package_keys = packages[["route_id", "stop_id"]].astype(str).drop_duplicates()
    stops_with_package = stop_keys.merge(package_keys, on=["route_id", "stop_id"], how="left", indicator=True)
    packages_with_stop = package_keys.merge(stop_keys, on=["route_id", "stop_id"], how="left", indicator=True)

    from_keys = stop_keys.rename(columns={"stop_id": "from_stop"})
    to_keys = stop_keys.rename(columns={"stop_id": "to_stop"})
    transition_from = transitions[["route_id", "from_stop"]].astype(str).merge(
        from_keys, on=["route_id", "from_stop"], how="left", indicator=True
    )
    transition_to = transitions[["route_id", "to_stop"]].astype(str).merge(
        to_keys, on=["route_id", "to_stop"], how="left", indicator=True
    )

    row = {
        "total_routes_in_routes_summary": len(routes),
        "total_routes_in_quality_report": len(quality),
        "total_stops": len(stops),
        "total_package_stop_rows": len(packages),
        "total_actual_transitions": len(transitions),
        "routes_missing_from_quality_report": len(route_ids - quality_ids),
        "stops_without_package_feature_count": int((stops_with_package["_merge"] == "left_only").sum()),
        "package_features_without_stop_count": int((packages_with_stop["_merge"] == "left_only").sum()),
        "transitions_with_missing_from_stop_count": int((transition_from["_merge"] == "left_only").sum()),
        "transitions_with_missing_to_stop_count": int((transition_to["_merge"] == "left_only").sum()),
    }
    write_single_row_csv(output_dir / "dataset_consistency_summary.csv", row)
    return row


# ---------------------------------------------------------------------------
# 2. Route validity and training usability analysis
# ---------------------------------------------------------------------------


def reason_not_usable(row: pd.Series) -> str:
    if not row["route_exists_in_actual_sequences_bool"]:
        return "missing_actual_sequence"
    if int(row.get("route_stop_count", 0)) != int(row.get("sequence_stop_count", 0)):
        return "stop_count_mismatch"
    if not row["sequence_matches_route_stops_bool"]:
        return "stop_id_mismatch"
    station_count = int(row.get("number_of_station_stops", 0))
    if station_count == 0:
        return "no_station"
    if station_count > 1:
        return "multiple_stations"
    return "other"


def route_validity_analysis(data: dict[str, pd.DataFrame], output_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    """Summarize first-round route usability and write details for unusable routes."""

    quality = data["quality"].copy()
    routes = data["routes"][["route_id", "route_score"]].copy()
    merged = quality.merge(routes, on="route_id", how="left")
    merged["route_exists_in_actual_sequences_bool"] = to_bool(merged["route_exists_in_actual_sequences"])
    merged["sequence_matches_route_stops_bool"] = to_bool(merged["sequence_matches_route_stops"])
    merged["has_single_station_bool"] = to_bool(merged["has_single_station"])
    merged["can_use_for_training_bool"] = to_bool(merged["can_use_for_training"])
    stop_count_matches = numeric(merged, "route_stop_count") == numeric(merged, "sequence_stop_count")

    summary = {
        "total_routes": len(merged),
        "routes_with_actual_sequence": int(merged["route_exists_in_actual_sequences_bool"].sum()),
        "routes_with_matching_stop_count": int(stop_count_matches.sum()),
        "routes_with_matching_stop_ids": int(merged["sequence_matches_route_stops_bool"].sum()),
        "routes_with_single_station": int(merged["has_single_station_bool"].sum()),
        "routes_can_use_for_training": int(merged["can_use_for_training_bool"].sum()),
        "routes_not_usable_for_training": int((~merged["can_use_for_training_bool"]).sum()),
        "usable_route_percentage": float(merged["can_use_for_training_bool"].mean() * 100) if len(merged) else 0.0,
    }
    write_single_row_csv(output_dir / "training_route_filter_summary.csv", summary)

    unusable = merged[~merged["can_use_for_training_bool"]].copy()
    unusable["reason_not_usable"] = unusable.apply(reason_not_usable, axis=1)
    detail_columns = [
        "route_id",
        "route_score",
        "route_stop_count",
        "sequence_stop_count",
        "number_of_station_stops",
        "has_single_station",
        "sequence_matches_route_stops",
        "reason_not_usable",
    ]
    unusable[detail_columns].to_csv(output_dir / "unusable_routes_detail.csv", index=False)
    return summary, merged


# ---------------------------------------------------------------------------
# 3-4. Outlier analyses
# ---------------------------------------------------------------------------


def compute_outlier_summary(frame: pd.DataFrame, columns: Iterable[str], include_zero_extreme: bool = False) -> pd.DataFrame:
    """Compute descriptive statistics plus IQR outlier counts for numeric columns."""

    rows: list[dict[str, Any]] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            row = {"feature": column, "count": 0}
            rows.append(row)
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = (values < lower) | (values > upper)
        row = {
            "feature": column,
            "count": int(values.count()),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "p01": float(values.quantile(0.01)),
            "p05": float(values.quantile(0.05)),
            "p25": float(q1),
            "p50": float(values.quantile(0.50)),
            "p75": float(q3),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "max": float(values.max()),
            "lower_bound": float(lower),
            "upper_bound": float(upper),
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": float(outliers.mean() * 100),
        }
        if include_zero_extreme:
            p99 = values.quantile(0.99)
            row["zero_value_count"] = int((values == 0).sum())
            row["zero_value_percentage"] = float((values == 0).mean() * 100)
            row["extreme_high_value_count"] = int((values > p99).sum())
            row["extreme_high_threshold_p99"] = float(p99)
        rows.append(row)
    return pd.DataFrame(rows)


def append_outlier_flags(frame: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    flag_lists = pd.Series([[] for _ in range(len(frame))], index=frame.index, dtype=object)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (values < lower) | (values > upper)
        flag_lists.loc[mask] = flag_lists.loc[mask].apply(lambda flags, name=column: flags + [name])
    return flag_lists.apply(lambda flags: ";".join(flags))


def route_level_outlier_analysis(routes: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Identify unusual route sizes, capacities, and missing-zone ratios."""

    routes = routes.copy()
    for column in ROUTE_OUTLIER_COLUMNS:
        routes[column] = pd.to_numeric(routes[column], errors="coerce")
    summary = compute_outlier_summary(routes, ROUTE_OUTLIER_COLUMNS)
    summary.to_csv(output_dir / "route_level_outlier_summary.csv", index=False)

    routes["outlier_flags"] = append_outlier_flags(routes, ROUTE_OUTLIER_COLUMNS)
    flagged = routes[routes["outlier_flags"] != ""].copy()
    flagged[
        ["route_id", "route_score", "number_of_stops", "executor_capacity_cm3", "missing_zone_ratio", "outlier_flags"]
    ].to_csv(output_dir / "route_level_outlier_routes.csv", index=False)
    return routes[["route_id", "outlier_flags", "number_of_stops", "missing_zone_ratio"]]


def stop_package_outlier_analysis(packages: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Identify extreme package/service feature values at stop level."""

    packages = packages.copy()
    for column in PACKAGE_OUTLIER_COLUMNS:
        packages[column] = pd.to_numeric(packages[column], errors="coerce")
    summary = compute_outlier_summary(packages, PACKAGE_OUTLIER_COLUMNS, include_zero_extreme=True)
    summary.to_csv(output_dir / "stop_package_outlier_summary.csv", index=False)

    packages["outlier_flags"] = append_outlier_flags(packages, PACKAGE_OUTLIER_COLUMNS)
    flagged = packages[packages["outlier_flags"] != ""].copy()
    flagged[
        [
            "route_id",
            "stop_id",
            "package_count",
            "total_planned_service_time",
            "total_package_volume_cm3",
            "time_window_package_count",
            "outlier_flags",
        ]
    ].to_csv(output_dir / "stop_package_outlier_stops.csv", index=False)
    return packages[["route_id", "stop_id", "outlier_flags"]]


# ---------------------------------------------------------------------------
# 5. Station and route sequence logic
# ---------------------------------------------------------------------------


def station_sequence_check(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """Check route start station and expected transition count logic."""

    routes = data["routes"].copy()
    quality = data["quality"][["route_id", "has_single_station"]].copy()
    transitions = data["transitions"].copy()
    transitions["position"] = pd.to_numeric(transitions["position"], errors="coerce")

    first_stops = (
        transitions.sort_values(["route_id", "position"])
        .groupby("route_id", as_index=False)
        .first()[["route_id", "from_stop"]]
        .rename(columns={"from_stop": "first_stop_in_actual_sequence"})
    )
    transition_counts = transitions.groupby("route_id").size().reset_index(name="transition_count")

    check = routes.merge(quality, on="route_id", how="left").merge(first_stops, on="route_id", how="left")
    check = check.merge(transition_counts, on="route_id", how="left")
    check["transition_count"] = check["transition_count"].fillna(0).astype(int)
    check["number_of_stops"] = pd.to_numeric(check["number_of_stops"], errors="coerce").fillna(0).astype(int)
    check["expected_transition_count"] = (check["number_of_stops"] - 1).clip(lower=0)
    check["station_is_first_stop"] = check["station_stop_id"].astype(str) == check["first_stop_in_actual_sequence"].astype(str)
    check["transition_count_matches_expected"] = check["transition_count"] == check["expected_transition_count"]
    columns = [
        "route_id",
        "station_stop_id",
        "number_of_station_stops",
        "has_single_station",
        "first_stop_in_actual_sequence",
        "station_is_first_stop",
        "transition_count",
        "expected_transition_count",
        "transition_count_matches_expected",
    ]
    check[columns].to_csv(output_dir / "station_sequence_check.csv", index=False)
    return check[columns]


# ---------------------------------------------------------------------------
# 6. Dropoff stops and package availability
# ---------------------------------------------------------------------------


def dropoff_package_check(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """Flag dropoff stops without packages and station stops that have packages."""

    stops = data["stops"].copy()
    packages = data["packages"][["route_id", "stop_id", "package_count"]].copy()
    packages["package_count"] = pd.to_numeric(packages["package_count"], errors="coerce").fillna(0)
    merged = stops.merge(packages, on=["route_id", "stop_id"], how="left")
    merged["package_count"] = merged["package_count"].fillna(0)
    merged["type_normalized"] = merged["type"].astype(str).str.lower()
    merged["is_dropoff"] = merged["type_normalized"] == "dropoff"
    merged["is_station"] = merged["type_normalized"] == "station"

    dropoff_detail = merged[merged["is_dropoff"] & (merged["package_count"] == 0)].copy()
    dropoff_detail[["route_id", "stop_id", "type", "package_count"]].to_csv(
        output_dir / "dropoff_zero_package_detail.csv", index=False
    )

    rows: list[dict[str, Any]] = []
    for route_id, group in merged.groupby("route_id", dropna=False):
        dropoffs = group[group["is_dropoff"]]
        stations = group[group["is_station"]]
        total_dropoffs = len(dropoffs)
        zero_dropoffs = int((dropoffs["package_count"] == 0).sum())
        station_stops_with_packages = int((stations["package_count"] > 0).sum())
        rows.append(
            {
                "route_id": route_id,
                "total_dropoff_stops": total_dropoffs,
                "dropoff_stops_with_zero_packages": zero_dropoffs,
                "dropoff_zero_package_ratio": zero_dropoffs / total_dropoffs if total_dropoffs else 0,
                "station_stops_with_packages": station_stops_with_packages,
                "station_package_count": float(stations["package_count"].sum()),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "dropoff_package_check.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# 7. Missing-value strategy recommendation
# ---------------------------------------------------------------------------


def feature_reliability_summary(data: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """Recommend defensible missing-value handling and feature reliability choices."""

    missing = data["missing"]
    stops = data["stops"]
    packages = data["packages"]

    zone_missing_count, zone_total, zone_ratio = missing_lookup(missing, "zone_id")
    time_window_missing, time_window_total, time_window_ratio = missing_lookup(missing, "package_time_window")
    service_missing, service_total, service_ratio = missing_lookup(missing, "planned_service_time")
    dimensions_missing, dimensions_total, dimensions_ratio = missing_lookup(missing, "package_dimensions")
    scan_missing, scan_total, scan_ratio = missing_lookup(missing, "scan_status")

    rows = [
        {
            "feature_name": "zone_id",
            "missing_count": zone_missing_count,
            "total_count": zone_total,
            "missing_ratio": zone_ratio,
            "proposed_handling": "Encode missing as UNKNOWN_ZONE and keep zone_missing flag.",
            "use_as_core_feature": zone_ratio <= 0.20,
            "explanation": "Zone is important spatial context; use as core only if missingness is not excessive.",
        },
        {
            "feature_name": "zone_missing",
            "missing_count": 0,
            "total_count": len(stops),
            "missing_ratio": 0.0,
            "proposed_handling": "Use binary flag directly.",
            "use_as_core_feature": True,
            "explanation": "The first-round cleaner creates this explicit indicator for missing zone IDs.",
        },
        {
            "feature_name": "package_time_window",
            "missing_count": time_window_missing,
            "total_count": time_window_total,
            "missing_ratio": time_window_ratio,
            "proposed_handling": "Treat missing time window as no specified time window.",
            "use_as_core_feature": False,
            "explanation": "Raw window timestamps are less suitable than the derived has_time_window flag.",
        },
        {
            "feature_name": "has_time_window",
            "missing_count": 0,
            "total_count": len(packages),
            "missing_ratio": 0.0,
            "proposed_handling": "Use binary flag directly.",
            "use_as_core_feature": True,
            "explanation": "No window means no specified constraint rather than a data error.",
        },
        {
            "feature_name": "planned_service_time",
            "missing_count": service_missing,
            "total_count": service_total,
            "missing_ratio": service_ratio,
            "proposed_handling": "Use total_planned_service_time; impute rare missing package-level values as zero or median.",
            "use_as_core_feature": service_ratio <= 0.01,
            "explanation": "Core workload feature if package-level missingness is near zero.",
        },
        {
            "feature_name": "package_dimensions",
            "missing_count": dimensions_missing,
            "total_count": dimensions_total,
            "missing_ratio": dimensions_ratio,
            "proposed_handling": "Use total_package_volume_cm3 with missing dimensions contributing zero volume; check outliers.",
            "use_as_core_feature": dimensions_ratio <= 0.05,
            "explanation": "Volume can be a useful capacity burden feature if dimension missingness is low.",
        },
        {
            "feature_name": "package_count",
            "missing_count": 0,
            "total_count": len(packages),
            "missing_ratio": 0.0,
            "proposed_handling": "Use numeric stop-level package count directly.",
            "use_as_core_feature": True,
            "explanation": "Package count is a reliable descriptive workload feature from stop-level aggregation.",
        },
        {
            "feature_name": "total_package_volume_cm3",
            "missing_count": dimensions_missing,
            "total_count": dimensions_total,
            "missing_ratio": dimensions_ratio,
            "proposed_handling": "Use as core or secondary workload feature after reviewing outliers.",
            "use_as_core_feature": dimensions_ratio <= 0.05,
            "explanation": "Derived from dimensions; reliability follows package dimension completeness.",
        },
        {
            "feature_name": "scan_status",
            "missing_count": scan_missing,
            "total_count": scan_total,
            "missing_ratio": scan_ratio,
            "proposed_handling": "Avoid raw scan_status as a core feature; aggregate missing/unknown statuses descriptively.",
            "use_as_core_feature": False,
            "explanation": "Scan status can leak outcome-like information and may have meaningful missingness.",
        },
        {
            "feature_name": "unknown_status_count",
            "missing_count": 0,
            "total_count": len(packages),
            "missing_ratio": 0.0,
            "proposed_handling": "Use only as an auxiliary descriptive feature.",
            "use_as_core_feature": False,
            "explanation": "Useful for data diagnostics but should not drive core route ordering decisions.",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "feature_reliability_summary.csv", index=False)
    return summary


# ---------------------------------------------------------------------------
# 8. Memory-safe travel time integrity check
# ---------------------------------------------------------------------------


def build_transition_route_map(transitions: pd.DataFrame, max_routes: int | None) -> dict[str, pd.DataFrame]:
    route_ids = transitions["route_id"].astype(str).drop_duplicates().tolist()
    if max_routes is not None:
        route_ids = route_ids[:max_routes]
    subset = transitions[transitions["route_id"].astype(str).isin(route_ids)].copy()
    subset["route_id"] = subset["route_id"].astype(str)
    return {route_id: group.copy() for route_id, group in subset.groupby("route_id", sort=False)}


def extract_travel_time(matrix: Any, from_stop: str, to_stop: str) -> Any:
    if not isinstance(matrix, dict):
        return None
    from_row = matrix.get(from_stop)
    if not isinstance(from_row, dict):
        return None
    return from_row.get(to_stop)


def travel_time_integrity_check(
    transitions: pd.DataFrame, data_root: Path, output_dir: Path, max_routes: int | None
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Stream raw travel_times.json route-by-route and check actual transition coverage."""

    route_transition_map = build_transition_route_map(transitions, max_routes)
    pending_routes = set(route_transition_map)
    travel_path = data_root / TRAVEL_TIMES_RELATIVE_PATH
    detail_path = output_dir / "actual_transition_travel_time_check.csv"
    fieldnames = ["route_id", "from_stop", "to_stop", "position", "travel_time_ij", "travel_time_missing"]
    route_rows: list[dict[str, Any]] = []
    all_travel_times: list[float] = []
    checked_routes: set[str] = set()

    with detail_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        if travel_path.exists():
            for route_id, matrix in stream_top_level_object(travel_path):
                if route_id not in pending_routes:
                    continue
                checked_routes.add(route_id)
                group = route_transition_map[route_id]
                missing_count = 0
                route_time_values: list[float] = []
                for _, transition in group.iterrows():
                    travel_time = extract_travel_time(matrix, str(transition["from_stop"]), str(transition["to_stop"]))
                    missing = is_missing(travel_time)
                    if missing:
                        missing_count += 1
                    else:
                        route_time_values.append(float(travel_time))
                        all_travel_times.append(float(travel_time))
                    writer.writerow(
                        {
                            "route_id": route_id,
                            "from_stop": transition["from_stop"],
                            "to_stop": transition["to_stop"],
                            "position": transition["position"],
                            "travel_time_ij": csv_safe(travel_time),
                            "travel_time_missing": 1 if missing else 0,
                        }
                    )
                route_rows.append(
                    {
                        "route_id": route_id,
                        "actual_transition_count_checked": len(group),
                        "missing_travel_time_count": missing_count,
                        "has_complete_actual_transition_travel_time": missing_count == 0,
                    }
                )
                pending_routes.remove(route_id)
                if not pending_routes:
                    break
        else:
            print(f"WARNING: travel_times.json not found at {travel_path}; marking checked transitions as missing.")

        # If route IDs from actual_transitions are not present in travel_times.json,
        # write their transitions as missing instead of silently dropping them.
        for route_id in sorted(pending_routes):
            group = route_transition_map[route_id]
            for _, transition in group.iterrows():
                writer.writerow(
                    {
                        "route_id": route_id,
                        "from_stop": transition["from_stop"],
                        "to_stop": transition["to_stop"],
                        "position": transition["position"],
                        "travel_time_ij": "",
                        "travel_time_missing": 1,
                    }
                )
            route_rows.append(
                {
                    "route_id": route_id,
                    "actual_transition_count_checked": len(group),
                    "missing_travel_time_count": len(group),
                    "has_complete_actual_transition_travel_time": False,
                }
            )

    route_completeness = pd.DataFrame(route_rows)
    total_checked = int(route_completeness["actual_transition_count_checked"].sum()) if len(route_completeness) else 0
    missing_total = int(route_completeness["missing_travel_time_count"].sum()) if len(route_completeness) else 0
    values = pd.Series(all_travel_times, dtype="float64")
    summary = {
        "total_actual_transitions_checked": total_checked,
        "transitions_with_travel_time": total_checked - missing_total,
        "transitions_missing_travel_time": missing_total,
        "missing_travel_time_ratio": missing_total / total_checked if total_checked else 0,
        "routes_with_any_missing_travel_time": int((route_completeness["missing_travel_time_count"] > 0).sum()) if len(route_completeness) else 0,
        "routes_with_complete_actual_transition_travel_time": int(route_completeness["has_complete_actual_transition_travel_time"].sum()) if len(route_completeness) else 0,
        "average_actual_transition_travel_time": float(values.mean()) if len(values) else 0,
        "median_actual_transition_travel_time": float(values.median()) if len(values) else 0,
        "p95_actual_transition_travel_time": float(values.quantile(0.95)) if len(values) else 0,
        "max_actual_transition_travel_time": float(values.max()) if len(values) else 0,
    }
    write_single_row_csv(output_dir / "travel_time_integrity_summary.csv", summary)
    return route_completeness, summary, pd.read_csv(detail_path)


# ---------------------------------------------------------------------------
# 9. Training-ready route recommendation
# ---------------------------------------------------------------------------


def training_ready_routes(
    data: dict[str, pd.DataFrame],
    route_flags: pd.DataFrame,
    station_check: pd.DataFrame,
    travel_completeness: pd.DataFrame,
    dropoff_check: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Recommend routes for later model training based on severe validity checks."""

    routes = data["routes"][["route_id", "route_score"]]
    quality = data["quality"][["route_id", "can_use_for_training", "has_single_station", "sequence_matches_route_stops"]].copy()
    quality["can_use_for_training_original"] = to_bool(quality["can_use_for_training"])
    quality["has_single_station"] = to_bool(quality["has_single_station"])
    quality["sequence_matches_route_stops"] = to_bool(quality["sequence_matches_route_stops"])

    recommendation = routes.merge(quality, on="route_id", how="left")
    recommendation = recommendation.merge(
        station_check[["route_id", "station_is_first_stop", "transition_count_matches_expected"]], on="route_id", how="left"
    )
    recommendation = recommendation.merge(
        travel_completeness[["route_id", "has_complete_actual_transition_travel_time"]], on="route_id", how="left"
    )
    recommendation = recommendation.merge(
        route_flags[["route_id", "outlier_flags", "number_of_stops", "missing_zone_ratio"]], on="route_id", how="left"
    )
    recommendation = recommendation.merge(
        dropoff_check[["route_id", "dropoff_stops_with_zero_packages", "station_stops_with_packages"]], on="route_id", how="left"
    )

    for column in [
        "can_use_for_training_original",
        "has_single_station",
        "sequence_matches_route_stops",
        "station_is_first_stop",
        "transition_count_matches_expected",
        "has_complete_actual_transition_travel_time",
    ]:
        recommendation[column] = to_bool(recommendation[column])

    recommendation["outlier_flags"] = recommendation["outlier_flags"].fillna("")
    recommendation["severe_route_outlier"] = (
        (pd.to_numeric(recommendation["number_of_stops"], errors="coerce").fillna(2) < 2)
        | (pd.to_numeric(recommendation["missing_zone_ratio"], errors="coerce").fillna(0) >= 1)
    )
    recommendation["severe_package_issue"] = pd.to_numeric(
        recommendation["dropoff_stops_with_zero_packages"], errors="coerce"
    ).fillna(0) > 0

    recommendation["recommended_for_training"] = (
        recommendation["can_use_for_training_original"]
        & recommendation["has_single_station"]
        & recommendation["sequence_matches_route_stops"]
        & recommendation["station_is_first_stop"]
        & recommendation["transition_count_matches_expected"]
        & recommendation["has_complete_actual_transition_travel_time"]
    )

    def exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not row["can_use_for_training_original"]:
            reasons.append("first_round_not_usable")
        if not row["has_single_station"]:
            reasons.append("station_count_issue")
        if not row["sequence_matches_route_stops"]:
            reasons.append("stop_id_mismatch")
        if not row["station_is_first_stop"]:
            reasons.append("station_not_first_stop")
        if not row["transition_count_matches_expected"]:
            reasons.append("transition_count_mismatch")
        if not row["has_complete_actual_transition_travel_time"]:
            reasons.append("missing_actual_transition_travel_time")
        if row["severe_route_outlier"]:
            reasons.append("severe_route_outlier_flag")
        if row["severe_package_issue"]:
            reasons.append("zero_package_dropoff_flag")
        return ";".join(reasons) if reasons else ""

    recommendation["exclusion_reason"] = recommendation.apply(exclusion_reason, axis=1)
    columns = [
        "route_id",
        "route_score",
        "can_use_for_training_original",
        "has_single_station",
        "sequence_matches_route_stops",
        "station_is_first_stop",
        "transition_count_matches_expected",
        "has_complete_actual_transition_travel_time",
        "severe_route_outlier",
        "severe_package_issue",
        "recommended_for_training",
        "exclusion_reason",
    ]
    recommendation[columns].to_csv(output_dir / "training_ready_routes.csv", index=False)
    return recommendation[columns]


# ---------------------------------------------------------------------------
# 10. Plots
# ---------------------------------------------------------------------------


def save_histogram(values: pd.Series, path: Path, title: str, xlabel: str, bins: int = 40) -> None:
    clean_values = pd.to_numeric(values, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clean_values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def create_plots(
    data: dict[str, pd.DataFrame], travel_detail: pd.DataFrame, training_ready: pd.DataFrame, output_dir: Path
) -> None:
    """Create simple matplotlib diagnostic plots under quality_outputs/plots/."""

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    routes = data["routes"]
    packages = data["packages"]

    score_counts = routes["route_score"].fillna("Missing").value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(score_counts.index.astype(str), score_counts.values)
    ax.set_title("Route Score Distribution")
    ax.set_xlabel("Route Score")
    ax.set_ylabel("Routes")
    fig.tight_layout()
    fig.savefig(plots_dir / "route_score_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(pd.to_numeric(routes["number_of_stops"], errors="coerce").dropna())
    ax.set_title("Number Of Stops Boxplot")
    ax.set_ylabel("Number Of Stops")
    fig.tight_layout()
    fig.savefig(plots_dir / "number_of_stops_boxplot.png", dpi=150)
    plt.close(fig)

    save_histogram(routes["number_of_stops"], plots_dir / "number_of_stops_histogram.png", "Number Of Stops", "Number Of Stops")
    save_histogram(
        routes["missing_zone_ratio"], plots_dir / "missing_zone_ratio_histogram.png", "Missing Zone Ratio", "Missing Zone Ratio"
    )
    save_histogram(packages["package_count"], plots_dir / "package_count_histogram.png", "Package Count", "Package Count")
    save_histogram(
        packages["total_planned_service_time"],
        plots_dir / "total_planned_service_time_histogram.png",
        "Total Planned Service Time",
        "Total Planned Service Time",
    )
    save_histogram(
        packages["total_package_volume_cm3"],
        plots_dir / "total_package_volume_histogram.png",
        "Total Package Volume",
        "Total Package Volume Cm3",
    )
    save_histogram(
        travel_detail["travel_time_ij"],
        plots_dir / "actual_transition_travel_time_histogram.png",
        "Actual Transition Travel Time",
        "Travel Time",
    )

    usability_counts = training_ready["recommended_for_training"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(usability_counts.index.astype(str), usability_counts.values)
    ax.set_title("Route Usability Recommendation")
    ax.set_xlabel("Recommended For Training")
    ax.set_ylabel("Routes")
    fig.tight_layout()
    fig.savefig(plots_dir / "route_usability_bar_chart.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 11-12. Orchestration and final console summary
# ---------------------------------------------------------------------------


def print_final_summary(
    data: dict[str, pd.DataFrame],
    route_filter: dict[str, Any],
    missing_features: pd.DataFrame,
    travel_summary: dict[str, Any],
    station_check_frame: pd.DataFrame,
    dropoff_check_frame: pd.DataFrame,
    training_ready_frame: pd.DataFrame,
) -> None:
    def feature_ratio(feature: str) -> float:
        match = missing_features[missing_features["feature_name"] == feature]
        return float(match["missing_ratio"].iloc[0]) if not match.empty else 0.0

    print("\nSecond-round data quality analysis complete.")
    print(f"Total routes: {len(data['routes'])}")
    print(f"Usable routes from first-round report: {route_filter['routes_can_use_for_training']}")
    print(f"Recommended training-ready routes: {int(to_bool(training_ready_frame['recommended_for_training']).sum())}")
    print(f"Total stops: {len(data['stops'])}")
    print(f"Total actual transitions: {len(data['transitions'])}")
    print(f"Zone missing ratio: {feature_ratio('zone_id'):.4f}")
    print(f"Planned service time missing ratio: {feature_ratio('planned_service_time'):.4f}")
    print(f"Package dimension missing ratio: {feature_ratio('package_dimensions'):.4f}")
    print(f"Scan status missing ratio: {feature_ratio('scan_status'):.4f}")
    print(f"Travel time missing ratio for actual transitions: {travel_summary['missing_travel_time_ratio']:.4f}")
    print(f"Routes where station is not first stop: {int((~to_bool(station_check_frame['station_is_first_stop'])).sum())}")
    print(
        "Routes with transition count mismatch: "
        f"{int((~to_bool(station_check_frame['transition_count_matches_expected'])).sum())}"
    )
    print(
        "Routes with zero-package dropoff stops: "
        f"{int((pd.to_numeric(dropoff_check_frame['dropoff_stops_with_zero_packages'], errors='coerce').fillna(0) > 0).sum())}"
    )


def run_analysis(
    input_dir: Path = PROCESSED_OUTPUTS,
    data_root: Path = DATA_ROOT,
    output_dir: Path = QUALITY_OUTPUTS,
    max_routes: int | None = None,
) -> None:
    input_dir = Path(input_dir)
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading first-round cleaned CSV files from: {input_dir}")
    print(f"Writing quality outputs to: {output_dir}")
    if max_routes is not None:
        print(f"Travel-time integrity check limited to first {max_routes} actual-transition routes.")

    data = read_cleaned_inputs(input_dir)
    dataset_consistency_summary(data, output_dir)
    route_filter_summary, _quality_with_routes = route_validity_analysis(data, output_dir)
    route_flags = route_level_outlier_analysis(data["routes"], output_dir)
    stop_package_outlier_analysis(data["packages"], output_dir)
    station_check_frame = station_sequence_check(data, output_dir)
    dropoff_check_frame = dropoff_package_check(data, output_dir)
    missing_features = feature_reliability_summary(data, output_dir)
    travel_completeness, travel_summary, travel_detail = travel_time_integrity_check(
        data["transitions"], data_root, output_dir, max_routes
    )
    training_ready_frame = training_ready_routes(
        data, route_flags, station_check_frame, travel_completeness, dropoff_check_frame, output_dir
    )
    create_plots(data, travel_detail, training_ready_frame, output_dir)
    print_final_summary(
        data,
        route_filter_summary,
        missing_features,
        travel_summary,
        station_check_frame,
        dropoff_check_frame,
        training_ready_frame,
    )


def main() -> None:
    args = parse_args()
    run_analysis(args.input_dir, args.data_root, args.output_dir, args.max_routes)


if __name__ == "__main__":
    main()
