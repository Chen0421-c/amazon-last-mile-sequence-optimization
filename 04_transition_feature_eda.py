#!/usr/bin/env python3
"""Transition-level feature EDA for dissertation feature selection.

This script reads cleaned CSV outputs only. It builds a transition-level feature
table and creates summary CSV files and matplotlib plots to justify feature
selection before machine learning.

It does not read raw JSON files and does not train any model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PROCESSED_DIR = Path("/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs")
DEFAULT_FINAL_CLEANED_DIR = DEFAULT_PROCESSED_DIR / "final_cleaned"
DEFAULT_OUTPUT_DIR = DEFAULT_PROCESSED_DIR / "feature_eda_outputs"
DEFAULT_MAX_ROWS = 100_000
RANDOM_STATE = 42

ROUTE_SCORE_ORDER = ["High", "Medium", "Low"]

NUMERIC_FEATURES = [
    "travel_time_ij",
    "route_progress",
    "same_zone_numeric",
    "zone_changed_numeric",
    "to_package_count",
    "to_total_planned_service_time",
    "to_has_time_window",
    "to_time_window_package_count",
    "to_total_package_volume_cm3",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Create transition-level EDA outputs for feature selection before machine learning."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory containing cleaned first-round CSV files. Default: {DEFAULT_PROCESSED_DIR}",
    )
    parser.add_argument(
        "--final-cleaned-dir",
        type=Path,
        default=DEFAULT_FINAL_CLEANED_DIR,
        help=f"Directory containing final cleaned travel-time CSV files. Default: {DEFAULT_FINAL_CLEANED_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where EDA CSV and PNG outputs will be saved. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction of transitions to sample before merge-heavy EDA.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_MAX_ROWS,
        help="Maximum transition rows to use by default. Default: 100000.",
    )
    parser.add_argument(
        "--save-full-table",
        action="store_true",
        help="Save transition_feature_table_full.csv. Without this flag, only a sampled table is saved.",
    )
    return parser.parse_args()


def require_csv(directory: Path, filename: str) -> Path:
    """Return a required CSV path or raise a clear error."""

    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return path


def read_inputs(processed_dir: Path, final_cleaned_dir: Path) -> dict[str, pd.DataFrame]:
    """Read cleaned CSV inputs. This function never reads raw JSON files."""

    return {
        "routes": pd.read_csv(require_csv(processed_dir, "routes_summary.csv")),
        "stops": pd.read_csv(require_csv(processed_dir, "stops_base_features.csv")),
        "packages": pd.read_csv(require_csv(processed_dir, "stop_package_features.csv")),
        "transitions_clean": pd.read_csv(
            require_csv(final_cleaned_dir, "actual_transition_travel_time_clean.csv")
        ),
        "complete_routes": pd.read_csv(
            require_csv(final_cleaned_dir, "travel_time_complete_routes.csv")
        ),
        "complete_route_transitions": pd.read_csv(
            require_csv(final_cleaned_dir, "actual_transition_travel_time_complete_routes.csv")
        ),
    }


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric values safely."""

    return pd.to_numeric(series, errors="coerce")


def sample_frame(
    frame: pd.DataFrame,
    sample_frac: float | None,
    max_rows: int | None,
) -> pd.DataFrame:
    """Sample rows for memory-conscious EDA."""

    sampled = frame.copy()

    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac must be greater than 0 and less than or equal to 1.")
        sampled = sampled.sample(frac=sample_frac, random_state=RANDOM_STATE)

    if max_rows is not None and len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE)

    return sampled.reset_index(drop=True)


def normalize_zone(series: pd.Series) -> pd.Series:
    """Normalize missing or blank zone values to UNKNOWN_ZONE."""

    normalized = series.fillna("UNKNOWN_ZONE").astype(str).str.strip()
    return normalized.replace(
        {
            "": "UNKNOWN_ZONE",
            "nan": "UNKNOWN_ZONE",
            "NaN": "UNKNOWN_ZONE",
            "None": "UNKNOWN_ZONE",
            "null": "UNKNOWN_ZONE",
        }
    )


def route_score_sort_key(series: pd.Series) -> pd.Series:
    """Sort route score categories in a stable High, Medium, Low order."""

    order = {score: index for index, score in enumerate(ROUTE_SCORE_ORDER)}
    return series.map(order).fillna(len(order))


def prepare_output_dirs(output_dir: Path) -> Path:
    """Create output directories and return the plots directory."""

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def safe_mean(series: pd.Series) -> float:
    values = to_numeric(series).dropna()
    return float(values.mean()) if len(values) else 0.0


def safe_median(series: pd.Series) -> float:
    values = to_numeric(series).dropna()
    return float(values.median()) if len(values) else 0.0


def write_dataset_summary(inputs: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    """Create the overall EDA dataset summary."""

    routes = inputs["routes"]
    stops = inputs["stops"]
    packages = inputs["packages"]
    transitions = inputs["transitions_clean"]
    complete_routes = inputs["complete_routes"]
    complete_route_transitions = inputs["complete_route_transitions"]

    route_score_distribution = routes["route_score"].fillna("Missing").value_counts().to_dict()

    package_count = (
        to_numeric(packages["package_count"])
        if "package_count" in packages.columns
        else pd.Series(dtype=float)
    )
    service_time = (
        to_numeric(packages["total_planned_service_time"])
        if "total_planned_service_time" in packages.columns
        else pd.Series(dtype=float)
    )
    has_time_window = (
        to_numeric(packages["has_time_window"]).fillna(0)
        if "has_time_window" in packages.columns
        else pd.Series(dtype=float)
    )

    row = {
        "total_routes": len(routes),
        "total_clean_transitions": len(transitions),
        "total_complete_routes": len(complete_routes),
        "total_complete_route_transitions": len(complete_route_transitions),
        "total_stops": len(stops),
        "total_package_stop_rows": len(packages),
        "route_score_distribution": json.dumps(route_score_distribution, sort_keys=True),
        "average_stops_per_route": safe_mean(routes["number_of_stops"]) if "number_of_stops" in routes.columns else 0.0,
        "average_package_count_per_stop": float(package_count.mean()) if len(package_count) else 0.0,
        "average_service_time_per_stop": float(service_time.mean()) if len(service_time) else 0.0,
        "percentage_stops_with_time_window": float(has_time_window.mean() * 100) if len(has_time_window) else 0.0,
    }

    summary = pd.DataFrame([row])
    summary.to_csv(output_dir / "eda_dataset_summary.csv", index=False)
    return summary


def build_transition_feature_table(
    transitions: pd.DataFrame,
    routes: pd.DataFrame,
    stops: pd.DataFrame,
    packages: pd.DataFrame,
) -> pd.DataFrame:
    """Build a transition-level feature table from cleaned CSV inputs."""

    frame = transitions.copy()

    frame["route_id"] = frame["route_id"].astype(str)
    frame["from_stop"] = frame["from_stop"].astype(str)
    frame["to_stop"] = frame["to_stop"].astype(str)
    frame["position"] = to_numeric(frame["position"])
    frame["travel_time_ij"] = to_numeric(frame["travel_time_ij"])

    route_columns = ["route_id", "route_score", "number_of_stops"]
    missing_route_columns = [col for col in route_columns if col not in routes.columns]
    if missing_route_columns:
        raise ValueError(f"routes_summary.csv is missing required columns: {missing_route_columns}")

    route_lookup = routes[route_columns].copy()
    route_lookup["route_id"] = route_lookup["route_id"].astype(str)
    frame = frame.merge(route_lookup, on="route_id", how="left")
    frame["number_of_stops"] = to_numeric(frame["number_of_stops"])

    denominator = (frame["number_of_stops"] - 1).replace(0, pd.NA)
    frame["route_progress"] = (frame["position"] / denominator).fillna(0)
    frame["route_progress"] = frame["route_progress"].clip(lower=0, upper=1)

    stop_columns = ["route_id", "stop_id", "zone_id", "type"]
    missing_stop_columns = [col for col in stop_columns if col not in stops.columns]
    if missing_stop_columns:
        raise ValueError(f"stops_base_features.csv is missing required columns: {missing_stop_columns}")

    stop_lookup = stops[stop_columns].copy()
    stop_lookup["route_id"] = stop_lookup["route_id"].astype(str)
    stop_lookup["stop_id"] = stop_lookup["stop_id"].astype(str)

    from_stop_lookup = stop_lookup.rename(
        columns={
            "stop_id": "from_stop",
            "zone_id": "from_zone",
            "type": "from_type",
        }
    )
    to_stop_lookup = stop_lookup.rename(
        columns={
            "stop_id": "to_stop",
            "zone_id": "to_zone",
            "type": "to_type",
        }
    )

    frame = frame.merge(from_stop_lookup, on=["route_id", "from_stop"], how="left")
    frame = frame.merge(to_stop_lookup, on=["route_id", "to_stop"], how="left")

    frame["from_zone"] = normalize_zone(frame["from_zone"])
    frame["to_zone"] = normalize_zone(frame["to_zone"])

    frame["zone_missing_in_transition"] = frame["from_zone"].eq("UNKNOWN_ZONE") | frame["to_zone"].eq(
        "UNKNOWN_ZONE"
    )
    frame["same_zone"] = (~frame["zone_missing_in_transition"]) & frame["from_zone"].eq(frame["to_zone"])
    frame["zone_changed"] = (~frame["zone_missing_in_transition"]) & (~frame["same_zone"])

    package_columns = [
        "route_id",
        "stop_id",
        "package_count",
        "total_planned_service_time",
        "has_time_window",
        "time_window_package_count",
        "total_package_volume_cm3",
    ]
    missing_package_columns = [col for col in package_columns if col not in packages.columns]
    if missing_package_columns:
        raise ValueError(f"stop_package_features.csv is missing required columns: {missing_package_columns}")

    optional_package_columns = [
        column
        for column in ["unknown_status_count", "delivered_count", "attempted_count", "rejected_count"]
        if column in packages.columns
    ]

    package_lookup = packages[package_columns + optional_package_columns].copy()
    package_lookup["route_id"] = package_lookup["route_id"].astype(str)
    package_lookup["stop_id"] = package_lookup["stop_id"].astype(str)
    package_lookup = package_lookup.rename(
        columns={
            "stop_id": "to_stop",
            "package_count": "to_package_count",
            "total_planned_service_time": "to_total_planned_service_time",
            "has_time_window": "to_has_time_window",
            "time_window_package_count": "to_time_window_package_count",
            "total_package_volume_cm3": "to_total_package_volume_cm3",
            "unknown_status_count": "to_unknown_status_count",
            "delivered_count": "to_delivered_count",
            "attempted_count": "to_attempted_count",
            "rejected_count": "to_rejected_count",
        }
    )

    frame = frame.merge(package_lookup, on=["route_id", "to_stop"], how="left")

    numeric_columns = [
        "to_package_count",
        "to_total_planned_service_time",
        "to_has_time_window",
        "to_time_window_package_count",
        "to_total_package_volume_cm3",
        "to_unknown_status_count",
        "to_delivered_count",
        "to_attempted_count",
        "to_rejected_count",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = to_numeric(frame[column]).fillna(0)

    frame["same_zone_numeric"] = frame["same_zone"].astype(int)
    frame["zone_changed_numeric"] = frame["zone_changed"].astype(int)
    frame["to_has_time_window"] = to_numeric(frame["to_has_time_window"]).fillna(0).astype(int)

    return frame


def save_feature_table(
    frame: pd.DataFrame,
    output_dir: Path,
    save_full_table: bool,
    max_rows: int | None,
) -> None:
    """Save a default sample table and optionally the full transition feature table."""

    if save_full_table:
        frame.to_csv(output_dir / "transition_feature_table_full.csv", index=False)

    sample = sample_frame(frame, sample_frac=None, max_rows=max_rows)
    sample.to_csv(output_dir / "transition_feature_table_sample.csv", index=False)


def save_histogram(series: pd.Series, output_path: Path, title: str, xlabel: str, bins: int = 50) -> None:
    """Save a histogram using matplotlib only."""

    values = to_numeric(series).dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_boxplot_by_group(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Save a grouped boxplot using matplotlib only."""

    grouped_values = []
    labels = []
    sorted_frame = frame.copy()

    if group_column == "route_score":
        sorted_frame["_sort_order"] = route_score_sort_key(sorted_frame[group_column])
        sorted_frame = sorted_frame.sort_values(["_sort_order", group_column])

    for group_name, group in sorted_frame.groupby(group_column, dropna=False):
        values = to_numeric(group[value_column]).dropna()
        if len(values) > 0:
            labels.append(str(group_name))
            grouped_values.append(values)

    fig, ax = plt.subplots(figsize=(8, 5))
    if grouped_values:
        ax.boxplot(grouped_values, labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(group_column.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_bar_plot(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Save a simple bar chart using matplotlib only."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(frame[x_column].astype(str), frame[y_column])
    ax.set_title(title)
    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_line_plot(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Save a simple line plot using matplotlib only."""

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame[x_column].astype(str), frame[y_column], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x_column.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def descriptive_statistics(series: pd.Series) -> pd.DataFrame:
    """Return descriptive statistics for a numeric series."""

    values = to_numeric(series)
    stats = values.describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return stats.reset_index().rename(columns={"index": "statistic", values.name: "value"})


def add_route_progress_bins(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 10 route progress bins for position-based EDA."""

    output = frame.copy()
    output["route_progress_bin"] = pd.cut(
        output["route_progress"],
        bins=10,
        include_lowest=True,
        duplicates="drop",
    )
    output["route_progress_bin"] = output["route_progress_bin"].astype(str)
    return output


def travel_time_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    """Analyze travel time as a core routing cost feature."""

    summary = descriptive_statistics(frame["travel_time_ij"])
    summary.to_csv(output_dir / "travel_time_summary.csv", index=False)

    save_histogram(
        frame["travel_time_ij"],
        plots_dir / "travel_time_distribution.png",
        "Travel Time Distribution",
        "Travel Time",
    )

    by_score = (
        frame.groupby("route_score", dropna=False)["travel_time_ij"]
        .agg(
            transition_count="count",
            mean="mean",
            median="median",
            p95=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )
    by_score["_sort_order"] = route_score_sort_key(by_score["route_score"])
    by_score = by_score.sort_values(["_sort_order", "route_score"]).drop(columns="_sort_order")
    by_score.to_csv(output_dir / "travel_time_by_route_score.csv", index=False)

    save_boxplot_by_group(
        frame,
        "route_score",
        "travel_time_ij",
        plots_dir / "travel_time_by_route_score_boxplot.png",
        "Travel Time By Route Score",
        "Travel Time",
    )


def zone_continuity_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> pd.DataFrame:
    """Analyze same-zone and cross-zone transition patterns."""

    overall = pd.DataFrame(
        [
            {
                "transition_count": len(frame),
                "same_zone_transition_count": int(frame["same_zone"].sum()),
                "zone_changed_transition_count": int(frame["zone_changed"].sum()),
                "zone_missing_transition_count": int(frame["zone_missing_in_transition"].sum()),
                "same_zone_ratio": float(frame["same_zone"].mean()) if len(frame) else 0.0,
            }
        ]
    )
    overall.to_csv(output_dir / "same_zone_transition_summary.csv", index=False)

    by_score = (
        frame.groupby("route_score", dropna=False)["same_zone"]
        .agg(
            transition_count="count",
            same_zone_transition_count="sum",
            same_zone_ratio="mean",
        )
        .reset_index()
    )
    by_score["_sort_order"] = route_score_sort_key(by_score["route_score"])
    by_score = by_score.sort_values(["_sort_order", "route_score"]).drop(columns="_sort_order")
    by_score.to_csv(output_dir / "same_zone_by_route_score.csv", index=False)

    save_bar_plot(
        by_score,
        "route_score",
        "same_zone_ratio",
        plots_dir / "same_zone_ratio_by_route_score.png",
        "Same-Zone Transition Ratio By Route Score",
        "Same-Zone Ratio",
    )

    travel_by_same_zone = (
        frame.groupby("same_zone", dropna=False)["travel_time_ij"]
        .agg(
            transition_count="count",
            mean="mean",
            median="median",
            p95=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )
    travel_by_same_zone.to_csv(output_dir / "travel_time_by_same_zone.csv", index=False)

    return overall


def time_window_position_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> pd.DataFrame:
    """Analyze where time-window stops occur along the route."""

    summary = (
        frame.groupby("to_has_time_window", dropna=False)["route_progress"]
        .agg(
            transition_count="count",
            mean="mean",
            median="median",
            p25=lambda values: values.quantile(0.25),
            p75=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "time_window_position_summary.csv", index=False)

    save_boxplot_by_group(
        frame,
        "to_has_time_window",
        "route_progress",
        plots_dir / "route_progress_by_time_window_boxplot.png",
        "Route Progress By Time Window Flag",
        "Route Progress",
    )

    binned = add_route_progress_bins(frame)
    time_window_rate = (
        binned.groupby("route_progress_bin", observed=True)["to_has_time_window"]
        .agg(
            transition_count="count",
            time_window_stop_rate="mean",
        )
        .reset_index()
    )
    time_window_rate.to_csv(output_dir / "time_window_rate_by_route_progress_bin.csv", index=False)

    save_line_plot(
        time_window_rate,
        "route_progress_bin",
        "time_window_stop_rate",
        plots_dir / "time_window_rate_by_route_progress_bin.png",
        "Time-Window Stop Rate By Route Progress",
        "Time-Window Stop Rate",
    )

    return summary


def package_service_burden_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    """Analyze package, service-time, and volume burden by route progress."""

    binned = add_route_progress_bins(frame)

    combined_summary = (
        binned.groupby("route_progress_bin", observed=True)[
            [
                "to_package_count",
                "to_total_planned_service_time",
                "to_total_package_volume_cm3",
            ]
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    combined_summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in combined_summary.columns
    ]
    combined_summary.to_csv(output_dir / "package_service_position_summary.csv", index=False)

    outputs = [
        (
            "to_package_count",
            "package_count_by_route_progress_bin.csv",
            "package_count_by_route_progress_bin.png",
            "Average Package Count",
        ),
        (
            "to_total_planned_service_time",
            "service_time_by_route_progress_bin.csv",
            "service_time_by_route_progress_bin.png",
            "Average Planned Service Time",
        ),
        (
            "to_total_package_volume_cm3",
            "package_volume_by_route_progress_bin.csv",
            "package_volume_by_route_progress_bin.png",
            "Average Package Volume",
        ),
    ]

    for column, csv_name, plot_name, ylabel in outputs:
        by_bin = (
            binned.groupby("route_progress_bin", observed=True)[column]
            .agg(
                transition_count="count",
                mean="mean",
                median="median",
                p95=lambda values: values.quantile(0.95),
            )
            .reset_index()
        )
        by_bin.to_csv(output_dir / csv_name, index=False)

        save_line_plot(
            by_bin,
            "route_progress_bin",
            "mean",
            plots_dir / plot_name,
            f"{ylabel} By Route Progress",
            ylabel,
        )


def route_score_feature_comparison(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> pd.DataFrame:
    """Compare operational features across route_score groups."""

    comparison = (
        frame.groupby("route_score", dropna=False)
        .agg(
            transition_count=("route_id", "count"),
            average_travel_time_ij=("travel_time_ij", "mean"),
            same_zone_ratio=("same_zone", "mean"),
            average_to_package_count=("to_package_count", "mean"),
            average_to_total_planned_service_time=("to_total_planned_service_time", "mean"),
            average_to_total_package_volume_cm3=("to_total_package_volume_cm3", "mean"),
            time_window_stop_ratio=("to_has_time_window", "mean"),
        )
        .reset_index()
    )
    comparison["_sort_order"] = route_score_sort_key(comparison["route_score"])
    comparison = comparison.sort_values(["_sort_order", "route_score"]).drop(columns="_sort_order")
    comparison.to_csv(output_dir / "route_score_feature_comparison.csv", index=False)

    plot_columns = [
        "average_travel_time_ij",
        "same_zone_ratio",
        "average_to_package_count",
        "average_to_total_planned_service_time",
        "average_to_total_package_volume_cm3",
        "time_window_stop_ratio",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.ravel(), plot_columns):
        ax.bar(comparison["route_score"].astype(str), comparison[column])
        ax.set_title(column.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    fig.savefig(plots_dir / "route_score_feature_comparison.png", dpi=150)
    plt.close(fig)

    return comparison


def correlation_analysis(frame: pd.DataFrame, output_dir: Path) -> None:
    """Compute Pearson and Spearman correlations for numeric candidate features."""

    available_columns = [column for column in NUMERIC_FEATURES if column in frame.columns]
    numeric_frame = frame[available_columns].apply(to_numeric)

    pearson = numeric_frame.corr(method="pearson")
    spearman = numeric_frame.corr(method="spearman")

    pearson.to_csv(output_dir / "numeric_feature_correlation_pearson.csv")
    spearman.to_csv(output_dir / "numeric_feature_correlation_spearman.csv")


def feature_justification_summary(
    frame: pd.DataFrame,
    same_zone_summary: pd.DataFrame,
    time_window_position_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    """Create a feature recommendation table based on observed EDA evidence."""

    travel_missing_rate = float(frame["travel_time_ij"].isna().mean()) if len(frame) else 1.0
    average_travel_time = float(frame["travel_time_ij"].mean()) if len(frame) else 0.0
    median_travel_time = float(frame["travel_time_ij"].median()) if len(frame) else 0.0

    same_zone_ratio = 0.0
    if not same_zone_summary.empty:
        same_zone_ratio = float(same_zone_summary.loc[0, "same_zone_ratio"])

    time_window_rate = float(frame["to_has_time_window"].mean()) if len(frame) else 0.0

    time_window_mean_difference = 0.0
    if len(time_window_position_summary["to_has_time_window"].dropna().unique()) >= 2:
        means = time_window_position_summary.set_index("to_has_time_window")["mean"].to_dict()
        time_window_mean_difference = abs(float(means.get(1, 0.0)) - float(means.get(0, 0.0)))

    average_package_count = float(frame["to_package_count"].mean()) if len(frame) else 0.0
    average_service_time = float(frame["to_total_planned_service_time"].mean()) if len(frame) else 0.0
    average_volume = float(frame["to_total_package_volume_cm3"].mean()) if len(frame) else 0.0
    volume_skew = float(frame["to_total_package_volume_cm3"].skew()) if len(frame) else 0.0

    rows = [
        {
            "feature_name": "travel_time_ij",
            "evidence_from_eda": (
                f"Missing rate after cleaning is {travel_missing_rate:.4f}; "
                f"mean is {average_travel_time:.2f}; median is {median_travel_time:.2f}."
            ),
            "recommended_use": "core_feature",
            "explanation": "Travel time is central to route cost and has very low missingness after multi-source verification.",
        },
        {
            "feature_name": "same_zone / zone_changed",
            "evidence_from_eda": f"Overall same-zone transition ratio is {same_zone_ratio:.4f}.",
            "recommended_use": "core_feature" if same_zone_ratio >= 0.20 else "secondary_feature",
            "explanation": "Zone continuity captures whether a transition stays in the same delivery area or crosses zones.",
        },
        {
            "feature_name": "to_has_time_window",
            "evidence_from_eda": (
                f"Time-window transition rate is {time_window_rate:.4f}; "
                f"route-progress mean difference is {time_window_mean_difference:.4f}."
            ),
            "recommended_use": "core_feature" if time_window_mean_difference >= 0.05 else "secondary_feature",
            "explanation": "Time-window stops can represent route-position risk and delivery timing constraints.",
        },
        {
            "feature_name": "to_package_count",
            "evidence_from_eda": f"Average destination package count is {average_package_count:.2f}.",
            "recommended_use": "core_feature" if average_package_count > 0 else "not_recommended",
            "explanation": "Package count is a direct stop workload feature.",
        },
        {
            "feature_name": "to_total_planned_service_time",
            "evidence_from_eda": f"Average destination planned service time is {average_service_time:.2f}.",
            "recommended_use": "core_feature" if average_service_time > 0 else "secondary_feature",
            "explanation": "Planned service time captures expected stop handling burden.",
        },
        {
            "feature_name": "to_total_package_volume_cm3",
            "evidence_from_eda": (
                f"Average destination package volume is {average_volume:.2f}; "
                f"skewness is {volume_skew:.2f}."
            ),
            "recommended_use": "secondary_feature",
            "explanation": "Package volume may be useful but can be highly skewed, so it should be reviewed before modeling.",
        },
        {
            "feature_name": "scan_status / unknown_status_count",
            "evidence_from_eda": "Earlier missing-value analysis showed high missingness for scan status.",
            "recommended_use": "auxiliary_only",
            "explanation": "Use scan-status-derived fields for diagnostics rather than as primary driver preference features.",
        },
    ]

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "feature_justification_summary.csv", index=False)
    return summary


def print_final_summary(frame: pd.DataFrame, recommendations: pd.DataFrame) -> None:
    """Print a concise final summary for the dissertation workflow."""

    print("\nTransition-level feature EDA complete.")
    print(f"Total clean transitions used: {len(frame)}")
    print(f"Route count: {frame['route_id'].nunique()}")
    print(f"Overall same-zone transition ratio: {frame['same_zone'].mean():.4f}")
    print(f"Average travel time: {frame['travel_time_ij'].mean():.4f}")
    print(f"Median travel time: {frame['travel_time_ij'].median():.4f}")
    print(f"Percentage of transitions to stops with time windows: {frame['to_has_time_window'].mean() * 100:.2f}%")
    print(f"Average package count: {frame['to_package_count'].mean():.4f}")
    print(f"Average service time: {frame['to_total_planned_service_time'].mean():.4f}")
    print(f"Average package volume: {frame['to_total_package_volume_cm3'].mean():.4f}")

    print("Route_score distribution:")
    for route_score, count in frame["route_score"].fillna("Missing").value_counts().items():
        print(f"  {route_score}: {count}")

    print("Key feature recommendations:")
    for _, row in recommendations.iterrows():
        print(f"  {row['feature_name']}: {row['recommended_use']}")


def run_eda(
    processed_dir: Path,
    final_cleaned_dir: Path,
    output_dir: Path,
    sample_frac: float | None,
    max_rows: int | None,
    save_full_table: bool,
) -> None:
    """Run the complete transition-level EDA workflow."""

    plots_dir = prepare_output_dirs(output_dir)

    inputs = read_inputs(processed_dir, final_cleaned_dir)
    write_dataset_summary(inputs, output_dir)

    if save_full_table:
        transition_source = inputs["transitions_clean"]
    else:
        transition_source = sample_frame(inputs["transitions_clean"], sample_frac, max_rows)

    feature_table = build_transition_feature_table(
        transition_source,
        inputs["routes"],
        inputs["stops"],
        inputs["packages"],
    )

    save_feature_table(feature_table, output_dir, save_full_table, max_rows)

    travel_time_analysis(feature_table, output_dir, plots_dir)
    same_zone_summary = zone_continuity_analysis(feature_table, output_dir, plots_dir)
    time_window_summary = time_window_position_analysis(feature_table, output_dir, plots_dir)
    package_service_burden_analysis(feature_table, output_dir, plots_dir)
    route_score_feature_comparison(feature_table, output_dir, plots_dir)
    correlation_analysis(feature_table, output_dir)

    recommendations = feature_justification_summary(
        feature_table,
        same_zone_summary,
        time_window_summary,
        output_dir,
    )

    print_final_summary(feature_table, recommendations)


def main() -> None:
    args = parse_args()
    run_eda(
        processed_dir=args.processed_dir,
        final_cleaned_dir=args.final_cleaned_dir,
        output_dir=args.output_dir,
        sample_frac=args.sample_frac,
        max_rows=args.max_rows,
        save_full_table=args.save_full_table,
    )


if __name__ == "__main__":
    main()
