#!/usr/bin/env python3
"""Transition-level feature EDA for dissertation feature selection.

This standalone script reads only cleaned CSV outputs, builds a transition-level
feature table, and writes summary CSVs/plots that justify feature choices before
any machine learning is trained.
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
DEFAULT_SAMPLE_ROWS = 100_000
RANDOM_STATE = 42

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


# ---------------------------------------------------------------------------
# CLI, reading, and table construction helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create transition-level EDA outputs for feature selection.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory containing first-round cleaned CSV files. Default: {DEFAULT_PROCESSED_DIR}",
    )
    parser.add_argument(
        "--final-cleaned-dir",
        type=Path,
        default=DEFAULT_FINAL_CLEANED_DIR,
        help=f"Directory containing final cleaned travel-time CSVs. Default: {DEFAULT_FINAL_CLEANED_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for transition feature EDA outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--sample-frac", type=float, default=None, help="Optional fraction of transitions to sample.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=DEFAULT_SAMPLE_ROWS,
        help="Maximum rows to keep in the sampled feature table. Default: 100000.",
    )
    parser.add_argument(
        "--save-full-table",
        action="store_true",
        help="Save the full transition feature table in addition to the sample.",
    )
    return parser.parse_args()


def require_csv(directory: Path, filename: str) -> Path:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return path


def read_inputs(processed_dir: Path, final_cleaned_dir: Path) -> dict[str, pd.DataFrame]:
    """Read only cleaned CSV files; this script never reads raw JSON data."""

    inputs = {
        "routes": pd.read_csv(require_csv(processed_dir, "routes_summary.csv")),
        "stops": pd.read_csv(require_csv(processed_dir, "stops_base_features.csv")),
        "packages": pd.read_csv(require_csv(processed_dir, "stop_package_features.csv")),
        "transitions_clean": pd.read_csv(require_csv(final_cleaned_dir, "actual_transition_travel_time_clean.csv")),
        "complete_routes": pd.read_csv(require_csv(final_cleaned_dir, "travel_time_complete_routes.csv")),
        "complete_route_transitions": pd.read_csv(
            require_csv(final_cleaned_dir, "actual_transition_travel_time_complete_routes.csv")
        ),
    }
    return inputs


def sample_transitions(transitions: pd.DataFrame, sample_frac: float | None, max_rows: int | None) -> pd.DataFrame:
    """Sample transitions before merge-heavy feature construction to control memory use."""

    sampled = transitions.copy()
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac must be greater than 0 and less than or equal to 1.")
        sampled = sampled.sample(frac=sample_frac, random_state=RANDOM_STATE)
    if max_rows is not None and len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=RANDOM_STATE)
    return sampled.reset_index(drop=True)


def normalize_zone(series: pd.Series) -> pd.Series:
    return series.fillna("UNKNOWN_ZONE").astype(str).str.strip().replace({"": "UNKNOWN_ZONE", "nan": "UNKNOWN_ZONE"})


def build_transition_feature_table(transitions: pd.DataFrame, routes: pd.DataFrame, stops: pd.DataFrame, packages: pd.DataFrame) -> pd.DataFrame:
    """Merge transitions with route, stop, and package features for EDA."""

    frame = transitions.copy()
    frame["route_id"] = frame["route_id"].astype(str)
    frame["from_stop"] = frame["from_stop"].astype(str)
    frame["to_stop"] = frame["to_stop"].astype(str)
    frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    frame["travel_time_ij"] = pd.to_numeric(frame["travel_time_ij"], errors="coerce")
    max_position = frame.groupby("route_id")["position"].transform("max")
    frame["route_progress"] = (frame["position"] / max_position.mask(max_position == 0)).fillna(0)

    route_cols = ["route_id", "route_score", "number_of_stops"]
    frame = frame.merge(routes[route_cols], on="route_id", how="left")

    stop_lookup = stops[["route_id", "stop_id", "zone_id", "type"]].copy()
    stop_lookup["route_id"] = stop_lookup["route_id"].astype(str)
    stop_lookup["stop_id"] = stop_lookup["stop_id"].astype(str)
    from_lookup = stop_lookup.rename(columns={"stop_id": "from_stop", "zone_id": "from_zone", "type": "from_type"})
    to_lookup = stop_lookup.rename(columns={"stop_id": "to_stop", "zone_id": "to_zone", "type": "to_type"})
    frame = frame.merge(from_lookup, on=["route_id", "from_stop"], how="left")
    frame = frame.merge(to_lookup, on=["route_id", "to_stop"], how="left")

    frame["from_zone"] = normalize_zone(frame["from_zone"])
    frame["to_zone"] = normalize_zone(frame["to_zone"])
    frame["zone_missing_in_transition"] = frame["from_zone"].eq("UNKNOWN_ZONE") | frame["to_zone"].eq("UNKNOWN_ZONE")
    frame["same_zone"] = (~frame["zone_missing_in_transition"]) & frame["from_zone"].eq(frame["to_zone"])
    frame["zone_changed"] = (~frame["zone_missing_in_transition"]) & ~frame["same_zone"]

    package_cols = [
        "route_id",
        "stop_id",
        "package_count",
        "total_planned_service_time",
        "has_time_window",
        "time_window_package_count",
        "total_package_volume_cm3",
    ]
    optional_cols = [column for column in ["unknown_status_count", "scan_status"] if column in packages.columns]
    package_lookup = packages[package_cols + optional_cols].rename(
        columns={
            "stop_id": "to_stop",
            "package_count": "to_package_count",
            "total_planned_service_time": "to_total_planned_service_time",
            "has_time_window": "to_has_time_window",
            "time_window_package_count": "to_time_window_package_count",
            "total_package_volume_cm3": "to_total_package_volume_cm3",
            "unknown_status_count": "to_unknown_status_count",
            "scan_status": "to_scan_status",
        }
    )
    package_lookup["route_id"] = package_lookup["route_id"].astype(str)
    package_lookup["to_stop"] = package_lookup["to_stop"].astype(str)
    frame = frame.merge(package_lookup, on=["route_id", "to_stop"], how="left")

    for column in [
        "number_of_stops",
        "to_package_count",
        "to_total_planned_service_time",
        "to_has_time_window",
        "to_time_window_package_count",
        "to_total_package_volume_cm3",
        "to_unknown_status_count",
    ]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    frame["same_zone_numeric"] = frame["same_zone"].astype(int)
    frame["zone_changed_numeric"] = frame["zone_changed"].astype(int)
    return frame


def save_feature_table(frame: pd.DataFrame, output_dir: Path, save_full_table: bool, max_rows: int | None) -> None:
    if save_full_table:
        frame.to_csv(output_dir / "transition_feature_table_full.csv", index=False)
    sample = sample_transitions(frame, sample_frac=None, max_rows=max_rows)
    sample.to_csv(output_dir / "transition_feature_table_sample.csv", index=False)


# ---------------------------------------------------------------------------
# Analysis and plotting helpers
# ---------------------------------------------------------------------------


def save_histogram(series: pd.Series, path: Path, title: str, xlabel: str, bins: int = 50) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_group_boxplot(frame: pd.DataFrame, group_col: str, value_col: str, path: Path, title: str) -> None:
    grouped = [(str(name), pd.to_numeric(group[value_col], errors="coerce").dropna()) for name, group in frame.groupby(group_col)]
    grouped = [(name, values) for name, values in grouped if len(values)]
    fig, ax = plt.subplots(figsize=(8, 5))
    if grouped:
        ax.boxplot([values for _name, values in grouped], labels=[name for name, _values in grouped], showfliers=False)
    ax.set_title(title)
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel(value_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_line_plot(frame: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(frame[x].astype(str), frame[y], marker="o")
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_bar_plot(frame: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(frame[x].astype(str), frame[y])
    ax.set_title(title)
    ax.set_xlabel(x.replace("_", " ").title())
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def descriptive_stats(series: pd.Series) -> pd.DataFrame:
    values = pd.to_numeric(series, errors="coerce")
    stats = values.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).reset_index()
    return stats.rename(columns={"index": "statistic", series.name or 0: "value"})


def add_progress_bins(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["route_progress_bin"] = pd.cut(frame["route_progress"], bins=10, include_lowest=True)
    frame["route_progress_bin"] = frame["route_progress_bin"].astype(str)
    return frame


# ---------------------------------------------------------------------------
# Required EDA outputs
# ---------------------------------------------------------------------------


def write_dataset_summary(inputs: dict[str, pd.DataFrame], output_dir: Path) -> pd.DataFrame:
    routes = inputs["routes"]
    packages = inputs["packages"]
    transitions = inputs["transitions_clean"]
    complete_routes = inputs["complete_routes"]
    complete_route_transitions = inputs["complete_route_transitions"]
    route_score_distribution = routes["route_score"].fillna("Missing").value_counts().to_dict()
    row = {
        "total_routes": len(routes),
        "total_clean_transitions": len(transitions),
        "total_complete_routes": len(complete_routes),
        "total_complete_route_transitions": len(complete_route_transitions),
        "total_stops": int(pd.to_numeric(routes["number_of_stops"], errors="coerce").sum()) if "number_of_stops" in routes else 0,
        "total_package_stop_rows": len(packages),
        "route_score_distribution": json.dumps(route_score_distribution, sort_keys=True),
        "average_stops_per_route": pd.to_numeric(routes["number_of_stops"], errors="coerce").mean(),
        "average_package_count_per_stop": pd.to_numeric(packages["package_count"], errors="coerce").mean(),
        "average_service_time_per_stop": pd.to_numeric(packages["total_planned_service_time"], errors="coerce").mean(),
        "percentage_stops_with_time_window": pd.to_numeric(packages["has_time_window"], errors="coerce").fillna(0).mean() * 100,
    }
    summary = pd.DataFrame([row])
    summary.to_csv(output_dir / "eda_dataset_summary.csv", index=False)
    return summary


def travel_time_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = descriptive_stats(frame["travel_time_ij"])
    summary.to_csv(output_dir / "travel_time_summary.csv", index=False)
    save_histogram(frame["travel_time_ij"], plots_dir / "travel_time_distribution.png", "Travel Time Distribution", "Travel Time")
    by_score = (
        frame.groupby("route_score", dropna=False)["travel_time_ij"]
        .agg(count="count", mean="mean", median="median", p95=lambda x: x.quantile(0.95))
        .reset_index()
    )
    by_score.to_csv(output_dir / "travel_time_by_route_score.csv", index=False)
    save_group_boxplot(frame, "route_score", "travel_time_ij", plots_dir / "travel_time_by_route_score_boxplot.png", "Travel Time By Route Score")
    return summary, by_score


def zone_continuity_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = pd.DataFrame(
        [
            {
                "transition_count": len(frame),
                "same_zone_transition_count": int(frame["same_zone"].sum()),
                "zone_changed_transition_count": int(frame["zone_changed"].sum()),
                "zone_missing_transition_count": int(frame["zone_missing_in_transition"].sum()),
                "same_zone_ratio": float(frame["same_zone"].mean()) if len(frame) else 0,
            }
        ]
    )
    overall.to_csv(output_dir / "same_zone_transition_summary.csv", index=False)
    by_score = (
        frame.groupby("route_score", dropna=False)["same_zone"]
        .agg(transition_count="count", same_zone_transition_count="sum", same_zone_ratio="mean")
        .reset_index()
    )
    by_score.to_csv(output_dir / "same_zone_by_route_score.csv", index=False)
    save_bar_plot(by_score, "route_score", "same_zone_ratio", plots_dir / "same_zone_ratio_by_route_score.png", "Same-Zone Ratio By Route Score", "Same-Zone Ratio")
    by_same_zone = (
        frame.groupby("same_zone", dropna=False)["travel_time_ij"]
        .agg(transition_count="count", mean="mean", median="median", p95=lambda x: x.quantile(0.95))
        .reset_index()
    )
    by_same_zone.to_csv(output_dir / "travel_time_by_same_zone.csv", index=False)
    return overall, by_score


def time_window_position_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> pd.DataFrame:
    summary = (
        frame.groupby("to_has_time_window", dropna=False)["route_progress"]
        .agg(count="count", mean="mean", median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
        .reset_index()
    )
    summary.to_csv(output_dir / "time_window_position_summary.csv", index=False)
    save_group_boxplot(frame, "to_has_time_window", "route_progress", plots_dir / "route_progress_by_time_window_boxplot.png", "Route Progress By Time Window")
    binned = add_progress_bins(frame)
    rate = (
        binned.groupby("route_progress_bin", observed=True)["to_has_time_window"]
        .agg(transition_count="count", time_window_stop_rate="mean")
        .reset_index()
    )
    rate.to_csv(output_dir / "time_window_rate_by_route_progress_bin.csv", index=False)
    save_line_plot(rate, "route_progress_bin", "time_window_stop_rate", plots_dir / "time_window_rate_by_route_progress_bin.png", "Time-Window Rate By Route Progress", "Time-Window Stop Rate")
    return summary


def package_service_burden_analysis(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> None:
    binned = add_progress_bins(frame)
    summary = (
        binned.groupby("route_progress_bin", observed=True)[
            ["to_package_count", "to_total_planned_service_time", "to_total_package_volume_cm3"]
        ]
        .agg(["count", "mean", "median"])
    )
    summary.columns = ["_".join(column).strip("_") for column in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(output_dir / "package_service_position_summary.csv", index=False)

    mappings = [
        ("to_package_count", "package_count_by_route_progress_bin.csv", "package_count_by_route_progress_bin.png", "Average Package Count"),
        (
            "to_total_planned_service_time",
            "service_time_by_route_progress_bin.csv",
            "service_time_by_route_progress_bin.png",
            "Average Service Time",
        ),
        (
            "to_total_package_volume_cm3",
            "package_volume_by_route_progress_bin.csv",
            "package_volume_by_route_progress_bin.png",
            "Average Package Volume",
        ),
    ]
    for column, csv_name, plot_name, ylabel in mappings:
        by_bin = binned.groupby("route_progress_bin", observed=True)[column].agg(count="count", mean="mean", median="median").reset_index()
        by_bin.to_csv(output_dir / csv_name, index=False)
        save_line_plot(by_bin, "route_progress_bin", "mean", plots_dir / plot_name, f"{ylabel} By Route Progress", ylabel)


def route_score_comparison(frame: pd.DataFrame, output_dir: Path, plots_dir: Path) -> pd.DataFrame:
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
    comparison.to_csv(output_dir / "route_score_feature_comparison.csv", index=False)
    plot_cols = [
        "average_travel_time_ij",
        "same_zone_ratio",
        "average_to_package_count",
        "average_to_total_planned_service_time",
        "average_to_total_package_volume_cm3",
        "time_window_stop_ratio",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, column in zip(axes.ravel(), plot_cols):
        ax.bar(comparison["route_score"].astype(str), comparison[column])
        ax.set_title(column.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plots_dir / "route_score_feature_comparison.png", dpi=150)
    plt.close(fig)
    return comparison


def correlation_analysis(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    available = [column for column in NUMERIC_FEATURES if column in frame.columns]
    numeric_frame = frame[available].apply(pd.to_numeric, errors="coerce")
    rows: list[dict[str, Any]] = []
    for method in ["pearson", "spearman"]:
        corr = numeric_frame.corr(method=method)
        for feature_x in corr.index:
            for feature_y in corr.columns:
                rows.append(
                    {
                        "method": method,
                        "feature_x": feature_x,
                        "feature_y": feature_y,
                        "correlation": corr.loc[feature_x, feature_y],
                    }
                )
    output = pd.DataFrame(rows)
    output.to_csv(output_dir / "numeric_feature_correlation.csv", index=False)
    return output


def feature_justification_summary(frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    missing_rate = float(frame["travel_time_ij"].isna().mean()) if len(frame) else 1
    same_zone_ratio = float(frame["same_zone"].mean()) if len(frame) else 0
    time_window_rate = float(frame["to_has_time_window"].mean()) if len(frame) else 0
    package_mean = float(frame["to_package_count"].mean()) if len(frame) else 0
    service_mean = float(frame["to_total_planned_service_time"].mean()) if len(frame) else 0
    volume_skew = float(frame["to_total_package_volume_cm3"].skew()) if len(frame) else 0
    rows = [
        {
            "feature_name": "travel_time_ij",
            "evidence_from_eda": f"Missing rate {missing_rate:.4f}; mean {frame['travel_time_ij'].mean():.2f}; median {frame['travel_time_ij'].median():.2f}.",
            "recommended_use": "core_feature" if missing_rate <= 0.05 else "secondary_feature",
            "explanation": "Direct transition cost signal for routing and hybrid cost functions.",
        },
        {
            "feature_name": "same_zone / zone_changed",
            "evidence_from_eda": f"Same-zone transition ratio {same_zone_ratio:.4f}.",
            "recommended_use": "core_feature" if same_zone_ratio >= 0.20 else "secondary_feature",
            "explanation": "Captures zone continuity and potential cross-zone penalty behavior.",
        },
        {
            "feature_name": "to_has_time_window",
            "evidence_from_eda": f"Transitions to time-window stops {time_window_rate:.4f}.",
            "recommended_use": "core_feature" if time_window_rate >= 0.05 else "secondary_feature",
            "explanation": "Represents delivery timing constraints and route-position risk.",
        },
        {
            "feature_name": "to_package_count",
            "evidence_from_eda": f"Average package count at destination stop {package_mean:.2f}.",
            "recommended_use": "core_feature" if package_mean > 0 else "not_recommended",
            "explanation": "Simple workload/burden signal for a stop.",
        },
        {
            "feature_name": "to_total_planned_service_time",
            "evidence_from_eda": f"Average planned service time at destination stop {service_mean:.2f}.",
            "recommended_use": "core_feature" if service_mean > 0 else "secondary_feature",
            "explanation": "Operational service burden that can affect route sequencing.",
        },
        {
            "feature_name": "to_total_package_volume_cm3",
            "evidence_from_eda": f"Volume skewness {volume_skew:.2f}; average {frame['to_total_package_volume_cm3'].mean():.2f}.",
            "recommended_use": "secondary_feature",
            "explanation": "Useful capacity/burden signal but often skewed, so validate transformations before modeling.",
        },
        {
            "feature_name": "scan_status / unknown_status_count",
            "evidence_from_eda": "Scan status is outcome-like or missing in prior cleaning; unknown_status_count may be diagnostic.",
            "recommended_use": "auxiliary_only",
            "explanation": "Use for data diagnostics rather than core preference-learning features.",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "feature_justification_summary.csv", index=False)
    return summary


def print_final_summary(frame: pd.DataFrame, recommendations: pd.DataFrame) -> None:
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
    print("Route score distribution:")
    for route_score, count in frame["route_score"].fillna("Missing").value_counts().items():
        print(f"  {route_score}: {count}")
    print("Key feature recommendations:")
    for _index, row in recommendations.iterrows():
        print(f"  {row['feature_name']}: {row['recommended_use']}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_eda(
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    final_cleaned_dir: Path = DEFAULT_FINAL_CLEANED_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sample_frac: float | None = None,
    max_rows: int | None = DEFAULT_SAMPLE_ROWS,
    save_full_table: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    inputs = read_inputs(processed_dir, final_cleaned_dir)
    write_dataset_summary(inputs, output_dir)
    transition_source = (
        inputs["transitions_clean"]
        if save_full_table
        else sample_transitions(inputs["transitions_clean"], sample_frac, max_rows)
    )
    feature_frame = build_transition_feature_table(
        transition_source, inputs["routes"], inputs["stops"], inputs["packages"]
    )
    save_feature_table(feature_frame, output_dir, save_full_table, max_rows)

    travel_time_analysis(feature_frame, output_dir, plots_dir)
    zone_continuity_analysis(feature_frame, output_dir, plots_dir)
    time_window_position_analysis(feature_frame, output_dir, plots_dir)
    package_service_burden_analysis(feature_frame, output_dir, plots_dir)
    route_score_comparison(feature_frame, output_dir, plots_dir)
    correlation_analysis(feature_frame, output_dir)
    recommendations = feature_justification_summary(feature_frame, output_dir)
    print_final_summary(feature_frame, recommendations)


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
