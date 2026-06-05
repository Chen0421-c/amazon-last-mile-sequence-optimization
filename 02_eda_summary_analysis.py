#!/usr/bin/env python3
"""EDA summary analysis for cleaned Amazon Last Mile CSV outputs.

This script reads the cleaned CSV files created by ``01_data_cleaning_pipeline.py``
and writes dissertation-friendly summary tables and simple plots. It does not
train any machine learning model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_PROCESSED_OUTPUTS_DIR = Path("/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs")
DEFAULT_EDA_OUTPUT_DIR = DEFAULT_PROCESSED_OUTPUTS_DIR / "eda_outputs"
ROUTE_SCORE_ORDER = ["High", "Medium", "Low"]
BURDEN_COLUMNS = ["package_count", "total_planned_service_time", "total_package_volume_cm3"]


class MissingInputError(FileNotFoundError):
    """Raised when a required cleaned CSV input is missing."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create EDA summary CSVs and plots from cleaned Amazon Last Mile outputs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_PROCESSED_OUTPUTS_DIR,
        help=f"Directory containing cleaned CSV files. Default: {DEFAULT_PROCESSED_OUTPUTS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_EDA_OUTPUT_DIR,
        help=f"Directory for EDA CSV/PNG outputs. Default: {DEFAULT_EDA_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction of actual transitions to sample for merge-heavy transition analyses.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional maximum number of actual transition rows for merge-heavy transition analyses.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed used when sampling transitions.")
    return parser.parse_args()


def require_csv(input_dir: Path, filename: str) -> Path:
    path = input_dir / filename
    if not path.exists():
        raise MissingInputError(f"Required input file not found: {path}")
    return path


def read_required_csv(input_dir: Path, filename: str, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(require_csv(input_dir, filename), **kwargs)


def to_bool(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean encodings to pandas boolean values."""

    if series.dtype == bool:
        return series
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["true", "1", "yes", "y"])


def safe_percent(counts: pd.Series) -> pd.Series:
    total = counts.sum()
    if total == 0:
        return counts.astype(float)
    return counts / total * 100


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


def save_histogram(series: pd.Series, path: Path, title: str, xlabel: str, bins: int = 30) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_frame(frame: pd.DataFrame, sample_frac: float | None, max_rows: int | None, random_state: int) -> pd.DataFrame:
    """Optionally sample transition rows before memory-heavy merges."""

    sampled = frame
    if sample_frac is not None:
        if not 0 < sample_frac <= 1:
            raise ValueError("--sample-frac must be greater than 0 and less than or equal to 1.")
        sampled = sampled.sample(frac=sample_frac, random_state=random_state)
    if max_rows is not None and len(sampled) > max_rows:
        sampled = sampled.sample(n=max_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def describe_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    numeric = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric.describe().reset_index().rename(columns={"index": "statistic"})


# ---------------------------------------------------------------------------
# EDA sections
# ---------------------------------------------------------------------------


def analyze_route_scores(routes: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    counts = routes["route_score"].fillna("Missing").value_counts(dropna=False)
    distribution = counts.rename_axis("route_score").reset_index(name="count")
    distribution["percentage"] = safe_percent(distribution["count"])
    score_order = {score: index for index, score in enumerate(ROUTE_SCORE_ORDER + ["Missing"])}
    distribution["_sort_order"] = distribution["route_score"].map(score_order).fillna(len(score_order))
    distribution = distribution.sort_values(["_sort_order", "route_score"]).drop(columns="_sort_order").reset_index(drop=True)
    distribution.to_csv(output_dir / "route_score_distribution.csv", index=False)
    save_bar_plot(
        distribution,
        "route_score",
        "count",
        output_dir / "route_score_distribution.png",
        "Route Score Distribution",
        "Routes",
    )
    return distribution


def analyze_data_quality(data_quality: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ["can_use_for_training", "has_single_station", "sequence_matches_route_stops"]:
        bool_values = to_bool(data_quality[column])
        counts = bool_values.value_counts(dropna=False).rename_axis("value").reset_index(name="count")
        counts["percentage"] = safe_percent(counts["count"])
        for _, row in counts.iterrows():
            rows.append(
                {
                    "metric": column,
                    "value": bool(row["value"]),
                    "count": int(row["count"]),
                    "percentage": float(row["percentage"]),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "data_quality_summary.csv", index=False)
    return summary


def analyze_route_sizes(routes: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = describe_numeric(routes, ["number_of_stops"])
    summary.to_csv(output_dir / "route_size_summary.csv", index=False)
    save_histogram(
        routes["number_of_stops"],
        output_dir / "route_size_distribution.png",
        "Route Size Distribution",
        "Number Of Stops",
    )
    return summary


def analyze_zone_missing(stops: pd.DataFrame, routes: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    stops = stops.copy()
    stops["zone_missing_bool"] = to_bool(stops["zone_missing"])
    overall = pd.DataFrame(
        [
            {
                "analysis_level": "overall",
                "route_score": "All",
                "total_stops": len(stops),
                "missing_zone_count": int(stops["zone_missing_bool"].sum()),
                "missing_zone_ratio": float(stops["zone_missing_bool"].mean()) if len(stops) else 0,
            }
        ]
    )

    stops_with_scores = stops.merge(routes[["route_id", "route_score"]], on="route_id", how="left")
    by_score = (
        stops_with_scores.groupby("route_score", dropna=False)["zone_missing_bool"]
        .agg(total_stops="count", missing_zone_count="sum", missing_zone_ratio="mean")
        .reset_index()
    )
    by_score.insert(0, "analysis_level", "by_route_score")
    summary = pd.concat([overall, by_score], ignore_index=True)
    summary.to_csv(output_dir / "zone_missing_summary.csv", index=False)
    return summary


def analyze_package_features(packages: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    packages = packages.copy()
    for column in BURDEN_COLUMNS + ["has_time_window"]:
        packages[column] = pd.to_numeric(packages[column], errors="coerce")

    feature_summary = describe_numeric(packages, BURDEN_COLUMNS)
    feature_summary.to_csv(output_dir / "package_feature_summary.csv", index=False)

    time_window_counts = packages["has_time_window"].fillna(0).astype(int).value_counts().rename_axis("has_time_window")
    time_window_summary = time_window_counts.reset_index(name="count")
    time_window_summary["percentage"] = safe_percent(time_window_summary["count"])
    time_window_summary = time_window_summary.sort_values("has_time_window").reset_index(drop=True)
    time_window_summary.to_csv(output_dir / "time_window_summary.csv", index=False)

    for column in BURDEN_COLUMNS:
        save_histogram(
            packages[column],
            output_dir / f"{column}_distribution.png",
            f"{column.replace('_', ' ').title()} Distribution",
            column.replace("_", " ").title(),
        )
    return feature_summary, time_window_summary


def build_transition_analysis_frame(
    transitions: pd.DataFrame,
    stops: pd.DataFrame,
    routes: pd.DataFrame,
    sample_frac: float | None,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    sampled = sample_frame(transitions, sample_frac, max_rows, random_state)
    stop_lookup = stops[["route_id", "stop_id", "zone_id"]].drop_duplicates(["route_id", "stop_id"])
    from_lookup = stop_lookup.rename(columns={"stop_id": "from_stop", "zone_id": "from_zone"})
    to_lookup = stop_lookup.rename(columns={"stop_id": "to_stop", "zone_id": "to_zone"})

    merged = sampled.merge(from_lookup, on=["route_id", "from_stop"], how="left")
    merged = merged.merge(to_lookup, on=["route_id", "to_stop"], how="left")
    merged = merged.merge(routes[["route_id", "route_score"]], on="route_id", how="left")
    from_zone = merged["from_zone"].fillna("").astype(str).str.strip()
    to_zone = merged["to_zone"].fillna("").astype(str).str.strip()
    merged["same_zone"] = (from_zone != "") & (to_zone != "") & (from_zone == to_zone)
    return merged


def analyze_transition_zones(transition_frame: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, float]:
    same_zone_ratio = float(transition_frame["same_zone"].mean()) if len(transition_frame) else 0
    overall = pd.DataFrame(
        [
            {
                "analysis_level": "overall",
                "route_score": "All",
                "transition_count": len(transition_frame),
                "same_zone_transition_count": int(transition_frame["same_zone"].sum()) if len(transition_frame) else 0,
                "same_zone_ratio": same_zone_ratio,
            }
        ]
    )
    by_score = (
        transition_frame.groupby("route_score", dropna=False)["same_zone"]
        .agg(transition_count="count", same_zone_transition_count="sum", same_zone_ratio="mean")
        .reset_index()
    )
    by_score.insert(0, "analysis_level", "by_route_score")
    summary = pd.concat([overall, by_score], ignore_index=True)
    summary.to_csv(output_dir / "transition_zone_summary.csv", index=False)

    plot_data = by_score.dropna(subset=["route_score"])
    save_bar_plot(
        plot_data,
        "route_score",
        "same_zone_ratio",
        output_dir / "same_zone_ratio_by_route_score.png",
        "Same-Zone Transition Ratio By Route Score",
        "Same-Zone Ratio",
    )
    return summary, same_zone_ratio


def add_route_progress(transitions: pd.DataFrame) -> pd.DataFrame:
    transitions = transitions.copy()
    transitions["position"] = pd.to_numeric(transitions["position"], errors="coerce")
    max_position = transitions.groupby("route_id")["position"].transform("max")
    denominator = max_position.mask(max_position == 0)
    transitions["route_progress"] = (transitions["position"] / denominator).fillna(0)
    return transitions


def merge_to_stop_package_features(transitions: pd.DataFrame, packages: pd.DataFrame) -> pd.DataFrame:
    package_lookup = packages.rename(columns={"stop_id": "to_stop"})[
        ["route_id", "to_stop", "package_count", "total_planned_service_time", "total_package_volume_cm3", "has_time_window"]
    ]
    merged = transitions.merge(package_lookup, on=["route_id", "to_stop"], how="left")
    for column in BURDEN_COLUMNS + ["has_time_window"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0)
    merged["has_time_window"] = merged["has_time_window"].astype(int)
    return merged


def analyze_time_window_positions(position_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    summary = (
        position_frame.groupby("has_time_window", dropna=False)["route_progress"]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .reset_index()
    )
    summary.to_csv(output_dir / "time_window_position_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = [group["route_progress"].dropna() for _, group in position_frame.groupby("has_time_window")]
    labels = [str(value) for value in sorted(position_frame["has_time_window"].dropna().unique())]
    if groups:
        ax.boxplot(groups, labels=labels)
    ax.set_title("Route Progress By Time Window Flag")
    ax.set_xlabel("Has Time Window")
    ax.set_ylabel("Route Progress")
    fig.tight_layout()
    fig.savefig(output_dir / "route_progress_by_time_window.png", dpi=150)
    plt.close(fig)
    return summary


def analyze_service_burden_positions(position_frame: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    frame = position_frame.copy()
    frame["route_progress_bin"] = pd.cut(frame["route_progress"], bins=10, include_lowest=True)
    grouped = frame.groupby("route_progress_bin", observed=True)
    summary = grouped[BURDEN_COLUMNS].mean().reset_index()
    summary.insert(1, "transition_count", grouped.size().to_numpy())
    summary["route_progress_bin"] = summary["route_progress_bin"].astype(str)
    summary.to_csv(output_dir / "service_package_position_summary.csv", index=False)

    for column in BURDEN_COLUMNS:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(summary["route_progress_bin"], summary[column], marker="o")
        ax.set_title(f"Average {column.replace('_', ' ').title()} By Route Progress")
        ax.set_xlabel("Route Progress Bin")
        ax.set_ylabel(f"Average {column.replace('_', ' ').title()}")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(output_dir / f"{column}_by_route_progress.png", dpi=150)
        plt.close(fig)
    return summary


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def run_eda(
    input_dir: Path = DEFAULT_PROCESSED_OUTPUTS_DIR,
    output_dir: Path = DEFAULT_EDA_OUTPUT_DIR,
    sample_frac: float | None = None,
    max_rows: int | None = None,
    random_state: int = 42,
) -> dict[str, object]:
    """Run all requested EDA outputs and return final summary values."""

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read only the cleaned CSVs generated by the previous pipeline step.
    routes = read_required_csv(input_dir, "routes_summary.csv")
    stops = read_required_csv(input_dir, "stops_base_features.csv")
    transitions = read_required_csv(input_dir, "actual_transitions.csv")
    packages = read_required_csv(input_dir, "stop_package_features.csv")
    data_quality = read_required_csv(input_dir, "data_quality_report.csv")
    missing_values = read_required_csv(input_dir, "missing_value_summary.csv")

    print(f"Loaded cleaned inputs from: {input_dir}")
    print(f"Saving EDA outputs to: {output_dir}")
    print(f"Missing-value summary rows loaded: {len(missing_values)}")

    route_score_distribution = analyze_route_scores(routes, output_dir)
    analyze_data_quality(data_quality, output_dir)
    analyze_route_sizes(routes, output_dir)
    analyze_zone_missing(stops, routes, output_dir)
    _package_feature_summary, time_window_summary = analyze_package_features(packages, output_dir)

    transition_frame = build_transition_analysis_frame(transitions, stops, routes, sample_frac, max_rows, random_state)
    _transition_zone_summary, same_zone_ratio = analyze_transition_zones(transition_frame, output_dir)

    sampled_transitions = sample_frame(transitions, sample_frac, max_rows, random_state)
    progress_frame = add_route_progress(sampled_transitions)
    position_frame = merge_to_stop_package_features(progress_frame, packages)
    analyze_time_window_positions(position_frame, output_dir)
    analyze_service_burden_positions(position_frame, output_dir)

    summary = {
        "total_routes": int(len(routes)),
        "usable_routes": int(to_bool(data_quality["can_use_for_training"]).sum()),
        "total_stops": int(len(stops)),
        "total_transitions": int(len(transitions)),
        "route_score_distribution": route_score_distribution.to_dict(orient="records"),
        "same_zone_transition_ratio": same_zone_ratio,
        "percentage_of_stops_with_time_windows": _percentage_time_window_stops(time_window_summary),
        "average_package_count_per_stop": float(pd.to_numeric(packages["package_count"], errors="coerce").mean()),
        "average_service_time_per_stop": float(pd.to_numeric(packages["total_planned_service_time"], errors="coerce").mean()),
    }
    print_final_summary(summary)
    return summary


def _percentage_time_window_stops(time_window_summary: pd.DataFrame) -> float:
    match = time_window_summary[time_window_summary["has_time_window"] == 1]
    if match.empty:
        return 0.0
    return float(match["percentage"].iloc[0])


def print_final_summary(summary: dict[str, object]) -> None:
    print("\nEDA complete.")
    print(f"Total routes: {summary['total_routes']}")
    print(f"Usable routes: {summary['usable_routes']}")
    print(f"Total stops: {summary['total_stops']}")
    print(f"Total transitions: {summary['total_transitions']}")
    print("Route score distribution:")
    for row in summary["route_score_distribution"]:  # type: ignore[index]
        print(f"  {row['route_score']}: {row['count']} ({row['percentage']:.2f}%)")
    print(f"Same-zone transition ratio: {summary['same_zone_transition_ratio']:.4f}")
    print(f"Percentage of stops with time windows: {summary['percentage_of_stops_with_time_windows']:.2f}%")
    print(f"Average package count per stop: {summary['average_package_count_per_stop']:.4f}")
    print(f"Average service time per stop: {summary['average_service_time_per_stop']:.4f}")


def main() -> None:
    args = parse_args()
    run_eda(args.input_dir, args.output_dir, args.sample_frac, args.max_rows, args.random_state)


if __name__ == "__main__":
    main()
