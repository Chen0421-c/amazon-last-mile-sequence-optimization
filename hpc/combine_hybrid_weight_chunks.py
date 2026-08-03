#!/usr/bin/env python3
"""Combine chunked hybrid weight-search outputs into final full-validation tables."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


WEIGHT_COLS = [
    "travel_weight",
    "preference_weight",
    "zone_weight",
    "time_window_weight",
    "workload_weight",
]

AVG_METRICS = [
    "actual_total_travel_time",
    "generated_total_travel_time",
    "travel_time_ratio_to_actual",
    "lcs_similarity",
    "position_match_ratio",
    "generated_same_zone_ratio",
    "actual_same_zone_ratio",
    "zone_change_count",
]

WEIGHT_RESULT_COLUMNS = [
    "weight_id",
    *WEIGHT_COLS,
    "route_count",
    "avg_actual_total_travel_time",
    "avg_generated_total_travel_time",
    "avg_travel_time_ratio_to_actual",
    "avg_lcs_similarity",
    "avg_position_match_ratio",
    "avg_generated_same_zone_ratio",
    "avg_actual_same_zone_ratio",
    "avg_zone_change_count",
    "valid_route_rate",
    "travel_score",
    "lcs_score",
    "same_zone_score",
    "position_score",
    "validation_score",
    "rank",
]

ROUTE_METRIC_COLUMNS = [
    "route_id",
    "method",
    "weight_id",
    *WEIGHT_COLS,
    "stop_count",
    "actual_total_travel_time",
    "generated_total_travel_time",
    "travel_time_ratio_to_actual",
    "lcs_similarity",
    "position_match_ratio",
    "generated_same_zone_ratio",
    "actual_same_zone_ratio",
    "zone_change_count",
    "route_valid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine route-level metrics from chunked hybrid weight-search jobs."
    )
    parser.add_argument("--chunk-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-chunks", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def score(values: pd.Series, higher: bool) -> np.ndarray:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
    out = np.zeros(arr.shape)
    mask = np.isfinite(arr)
    if not mask.any():
        return out
    finite = arr[mask]
    lo, hi = float(np.min(finite)), float(np.max(finite))
    if math.isclose(lo, hi):
        out[mask] = 1.0
    elif higher:
        out[mask] = (finite - lo) / (hi - lo)
    else:
        out[mask] = 1.0 - (finite - lo) / (hi - lo)
    return out


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(clean_json(payload), file_obj, indent=2)
        file_obj.write("\n")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite.")
    path.mkdir(parents=True, exist_ok=True)


def chunk_dirs(chunk_root: Path) -> list[Path]:
    return sorted(path for path in chunk_root.glob("chunk_*") if path.is_dir())


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required chunk output: {path}")
    return pd.read_csv(path)


def read_chunk_frames(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    route_metrics = []
    baseline_metrics = []
    run_summaries = []
    for path in paths:
        route_metrics.append(read_required_csv(path / "route_metrics_all_weights.csv"))
        baseline_path = path / "baseline_route_metrics.csv"
        if baseline_path.exists():
            baseline_metrics.append(pd.read_csv(baseline_path))
        summary_path = path / "hybrid_weight_search_run_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            summary.insert(0, "chunk_dir", str(path))
            run_summaries.append(summary)

    route_df = pd.concat(route_metrics, ignore_index=True)
    baseline_df = (
        pd.concat(baseline_metrics, ignore_index=True)
        if baseline_metrics
        else pd.DataFrame(columns=ROUTE_METRIC_COLUMNS)
    )
    summary_df = (
        pd.concat(run_summaries, ignore_index=True) if run_summaries else pd.DataFrame()
    )
    return route_df, baseline_df, summary_df


def build_weight_results(route_df: pd.DataFrame) -> pd.DataFrame:
    route_df = route_df.copy()
    for column in [*WEIGHT_COLS, *AVG_METRICS]:
        route_df[column] = pd.to_numeric(route_df[column], errors="coerce")
    route_df["route_valid_bool"] = bool_series(route_df["route_valid"])

    grouped = route_df.groupby(["weight_id", *WEIGHT_COLS], dropna=False)
    rows = []
    for key, group in grouped:
        row = dict(zip(["weight_id", *WEIGHT_COLS], key))
        row["route_count"] = int(len(group))
        for metric in AVG_METRICS:
            row[f"avg_{metric}"] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        row["valid_route_rate"] = float(group["route_valid_bool"].mean())
        rows.append(row)

    results = pd.DataFrame(rows)
    for column in WEIGHT_RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = np.nan

    results["travel_score"] = score(results["avg_travel_time_ratio_to_actual"], higher=False)
    results["lcs_score"] = score(results["avg_lcs_similarity"], higher=True)
    results["same_zone_score"] = score(results["avg_generated_same_zone_ratio"], higher=True)
    results["position_score"] = score(results["avg_position_match_ratio"], higher=True)
    results["validation_score"] = (
        0.50 * results["travel_score"]
        + 0.25 * results["lcs_score"]
        + 0.20 * results["same_zone_score"]
        + 0.05 * results["position_score"]
    )
    results = results.sort_values(
        [
            "validation_score",
            "avg_travel_time_ratio_to_actual",
            "avg_lcs_similarity",
            "avg_generated_same_zone_ratio",
        ],
        ascending=[False, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    results["rank"] = np.arange(1, len(results) + 1)
    return results.loc[:, WEIGHT_RESULT_COLUMNS]


def summarize_methods(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    rows["route_valid_bool"] = bool_series(rows["route_valid"])
    for metric in AVG_METRICS:
        rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    return (
        rows.groupby("method", dropna=False)
        .agg(
            route_count=("route_id", "count"),
            avg_actual_total_travel_time=("actual_total_travel_time", "mean"),
            avg_generated_total_travel_time=("generated_total_travel_time", "mean"),
            avg_travel_time_ratio_to_actual=("travel_time_ratio_to_actual", "mean"),
            avg_lcs_similarity=("lcs_similarity", "mean"),
            avg_position_match_ratio=("position_match_ratio", "mean"),
            avg_generated_same_zone_ratio=("generated_same_zone_ratio", "mean"),
            avg_actual_same_zone_ratio=("actual_same_zone_ratio", "mean"),
            avg_zone_change_count=("zone_change_count", "mean"),
            valid_route_rate=("route_valid_bool", "mean"),
        )
        .reset_index()
    )


def build_baseline_outputs(
    route_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    best_weight_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_parts = []
    if not baseline_df.empty and "method" in baseline_df.columns:
        keep_methods = {"travel_time_nearest_neighbor", "preference_greedy"}
        baseline_parts.append(baseline_df[baseline_df["method"].isin(keep_methods)].copy())

    best_rows = route_df[route_df["weight_id"].astype(str) == str(best_weight_id)].copy()
    if best_rows.empty:
        raise ValueError(f"No route metrics found for best weight_id={best_weight_id}")
    best_rows["method"] = "hybrid_greedy_best_weight"
    baseline_parts.append(best_rows)

    baseline_route_metrics = pd.concat(baseline_parts, ignore_index=True)
    for column in ROUTE_METRIC_COLUMNS:
        if column not in baseline_route_metrics.columns:
            baseline_route_metrics[column] = np.nan
        if column not in best_rows.columns:
            best_rows[column] = np.nan
    return (
        best_rows.loc[:, ROUTE_METRIC_COLUMNS],
        baseline_route_metrics.loc[:, ROUTE_METRIC_COLUMNS],
        summarize_methods(baseline_route_metrics),
    )


def main() -> int:
    args = parse_args()
    chunks = chunk_dirs(args.chunk_root)
    if not chunks:
        raise FileNotFoundError(f"No chunk_* output directories found in {args.chunk_root}")
    if args.expected_chunks is not None and len(chunks) != args.expected_chunks:
        raise ValueError(
            f"Expected {args.expected_chunks} chunk directories, found {len(chunks)}"
        )

    prepare_output_dir(args.output_dir, args.overwrite)
    route_df, baseline_df, chunk_summary = read_chunk_frames(chunks)
    if route_df.empty:
        raise ValueError("No route-level weight metrics were found.")

    weight_results = build_weight_results(route_df)
    best = weight_results.iloc[0].to_dict()
    best_weight_id = str(best["weight_id"])
    best_route_metrics, baseline_route_metrics, baseline_summary = build_baseline_outputs(
        route_df, baseline_df, best_weight_id
    )

    weight_results.to_csv(args.output_dir / "weight_grid_results.csv", index=False)
    pd.DataFrame([best]).to_csv(args.output_dir / "best_weight_summary.csv", index=False)
    write_json(args.output_dir / "best_weight_summary.json", best)
    best_route_metrics.to_csv(args.output_dir / "best_weight_route_metrics.csv", index=False)
    baseline_route_metrics.to_csv(args.output_dir / "baseline_route_metrics.csv", index=False)
    baseline_summary.to_csv(args.output_dir / "baseline_method_summary.csv", index=False)
    if not chunk_summary.empty:
        chunk_summary.to_csv(args.output_dir / "chunk_run_summaries.csv", index=False)

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "chunk_root": str(args.chunk_root),
        "output_dir": str(args.output_dir),
        "chunk_count": len(chunks),
        "route_count_processed": int(best_route_metrics["route_id"].nunique()),
        "weight_combinations_tested": int(weight_results["weight_id"].nunique()),
        "best_weight_id": best_weight_id,
        "combined_from_route_level_metrics": True,
    }
    pd.DataFrame([run_summary]).to_csv(
        args.output_dir / "hybrid_weight_search_run_summary.csv", index=False
    )
    write_json(args.output_dir / "hybrid_weight_search_run_summary.json", run_summary)

    print("Combined hybrid weight search chunks.")
    print(f"Chunk directories: {len(chunks)}")
    print(f"Routes processed: {run_summary['route_count_processed']}")
    print(f"Weights tested: {run_summary['weight_combinations_tested']}")
    print(f"Best weight_id: {best_weight_id}")
    print(f"Output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
