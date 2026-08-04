#!/usr/bin/env python3
"""Ablation and sensitivity analysis for the hybrid routing experiment.

This script analyses outputs already produced by:
- 12_optimize_hybrid_weights.py
- 13_route_generation_best_weights.py
- 15_statistical_tests.py, optionally

It does not regenerate routes. It creates dissertation-ready evidence about the
contribution of hybrid cost components and the trade-off between travel-time
routing and driver-like routing behaviour.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CONFIG = Path("config/config_final.yaml")
DEFAULT_BASE = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_OUTPUTS = DEFAULT_BASE / "final_experiment_outputs"
DEFAULT_WEIGHT_SEARCH_DIR = DEFAULT_OUTPUTS / "hybrid_weight_search_validation_full_300w"
DEFAULT_ROUTE_GENERATION_DIR = DEFAULT_OUTPUTS / "route_generation_best_weights"

WEIGHT_COLS = [
    "travel_weight",
    "preference_weight",
    "zone_weight",
    "time_window_weight",
    "workload_weight",
]

KEY_WEIGHT_METRICS = [
    "avg_travel_time_ratio_to_actual",
    "avg_lcs_similarity",
    "avg_position_match_ratio",
    "avg_generated_same_zone_ratio",
    "avg_zone_change_count",
    "validation_score",
    "rank",
]

LOWER_IS_BETTER = {
    "avg_travel_time_ratio_to_actual",
    "avg_zone_change_count",
    "travel_time_ratio_to_actual",
    "zone_change_count",
    "generated_total_travel_time",
}

ROUTE_SUMMARY_METRICS = [
    "avg_travel_time_ratio_to_actual",
    "avg_lcs_similarity",
    "avg_position_match_ratio",
    "avg_generated_same_zone_ratio",
    "avg_zone_change_count",
]

P1_PRELIMINARY_WEIGHTS = {
    "travel_weight": 0.35,
    "preference_weight": 0.40,
    "zone_weight": 0.15,
    "time_window_weight": 0.05,
    "workload_weight": 0.05,
}

DIAGNOSTIC_TARGETS = {
    "travel_only": {"travel_weight": 1.0, "preference_weight": 0.0, "zone_weight": 0.0, "time_window_weight": 0.0, "workload_weight": 0.0},
    "preference_only": {"travel_weight": 0.0, "preference_weight": 1.0, "zone_weight": 0.0, "time_window_weight": 0.0, "workload_weight": 0.0},
    "zone_only": {"travel_weight": 0.0, "preference_weight": 0.0, "zone_weight": 1.0, "time_window_weight": 0.0, "workload_weight": 0.0},
    "time_window_only": {"travel_weight": 0.0, "preference_weight": 0.0, "zone_weight": 0.0, "time_window_weight": 1.0, "workload_weight": 0.0},
    "workload_only": {"travel_weight": 0.0, "preference_weight": 0.0, "zone_weight": 0.0, "time_window_weight": 0.0, "workload_weight": 1.0},
    "balanced_all_components": {"travel_weight": 0.20, "preference_weight": 0.20, "zone_weight": 0.20, "time_window_weight": 0.20, "workload_weight": 0.20},
    "p1_preliminary_all_components": P1_PRELIMINARY_WEIGHTS,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse hybrid routing ablations and sensitivity results.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weight-search-dir", type=Path, default=None)
    parser.add_argument("--route-generation-dir", type=Path, default=None)
    parser.add_argument("--statistical-tests-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-nearest-target",
        action="store_true",
        help="Allow nearest weight-grid row if an exact diagnostic target is absent.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Please install pyyaml: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as file_obj:
        config = yaml.safe_load(file_obj) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config


def resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else path.resolve()


def resolve_paths(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Path | None]:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs"), dict) else {}
    route_dir = resolve_path(args.route_generation_dir or outputs.get("route_generation_best_weights_dir", DEFAULT_ROUTE_GENERATION_DIR))
    weight_dir = resolve_path(args.weight_search_dir or outputs.get("hybrid_weight_search_dir", DEFAULT_WEIGHT_SEARCH_DIR))
    stats_dir = resolve_path(args.statistical_tests_dir) if args.statistical_tests_dir else None
    if stats_dir is None:
        candidate = route_dir / "statistical_tests"
        if candidate.exists():
            stats_dir = candidate
    output_dir = resolve_path(args.output_dir or route_dir / "ablation_study")
    return {"weight_search_dir": weight_dir, "route_generation_dir": route_dir, "statistical_tests_dir": stats_dir, "output_dir": output_dir}


def prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def read_csv_optional(path: Path, label: str) -> pd.DataFrame | None:
    if not path.exists():
        warnings.warn(f"Optional {label} not found: {path}")
        return None
    return pd.read_csv(path)


def normalize_weight_dict(weights: dict[str, float]) -> dict[str, float]:
    return {col: round(float(weights.get(col, 0.0)), 10) for col in WEIGHT_COLS}


def weight_distance(row: pd.Series, target: dict[str, float]) -> tuple[float, float]:
    distances = [abs(float(row[col]) - float(target[col])) for col in WEIGHT_COLS]
    return float(sum(distances)), float(max(distances))


def find_weight_row(grid: pd.DataFrame, target_name: str, target: dict[str, float], allow_nearest: bool) -> dict[str, Any]:
    target = normalize_weight_dict(target)
    working = grid.copy()
    working["target_l1_distance"] = working.apply(lambda row: weight_distance(row, target)[0], axis=1)
    working["target_max_abs_distance"] = working.apply(lambda row: weight_distance(row, target)[1], axis=1)
    exact = working[np.isclose(working["target_l1_distance"], 0.0)]
    if exact.empty:
        if not allow_nearest:
            return {"target_name": target_name, "matched": False, "match_type": "missing_exact_target", **{f"target_{k}": v for k, v in target.items()}}
        selected = working.sort_values(["target_l1_distance", "target_max_abs_distance", "rank"], ascending=[True, True, True], kind="mergesort").iloc[0]
        match_type = "nearest_weight_in_grid"
    else:
        selected = exact.sort_values("rank", kind="mergesort").iloc[0]
        match_type = "exact_weight_in_grid"
    out: dict[str, Any] = {"target_name": target_name, "matched": True, "match_type": match_type}
    for col in WEIGHT_COLS:
        out[f"target_{col}"] = target[col]
        out[col] = float(selected[col])
    out["target_l1_distance"] = float(selected["target_l1_distance"])
    out["target_max_abs_distance"] = float(selected["target_max_abs_distance"])
    for col in ["weight_id", *KEY_WEIGHT_METRICS, "route_count", "valid_route_rate"]:
        if col in selected.index:
            out[col] = selected[col]
    return out


def redistribute_without(best: dict[str, float], component: str) -> dict[str, float]:
    out = dict(best)
    removed = float(out.get(component, 0.0))
    out[component] = 0.0
    remaining = [col for col in WEIGHT_COLS if col != component]
    remaining_sum = sum(float(out[col]) for col in remaining)
    if removed > 0 and remaining_sum > 0:
        for col in remaining:
            out[col] = float(out[col]) + removed * float(out[col]) / remaining_sum
    elif removed > 0:
        out["travel_weight"] = 1.0
    return normalize_weight_dict(out)


def build_ablation_targets(best: dict[str, float]) -> dict[str, dict[str, float]]:
    targets = dict(DIAGNOSTIC_TARGETS)
    targets["validation_selected_best"] = normalize_weight_dict(best)
    for component in WEIGHT_COLS:
        targets[f"selected_without_{component.replace('_weight', '')}"] = redistribute_without(best, component)
    return targets


def make_ablation_candidates(weight_grid: pd.DataFrame, best_summary: pd.DataFrame, allow_nearest: bool) -> pd.DataFrame:
    if best_summary.empty:
        raise ValueError("best_weight_summary.csv is empty.")
    best = {col: float(best_summary.iloc[0][col]) for col in WEIGHT_COLS}
    rows = [find_weight_row(weight_grid, name, target, allow_nearest) for name, target in build_ablation_targets(best).items()]
    df = pd.DataFrame(rows)
    if "rank" in df.columns:
        df = df.sort_values(["matched", "rank", "target_name"], ascending=[False, True, True], kind="mergesort")
    return df.reset_index(drop=True)


def component_family_summary(weight_grid: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "travel_positive": weight_grid["travel_weight"] > 0,
        "preference_positive": weight_grid["preference_weight"] > 0,
        "zone_positive": weight_grid["zone_weight"] > 0,
        "time_window_positive": weight_grid["time_window_weight"] > 0,
        "workload_positive": weight_grid["workload_weight"] > 0,
        "time_window_zero": weight_grid["time_window_weight"] == 0,
        "workload_zero": weight_grid["workload_weight"] == 0,
        "time_window_and_workload_zero": (weight_grid["time_window_weight"] == 0) & (weight_grid["workload_weight"] == 0),
        "all_five_components_positive": (weight_grid[WEIGHT_COLS] > 0).all(axis=1),
    }
    metric_cols = [col for col in KEY_WEIGHT_METRICS if col in weight_grid.columns]
    rows: list[dict[str, Any]] = []
    for family, mask in masks.items():
        subset = weight_grid[mask]
        row: dict[str, Any] = {"component_family": family, "candidate_count": int(len(subset))}
        if not subset.empty:
            best = subset.sort_values("rank", kind="mergesort").iloc[0]
            row["best_weight_id"] = best.get("weight_id", "")
            row["best_rank"] = best.get("rank", np.nan)
            row["best_validation_score"] = best.get("validation_score", np.nan)
            for col in WEIGHT_COLS:
                row[f"best_{col}"] = float(best[col])
            for metric in metric_cols:
                row[f"mean_{metric}"] = pd.to_numeric(subset[metric], errors="coerce").mean()
                row[f"median_{metric}"] = pd.to_numeric(subset[metric], errors="coerce").median()
        rows.append(row)
    return pd.DataFrame(rows)


def component_weight_correlation(weight_grid: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in KEY_WEIGHT_METRICS if col in weight_grid.columns]
    rows = []
    for weight_col in WEIGHT_COLS:
        for metric in metric_cols:
            x = pd.to_numeric(weight_grid[weight_col], errors="coerce")
            y = pd.to_numeric(weight_grid[metric], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3:
                pearson = np.nan
                spearman = np.nan
            else:
                pearson = x[valid].corr(y[valid], method="pearson")
                spearman = x[valid].corr(y[valid], method="spearman")
            rows.append(
                {
                    "weight_component": weight_col,
                    "metric": metric,
                    "metric_direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                    "n": int(valid.sum()),
                    "pearson_correlation": pearson,
                    "spearman_correlation": spearman,
                }
            )
    return pd.DataFrame(rows)


def route_generation_tradeoff(route_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if route_summary.empty:
        return pd.DataFrame(rows)
    for split, split_df in route_summary.groupby("split", dropna=False):
        methods = {row["method"]: row for _, row in split_df.iterrows()}
        hybrid = methods.get("hybrid_greedy_best_weight")
        if hybrid is None:
            continue
        for baseline_name in ["travel_time_nearest_neighbor", "preference_greedy"]:
            baseline = methods.get(baseline_name)
            if baseline is None:
                continue
            for metric in ROUTE_SUMMARY_METRICS:
                if metric not in hybrid.index or metric not in baseline.index:
                    continue
                hybrid_value = float(hybrid[metric])
                baseline_value = float(baseline[metric])
                if metric in LOWER_IS_BETTER:
                    improvement = baseline_value - hybrid_value
                else:
                    improvement = hybrid_value - baseline_value
                preferred = "hybrid_greedy_best_weight" if improvement > 0 else baseline_name
                percent = improvement / abs(baseline_value) * 100 if baseline_value != 0 else np.nan
                rows.append(
                    {
                        "split": split,
                        "comparison": f"hybrid_greedy_best_weight_vs_{baseline_name}",
                        "metric": metric,
                        "metric_direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                        "hybrid_value": hybrid_value,
                        "baseline_method": baseline_name,
                        "baseline_value": baseline_value,
                        "hybrid_improvement_vs_baseline": improvement,
                        "hybrid_percent_improvement_vs_baseline": percent,
                        "preferred_method_by_mean": preferred,
                    }
                )
    return pd.DataFrame(rows)


def load_statistical_summary(stats_dir: Path | None) -> pd.DataFrame | None:
    if stats_dir is None:
        return None
    return read_csv_optional(stats_dir / "pairwise_comparison_summary.csv", "pairwise_comparison_summary.csv")


def interpretation_summary(best_summary: pd.DataFrame, candidates: pd.DataFrame, tradeoff: pd.DataFrame, stats: pd.DataFrame | None) -> pd.DataFrame:
    best = best_summary.iloc[0]
    positive = [col for col in WEIGHT_COLS if float(best[col]) > 0]
    zero = [col for col in WEIGHT_COLS if math.isclose(float(best[col]), 0.0)]
    rows = [
        {
            "finding_id": "selected_weight_structure",
            "finding": f"The validation-selected weight uses positive terms for {', '.join(positive)} and zero terms for {', '.join(zero) or 'none'}.",
            "evidence": "best_weight_summary.csv",
        }
    ]
    p1 = candidates[candidates["target_name"] == "p1_preliminary_all_components"]
    if not p1.empty and bool(p1.iloc[0].get("matched", False)):
        rows.append(
            {
                "finding_id": "p1_preliminary_competitiveness",
                "finding": f"The P1 preliminary all-component setting remains a useful sensitivity baseline; it matched weight_id={p1.iloc[0].get('weight_id')} with rank={p1.iloc[0].get('rank')} and validation_score={p1.iloc[0].get('validation_score')}.",
                "evidence": "ablation_weight_candidates.csv",
            }
        )
    rows.append(
        {
            "finding_id": "zero_proxy_terms_interpretation",
            "finding": "Zero final weights for time-window and workload should be interpreted as limited additional value of the simplified proxy terms under the current validation objective, not as evidence that these operational constraints are theoretically irrelevant.",
            "evidence": "best_weight_summary.csv; component_family_summary.csv",
        }
    )
    if not tradeoff.empty:
        rows.append(
            {
                "finding_id": "hybrid_tradeoff",
                "finding": "The route-generation result should be framed as a trade-off: hybrid routing is expected to be close to the travel-time baseline while improving driver-like sequence and zone-continuity metrics.",
                "evidence": "route_generation_method_tradeoff_summary.csv",
            }
        )
    if stats is not None and not stats.empty:
        rows.append(
            {
                "finding_id": "statistical_support",
                "finding": "Paired route-level statistical tests are available and should be used to report which trade-offs are statistically supported.",
                "evidence": "pairwise_comparison_summary.csv",
            }
        )
    return pd.DataFrame(rows)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(json_safe(payload), file_obj, indent=2)
        file_obj.write("\n")


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args, load_config(args.config))
    weight_dir = paths["weight_search_dir"]
    route_dir = paths["route_generation_dir"]
    stats_dir = paths["statistical_tests_dir"]
    output_dir = paths["output_dir"]
    assert isinstance(weight_dir, Path)
    assert isinstance(route_dir, Path)
    assert isinstance(output_dir, Path)

    prepare_output(output_dir, args.overwrite)

    best_summary = read_csv_required(weight_dir / "best_weight_summary.csv", "best_weight_summary.csv")
    weight_grid = read_csv_required(weight_dir / "weight_grid_results.csv", "weight_grid_results.csv")
    route_summary = read_csv_required(route_dir / "method_summary_all_splits.csv", "method_summary_all_splits.csv")
    stats_summary = load_statistical_summary(stats_dir if isinstance(stats_dir, Path) else None)

    missing_cols = [col for col in WEIGHT_COLS if col not in weight_grid.columns]
    if missing_cols:
        raise ValueError(f"weight_grid_results.csv is missing columns: {missing_cols}")

    candidates = make_ablation_candidates(weight_grid, best_summary, args.allow_nearest_target)
    family = component_family_summary(weight_grid)
    corr = component_weight_correlation(weight_grid)
    tradeoff = route_generation_tradeoff(route_summary)
    interpretation = interpretation_summary(best_summary, candidates, tradeoff, stats_summary)

    candidates.to_csv(output_dir / "ablation_weight_candidates.csv", index=False)
    family.to_csv(output_dir / "component_family_summary.csv", index=False)
    corr.to_csv(output_dir / "component_weight_correlation.csv", index=False)
    tradeoff.to_csv(output_dir / "route_generation_method_tradeoff_summary.csv", index=False)
    interpretation.to_csv(output_dir / "ablation_interpretation_summary.csv", index=False)
    write_json(output_dir / "ablation_interpretation_summary.json", {"findings": interpretation.to_dict(orient="records")})

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "weight_search_dir": str(weight_dir),
        "route_generation_dir": str(route_dir),
        "statistical_tests_dir": str(stats_dir) if isinstance(stats_dir, Path) else None,
        "output_dir": str(output_dir),
        "allow_nearest_target": args.allow_nearest_target,
        "weight_grid_rows": int(len(weight_grid)),
        "ablation_candidates": int(len(candidates)),
        "matched_ablation_candidates": int(candidates["matched"].sum()) if "matched" in candidates.columns else 0,
        "component_family_rows": int(len(family)),
        "correlation_rows": int(len(corr)),
        "tradeoff_rows": int(len(tradeoff)),
    }
    pd.DataFrame([run_summary]).to_csv(output_dir / "ablation_study_run_summary.csv", index=False)
    write_json(output_dir / "ablation_study_run_summary.json", run_summary)

    print("Ablation study analysis complete.")
    print(f"Weight search directory: {weight_dir}")
    print(f"Route generation directory: {route_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Ablation candidates: {len(candidates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
