#!/usr/bin/env python3
"""Route-level statistical tests for final route-generation outputs.

This script reads route_metrics_validation.csv and route_metrics_test.csv from
13_route_generation_best_weights.py outputs, then compares routing methods with
paired route-level tests for dissertation reporting.
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

DEFAULT_COLAB_INPUT_DIR = Path(
    "/content/drive/MyDrive/dissertation/amazon_last_mile/"
    "final_experiment_outputs/route_generation_best_weights"
)
DEFAULT_DICC_INPUT_DIR = Path(
    "/home/user/chenziliang/dissertation/amazon_last_mile/"
    "final_experiment_outputs/route_generation_best_weights"
)
DEFAULT_CONFIG = Path("config/config_final.yaml")
VALID_SPLITS = ("validation", "test")
DEFAULT_COMPARISONS = (
    "hybrid_greedy_best_weight:travel_time_nearest_neighbor",
    "hybrid_greedy_best_weight:preference_greedy",
)
DEFAULT_METRICS = (
    "generated_total_travel_time",
    "travel_time_ratio_to_actual",
    "lcs_similarity",
    "position_match_ratio",
    "generated_same_zone_ratio",
    "zone_change_count",
)
LOWER_IS_BETTER = {
    "generated_total_travel_time",
    "travel_time_ratio_to_actual",
    "zone_change_count",
}
HIGHER_IS_BETTER = {
    "lcs_similarity",
    "position_match_ratio",
    "generated_same_zone_ratio",
    "route_valid",
    "valid_route_rate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired statistical tests for route-generation methods."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing route_metrics_validation.csv and route_metrics_test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: {input_dir}/statistical_tests.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=VALID_SPLITS,
        default=list(VALID_SPLITS),
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        default=list(DEFAULT_COMPARISONS),
        help="Method comparisons in METHOD_A:METHOD_B format.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["all"],
        help="Metrics to test, or 'all'.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        warnings.warn("pyyaml not available; config defaults ignored.")
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def resolve_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    p = Path(str(value)).expanduser()
    return p if p.is_absolute() else p.resolve()


def resolve_input_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.input_dir is not None:
        return resolve_path(args.input_dir)  # type: ignore[return-value]
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs"), dict) else {}
    candidates = []
    for key in ("route_generation_best_weights_dir", "route_generation_dir"):
        if outputs.get(key):
            candidates.append(resolve_path(outputs[key]))
    candidates.extend([DEFAULT_COLAB_INPUT_DIR, DEFAULT_DICC_INPUT_DIR])
    for p in candidates:
        if p is not None and p.exists():
            return p
    return DEFAULT_COLAB_INPUT_DIR


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def parse_comparison(text: str) -> tuple[str, str, str]:
    if ":" not in text:
        raise ValueError(f"Comparison must be METHOD_A:METHOD_B, got {text!r}")
    a, b = [part.strip() for part in text.split(":", 1)]
    if not a or not b:
        raise ValueError(f"Invalid comparison: {text!r}")
    return a, b, f"{a}_vs_{b}"


def metric_direction(metric: str) -> str:
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    return "higher_is_better"


def read_split(input_dir: Path, split: str) -> pd.DataFrame:
    path = input_dir / f"route_metrics_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing route metrics: {path}")
    df = pd.read_csv(path)
    required = {"route_id", "method"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    df["route_id"] = df["route_id"].astype(str)
    df["method"] = df["method"].astype(str)
    return df


def select_metrics(df: pd.DataFrame, requested: list[str]) -> list[str]:
    if requested == ["all"]:
        metrics = list(DEFAULT_METRICS)
    else:
        metrics = [m for m in requested if m != "all"]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        warnings.warn(f"Skipping unavailable metrics: {missing}")
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        raise ValueError("No requested metrics are available.")
    return metrics


def descriptive_stats(df: pd.DataFrame, split: str, metrics: list[str]) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method", dropna=False):
        for metric in metrics:
            x = pd.to_numeric(g[metric], errors="coerce").dropna()
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "metric": metric,
                    "n": int(len(x)),
                    "mean": float(x.mean()) if len(x) else math.nan,
                    "std": float(x.std(ddof=1)) if len(x) > 1 else math.nan,
                    "median": float(x.median()) if len(x) else math.nan,
                    "q1": float(x.quantile(0.25)) if len(x) else math.nan,
                    "q3": float(x.quantile(0.75)) if len(x) else math.nan,
                    "min": float(x.min()) if len(x) else math.nan,
                    "max": float(x.max()) if len(x) else math.nan,
                }
            )
    return pd.DataFrame(rows)


def paired_values(df: pd.DataFrame, method_a: str, method_b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["method"].isin([method_a, method_b])].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    grouped = sub.groupby(["route_id", "method"], as_index=False)[metric].mean()
    wide = grouped.pivot(index="route_id", columns="method", values=metric)
    if method_a not in wide.columns:
        wide[method_a] = np.nan
    if method_b not in wide.columns:
        wide[method_b] = np.nan
    wide = wide[[method_a, method_b]].dropna()
    return wide[method_a].to_numpy(float), wide[method_b].to_numpy(float)


def paired_t(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2 or np.isclose(np.std(diff, ddof=1), 0.0):
        return math.nan, math.nan
    try:
        from scipy import stats

        res = stats.ttest_rel(diff, np.zeros_like(diff), nan_policy="omit")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        mean = float(np.mean(diff))
        std = float(np.std(diff, ddof=1))
        t = mean / (std / math.sqrt(len(diff)))
        # Normal approximation fallback.
        cdf = 0.5 * (1.0 + math.erf(abs(t) / math.sqrt(2.0)))
        return t, float(2.0 * (1.0 - cdf))


def wilcoxon(diff: np.ndarray) -> tuple[float, float]:
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return math.nan, math.nan
    if np.allclose(diff, 0.0):
        return 0.0, 1.0
    try:
        from scipy import stats

        res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return math.nan, math.nan


def bootstrap_ci(values: np.ndarray, iterations: int, alpha: float, seed: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0 or iterations <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for i in range(iterations):
        means[i] = float(np.mean(rng.choice(values, size=len(values), replace=True)))
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def preferred(metric: str, mean_a: float, mean_b: float, method_a: str, method_b: str) -> str:
    if not math.isfinite(mean_a) or not math.isfinite(mean_b):
        return "undetermined"
    if math.isclose(mean_a, mean_b):
        return "tie"
    if metric_direction(metric) == "lower_is_better":
        return method_a if mean_a < mean_b else method_b
    return method_a if mean_a > mean_b else method_b


def test_metric(
    df: pd.DataFrame,
    split: str,
    method_a: str,
    method_b: str,
    comparison: str,
    metric: str,
    iterations: int,
    alpha: float,
    seed: int,
) -> dict[str, Any]:
    a, b = paired_values(df, method_a, method_b, metric)
    diff = a - b
    if metric_direction(metric) == "lower_is_better":
        improvement = -diff
    else:
        improvement = diff

    mean_a = float(np.mean(a)) if len(a) else math.nan
    mean_b = float(np.mean(b)) if len(b) else math.nan
    mean_diff = float(np.mean(diff)) if len(diff) else math.nan
    mean_improvement = float(np.mean(improvement)) if len(improvement) else math.nan
    denom = abs(mean_b) if math.isfinite(mean_b) and not math.isclose(mean_b, 0.0) else math.nan
    pct_improvement = float(mean_improvement / denom * 100.0) if math.isfinite(mean_improvement) and math.isfinite(denom) else math.nan
    diff_std = float(np.std(diff, ddof=1)) if len(diff) > 1 else math.nan
    imp_std = float(np.std(improvement, ddof=1)) if len(improvement) > 1 else math.nan
    dz_raw = float(mean_diff / diff_std) if math.isfinite(mean_diff) and math.isfinite(diff_std) and not math.isclose(diff_std, 0.0) else math.nan
    dz_imp = float(mean_improvement / imp_std) if math.isfinite(mean_improvement) and math.isfinite(imp_std) and not math.isclose(imp_std, 0.0) else math.nan
    t_stat, t_p = paired_t(diff)
    w_stat, w_p = wilcoxon(diff)
    ci_low, ci_high = bootstrap_ci(improvement, iterations, alpha, seed)
    pref = preferred(metric, mean_a, mean_b, method_a, method_b)
    interpretation = (
        f"{method_a} {'improves' if mean_improvement > 0 else 'is worse than' if mean_improvement < 0 else 'ties'} "
        f"{method_b} on {metric}; paired t-test "
        f"{'significant' if math.isfinite(t_p) and t_p < alpha else 'not significant' if math.isfinite(t_p) else 'unavailable'}."
    )
    return {
        "split": split,
        "comparison": comparison,
        "method_a": method_a,
        "method_b": method_b,
        "metric": metric,
        "metric_direction": metric_direction(metric),
        "n_pairs": int(len(diff)),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "median_a": float(np.median(a)) if len(a) else math.nan,
        "median_b": float(np.median(b)) if len(b) else math.nan,
        "mean_difference_a_minus_b": mean_diff,
        "median_difference_a_minus_b": float(np.median(diff)) if len(diff) else math.nan,
        "mean_improvement": mean_improvement,
        "percent_improvement_vs_b": pct_improvement,
        "cohens_dz_raw": dz_raw,
        "cohens_dz_improvement": dz_imp,
        "paired_t_statistic": t_stat,
        "paired_t_p_value": t_p,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_value": w_p,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_iterations": iterations,
        "alpha": alpha,
        "paired_t_significant": bool(math.isfinite(t_p) and t_p < alpha),
        "wilcoxon_significant": bool(math.isfinite(w_p) and w_p < alpha),
        "bootstrap_ci_excludes_zero": bool(math.isfinite(ci_low) and math.isfinite(ci_high) and ((ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0))),
        "preferred_method_by_mean": pref,
        "interpretation": interpretation,
    }


def combined_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts = []
    for split, df in frames.items():
        temp = df.copy()
        temp["route_id"] = split + "::" + temp["route_id"].astype(str)
        temp["source_split"] = split
        parts.append(temp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, tuple):
        return [clean_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    input_dir = resolve_input_dir(args, config)
    output_dir = resolve_path(args.output_dir) or input_dir / "statistical_tests"
    assert output_dir is not None
    prepare_output_dir(output_dir, args.overwrite)

    comparisons = [parse_comparison(c) for c in args.comparisons]
    frames: dict[str, pd.DataFrame] = {}
    descriptive_tables = []
    test_rows = []
    metrics_by_split: dict[str, list[str]] = {}

    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {args.splits}")

    for split in args.splits:
        df = read_split(input_dir, split)
        metrics = select_metrics(df, args.metrics)
        for metric in metrics:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
        frames[split] = df
        metrics_by_split[split] = metrics
        descriptive_tables.append(descriptive_stats(df, split, metrics))
        print(f"{split}: rows={len(df)}, routes={df['route_id'].nunique()}, methods={sorted(df['method'].unique())}")
        for method_a, method_b, label in comparisons:
            for metric in metrics:
                test_rows.append(test_metric(df, split, method_a, method_b, label, metric, args.bootstrap_iterations, args.alpha, args.seed))

    if len(frames) > 1:
        all_df = combined_frame(frames)
        metrics = select_metrics(all_df, args.metrics)
        descriptive_tables.append(descriptive_stats(all_df, "all_splits", metrics))
        for method_a, method_b, label in comparisons:
            for metric in metrics:
                test_rows.append(test_metric(all_df, "all_splits", method_a, method_b, label, metric, args.bootstrap_iterations, args.alpha, args.seed + 1000))

    descriptive = pd.concat(descriptive_tables, ignore_index=True)
    tests = pd.DataFrame(test_rows)
    summary_cols = [
        "split", "comparison", "method_a", "method_b", "metric", "metric_direction",
        "n_pairs", "mean_a", "mean_b", "mean_improvement", "percent_improvement_vs_b",
        "paired_t_p_value", "wilcoxon_p_value", "cohens_dz_improvement",
        "preferred_method_by_mean", "interpretation",
    ]
    summary = tests[[c for c in summary_cols if c in tests.columns]].copy()

    descriptive.to_csv(output_dir / "descriptive_stats_by_method.csv", index=False)
    tests.to_csv(output_dir / "paired_statistical_tests.csv", index=False)
    summary.to_csv(output_dir / "pairwise_comparison_summary.csv", index=False)

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": list(args.splits),
        "comparisons": [label for _, _, label in comparisons],
        "metrics_by_split": metrics_by_split,
        "bootstrap_iterations": args.bootstrap_iterations,
        "alpha": args.alpha,
        "seed": args.seed,
        "outputs": {
            "descriptive_stats_by_method": str(output_dir / "descriptive_stats_by_method.csv"),
            "paired_statistical_tests": str(output_dir / "paired_statistical_tests.csv"),
            "pairwise_comparison_summary": str(output_dir / "pairwise_comparison_summary.csv"),
        },
    }
    pd.DataFrame([run_summary]).to_csv(output_dir / "statistical_tests_run_summary.csv", index=False)
    with (output_dir / "statistical_tests_run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(clean_json(run_summary), fh, indent=2)
        fh.write("\n")

    print("\nStatistical tests complete.")
    print(f"Paired test rows: {len(tests)}")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
