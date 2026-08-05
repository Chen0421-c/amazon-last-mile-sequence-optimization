#!/usr/bin/env python3
"""Generate dissertation-ready final tables and figures.

This script consolidates final experiment outputs from the preference-learning,
subgroup-analysis, hybrid-weight-search, route-generation, statistical-test,
and ablation-study stages. It does not rerun experiments. It only reads already
produced CSV/JSON files and writes cleaned summary tables plus lightweight
figures for Chapter 4/5 reporting and defence slides.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - environment dependent
    plt = None
    _MATPLOTLIB_IMPORT_ERROR = exc
else:
    _MATPLOTLIB_IMPORT_ERROR = None

DEFAULT_BASE_DIR = Path(
    "/content/drive/MyDrive/dissertation/amazon_last_mile/final_experiment_outputs"
)

MODEL_NAMES = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}

METHOD_NAMES = {
    "hybrid_greedy_best_weight": "Hybrid greedy",
    "travel_time_nearest_neighbor": "Travel-time NN",
    "preference_greedy": "Preference greedy",
}

METRIC_LABELS = {
    "sampled_top1_accuracy": "Sampled top-1 accuracy",
    "sampled_top3_accuracy": "Sampled top-3 accuracy",
    "sampled_top5_accuracy": "Sampled top-5 accuracy",
    "roc_auc": "ROC-AUC",
    "average_precision": "Average precision",
    "f1": "F1",
    "log_loss": "Log loss",
    "avg_travel_time_ratio_to_actual": "Travel-time ratio to actual",
    "avg_lcs_similarity": "LCS similarity",
    "avg_position_match_ratio": "Position-match ratio",
    "avg_generated_same_zone_ratio": "Generated same-zone ratio",
    "avg_zone_change_count": "Zone-change count",
    "generated_total_travel_time": "Generated travel time",
    "travel_time_ratio_to_actual": "Travel-time ratio to actual",
    "lcs_similarity": "LCS similarity",
    "position_match_ratio": "Position-match ratio",
    "generated_same_zone_ratio": "Generated same-zone ratio",
    "zone_change_count": "Zone-change count",
    "validation_score": "Validation score",
}

WEIGHT_COLUMNS = [
    "travel_weight",
    "preference_weight",
    "zone_weight",
    "time_window_weight",
    "workload_weight",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate final experiment outputs into dissertation-ready tables and figures."
    )
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--subgroup-dir", type=Path, default=None)
    parser.add_argument("--weight-search-dir", type=Path, default=None)
    parser.add_argument("--route-generation-dir", type=Path, default=None)
    parser.add_argument("--statistical-tests-dir", type=Path, default=None)
    parser.add_argument("--ablation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fig-format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--top-n-weights", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    base = args.base_dir.expanduser().resolve()
    route_dir = args.route_generation_dir or base / "route_generation_best_weights"
    paths = {
        "base_dir": base,
        "model_dir": args.model_dir or base / "model_outputs_full_top3",
        "subgroup_dir": args.subgroup_dir or base / "model_subgroup_analysis_full_top3",
        "weight_search_dir": args.weight_search_dir
        or base / "hybrid_weight_search_validation_full_300w",
        "route_generation_dir": route_dir,
        "statistical_tests_dir": args.statistical_tests_dir
        or route_dir / "statistical_tests",
        "ablation_dir": args.ablation_dir or base / "ablation_study",
        "output_dir": args.output_dir or base / "final_tables_figures",
    }
    return {name: path.expanduser().resolve() for name, path in paths.items()}


def prepare_output_dir(path: Path, overwrite: bool) -> dict[str, Path]:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    subdirs = {
        "root": path,
        "tables_csv": path / "tables_csv",
        "tables_md": path / "tables_md",
        "figures": path / "figures",
        "narrative": path / "narrative",
    }
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    return subdirs


def read_csv_optional(path: Path, label: str, required: bool = False) -> pd.DataFrame | None:
    if not path.exists():
        message = f"{label} not found: {path}"
        if required:
            raise FileNotFoundError(message)
        warnings.warn(message)
        return None
    return pd.read_csv(path)


def safe_round_dataframe(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].round(decimals)
    return out


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    shown = df if max_rows is None else df.head(max_rows)
    try:
        return shown.to_markdown(index=False)
    except Exception:
        return shown.to_csv(index=False)


def save_table(
    df: pd.DataFrame,
    name: str,
    output_dirs: dict[str, Path],
    manifest: list[dict[str, Any]],
    decimals: int = 4,
) -> pd.DataFrame:
    clean = safe_round_dataframe(df, decimals=decimals)
    csv_path = output_dirs["tables_csv"] / f"{name}.csv"
    md_path = output_dirs["tables_md"] / f"{name}.md"
    clean.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(clean) + "\n", encoding="utf-8")
    manifest.append(
        {
            "artifact_type": "table",
            "name": name,
            "csv_path": str(csv_path),
            "markdown_path": str(md_path),
            "rows": int(len(clean)),
            "columns": int(len(clean.columns)),
        }
    )
    return clean


def save_json(path: Path, payload: Any) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert(value) for value in obj]
        if isinstance(obj, tuple):
            return [convert(value) for value in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            value = float(obj)
            return value if math.isfinite(value) else None
        if isinstance(obj, float):
            return obj if math.isfinite(obj) else None
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    path.write_text(json.dumps(convert(payload), indent=2) + "\n", encoding="utf-8")


def require_matplotlib(skip: bool) -> bool:
    if skip:
        return False
    if plt is None:
        warnings.warn(f"matplotlib is unavailable; figures skipped: {_MATPLOTLIB_IMPORT_ERROR}")
        return False
    return True


def finish_figure(path: Path, dpi: int, manifest: list[dict[str, Any]], name: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    manifest.append({"artifact_type": "figure", "name": name, "path": str(path)})


def prettify_method(series: pd.Series) -> pd.Series:
    return series.map(lambda value: METHOD_NAMES.get(str(value), str(value)))


def prettify_model(series: pd.Series) -> pd.Series:
    return series.map(lambda value: MODEL_NAMES.get(str(value), str(value)))


def build_model_comparison(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> pd.DataFrame | None:
    val = read_csv_optional(paths["model_dir"] / "model_comparison_validation.csv", "model comparison validation")
    test = read_csv_optional(paths["model_dir"] / "model_comparison_test.csv", "model comparison test")
    frames = []
    for split, df in (("validation", val), ("test", test)):
        if df is not None and not df.empty:
            tmp = df.copy()
            tmp.insert(0, "split", split)
            tmp["model_display"] = prettify_model(tmp["model_name"])
            frames.append(tmp)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    preferred_cols = [
        "split",
        "model_name",
        "model_display",
        "sampled_top1_accuracy",
        "sampled_top3_accuracy",
        "sampled_top5_accuracy",
        "roc_auc",
        "average_precision",
        "f1",
        "log_loss",
    ]
    cols = [col for col in preferred_cols if col in combined.columns]
    table = combined.loc[:, cols].sort_values(["split", "sampled_top1_accuracy"], ascending=[True, False])
    return save_table(table, "table_01_model_comparison", out_dirs, manifest)


def build_best_weight_table(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> pd.DataFrame | None:
    best = read_csv_optional(paths["weight_search_dir"] / "best_weight_summary.csv", "best weight summary")
    used = read_csv_optional(paths["route_generation_dir"] / "best_weight_used.csv", "best weight used")
    rows = []
    if best is not None and not best.empty:
        row = best.iloc[0].to_dict()
        row["source"] = "validation_weight_search"
        rows.append(row)
    if used is not None and not used.empty:
        row = used.iloc[0].to_dict()
        row["source"] = "route_generation_used"
        rows.append(row)
    if not rows:
        return None
    table = pd.DataFrame(rows)
    cols = ["source", "weight_id", *WEIGHT_COLUMNS, "validation_score", "rank", "route_count"]
    cols = [col for col in cols if col in table.columns]
    return save_table(table.loc[:, cols], "table_02_best_hybrid_weights", out_dirs, manifest)


def build_route_summary_table(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> pd.DataFrame | None:
    summary = read_csv_optional(paths["route_generation_dir"] / "method_summary_all_splits.csv", "route method summary")
    if summary is None or summary.empty:
        return None
    table = summary.copy()
    table["method_display"] = prettify_method(table["method"])
    preferred_cols = [
        "split",
        "method",
        "method_display",
        "route_count",
        "avg_travel_time_ratio_to_actual",
        "avg_lcs_similarity",
        "avg_position_match_ratio",
        "avg_generated_same_zone_ratio",
        "avg_actual_same_zone_ratio",
        "avg_zone_change_count",
        "valid_route_rate",
    ]
    cols = [col for col in preferred_cols if col in table.columns]
    return save_table(table.loc[:, cols], "table_03_route_generation_summary", out_dirs, manifest)


def build_statistical_summary(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> pd.DataFrame | None:
    stats = read_csv_optional(paths["statistical_tests_dir"] / "pairwise_comparison_summary.csv", "pairwise comparison summary")
    if stats is None or stats.empty:
        return None
    key_metrics = [
        "travel_time_ratio_to_actual",
        "lcs_similarity",
        "position_match_ratio",
        "generated_same_zone_ratio",
        "zone_change_count",
    ]
    table = stats[stats["metric"].isin(key_metrics)].copy()
    if table.empty:
        table = stats.copy()
    table["method_a_display"] = prettify_method(table["method_a"])
    table["method_b_display"] = prettify_method(table["method_b"])
    table["metric_display"] = table["metric"].map(lambda x: METRIC_LABELS.get(str(x), str(x)))
    cols = [
        "split",
        "comparison",
        "method_a_display",
        "method_b_display",
        "metric",
        "metric_display",
        "metric_direction",
        "n_pairs",
        "mean_a",
        "mean_b",
        "mean_improvement",
        "percent_improvement_vs_b",
        "paired_t_p_value",
        "wilcoxon_p_value",
        "cohens_dz_improvement",
        "preferred_method_by_mean",
    ]
    cols = [col for col in cols if col in table.columns]
    return save_table(table.loc[:, cols], "table_04_statistical_tests_summary", out_dirs, manifest)


def build_ablation_tables(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    candidates = read_csv_optional(paths["ablation_dir"] / "ablation_weight_candidates.csv", "ablation candidates")
    if candidates is not None and not candidates.empty:
        cols = [
            "candidate_label",
            "weight_id",
            *WEIGHT_COLUMNS,
            "rank",
            "validation_score",
            "avg_travel_time_ratio_to_actual",
            "avg_lcs_similarity",
            "avg_generated_same_zone_ratio",
            "avg_zone_change_count",
        ]
        cols = [col for col in cols if col in candidates.columns]
        outputs["candidates"] = save_table(
            candidates.loc[:, cols], "table_05_ablation_weight_candidates", out_dirs, manifest
        )
    family = read_csv_optional(paths["ablation_dir"] / "component_family_summary.csv", "component family summary")
    if family is not None and not family.empty:
        outputs["family"] = save_table(family, "table_06_component_family_summary", out_dirs, manifest)
    interpretation = read_csv_optional(paths["ablation_dir"] / "ablation_interpretation_summary.csv", "ablation interpretation summary")
    if interpretation is not None and not interpretation.empty:
        outputs["interpretation"] = save_table(
            interpretation, "table_07_ablation_interpretation", out_dirs, manifest
        )
    return outputs


def build_subgroup_tables(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    overall = read_csv_optional(paths["subgroup_dir"] / "overall_metrics.csv", "subgroup overall metrics")
    if overall is not None and not overall.empty:
        outputs["overall"] = save_table(overall, "table_08_preference_overall_metrics", out_dirs, manifest)
    subgroup_specs = {
        "route_progress": "subgroup_route_progress_bin.csv",
        "transition_zone": "subgroup_transition_zone_type.csv",
        "time_window": "subgroup_candidate_time_window_group.csv",
        "route_score": "subgroup_route_score.csv",
    }
    for key, filename in subgroup_specs.items():
        df = read_csv_optional(paths["subgroup_dir"] / filename, filename)
        if df is not None and not df.empty:
            outputs[key] = save_table(df, f"table_09_subgroup_{key}", out_dirs, manifest)
    return outputs


def build_feature_importance(paths: dict[str, Path], out_dirs: dict[str, Path], manifest: list[dict[str, Any]]) -> pd.DataFrame | None:
    fi = read_csv_optional(paths["model_dir"] / "feature_importance.csv", "feature importance")
    if fi is None or fi.empty:
        return None
    table = fi.copy()
    if "model_name" in table.columns:
        # Prefer the selected CatBoost model if present, otherwise keep top values overall.
        cat = table[table["model_name"].astype(str).str.lower() == "catboost"]
        if not cat.empty:
            table = cat
    sort_col = "importance" if "importance" in table.columns else None
    if sort_col:
        table = table.sort_values(sort_col, ascending=False)
    return save_table(table.head(20), "table_10_top_feature_importance", out_dirs, manifest)


def plot_model_comparison(table: pd.DataFrame | None, out_dirs: dict[str, Path], manifest: list[dict[str, Any]], fmt: str, dpi: int) -> None:
    if table is None or table.empty or "sampled_top1_accuracy" not in table.columns:
        return
    pivot = table.pivot_table(
        index="model_display", columns="split", values="sampled_top1_accuracy", aggfunc="first"
    )
    if pivot.empty:
        return
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_ylabel("Sampled top-1 accuracy")
    ax.set_xlabel("Model")
    ax.set_title("Full-scale preference model comparison")
    ax.legend(title="Split")
    finish_figure(out_dirs["figures"] / f"fig_01_model_top1.{fmt}", dpi, manifest, "fig_01_model_top1")


def plot_best_weights(table: pd.DataFrame | None, out_dirs: dict[str, Path], manifest: list[dict[str, Any]], fmt: str, dpi: int) -> None:
    if table is None or table.empty:
        return
    row = table.iloc[0]
    available = [col for col in WEIGHT_COLUMNS if col in table.columns]
    if not available:
        return
    values = [float(row[col]) for col in available]
    labels = [col.replace("_weight", "").replace("_", " ").title() for col in available]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    ax.set_ylabel("Weight")
    ax.set_title("Validation-selected hybrid cost weights")
    ax.tick_params(axis="x", rotation=25)
    finish_figure(out_dirs["figures"] / f"fig_02_best_weights.{fmt}", dpi, manifest, "fig_02_best_weights")


def plot_route_tradeoff(table: pd.DataFrame | None, out_dirs: dict[str, Path], manifest: list[dict[str, Any]], fmt: str, dpi: int) -> None:
    if table is None or table.empty:
        return
    needed = {"avg_travel_time_ratio_to_actual", "avg_generated_same_zone_ratio", "method_display"}
    if not needed.issubset(table.columns):
        return
    df = table[table.get("split", "") == "test"].copy()
    if df.empty:
        df = table.copy()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["avg_travel_time_ratio_to_actual"], df["avg_generated_same_zone_ratio"])
    for _, row in df.iterrows():
        ax.annotate(
            str(row["method_display"]),
            (row["avg_travel_time_ratio_to_actual"], row["avg_generated_same_zone_ratio"]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlabel("Travel-time ratio to actual")
    ax.set_ylabel("Generated same-zone ratio")
    ax.set_title("Route-generation trade-off on test routes")
    finish_figure(out_dirs["figures"] / f"fig_03_route_tradeoff_test.{fmt}", dpi, manifest, "fig_03_route_tradeoff_test")


def plot_zone_changes(table: pd.DataFrame | None, out_dirs: dict[str, Path], manifest: list[dict[str, Any]], fmt: str, dpi: int) -> None:
    if table is None or table.empty or "avg_zone_change_count" not in table.columns:
        return
    df = table[table.get("split", "") == "test"].copy()
    if df.empty:
        df = table.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df["method_display"], df["avg_zone_change_count"])
    ax.set_ylabel("Average zone-change count")
    ax.set_xlabel("Method")
    ax.set_title("Zone changes by route-generation method")
    ax.tick_params(axis="x", rotation=25)
    finish_figure(out_dirs["figures"] / f"fig_04_zone_changes_test.{fmt}", dpi, manifest, "fig_04_zone_changes_test")


def plot_ablation_family(table: pd.DataFrame | None, out_dirs: dict[str, Path], manifest: list[dict[str, Any]], fmt: str, dpi: int) -> None:
    if table is None or table.empty:
        return
    score_col = None
    for candidate in ["mean_validation_score", "avg_validation_score", "validation_score_mean", "validation_score"]:
        if candidate in table.columns:
            score_col = candidate
            break
    family_col = None
    for candidate in ["component_family", "family", "group", "candidate_group"]:
        if candidate in table.columns:
            family_col = candidate
            break
    if score_col is None or family_col is None:
        return
    df = table.sort_values(score_col, ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(df[family_col].astype(str), df[score_col])
    ax.set_ylabel("Mean validation score")
    ax.set_xlabel("Component family")
    ax.set_title("Hybrid-weight component family sensitivity")
    ax.tick_params(axis="x", rotation=35)
    finish_figure(out_dirs["figures"] / f"fig_05_ablation_family.{fmt}", dpi, manifest, "fig_05_ablation_family")


def plot_subgroup_progress(table: pd.DataFrame | None, out_dirs: dict[str, Path], manifest: list[dict[str, Any]], fmt: str, dpi: int) -> None:
    if table is None or table.empty:
        return
    top_col = None
    for candidate in ["sampled_top1_accuracy", "top1_accuracy", "sampled_top1"]:
        if candidate in table.columns:
            top_col = candidate
            break
    group_col = None
    for candidate in ["group_value", "route_progress_bin", "subgroup_value"]:
        if candidate in table.columns:
            group_col = candidate
            break
    if top_col is None or group_col is None:
        return
    df = table[table.get("split", "") == "test"].copy()
    if df.empty:
        df = table.copy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df[group_col].astype(str), df[top_col], marker="o")
    ax.set_ylabel("Sampled top-1 accuracy")
    ax.set_xlabel("Route progress bin")
    ax.set_title("Preference model accuracy by route progress")
    ax.tick_params(axis="x", rotation=25)
    finish_figure(out_dirs["figures"] / f"fig_06_subgroup_route_progress.{fmt}", dpi, manifest, "fig_06_subgroup_route_progress")


def write_narrative_summary(
    out_dirs: dict[str, Path],
    model_table: pd.DataFrame | None,
    best_weight_table: pd.DataFrame | None,
    route_table: pd.DataFrame | None,
    stats_table: pd.DataFrame | None,
    ablation_outputs: dict[str, pd.DataFrame],
    manifest: list[dict[str, Any]],
) -> None:
    lines: list[str] = []
    lines.append("# Final Experiment Result Summary")
    lines.append("")
    lines.append(f"Generated at: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")

    if model_table is not None and not model_table.empty and "sampled_top1_accuracy" in model_table.columns:
        test = model_table[model_table["split"] == "test"]
        if not test.empty:
            best = test.sort_values("sampled_top1_accuracy", ascending=False).iloc[0]
            lines.append("## Preference model")
            lines.append(
                f"- Best test preference model in the consolidated table: {best.get('model_display', best.get('model_name'))} "
                f"with sampled top-1 accuracy {best['sampled_top1_accuracy']:.4f}."
            )
            lines.append("")

    if best_weight_table is not None and not best_weight_table.empty:
        row = best_weight_table.iloc[0]
        weight_text = ", ".join(
            f"{col.replace('_weight', '')}={float(row[col]):.2f}"
            for col in WEIGHT_COLUMNS
            if col in row and pd.notna(row[col])
        )
        lines.append("## Hybrid weights")
        lines.append(f"- Validation-selected hybrid weights: {weight_text}.")
        lines.append("")

    if route_table is not None and not route_table.empty:
        test = route_table[route_table["split"] == "test"] if "split" in route_table.columns else route_table
        lines.append("## Route generation")
        if not test.empty:
            hybrid = test[test["method"] == "hybrid_greedy_best_weight"]
            nn = test[test["method"] == "travel_time_nearest_neighbor"]
            pref = test[test["method"] == "preference_greedy"]
            if not hybrid.empty and not nn.empty:
                h = hybrid.iloc[0]
                n = nn.iloc[0]
                lines.append(
                    "- Compared with travel-time nearest neighbour, the hybrid method trades a small travel-time increase "
                    f"({h['avg_travel_time_ratio_to_actual']:.4f} vs {n['avg_travel_time_ratio_to_actual']:.4f}) "
                    "for much stronger zone continuity."
                )
            if not hybrid.empty and not pref.empty:
                h = hybrid.iloc[0]
                p = pref.iloc[0]
                lines.append(
                    "- Compared with preference-only greedy routing, the hybrid method substantially reduces travel-time ratio "
                    f"({h['avg_travel_time_ratio_to_actual']:.4f} vs {p['avg_travel_time_ratio_to_actual']:.4f})."
                )
        lines.append("")

    if stats_table is not None and not stats_table.empty:
        lines.append("## Statistical testing")
        lines.append("- Paired route-level tests are available in table_04_statistical_tests_summary.csv.")
        lines.append("- These tests support the routing trade-off interpretation rather than a pure travel-time dominance claim.")
        lines.append("")

    if ablation_outputs:
        lines.append("## Ablation and sensitivity")
        lines.append("- Ablation tables summarize which cost components were selected by validation tuning and which manually designed settings remained competitive.")
        lines.append("- Zero final weights for time-window/workload should be interpreted as limited additional value of the simplified proxy terms under the current validation objective, not as theoretical irrelevance.")
        lines.append("")

    path = out_dirs["narrative"] / "final_experiment_summary.md"
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    manifest.append({"artifact_type": "narrative", "name": "final_experiment_summary", "path": str(path)})


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)
    out_dirs = prepare_output_dir(paths["output_dir"], args.overwrite)
    manifest: list[dict[str, Any]] = []

    print("Final table/figure generation inputs:")
    for key, value in paths.items():
        print(f"  {key}: {value}")

    model_table = build_model_comparison(paths, out_dirs, manifest)
    best_weight_table = build_best_weight_table(paths, out_dirs, manifest)
    route_table = build_route_summary_table(paths, out_dirs, manifest)
    stats_table = build_statistical_summary(paths, out_dirs, manifest)
    ablation_outputs = build_ablation_tables(paths, out_dirs, manifest)
    subgroup_outputs = build_subgroup_tables(paths, out_dirs, manifest)
    feature_table = build_feature_importance(paths, out_dirs, manifest)

    if require_matplotlib(args.skip_figures):
        plot_model_comparison(model_table, out_dirs, manifest, args.fig_format, args.dpi)
        plot_best_weights(best_weight_table, out_dirs, manifest, args.fig_format, args.dpi)
        plot_route_tradeoff(route_table, out_dirs, manifest, args.fig_format, args.dpi)
        plot_zone_changes(route_table, out_dirs, manifest, args.fig_format, args.dpi)
        plot_ablation_family(ablation_outputs.get("family"), out_dirs, manifest, args.fig_format, args.dpi)
        plot_subgroup_progress(subgroup_outputs.get("route_progress"), out_dirs, manifest, args.fig_format, args.dpi)

    write_narrative_summary(
        out_dirs,
        model_table,
        best_weight_table,
        route_table,
        stats_table,
        ablation_outputs,
        manifest,
    )

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "paths": {key: str(value) for key, value in paths.items()},
        "artifact_count": len(manifest),
        "tables_created": sum(1 for item in manifest if item["artifact_type"] == "table"),
        "figures_created": sum(1 for item in manifest if item["artifact_type"] == "figure"),
        "skip_figures": bool(args.skip_figures),
    }
    save_json(out_dirs["root"] / "final_tables_figures_run_summary.json", run_summary)
    pd.DataFrame(manifest).to_csv(out_dirs["root"] / "artifact_manifest.csv", index=False)

    print("\nFinal tables and figures generated.")
    print(f"Output directory: {paths['output_dir']}")
    print(f"Artifacts created: {len(manifest)}")
    print(f"Tables: {run_summary['tables_created']}")
    print(f"Figures: {run_summary['figures_created']}")
    print(f"Manifest: {out_dirs['root'] / 'artifact_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
