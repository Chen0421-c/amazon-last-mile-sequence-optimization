#!/usr/bin/env python3
"""Generate final routes using validation-selected hybrid weights.

This script loads the trained preference model and the best weights produced by
12_optimize_hybrid_weights.py, then generates final routes for validation/test
splits using travel-time nearest neighbour, preference greedy, and the selected
hybrid greedy method. It writes generated sequences, route-level metrics, method
summaries, and run summaries for dissertation reporting.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

VALID_SPLITS = ("validation", "test", "train")
DEFAULT_CONFIG = Path("config/config_final.yaml")
DEFAULT_BASE = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_MODEL_DIR = DEFAULT_BASE / "final_experiment_outputs/model_outputs_full_top3"
DEFAULT_WEIGHT_DIR = DEFAULT_BASE / "final_experiment_outputs/hybrid_weight_search_validation_full_300w"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE / "final_experiment_outputs/route_generation_best_weights"
METHODS = ("travel_time_nearest_neighbor", "preference_greedy", "hybrid_greedy_best_weight")
WEIGHT_COLUMNS = ["travel_weight", "preference_weight", "zone_weight", "time_window_weight", "workload_weight"]
ROUTE_METRIC_COLUMNS = [
    "split", "route_id", "method", "weight_id", *WEIGHT_COLUMNS,
    "stop_count", "actual_total_travel_time", "generated_total_travel_time",
    "travel_time_ratio_to_actual", "lcs_similarity", "position_match_ratio",
    "generated_same_zone_ratio", "actual_same_zone_ratio", "zone_change_count",
    "route_valid",
]
SEQUENCE_COLUMNS = ["split", "route_id", "method", "position", "stop_id"]


def load_repo_script(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


HYBRID = load_repo_script(REPO_ROOT / "12_optimize_hybrid_weights.py", "hybrid_weight_helpers")
PAIRWISE = HYBRID.PAIRWISE


class CsvWriter:
    def __init__(self, path: Path, columns: Sequence[str]):
        self.file_obj = path.open("w", encoding="utf-8", newline="")
        self.columns = list(columns)
        self.writer = csv.DictWriter(self.file_obj, fieldnames=self.columns)
        self.writer.writeheader()

    def write(self, row: dict[str, Any]) -> None:
        self.writer.writerow({col: row.get(col, "") for col in self.columns})

    def close(self) -> None:
        self.file_obj.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final routes using validation-selected hybrid weights.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-output-dir", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-columns", type=Path, default=None)
    parser.add_argument("--weight-search-dir", type=Path, default=None)
    parser.add_argument("--best-weight-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=VALID_SPLITS, default=["validation", "test"])
    parser.add_argument("--route-ids", type=Path, default=None, help="Custom route IDs; use only with one split.")
    parser.add_argument("--max-routes", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHODS + ("all",), default=["all"])
    parser.add_argument("--no-save-sequences", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def expand_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else path.resolve()


def resolve_dirs(args: argparse.Namespace, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    outputs = config.get("outputs", {}) if isinstance(config.get("outputs"), dict) else {}
    model_dir = expand_path(args.model_output_dir or outputs.get("model_full_dir", DEFAULT_MODEL_DIR))
    weight_dir = expand_path(args.weight_search_dir or outputs.get("hybrid_weight_search_validation_full_300w_dir", DEFAULT_WEIGHT_DIR))
    output_dir = expand_path(args.output_dir or outputs.get("route_generation_best_weights_dir", DEFAULT_OUTPUT_DIR))
    return model_dir, weight_dir, output_dir


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def normalize_methods(raw: Sequence[str]) -> list[str]:
    return list(METHODS) if "all" in raw else list(dict.fromkeys(raw))


def read_best_weight(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"Best weight file not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Best weight file is empty: {path}")
    row = df.iloc[0].to_dict()
    missing = [col for col in WEIGHT_COLUMNS if col not in row]
    if missing:
        raise ValueError(f"Best weight file is missing columns: {missing}")
    weight = {col: float(row[col]) for col in WEIGHT_COLUMNS}
    weight["weight_id"] = str(row.get("weight_id", "best_weight"))
    return weight, str(path)


def resolve_split_paths(args, config, split: str, model_dir: Path, output_dir: Path) -> dict[str, Any]:
    if args.route_ids and len(args.splits) > 1:
        raise ValueError("--route-ids can only be used when exactly one split is requested.")
    pseudo_args = SimpleNamespace(
        model_output_dir=model_dir,
        model_path=args.model_path,
        feature_columns=args.feature_columns,
        output_dir=output_dir,
        split=split,
        route_ids=args.route_ids if args.route_ids else None,
    )
    return HYBRID.resolve_paths(pseudo_args, config)


def build_sequences(route_ids, transitions) -> tuple[dict[str, list[str]], dict[str, str]]:
    sequences, invalid = {}, {}
    for route_id in route_ids:
        seq = PAIRWISE.build_sequence(transitions.get(route_id, []))
        if len(seq) < 2:
            invalid[route_id] = "missing_or_short_sequence"
        elif len(set(seq)) != len(seq):
            invalid[route_id] = "duplicate_stop_in_sequence"
        else:
            sequences[route_id] = [str(stop) for stop in seq]
    return sequences, invalid


def select_sources(route_ids, source_lookup) -> tuple[dict[str, list[str]], list[str]]:
    route_sources, missing = {}, []
    for route_id in route_ids:
        srcs = source_lookup.get(route_id, [])
        if srcs:
            route_sources[route_id] = srcs
        else:
            missing.append(route_id)
    return route_sources, missing


def add_split_and_columns(row: dict[str, Any], split: str) -> dict[str, Any]:
    row["split"] = split
    for col in ROUTE_METRIC_COLUMNS:
        row.setdefault(col, "")
    return row


def write_sequence(writer, split: str, route_id: str, method: str, sequence: Sequence[str]) -> None:
    if writer is None:
        return
    for pos, stop_id in enumerate(sequence):
        writer.write({"split": split, "route_id": route_id, "method": method, "position": pos, "stop_id": stop_id})


def summarize_methods(route_metrics: pd.DataFrame) -> pd.DataFrame:
    if route_metrics.empty:
        return pd.DataFrame()
    return (
        route_metrics.groupby(["split", "method"], dropna=False)
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
            valid_route_rate=("route_valid", "mean"),
        )
        .reset_index()
    )


def generate_for_split(split, args, config, model, feature_columns, best_weight, model_dir, output_dir, methods):
    paths = resolve_split_paths(args, config, split, model_dir, output_dir)
    print(f"\n=== Generating routes for split={split} ===")
    route_ids = HYBRID.select_route_ids(paths, split, args.max_routes)
    if not route_ids:
        raise ValueError(f"No route IDs selected for split={split}.")
    requested = set(route_ids)

    route_features = PAIRWISE.load_route_features(paths["routes_summary"])
    stops = PAIRWISE.load_stop_features(paths["stops_base_features"], requested)
    packages = PAIRWISE.load_package_features(paths["stop_package_features"], requested)
    transitions = PAIRWISE.load_transitions(paths["transitions"], requested)
    source_lookup = HYBRID.load_source_lookup(paths["source_lookup"], requested)

    sequences, invalid_sequences = build_sequences(route_ids, transitions)
    valid_route_ids = [rid for rid in route_ids if rid in sequences]
    route_sources, missing_sources = select_sources(valid_route_ids, source_lookup)
    processable = [rid for rid in valid_route_ids if rid not in set(missing_sources)]

    print(f"Selected routes: {len(route_ids)}")
    print(f"Processable routes: {len(processable)}")
    print(f"Skipped invalid sequence: {len(invalid_sequences)}")
    print(f"Skipped missing source: {len(missing_sources)}")

    metric_rows = []
    seq_writer = None if args.no_save_sequences else CsvWriter(output_dir / f"generated_routes_{split}.csv", SEQUENCE_COLUMNS)

    def record(route_id, label, actual, generated, matrix, weight=None):
        row = HYBRID.evaluate(route_id, label, actual, generated, matrix, stops, packages, weight)
        metric_rows.append(add_split_and_columns(row, split))
        write_sequence(seq_writer, split, route_id, label, generated)

    def handle(route_id, matrix, source):
        if args.verbose:
            print(f"Generating {split} route {route_id} from {source}")
        actual = sequences[route_id]
        cache = {}
        if "travel_time_nearest_neighbor" in methods:
            gen = HYBRID.generate_route(route_id, "travel_time_nearest_neighbor", actual, matrix, model, feature_columns, route_features, stops, packages, {}, cache)
            record(route_id, "travel_time_nearest_neighbor", actual, gen, matrix)
        if "preference_greedy" in methods:
            gen = HYBRID.generate_route(route_id, "preference_greedy", actual, matrix, model, feature_columns, route_features, stops, packages, {}, cache)
            record(route_id, "preference_greedy", actual, gen, matrix)
        if "hybrid_greedy_best_weight" in methods:
            gen = HYBRID.generate_route(route_id, "hybrid_greedy", actual, matrix, model, feature_columns, route_features, stops, packages, best_weight, cache)
            record(route_id, "hybrid_greedy_best_weight", actual, gen, matrix, best_weight)

    try:
        processed, missing_matrices = HYBRID.stream_matrices(processable, route_sources, paths["matrix_paths"], handle, args.verbose)
    finally:
        if seq_writer is not None:
            seq_writer.close()

    if not processed:
        raise RuntimeError(f"No routes could be processed for split={split}.")

    route_metrics = pd.DataFrame(metric_rows)
    for col in ROUTE_METRIC_COLUMNS:
        if col not in route_metrics.columns:
            route_metrics[col] = np.nan
    route_metrics = route_metrics.loc[:, ROUTE_METRIC_COLUMNS]
    method_summary = summarize_methods(route_metrics)

    route_metrics_path = output_dir / f"route_metrics_{split}.csv"
    method_summary_path = output_dir / f"method_summary_{split}.csv"
    route_metrics.to_csv(route_metrics_path, index=False)
    method_summary.to_csv(method_summary_path, index=False)

    print(f"Completed split={split}: processed {len(processed)} route(s)")
    return {
        "split": split,
        "route_count_requested": len(route_ids),
        "route_count_processable": len(processable),
        "route_count_processed": len(processed),
        "routes_skipped_invalid_sequence": len(invalid_sequences),
        "routes_skipped_missing_source": len(missing_sources),
        "routes_skipped_missing_matrix": len(missing_matrices),
        "route_metrics_path": str(route_metrics_path),
        "method_summary_path": str(method_summary_path),
        "generated_routes_path": str(output_dir / f"generated_routes_{split}.csv") if not args.no_save_sequences else "",
    }


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(clean_json(payload), f, indent=2)
        f.write("\n")


def main() -> int:
    args = parse_args()
    config = HYBRID.load_config(args.config)
    model_dir, weight_dir, output_dir = resolve_dirs(args, config)
    model_path = expand_path(args.model_path or model_dir / "models/best_model.joblib")
    feature_path = expand_path(args.feature_columns or model_dir / "feature_columns.json")
    best_weight_path = expand_path(args.best_weight_file or weight_dir / "best_weight_summary.csv")
    methods = normalize_methods(args.methods)

    prepare_output_dir(output_dir, args.overwrite)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {feature_path}")

    print("Runtime inputs:")
    print(f"  model_path: {model_path}")
    print(f"  feature_columns: {feature_path}")
    print(f"  best_weight_file: {best_weight_path}")
    print(f"  output_dir: {output_dir}")
    print(f"  splits: {args.splits}")
    print(f"  methods: {methods}")

    model = joblib.load(model_path)
    feature_columns, feature_source = HYBRID.load_feature_columns(model_dir, feature_path)
    best_weight, best_weight_source = read_best_weight(best_weight_path)
    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print("Best weights: " + ", ".join(f"{c}={best_weight[c]:.4f}" for c in WEIGHT_COLUMNS))

    pd.DataFrame([best_weight]).to_csv(output_dir / "best_weight_used.csv", index=False)
    write_json(output_dir / "best_weight_used.json", best_weight)

    split_summaries, route_metric_frames, method_summary_frames = [], [], []
    for split in args.splits:
        summary = generate_for_split(split, args, config, model, feature_columns, best_weight, model_dir, output_dir, methods)
        split_summaries.append(summary)
        route_metric_frames.append(pd.read_csv(summary["route_metrics_path"]))
        method_summary_frames.append(pd.read_csv(summary["method_summary_path"]))

    pd.concat(route_metric_frames, ignore_index=True).to_csv(output_dir / "route_metrics_all_splits.csv", index=False)
    pd.concat(method_summary_frames, ignore_index=True).to_csv(output_dir / "method_summary_all_splits.csv", index=False)

    run_summary = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "model_output_dir": str(model_dir),
        "model_path": str(model_path),
        "feature_columns_source": feature_source,
        "feature_columns_used": list(feature_columns),
        "weight_search_dir": str(weight_dir),
        "best_weight_source": best_weight_source,
        "best_weight": best_weight,
        "output_dir": str(output_dir),
        "splits": list(args.splits),
        "methods": list(methods),
        "max_routes": args.max_routes,
        "save_sequences": not args.no_save_sequences,
        "split_summaries": split_summaries,
    }
    pd.DataFrame(split_summaries).to_csv(output_dir / "route_generation_run_summary.csv", index=False)
    write_json(output_dir / "route_generation_run_summary.json", run_summary)

    print("\nRoute generation with best weights complete.")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
