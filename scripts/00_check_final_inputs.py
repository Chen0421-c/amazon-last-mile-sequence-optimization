#!/usr/bin/env python3
"""Check final experiment input files before full-scale runs.

This script is intentionally lightweight. It checks file existence, reads small
samples, verifies route split consistency, and can optionally scan selected large
files in chunks for route coverage.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - message is tested by running script manually.
    print("Please install pyyaml: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("Please install pandas: pip install pandas", file=sys.stderr)
    sys.exit(1)


CRITICAL_FILE_KEYS = [
    "route_splits",
    "train_route_ids",
    "validation_route_ids",
    "test_route_ids",
    "stops_base_features",
    "stop_package_features",
    "actual_transitions_with_travel_time",
]

OPTIONAL_FILE_KEYS = ["complete_routes"]


class CheckContext:
    """Track warnings and failures during checks."""

    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"WARNING: {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"FAIL: {message}")


def load_config(config_path: str | Path) -> Dict[str, Any]:
    path = expand_path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return data


def expand_path(path_value: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path_value)))).resolve()


def resolve_path_entry(entry: Any) -> Tuple[Optional[Path], List[Path], str]:
    """Resolve a config path entry.

    Returns selected path, candidates checked, and a status string.
    Supported forms:
    - string path
    - list of path strings
    - {candidates: [path1, path2, ...]}
    """
    if entry is None:
        return None, [], "missing_config_entry"

    if isinstance(entry, dict):
        candidates_raw = entry.get("candidates", [])
    elif isinstance(entry, list):
        candidates_raw = entry
    else:
        candidates_raw = [entry]

    candidates = [expand_path(p) for p in candidates_raw]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates, "found"
    return None, candidates, "missing"


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def read_sample(path: Path, sample_rows: int) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        return pd.read_csv(path, nrows=sample_rows), None
    except Exception as exc:  # noqa: BLE001 - user-facing checker should keep going.
        return None, str(exc)


def get_required_columns(config: Dict[str, Any], file_key: str) -> Tuple[List[str], List[str]]:
    required_columns = config.get("required_columns", {}) or {}
    mapping = {
        "route_splits": "route_splits",
        "train_route_ids": "route_id_files",
        "validation_route_ids": "route_id_files",
        "test_route_ids": "route_id_files",
        "stops_base_features": "stop_files",
        "stop_package_features": "stop_files",
        "actual_transitions_with_travel_time": "transition_files",
    }
    group = mapping.get(file_key)
    if not group:
        return [], []
    group_config = required_columns.get(group, {}) or {}
    return list(group_config.get("required", []) or []), list(group_config.get("optional", []) or [])


def check_file_entry(
    file_key: str,
    entry: Any,
    config: Dict[str, Any],
    sample_rows: int,
    ctx: CheckContext,
) -> Dict[str, Any]:
    selected_path, candidates, status = resolve_path_entry(entry)
    row: Dict[str, Any] = {
        "file_key": file_key,
        "status": status,
        "selected_path": str(selected_path) if selected_path else "",
        "candidates_checked": " | ".join(str(p) for p in candidates),
        "exists": bool(selected_path and selected_path.exists()),
        "size_mb": file_size_mb(selected_path) if selected_path else 0.0,
        "columns": "",
        "missing_required_columns": "",
        "missing_optional_columns": "",
        "read_error": "",
    }

    if selected_path is None:
        message = f"{file_key} is missing. Checked: {row['candidates_checked']}"
        if file_key in CRITICAL_FILE_KEYS:
            ctx.fail(message)
        else:
            ctx.warn(message)
        return row

    df, error = read_sample(selected_path, sample_rows)
    if error:
        row["read_error"] = error
        if file_key in CRITICAL_FILE_KEYS:
            ctx.fail(f"Could not read {file_key}: {error}")
        else:
            ctx.warn(f"Could not read {file_key}: {error}")
        return row

    if df is None:
        ctx.warn(f"No sample dataframe returned for {file_key}")
        return row

    columns = list(df.columns)
    row["columns"] = " | ".join(columns)

    required, optional = get_required_columns(config, file_key)
    missing_required = [col for col in required if col not in columns]
    missing_optional = [col for col in optional if col not in columns]
    row["missing_required_columns"] = " | ".join(missing_required)
    row["missing_optional_columns"] = " | ".join(missing_optional)

    if missing_required:
        if file_key in {"train_route_ids", "validation_route_ids", "test_route_ids"}:
            ctx.warn(
                f"{file_key} missing required columns {missing_required}. "
                "If it has one column, it will be treated as route_id later."
            )
        else:
            ctx.warn(f"{file_key} missing required columns {missing_required}")
    if missing_optional:
        ctx.warn(f"{file_key} missing optional columns {missing_optional}")

    print(f"PASS: {file_key} -> {selected_path} ({row['size_mb']:.2f} MB)")
    return row


def detect_route_id_column(df: pd.DataFrame, file_label: str, ctx: CheckContext) -> Optional[str]:
    if "route_id" in df.columns:
        return "route_id"
    if len(df.columns) == 1:
        fallback = str(df.columns[0])
        ctx.warn(f"{file_label} has no route_id column; treating single column '{fallback}' as route_id")
        return fallback
    ctx.fail(f"{file_label} has no route_id column and more than one column")
    return None


def read_route_ids(path: Path, label: str, ctx: CheckContext) -> set[str]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        ctx.fail(f"Could not read {label}: {exc}")
        return set()
    route_col = detect_route_id_column(df, label, ctx)
    if route_col is None:
        return set()
    route_ids = set(df[route_col].dropna().astype(str))
    print(f"{label}: {len(route_ids)} unique route IDs")
    return route_ids


def check_route_splits(
    resolved_paths: Dict[str, Optional[Path]],
    output_dir: Path,
    ctx: CheckContext,
) -> Tuple[str, pd.DataFrame, Optional[pd.DataFrame]]:
    train = read_route_ids(resolved_paths["train_route_ids"], "train_route_ids", ctx) if resolved_paths.get("train_route_ids") else set()
    val = read_route_ids(resolved_paths["validation_route_ids"], "validation_route_ids", ctx) if resolved_paths.get("validation_route_ids") else set()
    test = read_route_ids(resolved_paths["test_route_ids"], "test_route_ids", ctx) if resolved_paths.get("test_route_ids") else set()

    overlap_rows = []
    comparisons = [
        ("train_validation", train, val),
        ("train_test", train, test),
        ("validation_test", val, test),
    ]
    overlap_status = "PASS"
    for name, left, right in comparisons:
        overlap = left.intersection(right)
        overlap_rows.append(
            {
                "comparison": name,
                "overlap_count": len(overlap),
                "sample_overlapping_route_ids": ",".join(sorted(list(overlap))[:20]),
            }
        )
        if overlap:
            overlap_status = "FAIL"
            ctx.fail(f"Route split overlap detected for {name}: {len(overlap)} routes")

    overlap_report = pd.DataFrame(overlap_rows)
    overlap_report.to_csv(output_dir / "route_split_overlap_report.csv", index=False)

    route_score_distribution = None
    route_splits_path = resolved_paths.get("route_splits")
    if route_splits_path:
        try:
            route_splits = pd.read_csv(route_splits_path)
            if {"route_id", "split"}.issubset(route_splits.columns):
                print("route_splits split counts:")
                print(route_splits["split"].value_counts(dropna=False).to_string())
                if "route_score" in route_splits.columns:
                    route_score_distribution = (
                        route_splits.groupby(["split", "route_score"], dropna=False)
                        .size()
                        .reset_index(name="count")
                    )
                    route_score_distribution.to_csv(output_dir / "route_score_distribution_by_split.csv", index=False)
            else:
                ctx.warn("route_splits.csv does not contain both route_id and split columns")
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"Could not analyze route_splits.csv: {exc}")

    return overlap_status, overlap_report, route_score_distribution


def read_small_or_sample(path: Path, route_col: str = "route_id", max_full_mb: float = 50.0) -> Dict[str, Any]:
    result = {"unique_route_id_count": "", "note": ""}
    if file_size_mb(path) <= max_full_mb:
        try:
            df = pd.read_csv(path)
            if route_col in df.columns:
                result["unique_route_id_count"] = int(df[route_col].astype(str).nunique())
            else:
                result["note"] = f"column {route_col} not found"
        except Exception as exc:  # noqa: BLE001
            result["note"] = f"could not read full file: {exc}"
    else:
        result["note"] = "large file; full unique count not computed"
    return result


def sample_route_ids(route_ids: Sequence[str], limit: int = 20) -> List[str]:
    return list(route_ids[:limit])


def scan_routes_in_csv(path: Path, route_ids: set[str], chunk_size: int) -> set[str]:
    found: set[str] = set()
    try:
        for chunk in pd.read_csv(path, usecols=lambda c: c == "route_id", chunksize=chunk_size):
            if "route_id" not in chunk.columns:
                return found
            values = set(chunk["route_id"].dropna().astype(str))
            found.update(values.intersection(route_ids))
            if found == route_ids:
                break
    except Exception:
        return found
    return found


def run_deep_coverage_check(
    resolved_paths: Dict[str, Optional[Path]],
    output_dir: Path,
    chunk_size: int,
    ctx: CheckContext,
) -> None:
    split_samples: Dict[str, List[str]] = {}
    for key in ["train_route_ids", "validation_route_ids", "test_route_ids"]:
        path = resolved_paths.get(key)
        if not path:
            continue
        ids = sorted(read_route_ids(path, key, ctx))
        split_samples[key.replace("_route_ids", "")] = sample_route_ids(ids, 20)

    all_sampled = set(route_id for ids in split_samples.values() for route_id in ids)
    coverage_rows = []
    files_to_scan = [
        "actual_transitions_with_travel_time",
        "stops_base_features",
        "stop_package_features",
    ]
    for file_key in files_to_scan:
        path = resolved_paths.get(file_key)
        if not path:
            continue
        found = scan_routes_in_csv(path, all_sampled, chunk_size)
        for split, route_ids in split_samples.items():
            for route_id in route_ids:
                coverage_rows.append(
                    {
                        "file_key": file_key,
                        "split": split,
                        "route_id": route_id,
                        "found": route_id in found,
                    }
                )

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "sample_route_coverage_report.csv", index=False)
    if not coverage.empty and not coverage["found"].all():
        ctx.warn("Some sampled route_ids were not found during deep coverage check")


def get_output_dir(config: Dict[str, Any], output_dir_arg: Optional[str]) -> Path:
    if output_dir_arg:
        output_dir = expand_path(output_dir_arg)
    else:
        outputs = config.get("outputs", {}) or {}
        output_dir = expand_path(outputs.get("logs", "final_experiment_outputs/logs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def build_file_entries(config: Dict[str, Any]) -> Dict[str, Any]:
    paths = config.get("paths", {}) or {}
    entries = {key: paths.get(key) for key in CRITICAL_FILE_KEYS + OPTIONAL_FILE_KEYS}
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Check final experiment input files.")
    parser.add_argument("--config", default="config/config_final.yaml", help="Path to YAML config file")
    parser.add_argument("--sample-rows", type=int, default=5, help="Number of rows to read for sample checks")
    parser.add_argument("--deep-check", action="store_true", help="Scan selected large files for sampled route coverage")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Chunk size for deep CSV scans")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for reports")
    args = parser.parse_args()

    ctx = CheckContext()
    try:
        config = load_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: Could not load config: {exc}", file=sys.stderr)
        return 1

    output_dir = get_output_dir(config, args.output_dir)
    entries = build_file_entries(config)

    report_rows: List[Dict[str, Any]] = []
    resolved_paths: Dict[str, Optional[Path]] = {}

    for file_key, entry in entries.items():
        selected_path, _, _ = resolve_path_entry(entry)
        resolved_paths[file_key] = selected_path
        report_rows.append(check_file_entry(file_key, entry, config, args.sample_rows, ctx))

    file_report = pd.DataFrame(report_rows)
    file_report.to_csv(output_dir / "final_input_check_report.csv", index=False)

    complete_routes_path = resolved_paths.get("complete_routes")
    if complete_routes_path:
        complete_info = read_small_or_sample(complete_routes_path)
        if complete_info.get("unique_route_id_count") != "":
            print(f"complete_routes unique route_id count: {complete_info['unique_route_id_count']}")
        elif complete_info.get("note"):
            print(f"complete_routes note: {complete_info['note']}")

    overlap_status = "UNKNOWN"
    if all(resolved_paths.get(key) for key in ["train_route_ids", "validation_route_ids", "test_route_ids"]):
        overlap_status, _, _ = check_route_splits(resolved_paths, output_dir, ctx)
    else:
        ctx.fail("Cannot check route split overlaps because one or more route_id files are missing")
        overlap_status = "FAIL"

    if args.deep_check:
        run_deep_coverage_check(resolved_paths, output_dir, args.chunk_size, ctx)

    missing_files = int((file_report["exists"] == False).sum())  # noqa: E712
    print("\nFinal input check complete.")
    print(f"Files checked: {len(file_report)}")
    print(f"Missing files: {missing_files}")
    print(f"Warnings: {len(ctx.warnings)}")
    print(f"Route split overlap status: {overlap_status}")
    print(f"Output directory: {output_dir}")

    critical_missing = any(
        row["file_key"] in CRITICAL_FILE_KEYS and not row["exists"] for row in report_rows
    )
    if critical_missing or ctx.failures or overlap_status == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
