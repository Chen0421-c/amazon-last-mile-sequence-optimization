#!/usr/bin/env python3
"""Create route-level train/validation/test splits for the Amazon Last Mile project.

This script creates modelling splits from the route subset with complete actual
transition travel-time coverage. Splitting is done at route level, not transition
level, to avoid data leakage between transitions from the same route.

Inputs:
- final_cleaned/travel_time_complete_routes.csv
- processed_outputs/routes_summary.csv

Outputs under final_cleaned/
- route_splits.csv
- train_route_ids.csv
- validation_route_ids.csv
- test_route_ids.csv
- split_summary.csv
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_PROCESSED_DIR = Path("/content/drive/MyDrive/dissertation/amazon_last_mile/processed_outputs")
DEFAULT_FINAL_CLEANED_DIR = DEFAULT_PROCESSED_DIR / "final_cleaned"
DEFAULT_ROUTES_SUMMARY_PATH = DEFAULT_PROCESSED_DIR / "routes_summary.csv"

SPLITS = ["train", "validation", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create route-level train/validation/test splits for modelling."
    )
    parser.add_argument(
        "--final-cleaned-dir",
        type=Path,
        default=DEFAULT_FINAL_CLEANED_DIR,
        help=f"Directory containing travel_time_complete_routes.csv. Default: {DEFAULT_FINAL_CLEANED_DIR}",
    )
    parser.add_argument(
        "--routes-summary-path",
        type=Path,
        default=DEFAULT_ROUTES_SUMMARY_PATH,
        help=f"Path to routes_summary.csv. Default: {DEFAULT_ROUTES_SUMMARY_PATH}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting. Default: 42.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training route ratio. Default: 0.70.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation route ratio. Default: 0.15.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test route ratio. Default: 0.15.",
    )
    return parser.parse_args()


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    return path


def clean_route_score(series: pd.Series) -> pd.Series:
    """Normalize missing route_score values."""

    cleaned = series.fillna("Missing").astype(str).str.strip()
    cleaned = cleaned.replace(
        {
            "": "Missing",
            "nan": "Missing",
            "NaN": "Missing",
            "None": "Missing",
            "null": "Missing",
        }
    )
    return cleaned


def largest_remainder_counts(total: int, ratios: list[float]) -> list[int]:
    """Allocate an integer total according to ratios using the largest remainder method."""

    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)

    remainders = [
        (raw_value - count, index)
        for index, (raw_value, count) in enumerate(zip(raw, counts))
    ]
    remainders.sort(reverse=True)

    for _remainder, index in remainders[:remaining]:
        counts[index] += 1

    return counts


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> list[float]:
    ratios = [train_ratio, val_ratio, test_ratio]

    if any(ratio < 0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    total = sum(ratios)
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0. Got {total:.6f} from "
            f"{train_ratio}, {val_ratio}, {test_ratio}."
        )

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be greater than zero.")

    return ratios


def load_complete_routes(final_cleaned_dir: Path) -> pd.DataFrame:
    """Load the route list used for final modelling."""

    path = require_file(final_cleaned_dir / "travel_time_complete_routes.csv")
    routes = pd.read_csv(path)

    if "route_id" not in routes.columns:
        raise ValueError(f"{path} must contain a route_id column.")

    routes = routes[["route_id"]].copy()
    routes["route_id"] = routes["route_id"].astype(str)
    routes = routes.drop_duplicates("route_id").reset_index(drop=True)

    if routes.empty:
        raise ValueError("No complete routes found in travel_time_complete_routes.csv.")

    return routes


def load_route_scores(routes_summary_path: Path) -> pd.DataFrame:
    """Load route_score from routes_summary.csv if available."""

    path = require_file(routes_summary_path)
    routes_summary = pd.read_csv(path)

    if "route_id" not in routes_summary.columns:
        raise ValueError(f"{path} must contain a route_id column.")

    if "route_score" not in routes_summary.columns:
        route_scores = routes_summary[["route_id"]].copy()
        route_scores["route_score"] = "Missing"
    else:
        route_scores = routes_summary[["route_id", "route_score"]].copy()

    route_scores["route_id"] = route_scores["route_id"].astype(str)
    route_scores["route_score"] = clean_route_score(route_scores["route_score"])
    route_scores = route_scores.drop_duplicates("route_id").reset_index(drop=True)

    return route_scores


def build_modelling_route_table(
    final_cleaned_dir: Path,
    routes_summary_path: Path,
) -> pd.DataFrame:
    """Create one modelling table with route_id and route_score."""

    complete_routes = load_complete_routes(final_cleaned_dir)
    route_scores = load_route_scores(routes_summary_path)

    route_table = complete_routes.merge(route_scores, on="route_id", how="left")
    route_table["route_score"] = clean_route_score(route_table["route_score"])

    route_table = route_table.drop_duplicates("route_id").reset_index(drop=True)

    return route_table


def stratified_route_split(
    route_table: pd.DataFrame,
    ratios: list[float],
    seed: int,
) -> pd.DataFrame:
    """Create a route-level stratified split by route_score."""

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []

    for route_score, group in route_table.groupby("route_score", dropna=False):
        route_ids = sorted(group["route_id"].astype(str).tolist())
        rng.shuffle(route_ids)

        train_count, val_count, test_count = largest_remainder_counts(len(route_ids), ratios)
        counts = {
            "train": train_count,
            "validation": val_count,
            "test": test_count,
        }

        start = 0
        for split in SPLITS:
            end = start + counts[split]
            for route_id in route_ids[start:end]:
                rows.append(
                    {
                        "route_id": route_id,
                        "route_score": str(route_score),
                        "split": split,
                    }
                )
            start = end

    split_table = pd.DataFrame(rows)
    split_table = rebalance_to_global_targets(split_table, ratios, seed)
    split_table = split_table.sort_values(["split", "route_score", "route_id"]).reset_index(
        drop=True
    )

    return split_table


def rebalance_to_global_targets(
    split_table: pd.DataFrame,
    ratios: list[float],
    seed: int,
) -> pd.DataFrame:
    """Adjust split totals to match the global target counts exactly."""

    total_routes = len(split_table)
    target_counts_raw = largest_remainder_counts(total_routes, ratios)
    target_counts = dict(zip(SPLITS, target_counts_raw))

    output = split_table.copy()
    rng = random.Random(seed + 999)

    def current_counts() -> dict[str, int]:
        counts = output["split"].value_counts().to_dict()
        return {split: int(counts.get(split, 0)) for split in SPLITS}

    counts = current_counts()

    while any(counts[split] != target_counts[split] for split in SPLITS):
        overfull = [split for split in SPLITS if counts[split] > target_counts[split]]
        underfull = [split for split in SPLITS if counts[split] < target_counts[split]]

        if not overfull or not underfull:
            break

        source_split = max(overfull, key=lambda split: counts[split] - target_counts[split])
        target_split = max(underfull, key=lambda split: target_counts[split] - counts[split])

        candidate_indices = output.index[output["split"] == source_split].tolist()
        if not candidate_indices:
            break

        chosen_index = rng.choice(candidate_indices)
        output.loc[chosen_index, "split"] = target_split

        counts = current_counts()

    return output


def validate_split(
    split_table: pd.DataFrame,
    expected_route_ids: Iterable[str],
) -> None:
    """Validate split integrity."""

    expected_route_ids = set(str(route_id) for route_id in expected_route_ids)
    assigned_route_ids = set(split_table["route_id"].astype(str))

    if expected_route_ids != assigned_route_ids:
        missing = expected_route_ids - assigned_route_ids
        extra = assigned_route_ids - expected_route_ids
        raise ValueError(
            "Assigned route IDs do not match expected complete route IDs. "
            f"Missing: {len(missing)}, Extra: {len(extra)}."
        )

    duplicate_count = int(split_table["route_id"].duplicated().sum())
    if duplicate_count > 0:
        raise ValueError(f"Route split contains duplicated route_id values: {duplicate_count}")

    split_sets = {
        split: set(split_table.loc[split_table["split"] == split, "route_id"].astype(str))
        for split in SPLITS
    }

    for i, split_a in enumerate(SPLITS):
        for split_b in SPLITS[i + 1:]:
            overlap = split_sets[split_a] & split_sets[split_b]
            if overlap:
                raise ValueError(
                    f"Route IDs overlap between {split_a} and {split_b}: {len(overlap)}"
                )

    missing_splits = set(split_table["split"].unique()) - set(SPLITS)
    if missing_splits:
        raise ValueError(f"Unexpected split labels found: {sorted(missing_splits)}")


def build_split_summary(split_table: pd.DataFrame) -> pd.DataFrame:
    """Create summary rows for split counts and route_score distributions."""

    total_routes = len(split_table)
    rows: list[dict[str, object]] = []

    for split in SPLITS:
        split_rows = split_table[split_table["split"] == split]
        split_count = len(split_rows)

        rows.append(
            {
                "summary_type": "split_total",
                "split": split,
                "route_score": "ALL",
                "route_count": split_count,
                "percentage_of_all_routes": split_count / total_routes if total_routes else 0.0,
                "percentage_within_split": 1.0 if split_count else 0.0,
            }
        )

    for split in SPLITS:
        split_rows = split_table[split_table["split"] == split]
        split_count = len(split_rows)

        score_counts = (
            split_rows["route_score"].fillna("Missing").astype(str).value_counts().sort_index()
        )

        for route_score, route_count in score_counts.items():
            rows.append(
                {
                    "summary_type": "route_score_distribution",
                    "split": split,
                    "route_score": route_score,
                    "route_count": int(route_count),
                    "percentage_of_all_routes": route_count / total_routes if total_routes else 0.0,
                    "percentage_within_split": route_count / split_count if split_count else 0.0,
                }
            )

    summary = pd.DataFrame(rows)
    return summary


def save_outputs(split_table: pd.DataFrame, summary: pd.DataFrame, final_cleaned_dir: Path) -> None:
    """Save split outputs."""

    final_cleaned_dir.mkdir(parents=True, exist_ok=True)

    split_table.to_csv(final_cleaned_dir / "route_splits.csv", index=False)
    summary.to_csv(final_cleaned_dir / "split_summary.csv", index=False)

    for split in SPLITS:
        split_rows = split_table[split_table["split"] == split].copy()
        split_rows[["route_id", "route_score"]].to_csv(
            final_cleaned_dir / f"{split}_route_ids.csv",
            index=False,
        )


def print_summary(split_table: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Print a concise console summary."""

    print("\nRoute-level train/validation/test split complete.")
    print(f"Total routes assigned: {len(split_table)}")

    split_totals = summary[summary["summary_type"] == "split_total"].copy()
    for _index, row in split_totals.iterrows():
        print(
            f"{row['split']}: {int(row['route_count'])} routes "
            f"({float(row['percentage_of_all_routes']) * 100:.2f}%)"
        )

    print("\nRoute score distribution by split:")
    distribution = summary[summary["summary_type"] == "route_score_distribution"].copy()
    if distribution.empty:
        print("No route_score distribution available.")
    else:
        pivot = distribution.pivot_table(
            index="route_score",
            columns="split",
            values="route_count",
            aggfunc="sum",
            fill_value=0,
        )
        pivot = pivot.reindex(columns=SPLITS, fill_value=0)
        print(pivot)

    print("\nSaved files:")
    print("- route_splits.csv")
    print("- train_route_ids.csv")
    print("- validation_route_ids.csv")
    print("- test_route_ids.csv")
    print("- split_summary.csv")


def main() -> None:
    args = parse_args()
    ratios = validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    route_table = build_modelling_route_table(
        final_cleaned_dir=args.final_cleaned_dir,
        routes_summary_path=args.routes_summary_path,
    )

    split_table = stratified_route_split(
        route_table=route_table,
        ratios=ratios,
        seed=args.seed,
    )

    validate_split(
        split_table=split_table,
        expected_route_ids=route_table["route_id"].astype(str).tolist(),
    )

    summary = build_split_summary(split_table)
    save_outputs(split_table, summary, args.final_cleaned_dir)
    print_summary(split_table, summary)


if __name__ == "__main__":
    main()
