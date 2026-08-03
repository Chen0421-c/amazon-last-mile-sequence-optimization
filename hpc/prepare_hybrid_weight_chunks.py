#!/usr/bin/env python3
"""Prepare route-id chunks for DICC hybrid weight search array jobs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a route_id CSV into deterministic chunk CSV files."
    )
    parser.add_argument("--route-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=150)
    parser.add_argument("--prefix", default="validation_chunk")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_route_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        if reader.fieldnames is None:
            return []
        field = "route_id" if "route_id" in reader.fieldnames else reader.fieldnames[0]
        route_ids = [str(row.get(field, "")).strip() for row in reader]
    return [route_id for route_id in dict.fromkeys(route_ids) if route_id]


def write_route_ids(path: Path, route_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=["route_id"])
        writer.writeheader()
        for route_id in route_ids:
            writer.writerow({"route_id": route_id})


def main() -> int:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if not args.route_ids.exists():
        raise FileNotFoundError(f"Route ID file not found: {args.route_ids}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output_dir.glob(f"{args.prefix}_*.csv"))
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Chunk files already exist in {args.output_dir}. Use --overwrite."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    route_ids = read_route_ids(args.route_ids)
    if not route_ids:
        raise ValueError(f"No route IDs found in {args.route_ids}")

    manifest_rows = []
    for index, start in enumerate(range(0, len(route_ids), args.chunk_size)):
        chunk = route_ids[start : start + args.chunk_size]
        path = args.output_dir / f"{args.prefix}_{index:03d}.csv"
        write_route_ids(path, chunk)
        manifest_rows.append(
            {
                "chunk_index": index,
                "chunk_file": str(path),
                "route_count": len(chunk),
                "first_route_id": chunk[0],
                "last_route_id": chunk[-1],
            }
        )

    manifest_path = args.output_dir / f"{args.prefix}_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=[
                "chunk_index",
                "chunk_file",
                "route_count",
                "first_route_id",
                "last_route_id",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Input routes: {len(route_ids)}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunks written: {len(manifest_rows)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
