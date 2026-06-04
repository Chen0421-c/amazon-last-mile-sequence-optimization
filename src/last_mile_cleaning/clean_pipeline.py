"""Memory-safe cleaning pipeline for the Amazon Last Mile dataset.

The source files in the Amazon Last Mile Routing Research Challenge are large
JSON objects keyed by route identifier.  This module deliberately streams one
route entry at a time instead of loading complete JSON files into memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed_outputs"

SOURCE_FILES = {
    "routes": "route_data.json",
    "packages": "package_data.json",
    "travel_times": "travel_times.json",
    "actual_sequences": "actual_sequences.json",
    "invalid_sequence_scores": "invalid_sequence_scores.json",
}

CSV_FILENAMES = {
    "routes": "cleaned_routes.csv",
    "stops": "cleaned_stops.csv",
    "packages": "cleaned_packages.csv",
    "travel_times": "cleaned_travel_times.csv",
    "actual_sequences": "cleaned_actual_sequences.csv",
    "invalid_sequence_scores": "cleaned_invalid_sequence_scores.csv",
}


@dataclass(frozen=True)
class CleaningSummary:
    """Counts written by the cleaning pipeline."""

    routes: int = 0
    stops: int = 0
    packages: int = 0
    travel_times: int = 0
    actual_sequences: int = 0
    invalid_sequence_scores: int = 0


class TopLevelJsonObjectStreamer:
    """Yield top-level ``key, value`` pairs from a large JSON object.

    Python's standard JSON parser accepts non-standard constants such as ``NaN``
    by default.  The decoder used here maps ``NaN``, ``Infinity``, and
    ``-Infinity`` to ``None`` so downstream CSVs contain empty cells instead of
    unsafe floating-point sentinel values.
    """

    def __init__(self, file_obj: TextIO, chunk_size: int = 1024 * 1024) -> None:
        self.file_obj = file_obj
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder(parse_constant=lambda _constant: None)
        self.buffer = ""
        self.position = 0
        self.eof = False

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        self._consume_expected("{")
        next_char = self._peek_non_whitespace()
        if next_char == "}":
            self.position += 1
            return

        while True:
            key = self._parse_next_value()
            if not isinstance(key, str):
                raise ValueError("Top-level JSON object keys must be strings.")
            self._consume_expected(":")
            value = self._parse_next_value()
            yield key, sanitize_value(value)
            self._discard_consumed_buffer()

            separator = self._peek_non_whitespace()
            if separator == ",":
                self.position += 1
                continue
            if separator == "}":
                self.position += 1
                return
            raise ValueError(f"Expected ',' or '}}' after route entry, found {separator!r}.")

    def _read_more(self) -> None:
        if self.eof:
            return
        chunk = self.file_obj.read(self.chunk_size)
        if chunk == "":
            self.eof = True
        else:
            self.buffer += chunk

    def _discard_consumed_buffer(self) -> None:
        if self.position > self.chunk_size:
            self.buffer = self.buffer[self.position :]
            self.position = 0

    def _peek_non_whitespace(self) -> str:
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if self.eof:
                raise ValueError("Unexpected end of JSON file.")
            self._discard_consumed_buffer()
            self._read_more()

    def _consume_expected(self, expected: str) -> None:
        actual = self._peek_non_whitespace()
        if actual != expected:
            raise ValueError(f"Expected {expected!r}, found {actual!r}.")
        self.position += 1

    def _parse_next_value(self) -> Any:
        while True:
            self._peek_non_whitespace()
            try:
                value, new_position = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError as exc:
                if self.eof:
                    raise ValueError(f"Could not parse JSON value near byte {self.position}.") from exc
                self._read_more()
                continue
            self.position = new_position
            return value


def stream_top_level_object(path: Path) -> Iterator[tuple[str, Any]]:
    """Stream route-level entries from a top-level JSON object."""

    with path.open("r", encoding="utf-8") as file_obj:
        yield from TopLevelJsonObjectStreamer(file_obj)


def sanitize_value(value: Any) -> Any:
    """Recursively replace unsafe missing values with ``None`` for CSV output."""

    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        return None
    if isinstance(value, Mapping):
        return {key: sanitize_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def scalar_or_json(value: Any) -> Any:
    """Return scalar values directly and encode nested values compactly as JSON."""

    value = sanitize_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def flatten_mapping(mapping: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries while safely serializing lists/complex values."""

    flattened: dict[str, Any] = {}
    for key, value in mapping.items():
        output_key = f"{prefix}{key}" if prefix == "" else f"{prefix}_{key}"
        if isinstance(value, Mapping):
            flattened.update(flatten_mapping(value, output_key))
        else:
            flattened[output_key] = scalar_or_json(value)
    return flattened


class IncrementalCsvWriter:
    """Write CSV rows without keeping all rows in memory.

    The writer buffers only rows encountered before the schema is known.  When
    new columns appear later, the file is transparently rewritten from a small
    temporary copy so all rows keep the same header.
    """

    def __init__(self, path: Path, base_fieldnames: Sequence[str]) -> None:
        self.path = path
        self.base_fieldnames = list(base_fieldnames)
        self.fieldnames = list(base_fieldnames)
        self.file_obj: TextIO | None = None
        self.writer: csv.DictWriter[str] | None = None
        self.rows_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._open_writer()

    def __enter__(self) -> "IncrementalCsvWriter":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def write_row(self, row: Mapping[str, Any]) -> None:
        row = {key: scalar_or_json(value) for key, value in row.items()}
        new_columns = [key for key in row if key not in self.fieldnames]
        if new_columns:
            self._add_columns(new_columns)
        assert self.writer is not None
        self.writer.writerow({field: row.get(field) for field in self.fieldnames})
        self.rows_written += 1

    def close(self) -> None:
        if self.file_obj is not None:
            self.file_obj.close()
            self.file_obj = None
            self.writer = None

    def _open_writer(self) -> None:
        self.file_obj = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file_obj, fieldnames=self.fieldnames, extrasaction="ignore")
        self.writer.writeheader()

    def _add_columns(self, new_columns: Sequence[str]) -> None:
        self.close()
        old_path = self.path.with_suffix(self.path.suffix + ".tmp")
        self.path.replace(old_path)
        self.fieldnames.extend(new_columns)
        with old_path.open("r", encoding="utf-8", newline="") as old_file, self.path.open(
            "w", encoding="utf-8", newline=""
        ) as new_file:
            reader = csv.DictReader(old_file)
            writer = csv.DictWriter(new_file, fieldnames=self.fieldnames)
            writer.writeheader()
            for old_row in reader:
                writer.writerow({field: old_row.get(field) for field in self.fieldnames})
        old_path.unlink()
        self.file_obj = self.path.open("a", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file_obj, fieldnames=self.fieldnames, extrasaction="ignore")


def clean_dataset(data_root: Path = DEFAULT_DATA_ROOT, output_dir: Path = DEFAULT_OUTPUT_DIR) -> CleaningSummary:
    """Clean available Amazon Last Mile JSON files into normalized CSV outputs."""

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_counts = _write_routes_and_stops(data_root / SOURCE_FILES["routes"], output_dir)
    package_count = _write_packages(data_root / SOURCE_FILES["packages"], output_dir)
    travel_time_count = _write_travel_times(data_root / SOURCE_FILES["travel_times"], output_dir)
    sequence_count = _write_actual_sequences(data_root / SOURCE_FILES["actual_sequences"], output_dir)
    invalid_score_count = _write_invalid_sequence_scores(
        data_root / SOURCE_FILES["invalid_sequence_scores"], output_dir
    )

    return CleaningSummary(
        routes=route_counts["routes"],
        stops=route_counts["stops"],
        packages=package_count,
        travel_times=travel_time_count,
        actual_sequences=sequence_count,
        invalid_sequence_scores=invalid_score_count,
    )


def _write_routes_and_stops(path: Path, output_dir: Path) -> dict[str, int]:
    counts = {"routes": 0, "stops": 0}
    with IncrementalCsvWriter(output_dir / CSV_FILENAMES["routes"], ["route_id"]) as route_writer, IncrementalCsvWriter(
        output_dir / CSV_FILENAMES["stops"], ["route_id", "stop_id"]
    ) as stop_writer:
        for route_id, route in _stream_if_exists(path):
            if not isinstance(route, Mapping):
                continue
            route_row = {"route_id": route_id}
            route_row.update(flatten_mapping({key: value for key, value in route.items() if key != "stops"}))
            route_writer.write_row(route_row)
            counts["routes"] += 1

            stops = route.get("stops", {})
            if isinstance(stops, Mapping):
                for stop_id, stop in stops.items():
                    stop_row = {"route_id": route_id, "stop_id": stop_id}
                    if isinstance(stop, Mapping):
                        stop_row.update(flatten_mapping(stop))
                    else:
                        stop_row["value"] = stop
                    stop_writer.write_row(stop_row)
                    counts["stops"] += 1
    return counts


def _write_packages(path: Path, output_dir: Path) -> int:
    count = 0
    with IncrementalCsvWriter(
        output_dir / CSV_FILENAMES["packages"], ["route_id", "stop_id", "package_id"]
    ) as writer:
        for route_id, route_packages in _stream_if_exists(path):
            if not isinstance(route_packages, Mapping):
                continue
            for stop_id, stop_packages in route_packages.items():
                if not isinstance(stop_packages, Mapping):
                    continue
                for package_id, package in stop_packages.items():
                    row = {"route_id": route_id, "stop_id": stop_id, "package_id": package_id}
                    if isinstance(package, Mapping):
                        row.update(flatten_mapping(package))
                    else:
                        row["value"] = package
                    writer.write_row(row)
                    count += 1
    return count


def _write_travel_times(path: Path, output_dir: Path) -> int:
    count = 0
    with IncrementalCsvWriter(
        output_dir / CSV_FILENAMES["travel_times"], ["route_id", "from_stop_id", "to_stop_id", "travel_time_seconds"]
    ) as writer:
        for route_id, matrix in _stream_if_exists(path):
            if not isinstance(matrix, Mapping):
                continue
            for from_stop_id, destinations in matrix.items():
                if not isinstance(destinations, Mapping):
                    continue
                for to_stop_id, travel_time_seconds in destinations.items():
                    writer.write_row(
                        {
                            "route_id": route_id,
                            "from_stop_id": from_stop_id,
                            "to_stop_id": to_stop_id,
                            "travel_time_seconds": travel_time_seconds,
                        }
                    )
                    count += 1
    return count


def _write_actual_sequences(path: Path, output_dir: Path) -> int:
    count = 0
    with IncrementalCsvWriter(
        output_dir / CSV_FILENAMES["actual_sequences"], ["route_id", "stop_id", "actual_sequence"]
    ) as writer:
        for route_id, sequence_data in _stream_if_exists(path):
            actual = sequence_data.get("actual") if isinstance(sequence_data, Mapping) else sequence_data
            if not isinstance(actual, Mapping):
                continue
            for stop_id, sequence in actual.items():
                writer.write_row({"route_id": route_id, "stop_id": stop_id, "actual_sequence": sequence})
                count += 1
    return count


def _write_invalid_sequence_scores(path: Path, output_dir: Path) -> int:
    count = 0
    with IncrementalCsvWriter(
        output_dir / CSV_FILENAMES["invalid_sequence_scores"], ["route_id"]
    ) as writer:
        for route_id, score_data in _stream_if_exists(path):
            row = {"route_id": route_id}
            if isinstance(score_data, Mapping):
                row.update(flatten_mapping(score_data))
            else:
                row["score"] = score_data
            writer.write_row(row)
            count += 1
    return count


def _stream_if_exists(path: Path) -> Iterator[tuple[str, Any]]:
    if not path.exists():
        print(f"Skipping missing source file: {path}")
        return
    yield from stream_top_level_object(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean Amazon Last Mile JSON files into route-normalized CSV outputs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Directory containing Amazon JSON files. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for cleaned CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> CleaningSummary:
    args = build_arg_parser().parse_args(argv)
    summary = clean_dataset(args.data_root, args.output_dir)
    print("Cleaning complete:")
    for key, value in summary.__dict__.items():
        print(f"  {key}: {value}")
    print(f"Outputs written to: {args.output_dir}")
    return summary


if __name__ == "__main__":
    main()
