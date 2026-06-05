"""Memory-safe cleaning pipeline for the Amazon Last Mile dataset.

The Amazon Last Mile Routing Research Challenge files are large top-level JSON
objects keyed by route identifier. This module streams one route at a time,
normalizes non-standard JSON ``NaN``/Infinity values to ``None``, and writes CSV
outputs for data cleaning and EDA preparation without training any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

DEFAULT_DATA_ROOT = Path("/content/drive/MyDrive/dissertation/amazon_last_mile")
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "processed_outputs"
PROGRESS_EVERY = 500

# The challenge data is split into build/apply/score folders. Each entry below
# describes one logical dataset slice and its route/package/travel/score files.
DATASET_PARTS = [
    {
        "name": "training_build",
        "relative_dir": "almrrc2021-data-training/model_build_inputs",
        "route_file": "route_data.json",
        "package_file": "package_data.json",
        "travel_time_file": "travel_times.json",
        "actual_sequence_file": "actual_sequences.json",
        "invalid_sequence_score_file": "invalid_sequence_scores.json",
    },
    {
        "name": "training_apply",
        "relative_dir": "almrrc2021-data-training/model_apply_inputs",
        "route_file": "new_route_data.json",
        "package_file": "new_package_data.json",
        "travel_time_file": "new_travel_times.json",
        "actual_sequence_file": None,
        "invalid_sequence_score_file": None,
    },
    {
        "name": "training_score",
        "relative_dir": "almrrc2021-data-training/model_score_inputs",
        "route_file": None,
        "package_file": None,
        "travel_time_file": None,
        "actual_sequence_file": "new_actual_sequences.json",
        "invalid_sequence_score_file": "new_invalid_sequence_scores.json",
    },
    {
        "name": "evaluation_apply",
        "relative_dir": "almrrc2021-data-evaluation/model_apply_inputs",
        "route_file": "eval_route_data.json",
        "package_file": "eval_package_data.json",
        "travel_time_file": "eval_travel_times.json",
        "actual_sequence_file": None,
        "invalid_sequence_score_file": None,
    },
    {
        "name": "evaluation_score",
        "relative_dir": "almrrc2021-data-evaluation/model_score_inputs",
        "route_file": None,
        "package_file": None,
        "travel_time_file": None,
        "actual_sequence_file": "eval_actual_sequences.json",
        "invalid_sequence_score_file": "eval_invalid_sequence_scores.json",
    },
]

OUTPUT_FILES = {
    "routes_summary": "routes_summary.csv",
    "stops_base_features": "stops_base_features.csv",
    "actual_transitions": "actual_transitions.csv",
    "stop_package_features": "stop_package_features.csv",
    "data_quality_report": "data_quality_report.csv",
    "missing_value_summary": "missing_value_summary.csv",
}


@dataclass
class CleaningSummary:
    """High-level row counts printed after the pipeline completes."""

    routes_processed: int = 0
    usable_routes: int = 0
    routes_summary_rows: int = 0
    stops_base_features_rows: int = 0
    actual_transitions_rows: int = 0
    stop_package_features_rows: int = 0
    data_quality_report_rows: int = 0
    missing_value_summary_rows: int = 0


class TopLevelJsonObjectStreamer:
    """Yield top-level ``key, value`` pairs from a large JSON object.

    Python's standard JSON decoder accepts non-standard constants such as
    ``NaN``. The decoder below maps ``NaN``, ``Infinity``, and ``-Infinity`` to
    ``None`` while reading chunks from disk, so the original JSON files are not
    modified and unsafe values never reach the CSV outputs.
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

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, Mapping):
        return {key: sanitize_value(nested_value) for key, nested_value in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    return value


def is_missing(value: Any) -> bool:
    """Return True for values that should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def bool_int(value: bool) -> int:
    """Encode booleans as 0/1 for CSV feature files."""

    return 1 if value else 0


def csv_value(value: Any) -> Any:
    """Return a scalar CSV-safe value, serializing unexpected nested values."""

    value = sanitize_value(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class CsvOutput:
    """Small wrapper that writes fixed-schema CSV rows and counts them."""

    def __init__(self, path: Path, fieldnames: Sequence[str]) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file_obj = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file_obj, fieldnames=fieldnames, extrasaction="ignore")
        self.writer.writeheader()
        self.rows_written = 0

    def write(self, row: Mapping[str, Any]) -> None:
        self.writer.writerow({key: csv_value(row.get(key)) for key in self.writer.fieldnames or []})
        self.rows_written += 1

    def close(self) -> None:
        self.file_obj.close()

    def __enter__(self) -> "CsvOutput":
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()


def clean_dataset(
    data_root: Path = DEFAULT_DATA_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    progress_every: int = PROGRESS_EVERY,
) -> CleaningSummary:
    """Create all requested cleaned CSV outputs under ``output_dir``.

    The function makes multiple streaming passes over the source files. It stores
    only compact per-route metadata in a temporary SQLite database located in the
    output directory, then deletes that temporary database before returning.
    """

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "_cleaning_state.sqlite"
    if state_path.exists():
        state_path.unlink()

    connection = sqlite3.connect(state_path)
    try:
        _initialize_state_database(connection)
        summary = CleaningSummary()

        # Pass 1: route files. Write route-level summaries and stop-level base
        # features while recording route stop sets for later quality checks.
        with CsvOutput(output_dir / OUTPUT_FILES["routes_summary"], ROUTES_SUMMARY_FIELDS) as routes_csv, CsvOutput(
            output_dir / OUTPUT_FILES["stops_base_features"], STOPS_BASE_FIELDS
        ) as stops_csv:
            for part in DATASET_PARTS:
                route_file = part.get("route_file")
                if route_file is None:
                    continue
                path = data_root / str(part["relative_dir"]) / str(route_file)
                processed = _process_route_file(path, routes_csv, stops_csv, connection, progress_every)
                summary.routes_processed += processed
        summary.routes_summary_rows = routes_csv.rows_written
        summary.stops_base_features_rows = stops_csv.rows_written

        # Pass 2: actual sequence files. Write positive transition labels and
        # keep sequence stop sets for route-level quality checks.
        with CsvOutput(output_dir / OUTPUT_FILES["actual_transitions"], ACTUAL_TRANSITION_FIELDS) as transitions_csv:
            for part in DATASET_PARTS:
                sequence_file = part.get("actual_sequence_file")
                if sequence_file is None:
                    continue
                path = data_root / str(part["relative_dir"]) / str(sequence_file)
                _process_actual_sequence_file(path, transitions_csv, connection, progress_every)
        summary.actual_transitions_rows = transitions_csv.rows_written

        # Pass 3: package files. Write package aggregates at stop level and store
        # package stop sets for route-level quality checks and missing summaries.
        package_missing_totals = _empty_package_missing_totals()
        with CsvOutput(output_dir / OUTPUT_FILES["stop_package_features"], STOP_PACKAGE_FIELDS) as package_csv:
            for part in DATASET_PARTS:
                package_file = part.get("package_file")
                if package_file is None:
                    continue
                path = data_root / str(part["relative_dir"]) / str(package_file)
                _process_package_file(path, package_csv, connection, package_missing_totals, progress_every)
        summary.stop_package_features_rows = package_csv.rows_written

        # Pass 4: final reports based on compact route metadata.
        with CsvOutput(output_dir / OUTPUT_FILES["data_quality_report"], DATA_QUALITY_FIELDS) as quality_csv:
            summary.usable_routes = _write_data_quality_report(connection, quality_csv)
        summary.data_quality_report_rows = quality_csv.rows_written

        with CsvOutput(output_dir / OUTPUT_FILES["missing_value_summary"], MISSING_VALUE_FIELDS) as missing_csv:
            _write_missing_value_summary(connection, package_missing_totals, missing_csv)
        summary.missing_value_summary_rows = missing_csv.rows_written

        _print_final_summary(summary)
        return summary
    finally:
        connection.close()
        if state_path.exists():
            state_path.unlink()


ROUTES_SUMMARY_FIELDS = [
    "route_id",
    "station_code",
    "date_YYYY_MM_DD",
    "departure_time_utc",
    "executor_capacity_cm3",
    "route_score",
    "number_of_stops",
    "station_stop_id",
    "number_of_station_stops",
    "number_of_dropoff_stops",
    "missing_zone_count",
    "missing_zone_ratio",
]

STOPS_BASE_FIELDS = [
    "route_id",
    "stop_id",
    "lat",
    "lng",
    "type",
    "zone_id",
    "zone_missing",
    "is_station",
    "is_dropoff",
]

ACTUAL_TRANSITION_FIELDS = ["route_id", "from_stop", "to_stop", "position", "label"]

STOP_PACKAGE_FIELDS = [
    "route_id",
    "stop_id",
    "package_count",
    "total_planned_service_time",
    "has_time_window",
    "time_window_package_count",
    "total_package_volume_cm3",
    "delivered_count",
    "attempted_count",
    "rejected_count",
    "unknown_status_count",
]

DATA_QUALITY_FIELDS = [
    "route_id",
    "route_exists_in_actual_sequences",
    "sequence_stop_count",
    "route_stop_count",
    "sequence_matches_route_stops",
    "package_stop_count",
    "package_stops_not_in_route_count",
    "number_of_station_stops",
    "has_single_station",
    "can_use_for_training",
]

MISSING_VALUE_FIELDS = ["field", "missing_count", "total_count", "missing_ratio"]


def _initialize_state_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE route_state (
            route_id TEXT PRIMARY KEY,
            route_stop_count INTEGER NOT NULL,
            route_stop_ids_json TEXT NOT NULL,
            number_of_station_stops INTEGER NOT NULL,
            missing_zone_count INTEGER NOT NULL
        );
        CREATE TABLE sequence_state (
            route_id TEXT PRIMARY KEY,
            sequence_stop_count INTEGER NOT NULL,
            sequence_stop_ids_json TEXT NOT NULL
        );
        CREATE TABLE package_state (
            route_id TEXT PRIMARY KEY,
            package_stop_count INTEGER NOT NULL,
            package_stop_ids_json TEXT NOT NULL
        );
        """
    )
    connection.commit()


def _stream_if_exists(path: Path) -> Iterator[tuple[str, Any]]:
    if not path.exists():
        print(f"Skipping missing source file: {path}")
        return
    print(f"Processing source file: {path}")
    yield from stream_top_level_object(path)


def _process_route_file(
    path: Path,
    routes_csv: CsvOutput,
    stops_csv: CsvOutput,
    connection: sqlite3.Connection,
    progress_every: int,
) -> int:
    processed = 0
    for route_id, route in _stream_if_exists(path):
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"  processed {processed} routes from {path.name}")
        if not isinstance(route, Mapping):
            continue

        stops = route.get("stops")
        stops = stops if isinstance(stops, Mapping) else {}
        stop_ids = sorted(str(stop_id) for stop_id in stops)
        number_of_stops = len(stop_ids)
        station_stop_ids: list[str] = []
        dropoff_count = 0
        missing_zone_count = 0

        for stop_id, stop in stops.items():
            stop = stop if isinstance(stop, Mapping) else {}
            stop_type = stop.get("type")
            is_station = str(stop_type).lower() == "station"
            is_dropoff = str(stop_type).lower() == "dropoff"
            zone_missing = is_missing(stop.get("zone_id"))
            if is_station:
                station_stop_ids.append(str(stop_id))
            if is_dropoff:
                dropoff_count += 1
            if zone_missing:
                missing_zone_count += 1
            stops_csv.write(
                {
                    "route_id": route_id,
                    "stop_id": stop_id,
                    "lat": stop.get("lat"),
                    "lng": stop.get("lng"),
                    "type": stop_type,
                    "zone_id": stop.get("zone_id"),
                    "zone_missing": bool_int(zone_missing),
                    "is_station": bool_int(is_station),
                    "is_dropoff": bool_int(is_dropoff),
                }
            )

        routes_csv.write(
            {
                "route_id": route_id,
                "station_code": route.get("station_code"),
                "date_YYYY_MM_DD": route.get("date_YYYY_MM_DD"),
                "departure_time_utc": route.get("departure_time_utc"),
                "executor_capacity_cm3": route.get("executor_capacity_cm3"),
                "route_score": route.get("route_score"),
                "number_of_stops": number_of_stops,
                "station_stop_id": station_stop_ids[0] if station_stop_ids else None,
                "number_of_station_stops": len(station_stop_ids),
                "number_of_dropoff_stops": dropoff_count,
                "missing_zone_count": missing_zone_count,
                "missing_zone_ratio": missing_zone_count / number_of_stops if number_of_stops else 0,
            }
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO route_state (
                route_id, route_stop_count, route_stop_ids_json,
                number_of_station_stops, missing_zone_count
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (route_id, number_of_stops, json.dumps(stop_ids), len(station_stop_ids), missing_zone_count),
        )
    connection.commit()
    return processed


def _process_actual_sequence_file(
    path: Path,
    transitions_csv: CsvOutput,
    connection: sqlite3.Connection,
    progress_every: int,
) -> None:
    processed = 0
    for route_id, sequence_data in _stream_if_exists(path):
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"  processed {processed} actual-sequence routes from {path.name}")
        actual = sequence_data.get("actual") if isinstance(sequence_data, Mapping) else None
        actual = actual if isinstance(actual, Mapping) else {}
        ordered_stops = sorted(actual.items(), key=lambda item: (item[1] if item[1] is not None else math.inf, str(item[0])))
        ordered_stop_ids = [str(stop_id) for stop_id, _position in ordered_stops]

        for transition_position, (from_stop, to_stop) in enumerate(zip(ordered_stop_ids, ordered_stop_ids[1:])):
            transitions_csv.write(
                {
                    "route_id": route_id,
                    "from_stop": from_stop,
                    "to_stop": to_stop,
                    "position": transition_position,
                    "label": 1,
                }
            )

        connection.execute(
            """
            INSERT OR REPLACE INTO sequence_state (
                route_id, sequence_stop_count, sequence_stop_ids_json
            ) VALUES (?, ?, ?)
            """,
            (route_id, len(ordered_stop_ids), json.dumps(sorted(ordered_stop_ids))),
        )
    connection.commit()


def _process_package_file(
    path: Path,
    package_csv: CsvOutput,
    connection: sqlite3.Connection,
    missing_totals: dict[str, dict[str, int]],
    progress_every: int,
) -> None:
    processed = 0
    for route_id, route_packages in _stream_if_exists(path):
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"  processed {processed} package routes from {path.name}")
        route_packages = route_packages if isinstance(route_packages, Mapping) else {}
        package_stop_ids: list[str] = []

        for stop_id, stop_packages in route_packages.items():
            if not isinstance(stop_packages, Mapping):
                continue
            package_stop_ids.append(str(stop_id))
            aggregate = _aggregate_stop_packages(stop_packages, missing_totals)
            package_csv.write({"route_id": route_id, "stop_id": stop_id, **aggregate})

        connection.execute(
            """
            INSERT OR REPLACE INTO package_state (
                route_id, package_stop_count, package_stop_ids_json
            ) VALUES (?, ?, ?)
            """,
            (route_id, len(package_stop_ids), json.dumps(sorted(package_stop_ids))),
        )
    connection.commit()


def _aggregate_stop_packages(
    stop_packages: Mapping[str, Any], missing_totals: dict[str, dict[str, int]]
) -> dict[str, Any]:
    package_count = 0
    total_service_time = 0.0
    time_window_count = 0
    total_volume = 0.0
    delivered_count = 0
    attempted_count = 0
    rejected_count = 0
    unknown_status_count = 0

    for package in stop_packages.values():
        package = package if isinstance(package, Mapping) else {}
        package_count += 1

        service_time = package.get("planned_service_time_seconds")
        _add_missing_observation(missing_totals, "planned_service_time", is_missing(service_time))
        if not is_missing(service_time):
            total_service_time += float(service_time)

        time_window = package.get("time_window")
        time_window = time_window if isinstance(time_window, Mapping) else {}
        has_valid_window = not is_missing(time_window.get("start_time_utc")) and not is_missing(time_window.get("end_time_utc"))
        _add_missing_observation(missing_totals, "package_time_window", not has_valid_window)
        if has_valid_window:
            time_window_count += 1

        dimensions = package.get("dimensions")
        dimensions = dimensions if isinstance(dimensions, Mapping) else {}
        length = dimensions.get("length_cm", dimensions.get("depth_cm"))
        width = dimensions.get("width_cm")
        height = dimensions.get("height_cm")
        dimensions_missing = is_missing(length) or is_missing(width) or is_missing(height)
        _add_missing_observation(missing_totals, "package_dimensions", dimensions_missing)
        if not dimensions_missing:
            total_volume += float(length) * float(width) * float(height)

        scan_status = package.get("scan_status")
        _add_missing_observation(missing_totals, "scan_status", is_missing(scan_status))
        normalized_status = "" if is_missing(scan_status) else str(scan_status).lower()
        if "delivered" in normalized_status:
            delivered_count += 1
        elif "attempt" in normalized_status:
            attempted_count += 1
        elif "reject" in normalized_status:
            rejected_count += 1
        else:
            unknown_status_count += 1

    return {
        "package_count": package_count,
        "total_planned_service_time": total_service_time,
        "has_time_window": bool_int(time_window_count > 0),
        "time_window_package_count": time_window_count,
        "total_package_volume_cm3": total_volume,
        "delivered_count": delivered_count,
        "attempted_count": attempted_count,
        "rejected_count": rejected_count,
        "unknown_status_count": unknown_status_count,
    }


def _empty_package_missing_totals() -> dict[str, dict[str, int]]:
    return {
        "package_time_window": {"missing": 0, "total": 0},
        "planned_service_time": {"missing": 0, "total": 0},
        "package_dimensions": {"missing": 0, "total": 0},
        "scan_status": {"missing": 0, "total": 0},
    }


def _add_missing_observation(totals: dict[str, dict[str, int]], field: str, missing: bool) -> None:
    totals[field]["total"] += 1
    if missing:
        totals[field]["missing"] += 1


def _write_data_quality_report(connection: sqlite3.Connection, quality_csv: CsvOutput) -> int:
    usable_routes = 0
    rows = connection.execute(
        """
        SELECT
            r.route_id,
            r.route_stop_count,
            r.route_stop_ids_json,
            r.number_of_station_stops,
            COALESCE(s.sequence_stop_count, 0),
            s.sequence_stop_ids_json,
            COALESCE(p.package_stop_count, 0),
            p.package_stop_ids_json
        FROM route_state r
        LEFT JOIN sequence_state s ON r.route_id = s.route_id
        LEFT JOIN package_state p ON r.route_id = p.route_id
        ORDER BY r.route_id
        """
    )
    for row in rows:
        (
            route_id,
            route_stop_count,
            route_stop_ids_json,
            number_of_station_stops,
            sequence_stop_count,
            sequence_stop_ids_json,
            package_stop_count,
            package_stop_ids_json,
        ) = row
        route_stop_ids = set(json.loads(route_stop_ids_json))
        sequence_stop_ids = set(json.loads(sequence_stop_ids_json)) if sequence_stop_ids_json else set()
        package_stop_ids = set(json.loads(package_stop_ids_json)) if package_stop_ids_json else set()
        route_exists_in_actual_sequences = sequence_stop_ids_json is not None
        sequence_matches_route_stops = route_stop_ids == sequence_stop_ids
        package_stops_not_in_route_count = len(package_stop_ids - route_stop_ids)
        has_single_station = number_of_station_stops == 1
        can_use_for_training = (
            route_exists_in_actual_sequences
            and sequence_stop_count == route_stop_count
            and sequence_matches_route_stops
            and has_single_station
        )
        if can_use_for_training:
            usable_routes += 1
        quality_csv.write(
            {
                "route_id": route_id,
                "route_exists_in_actual_sequences": route_exists_in_actual_sequences,
                "sequence_stop_count": sequence_stop_count,
                "route_stop_count": route_stop_count,
                "sequence_matches_route_stops": sequence_matches_route_stops,
                "package_stop_count": package_stop_count,
                "package_stops_not_in_route_count": package_stops_not_in_route_count,
                "number_of_station_stops": number_of_station_stops,
                "has_single_station": has_single_station,
                "can_use_for_training": can_use_for_training,
            }
        )
    return usable_routes


def _write_missing_value_summary(
    connection: sqlite3.Connection, package_missing_totals: dict[str, dict[str, int]], missing_csv: CsvOutput
) -> None:
    zone_row = connection.execute(
        "SELECT COALESCE(SUM(missing_zone_count), 0), COALESCE(SUM(route_stop_count), 0) FROM route_state"
    ).fetchone()
    missing_csv.write(_missing_summary_row("zone_id", int(zone_row[0]), int(zone_row[1])))
    for field, totals in package_missing_totals.items():
        missing_csv.write(_missing_summary_row(field, totals["missing"], totals["total"]))


def _missing_summary_row(field: str, missing_count: int, total_count: int) -> dict[str, Any]:
    return {
        "field": field,
        "missing_count": missing_count,
        "total_count": total_count,
        "missing_ratio": missing_count / total_count if total_count else 0,
    }


def _print_final_summary(summary: CleaningSummary) -> None:
    print("Cleaning complete.")
    print(f"Number of routes processed: {summary.routes_processed}")
    print(f"Number of usable routes: {summary.usable_routes}")
    print("Rows written:")
    for output_name in OUTPUT_FILES:
        attribute = f"{output_name}_rows"
        print(f"  {OUTPUT_FILES[output_name]}: {getattr(summary, attribute)}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create cleaned Amazon Last Mile CSVs for EDA preparation.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Dataset root directory. Default: {DEFAULT_DATA_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated CSV outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=PROGRESS_EVERY,
        help="Print progress after this many routes per source file. Default: 500",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> CleaningSummary:
    args = build_arg_parser().parse_args(argv)
    return clean_dataset(args.data_root, args.output_dir, args.progress_every)


if __name__ == "__main__":
    main()
