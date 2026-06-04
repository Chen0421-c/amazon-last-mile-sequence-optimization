from __future__ import annotations

import csv
from pathlib import Path

from last_mile_cleaning.clean_pipeline import clean_dataset, stream_top_level_object


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_stream_top_level_object_converts_non_standard_nan(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    write(source, '{"route_1":{"value":NaN},"route_2":{"value":Infinity},"route_3":{"value":-Infinity}}')

    rows = list(stream_top_level_object(source))

    assert rows == [
        ("route_1", {"value": None}),
        ("route_2", {"value": None}),
        ("route_3", {"value": None}),
    ]


def test_clean_dataset_writes_normalized_csvs(tmp_path: Path) -> None:
    data_root = tmp_path / "raw"
    output_dir = tmp_path / "processed_outputs"
    data_root.mkdir()

    write(
        data_root / "route_data.json",
        """
        {
          "route_1": {
            "station_code": "DXX1",
            "route_score": NaN,
            "stops": {
              "AA": {"lat": 1.0, "lng": 2.0, "type": "Station"},
              "BB": {"lat": 3.0, "lng": 4.0, "zone_id": "Z1"}
            }
          }
        }
        """,
    )
    write(
        data_root / "package_data.json",
        """
        {"route_1":{"BB":{"pkg_1":{"planned_service_time_seconds":30,"dimensions":{"height_cm":10}}}}}
        """,
    )
    write(data_root / "travel_times.json", '{"route_1":{"AA":{"AA":0,"BB":12},"BB":{"AA":11,"BB":0}}}')
    write(data_root / "actual_sequences.json", '{"route_1":{"actual":{"AA":0,"BB":1}}}')
    write(data_root / "invalid_sequence_scores.json", '{"route_1":{"score":0.25}}')

    summary = clean_dataset(data_root, output_dir)

    assert summary.routes == 1
    assert summary.stops == 2
    assert summary.packages == 1
    assert summary.travel_times == 4
    assert summary.actual_sequences == 2
    assert summary.invalid_sequence_scores == 1

    with (output_dir / "cleaned_routes.csv").open(newline="", encoding="utf-8") as file_obj:
        routes = list(csv.DictReader(file_obj))
    assert routes[0]["route_id"] == "route_1"
    assert routes[0]["route_score"] == ""

    with (output_dir / "cleaned_packages.csv").open(newline="", encoding="utf-8") as file_obj:
        packages = list(csv.DictReader(file_obj))
    assert packages[0]["dimensions_height_cm"] == "10"
