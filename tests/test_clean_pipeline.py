from __future__ import annotations

import csv
from pathlib import Path

from last_mile_cleaning.clean_pipeline import clean_dataset, stream_top_level_object


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def test_stream_top_level_object_converts_non_standard_nan(tmp_path: Path) -> None:
    source = tmp_path / "sample.json"
    write(source, '{"route_1":{"value":NaN},"route_2":{"value":Infinity},"route_3":{"value":-Infinity}}')

    rows = list(stream_top_level_object(source))

    assert rows == [
        ("route_1", {"value": None}),
        ("route_2", {"value": None}),
        ("route_3", {"value": None}),
    ]


def test_clean_dataset_writes_required_dissertation_outputs(tmp_path: Path) -> None:
    data_root = tmp_path / "amazon_last_mile"
    output_dir = data_root / "processed_outputs"
    build_dir = data_root / "almrrc2021-data-training" / "model_build_inputs"

    write(
        build_dir / "route_data.json",
        """
        {
          "route_1": {
            "station_code": "DXX1",
            "date_YYYY_MM_DD": "2021-07-01",
            "departure_time_utc": "08:00:00",
            "executor_capacity_cm3": 1000,
            "route_score": "High",
            "stops": {
              "AA": {"lat": 1.0, "lng": 2.0, "type": "Station", "zone_id": NaN},
              "BB": {"lat": 3.0, "lng": 4.0, "type": "Dropoff", "zone_id": "Z1"},
              "CC": {"lat": 5.0, "lng": 6.0, "type": "Dropoff", "zone_id": null}
            }
          }
        }
        """,
    )
    write(
        build_dir / "actual_sequences.json",
        '{"route_1":{"actual":{"AA":0,"BB":1,"CC":2}}}',
    )
    write(
        build_dir / "package_data.json",
        """
        {
          "route_1": {
            "BB": {
              "pkg_1": {
                "planned_service_time_seconds": 30,
                "time_window": {"start_time_utc": "09:00:00", "end_time_utc": "10:00:00"},
                "dimensions": {"length_cm": 2, "width_cm": 3, "height_cm": 4},
                "scan_status": "DELIVERED"
              },
              "pkg_2": {
                "planned_service_time_seconds": null,
                "time_window": {"start_time_utc": NaN, "end_time_utc": NaN},
                "dimensions": {},
                "scan_status": null
              }
            }
          }
        }
        """,
    )
    write(build_dir / "travel_times.json", '{"route_1":{"AA":{"BB":12}}}')
    write(build_dir / "invalid_sequence_scores.json", '{"route_1":{"score":0.25}}')

    summary = clean_dataset(data_root, output_dir, progress_every=500)

    assert summary.routes_processed == 1
    assert summary.usable_routes == 1
    assert summary.routes_summary_rows == 1
    assert summary.stops_base_features_rows == 3
    assert summary.actual_transitions_rows == 2
    assert summary.stop_package_features_rows == 1
    assert summary.data_quality_report_rows == 1
    assert summary.missing_value_summary_rows == 5

    routes = read_csv(output_dir / "routes_summary.csv")
    assert routes[0]["route_id"] == "route_1"
    assert routes[0]["number_of_stops"] == "3"
    assert routes[0]["station_stop_id"] == "AA"
    assert routes[0]["number_of_station_stops"] == "1"
    assert routes[0]["number_of_dropoff_stops"] == "2"
    assert routes[0]["missing_zone_count"] == "2"

    stops = read_csv(output_dir / "stops_base_features.csv")
    assert stops[0]["zone_missing"] == "1"
    assert stops[0]["is_station"] == "1"
    assert stops[1]["is_dropoff"] == "1"

    transitions = read_csv(output_dir / "actual_transitions.csv")
    assert transitions == [
        {"route_id": "route_1", "from_stop": "AA", "to_stop": "BB", "position": "0", "label": "1"},
        {"route_id": "route_1", "from_stop": "BB", "to_stop": "CC", "position": "1", "label": "1"},
    ]

    packages = read_csv(output_dir / "stop_package_features.csv")
    assert packages[0]["package_count"] == "2"
    assert packages[0]["total_planned_service_time"] == "30.0"
    assert packages[0]["has_time_window"] == "1"
    assert packages[0]["time_window_package_count"] == "1"
    assert packages[0]["total_package_volume_cm3"] == "24.0"
    assert packages[0]["delivered_count"] == "1"
    assert packages[0]["unknown_status_count"] == "1"

    quality = read_csv(output_dir / "data_quality_report.csv")
    assert quality[0]["route_exists_in_actual_sequences"] == "True"
    assert quality[0]["sequence_matches_route_stops"] == "True"
    assert quality[0]["can_use_for_training"] == "True"

    missing = {row["field"]: row for row in read_csv(output_dir / "missing_value_summary.csv")}
    assert missing["zone_id"]["missing_count"] == "2"
    assert missing["package_time_window"]["missing_count"] == "1"
    assert missing["planned_service_time"]["missing_count"] == "1"
    assert missing["package_dimensions"]["missing_count"] == "1"
    assert missing["scan_status"]["missing_count"] == "1"
