from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def load_quality_module():
    module_path = Path(__file__).resolve().parents[1] / "02_data_quality_and_outlier_analysis.py"
    spec = importlib.util.spec_from_file_location("quality_analysis", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_route_validity_analysis_writes_empty_unusable_detail(tmp_path: Path) -> None:
    quality_analysis = load_quality_module()
    data = {
        "routes": pd.DataFrame([{"route_id": "route_1", "route_score": "High"}]),
        "quality": pd.DataFrame(
            [
                {
                    "route_id": "route_1",
                    "route_exists_in_actual_sequences": True,
                    "route_stop_count": 3,
                    "sequence_stop_count": 3,
                    "sequence_matches_route_stops": True,
                    "number_of_station_stops": 1,
                    "has_single_station": True,
                    "can_use_for_training": True,
                }
            ]
        ),
    }

    summary, _merged = quality_analysis.route_validity_analysis(data, tmp_path)

    assert summary["routes_not_usable_for_training"] == 0
    detail = pd.read_csv(tmp_path / "unusable_routes_detail.csv")
    assert list(detail.columns) == [
        "route_id",
        "route_score",
        "route_stop_count",
        "sequence_stop_count",
        "number_of_station_stops",
        "has_single_station",
        "sequence_matches_route_stops",
        "reason_not_usable",
    ]
    assert detail.empty


def test_route_validity_analysis_reason_is_single_string(tmp_path: Path) -> None:
    quality_analysis = load_quality_module()
    data = {
        "routes": pd.DataFrame([{"route_id": "route_1", "route_score": "Low"}]),
        "quality": pd.DataFrame(
            [
                {
                    "route_id": "route_1",
                    "route_exists_in_actual_sequences": False,
                    "route_stop_count": 3,
                    "sequence_stop_count": 0,
                    "sequence_matches_route_stops": False,
                    "number_of_station_stops": 1,
                    "has_single_station": True,
                    "can_use_for_training": False,
                }
            ]
        ),
    }

    summary, _merged = quality_analysis.route_validity_analysis(data, tmp_path)

    assert summary["routes_not_usable_for_training"] == 1
    detail = pd.read_csv(tmp_path / "unusable_routes_detail.csv")
    assert len(detail) == 1
    assert detail.loc[0, "reason_not_usable"] == "missing_actual_sequence"
    assert isinstance(detail.loc[0, "reason_not_usable"], str)
