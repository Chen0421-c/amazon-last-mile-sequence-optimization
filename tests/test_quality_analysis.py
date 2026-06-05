from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

DETAIL_COLUMNS = [
    "route_id",
    "route_score",
    "route_stop_count",
    "sequence_stop_count",
    "number_of_station_stops",
    "has_single_station",
    "sequence_matches_route_stops",
    "reason_not_usable",
]


def load_quality_module():
    module_path = Path(__file__).resolve().parents[1] / "02_data_quality_and_outlier_analysis.py"
    spec = importlib.util.spec_from_file_location("quality_analysis", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_data(quality_row: dict[str, object]) -> dict[str, object]:
    route_id = str(quality_row["route_id"])
    return {
        "routes": pd.DataFrame([{"route_id": route_id, "route_score": "High"}]),
        "quality": pd.DataFrame([quality_row]),
    }


def read_detail(tmp_path: Path):
    return pd.read_csv(tmp_path / "unusable_routes_detail.csv")


def test_route_validity_analysis_writes_empty_unusable_detail(tmp_path: Path) -> None:
    quality_analysis = load_quality_module()
    data = make_data(
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
    )

    summary, _merged = quality_analysis.route_validity_analysis(data, tmp_path)

    assert summary["routes_not_usable_for_training"] == 0
    detail = read_detail(tmp_path)
    assert list(detail.columns) == DETAIL_COLUMNS
    assert detail.empty


@pytest.mark.parametrize(
    ("quality_updates", "expected_reason"),
    [
        (
            {
                "route_exists_in_actual_sequences": False,
                "route_stop_count": 3,
                "sequence_stop_count": 0,
                "sequence_matches_route_stops": False,
                "number_of_station_stops": 1,
            },
            "missing_actual_sequence",
        ),
        (
            {
                "route_exists_in_actual_sequences": True,
                "route_stop_count": 3,
                "sequence_stop_count": 2,
                "sequence_matches_route_stops": True,
                "number_of_station_stops": 1,
            },
            "stop_count_mismatch",
        ),
        (
            {
                "route_exists_in_actual_sequences": True,
                "route_stop_count": 3,
                "sequence_stop_count": 3,
                "sequence_matches_route_stops": True,
                "number_of_station_stops": 2,
            },
            "multiple_stations",
        ),
    ],
)
def test_route_validity_analysis_reason_is_single_string(
    tmp_path: Path, quality_updates: dict[str, object], expected_reason: str
) -> None:
    quality_analysis = load_quality_module()
    quality_row = {
        "route_id": "route_1",
        "route_exists_in_actual_sequences": True,
        "route_stop_count": 3,
        "sequence_stop_count": 3,
        "sequence_matches_route_stops": True,
        "number_of_station_stops": 1,
        "has_single_station": True,
        "can_use_for_training": False,
    }
    quality_row.update(quality_updates)

    summary, _merged = quality_analysis.route_validity_analysis(make_data(quality_row), tmp_path)

    assert summary["routes_not_usable_for_training"] == 1
    detail = read_detail(tmp_path)
    assert len(detail) == 1
    assert detail.loc[0, "reason_not_usable"] == expected_reason
    assert isinstance(detail.loc[0, "reason_not_usable"], str)
