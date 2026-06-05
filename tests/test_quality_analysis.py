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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_travel_time_integrity_check_streams_multiple_sources(tmp_path: Path) -> None:
    quality_analysis = load_quality_module()
    data_root = tmp_path / "amazon_last_mile"
    output_dir = tmp_path / "quality_outputs"
    output_dir.mkdir()
    transitions = pd.DataFrame(
        [
            {"route_id": "build_route", "from_stop": "A", "to_stop": "B", "position": 0},
            {"route_id": "apply_route", "from_stop": "C", "to_stop": "D", "position": 0},
            {"route_id": "eval_route", "from_stop": "E", "to_stop": "F", "position": 0},
            {"route_id": "missing_route", "from_stop": "G", "to_stop": "H", "position": 0},
        ]
    )
    write_text(
        data_root / "almrrc2021-data-training" / "model_build_inputs" / "travel_times.json",
        '{"build_route":{"A":{"B":11}}}',
    )
    write_text(
        data_root / "almrrc2021-data-training" / "model_apply_inputs" / "new_travel_times.json",
        '{"apply_route":{"C":{"D":22}}}',
    )
    write_text(
        data_root / "almrrc2021-data-evaluation" / "model_apply_inputs" / "eval_travel_times.json",
        '{"eval_route":{"E":{"F":33}}}',
    )

    route_completeness, summary, detail = quality_analysis.travel_time_integrity_check(
        transitions, data_root, output_dir, max_routes=None
    )

    source_by_route = dict(zip(detail["route_id"], detail["travel_time_source"]))
    assert source_by_route == {
        "build_route": "training_build",
        "apply_route": "training_apply",
        "eval_route": "evaluation_apply",
        "missing_route": "missing_source",
    }
    assert summary["total_actual_transitions_checked"] == 4
    assert summary["transitions_with_travel_time"] == 3
    assert summary["transitions_missing_travel_time"] == 1
    assert summary["missing_travel_time_ratio"] == 0.25
    complete_by_route = dict(zip(route_completeness["route_id"], route_completeness["has_complete_actual_transition_travel_time"]))
    assert complete_by_route == {
        "build_route": True,
        "apply_route": True,
        "eval_route": True,
        "missing_route": False,
    }
