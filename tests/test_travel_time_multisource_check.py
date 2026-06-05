from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")


def load_multisource_module():
    module_path = Path(__file__).resolve().parents[1] / "03_travel_time_multisource_check.py"
    spec = importlib.util.spec_from_file_location("travel_time_multisource", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_multisource_travel_time_outputs(tmp_path: Path) -> None:
    module = load_multisource_module()
    input_dir = tmp_path / "processed_outputs"
    data_root = tmp_path / "amazon_last_mile"
    output_dir = input_dir / "travel_time_multisource_outputs"
    input_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {"route_id": "build_route", "from_stop": "A", "to_stop": "B", "position": 0},
            {"route_id": "apply_route", "from_stop": "C", "to_stop": "D", "position": 0},
            {"route_id": "eval_route", "from_stop": "E", "to_stop": "F", "position": 0},
            {"route_id": "duplicate_route", "from_stop": "G", "to_stop": "H", "position": 0},
            {"route_id": "missing_route", "from_stop": "I", "to_stop": "J", "position": 0},
        ]
    ).to_csv(input_dir / "actual_transitions.csv", index=False)
    write_text(
        data_root / "almrrc2021-data-training" / "model_build_inputs" / "travel_times.json",
        '{"build_route":{"A":{"B":11}},"duplicate_route":{"G":{"H":44}}}',
    )
    write_text(
        data_root / "almrrc2021-data-training" / "model_apply_inputs" / "new_travel_times.json",
        '{"apply_route":{"C":{"D":22}},"duplicate_route":{"G":{"H":45}}}',
    )
    write_text(
        data_root / "almrrc2021-data-evaluation" / "model_apply_inputs" / "eval_travel_times.json",
        '{"eval_route":{"E":{"F":33}}}',
    )

    module.run_check(input_dir=input_dir, data_root=data_root, output_dir=output_dir, max_routes=None)

    lookup = pd.read_csv(output_dir / "route_travel_time_source_lookup.csv")
    lookup_by_route = lookup.set_index("route_id")
    assert lookup_by_route.loc["build_route", "travel_time_source"] == "training_build"
    assert lookup_by_route.loc["apply_route", "travel_time_source"] == "training_apply"
    assert lookup_by_route.loc["eval_route", "travel_time_source"] == "evaluation_apply"
    assert lookup_by_route.loc["duplicate_route", "travel_time_source"] == "multiple_sources"
    assert lookup_by_route.loc["duplicate_route", "source_count"] == 2
    assert lookup_by_route.loc["missing_route", "travel_time_source"] == "missing_source"

    detail = pd.read_csv(output_dir / "actual_transition_travel_time_check_multisource.csv")
    detail_by_route = detail.set_index("route_id")
    assert detail_by_route.loc["duplicate_route", "travel_time_source"] == "multiple_sources"
    assert detail_by_route.loc["duplicate_route", "travel_time_missing"] == 0
    assert detail_by_route.loc["missing_route", "travel_time_source"] == "missing_source"
    assert detail_by_route.loc["missing_route", "travel_time_missing"] == 1

    summary = pd.read_csv(output_dir / "travel_time_integrity_summary_multisource.csv").iloc[0]
    assert summary["total_routes_checked"] == 5
    assert summary["routes_in_multiple_sources"] == 1
    assert summary["routes_missing_from_all_sources"] == 1
    assert summary["transitions_with_travel_time"] == 4
    assert summary["transitions_missing_travel_time"] == 1
