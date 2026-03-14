from __future__ import annotations

import json
from pathlib import Path

from benchmarks.arc_agi_2_adapter import ArcAgi2Adapter
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


ARC_009D5C81_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 8, 0, 0, 0, 8, 0, 8, 0, 8],
    [0, 0, 0, 0, 0, 8, 0, 8, 0, 8, 0, 0, 0, 8],
    [0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8, 8],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

ARC_009D5C81_OUTPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 7, 0, 0, 0, 7, 0, 7, 0, 7],
    [0, 0, 0, 0, 0, 7, 0, 7, 0, 7, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

ARC_00576224_INPUT = [
    [3, 2],
    [7, 8],
]

ARC_00576224_OUTPUT = [
    [3, 2, 3, 2, 3, 2],
    [7, 8, 7, 8, 7, 8],
    [2, 3, 2, 3, 2, 3],
    [8, 7, 8, 7, 8, 7],
    [3, 2, 3, 2, 3, 2],
    [7, 8, 7, 8, 7, 8],
]

ARC_CONNECT_COLOR_PAIRS_INPUT = [
    [0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 2, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 1, 0, 0, 0, 1],
    [0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

ARC_CONNECT_COLOR_PAIRS_OUTPUT = [
    [0, 0, 0, 0, 0, 0],
    [0, 2, 2, 2, 2, 0],
    [0, 0, 0, 3, 0, 0],
    [0, 1, 1, 3, 1, 1],
    [0, 0, 0, 3, 0, 0],
    [0, 0, 0, 0, 0, 0],
]

ARC_PERIODIC_CONSENSUS_INPUT = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
    [1, 9, 0, 2],
    [3, 4, 3, 0],
]

ARC_PERIODIC_CONSENSUS_OUTPUT = [
    [1, 2, 1, 2],
    [3, 4, 3, 4],
    [1, 2, 1, 2],
    [3, 4, 3, 4],
]

ARC_PERIODIC_CONSENSUS_WITH_FRINGE_INPUT = [
    [1, 2, 0, 1, 2, 0, 9],
    [3, 4, 0, 3, 4, 0, 9],
    [0, 0, 0, 0, 0, 0, 9],
    [1, 2, 0, 1, 2, 0, 9],
    [3, 4, 0, 3, 4, 0, 9],
    [0, 0, 0, 0, 0, 0, 9],
    [9, 9, 9, 9, 9, 9, 9],
]

ARC_PERIODIC_CONSENSUS_WITH_FRINGE_OUTPUT = [
    [1, 2, 0, 1, 2, 0, 0],
    [3, 4, 0, 3, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [1, 2, 0, 1, 2, 0, 0],
    [3, 4, 0, 3, 4, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]

ARC_FILL_ENCLOSED_INPUT = [
    [2, 2, 2, 2, 2, 2, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 2, 0, 0],
    [2, 0, 0, 2, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 2, 0, 0],
    [2, 0, 0, 0, 0, 0, 2, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

ARC_FILL_ENCLOSED_OUTPUT = [
    [2, 2, 2, 2, 2, 2, 2, 0, 0],
    [2, 4, 4, 4, 4, 4, 2, 0, 0],
    [2, 4, 4, 4, 4, 4, 2, 0, 0],
    [2, 4, 4, 2, 4, 4, 2, 0, 0],
    [2, 4, 4, 4, 4, 4, 2, 0, 0],
    [2, 4, 4, 4, 4, 4, 2, 0, 0],
    [2, 2, 2, 2, 2, 2, 2, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

ARC_PACK_COMPONENTS_DIAGONAL_INPUT = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 0, 8, 8, 0, 6, 0, 0, 0, 0],
    [7, 0, 8, 8, 0, 6, 0, 3, 3, 0],
    [7, 0, 8, 8, 0, 6, 0, 3, 3, 0],
]

ARC_PACK_COMPONENTS_DIAGONAL_OUTPUT = [
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 6, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 6, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 3, 3, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]

ARC_SELF_PATTERN_COMPLEMENT_INPUT = [
    [0, 0, 3],
    [3, 3, 0],
    [0, 3, 0],
]

ARC_SELF_PATTERN_COMPLEMENT_OUTPUT = [
    [0, 0, 0, 0, 0, 0, 3, 3, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 3],
    [0, 0, 0, 0, 0, 0, 3, 0, 3],
    [3, 3, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 3, 0, 0, 3, 0, 0, 0],
    [3, 0, 3, 3, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 3, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 0, 0],
]


def _arc_eval_task(task_id: str) -> dict[str, object]:
    dataset_path = Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation") / f"{task_id}.json"
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def test_knowledgeverse_arc_query_returns_gpu_output_grid(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_query")
    result = kv.execute_task(
        task={
            "type": "ARC_TASK",
            "task_id": "009d5c81",
            "query": "solve arc transformation task",
            "training_examples": [],
            "input_grid": ARC_009D5C81_INPUT,
            "expected_output": ARC_009D5C81_OUTPUT,
        },
        route={"specialist": "visual", "galaxy_names": ["Drawing", "Grammar", "Tool"]},
        specialist="visual",
        domain_hint="visual",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["program_id"] == Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID
    assert result["match"]["id"] == "arc_eval_009d5c81"
    assert result["output_grid"] == ARC_009D5C81_OUTPUT
    assert any(
        ("GPU primitive plan" in step) or ("GPU grid transform" in step)
        for step in result["reasoning_trace"]
    )


def test_knowledgeverse_arc_checker_tile_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_checker_plan")
    result = kv.execute_task(
        task={
            "type": "ARC_TASK",
            "task_id": "00576224",
            "query": "solve arc transformation task",
            "training_examples": [],
            "input_grid": ARC_00576224_INPUT,
            "expected_output": ARC_00576224_OUTPUT,
        },
        route={"specialist": "visual", "galaxy_names": ["Drawing", "Grammar", "Tool"]},
        specialist="visual",
        domain_hint="visual",
    )

    assert result["status"] == "ok"
    assert result["gpu_execution"] is True
    assert result["program_id"] == Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID
    assert result["match"]["id"] == "arc_eval_00576224"
    assert result["output_grid"] == ARC_00576224_OUTPUT
    assert any("checker_tile_repeat_hflip_rows" in step for step in result["reasoning_trace"])


def test_knowledgeverse_arc_connect_color_pairs_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_connect_pairs")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=ARC_CONNECT_COLOR_PAIRS_INPUT,
        primitive_plan=[{"op": "connect_color_pairs"}],
    )

    assert output == ARC_CONNECT_COLOR_PAIRS_OUTPUT


def test_knowledgeverse_arc_periodic_consensus_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_periodic_consensus")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=ARC_PERIODIC_CONSENSUS_INPUT,
        primitive_plan=[{"op": "periodic_consensus_cleanup"}],
    )

    assert output == ARC_PERIODIC_CONSENSUS_OUTPUT


def test_knowledgeverse_arc_periodic_consensus_zeroes_incomplete_fringe_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_periodic_consensus_fringe")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=ARC_PERIODIC_CONSENSUS_WITH_FRINGE_INPUT,
        primitive_plan=[{"op": "periodic_consensus_cleanup"}],
    )

    assert output == ARC_PERIODIC_CONSENSUS_WITH_FRINGE_OUTPUT


def test_knowledgeverse_arc_fill_enclosed_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_fill_enclosed")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=ARC_FILL_ENCLOSED_INPUT,
        primitive_plan=[{"op": "fill_enclosed_by_size"}],
    )

    assert output == ARC_FILL_ENCLOSED_OUTPUT


def test_knowledgeverse_arc_pack_components_diagonal_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_pack_components")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=ARC_PACK_COMPONENTS_DIAGONAL_INPUT,
        primitive_plan=[{"op": "pack_color_components_diagonal"}],
    )

    assert output == ARC_PACK_COMPONENTS_DIAGONAL_OUTPUT


def test_knowledgeverse_arc_self_pattern_complement_tiling_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_self_pattern_complement")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=ARC_SELF_PATTERN_COMPLEMENT_INPUT,
        primitive_plan=[{"op": "self_pattern_complement_tiling"}],
    )

    assert output == ARC_SELF_PATTERN_COMPLEMENT_OUTPUT


def test_knowledgeverse_arc_separator_bridge_projection_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_separator_bridge")
    task = _arc_eval_task("05a7bcf2")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=task["test"][0]["input"],  # type: ignore[index]
        primitive_plan=[{"op": "separator_bridge_projection"}],
    )

    assert output == task["test"][0]["output"]  # type: ignore[index]


def test_knowledgeverse_arc_anchor_spiral_pair_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_anchor_spiral")
    task = _arc_eval_task("08573cc6")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=task["test"][0]["input"],  # type: ignore[index]
        primitive_plan=[{"op": "anchor_spiral_pair"}],
    )

    assert output == task["test"][0]["output"]  # type: ignore[index]


def test_knowledgeverse_arc_marker_axis_crop_plan_executes_on_gpu(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_marker_axis_crop")
    task = _arc_eval_task("0934a4d8")
    output = kv._execute_arc_primitive_plan_gpu(
        input_grid=task["test"][0]["input"],  # type: ignore[index]
        primitive_plan=[{"op": "marker_axis_crop"}],
    )

    assert output == task["test"][0]["output"]  # type: ignore[index]


def test_arc_adapter_prefers_knowledgeverse_query_path(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_adapter")
    adapter = ArcAgi2Adapter(knowledgeverse=kv)

    def _legacy_must_not_run(_task):
        raise AssertionError("legacy_arc_solver_should_not_run")

    adapter._solve_task_ptx_only = _legacy_must_not_run  # type: ignore[method-assign]
    solved = adapter.solve_task(
        {
            "id": "00576224",
            "train": [],
            "test": [{"input": ARC_00576224_INPUT, "output": ARC_00576224_OUTPUT}],
        }
    )

    assert solved["gpu_execution"] is True
    assert solved["correct"] is True
    assert solved["predicted"] == ARC_00576224_OUTPUT
    assert solved["program_id"] == Knowledgeverse.GPU_ARC_REASONING_PROGRAM_ID


def test_arc_first_ten_eval_tasks_stay_green_on_gpu_path(tmp_path) -> None:
    dataset_path = Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation")
    assert dataset_path.exists()

    kv = Knowledgeverse(storage_root=tmp_path / "kv_arc_eval10")
    summary = ARCAGI2Benchmark(
        knowledgeverse=kv,
        dataset_path=str(dataset_path),
        max_tasks=10,
        tablet_boundary=None,
    ).run_benchmark(use_enriched=True)

    assert summary["correct"] == 10
    assert summary["total_tasks"] == 10
    assert summary["accuracy"] == 1.0
    assert all(result.get("solver") == "knowledgeverse_gpu_query" for result in summary["results"])
