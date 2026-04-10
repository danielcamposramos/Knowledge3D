"""Closed-loop ARC scoring harness over the canonical GAME_2D tablet/WINE path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc_submission_formatter import (
    format_arc_submission,
    validate_arc_submission,
    write_arc_submission,
)
from benchmarks.arc_task_galaxy_seeder import ARC_GRID_POLY_SYMBOL, seed_task
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import OP_POLY_BUILD
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.arc2_wine import arc2_game_envelope


TAG_POLY = 0x04


def decode_grid_from_star(bridge: Any, root_idx: int) -> list[list[int]] | None:
    """Test-only helper to reconstruct a compact grid STAR leaf."""
    if int(root_idx) < 0:
        return None
    node = bridge.read_cas_node(int(root_idx))
    if int(node["opcode"]) != int(OP_POLY_BUILD):
        return None
    if ((int(node["flags"]) >> 8) & 0xFF) != TAG_POLY:
        return None
    if (int(node["next"]) & 0xFFFF) != ARC_GRID_POLY_SYMBOL:
        return None
    coeff_count = ((int(node["next"]) >> 16) & 0xFFFF) + 1
    coeffs = bridge.read_cas_coefficients(int(node["payload"]), int(coeff_count))
    if len(coeffs) < 2:
        return None
    height = max(0, int(round(coeffs[0])))
    width = max(0, int(round(coeffs[1])))
    expected = 2 + (height * width)
    if not height or not width or len(coeffs) < expected:
        return None
    values = [max(0, int(round(value))) for value in coeffs[2:expected]]
    rows: list[list[int]] = []
    cursor = 0
    for _ in range(height):
        rows.append(values[cursor: cursor + width])
        cursor += width
    return rows


def decode_arc_predictions(result: Any) -> list[list[list[int]]]:
    """Extract up to two ARC grids from a task result or tablet submit payload."""
    predictions: list[list[list[int]]] = []
    for container in _prediction_containers(result):
        for key in (
            "predicted_grid",
            "output_grid",
            "answer_grid",
            "answer",
            "response",
            "result",
            "alt_grid",
            "second_prediction",
            "predictions",
        ):
            if not isinstance(container, dict) or key not in container:
                continue
            for grid in _extract_grids(container.get(key)):
                if grid not in predictions:
                    predictions.append(grid)
                if len(predictions) >= 2:
                    return predictions[:2]
    return predictions[:2]


def _prediction_containers(result: Any) -> list[dict[str, Any]]:
    containers: list[dict[str, Any]] = []
    queue = [result]
    seen: set[int] = set()
    while queue:
        current = queue.pop(0)
        if not isinstance(current, dict):
            continue
        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)
        containers.append(current)
        for key in ("emitted", "task_result", "response", "raw_response"):
            nested = current.get(key)
            if isinstance(nested, dict):
                queue.append(nested)
    return containers


def _extract_grids(value: Any) -> list[list[list[int]]]:
    direct = _coerce_grid(value)
    if direct is not None:
        return [direct]
    grids: list[list[list[int]]] = []
    if isinstance(value, list):
        for item in value:
            item_grid = _coerce_grid(item)
            if item_grid is not None and item_grid not in grids:
                grids.append(item_grid)
    return grids


def _coerce_grid(value: Any) -> list[list[int]] | None:
    if _is_grid(value):
        return [[int(cell) for cell in row] for row in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return _coerce_grid(parsed)
    return None


def _is_grid(value: Any) -> bool:
    if not isinstance(value, list) or not value:
        return False
    for row in value:
        if not isinstance(row, list):
            return False
        for cell in row:
            if not isinstance(cell, int):
                return False
    return True


def _resolve_tasks_dir(tasks_dir: str | Path | None) -> Path:
    if tasks_dir is not None:
        return Path(tasks_dir)
    candidates = [
        Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/arc_agi_2/evaluation"),
        Path("../Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
        Path("../Knowledge3D.local/datasets/arc_agi/ARC-AGI-master/data/evaluation"),
        Path("../Knowledge3D.local/datasets/arc_agi_2/evaluation"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("arc_dataset_missing: no ARC evaluation directory found at the canonical dataset roots")


def _load_task_files(tasks_dir: str | Path | None, max_tasks: int) -> tuple[Path, list[Path]]:
    tasks_root = _resolve_tasks_dir(tasks_dir)
    files = sorted(tasks_root.glob("*.json"))[: max(1, int(max_tasks))]
    if not files:
        raise FileNotFoundError(f"arc_dataset_empty: no ARC task JSON files found in {tasks_root}")
    return tasks_root, files


def _solve_task(
    *,
    tablet: HeadlessTabletMPC,
    task_id: str,
    task_json: dict[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    train_rows = list(task_json.get("train") or [])
    test_rows = list(task_json.get("test") or [])
    rows: list[dict[str, Any]] = []
    correct_predictions = 0
    for sample_idx, sample in enumerate(test_rows):
        if not isinstance(sample, dict):
            continue
        expected = sample.get("output")
        envelope = arc2_game_envelope(
            task_id=f"{task_id}:{int(sample_idx)}",
            training_examples=train_rows,
            input_grid=sample.get("input"),
            expected_output=expected,
        )
        tablet_result = tablet.submit(envelope, use_enriched=True)
        emitted = dict(tablet_result["emitted"])
        task_result = dict(emitted.get("task_result") or {})
        predictions = decode_arc_predictions(tablet_result)
        primary_prediction = predictions[0] if predictions else emitted.get("output_grid")
        secondary_prediction = predictions[1] if len(predictions) > 1 else None
        correct = bool(emitted.get("correct", False))
        correct_predictions += int(correct)
        rows.append(
            {
                "task_id": str(task_id),
                "sample_index": int(sample_idx),
                "input_grid": sample.get("input"),
                "predicted": primary_prediction,
                "secondary_prediction": secondary_prediction,
                "predictions": predictions,
                "expected": expected,
                "correct": correct,
                "match_type": str(
                    task_result.get("program_type")
                    or task_result.get("solver")
                    or emitted.get("actual_result_kind")
                    or "knowledgeverse_game2d"
                ),
                "pair_idx": None,
                "predicted_root": None,
                "star_id": str(task_result.get("winner_star_id") or task_result.get("winner_star") or ""),
                "task_transform": str(task_result.get("program_type") or task_result.get("solver") or "knowledgeverse"),
                "route_family": str(emitted.get("route_family") or ""),
                "failure_code": str(emitted.get("failure_code") or ""),
                "actual_result_kind": str(emitted.get("actual_result_kind") or ""),
                "tablet_contract": tablet_result.get("tablet_contract"),
            }
        )
    return correct_predictions, len(rows), rows


def run_evaluation(
    tasks_dir: str | Path | None,
    max_tasks: int = 50,
    *,
    submission_output: str | Path | None = None,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    tasks_root, files = _load_task_files(tasks_dir, max_tasks)
    kv = Knowledgeverse()
    tablet = HeadlessTabletMPC(knowledgeverse=kv)
    all_rows: list[dict[str, Any]] = []
    task_summaries: list[dict[str, Any]] = []
    correct = 0
    total_inputs = 0

    for path in files:
        task_json = json.loads(path.read_text(encoding="utf-8"))
        task_id = str(path.stem)
        seed_task(task_json, galaxy_manager=kv.galaxy_manager, task_id=task_id)
        task_correct, task_total, rows = _solve_task(tablet=tablet, task_id=task_id, task_json=task_json)
        correct += int(task_correct)
        total_inputs += int(task_total)
        all_rows.extend(rows)
        task_summaries.append(
            {
                "task_id": task_id,
                "correct": int(task_correct),
                "total": int(task_total),
                "score": float(task_correct / task_total) if task_total else 0.0,
                "match_types": [str(row["match_type"]) for row in rows],
                "task_transform": str(rows[0]["task_transform"]) if rows else "knowledgeverse",
            }
        )

    summary = {
        "tasks": len(files),
        "correct": int(correct),
        "total_inputs": int(total_inputs),
        "score": float(correct / total_inputs) if total_inputs else 0.0,
        "dataset_path": str(tasks_root),
        "task_summaries": task_summaries,
        "results": all_rows,
    }
    if submission_output is not None:
        submission = format_arc_submission(all_rows)
        errors = validate_arc_submission(submission)
        if errors:
            raise RuntimeError("ARC submission validation failed:\n- " + "\n- ".join(errors))
        write_arc_submission(submission_output, submission)
        summary["submission_output"] = str(submission_output)
    if summary_output is not None:
        output_path = Path(summary_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "tasks": summary["tasks"],
                "correct": summary["correct"],
                "total_inputs": summary["total_inputs"],
                "score": summary["score"],
            },
            indent=2,
        )
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--submission-output", default=None)
    parser.add_argument("--summary-output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_evaluation(
        args.tasks_dir,
        max_tasks=max(1, int(args.max_tasks)),
        submission_output=args.submission_output,
        summary_output=args.summary_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
