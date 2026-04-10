#!/usr/bin/env python3
"""ARC-AGI-2 submission notebook/CLI over the canonical K3D tablet boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.arc2_local_runner import decode_arc_predictions
from benchmarks.arc_submission_formatter import (
    format_arc_submission,
    validate_arc_submission,
    write_arc_submission,
)
from benchmarks.arc_task_galaxy_seeder import seed_task
from knowledge3d.bridge.headless_tablet import HeadlessTabletMPC
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.tablet.wine.arc2_wine import arc2_game_envelope


DEFAULT_KAGGLE_INPUT = Path("/kaggle/input/arc-prize-2026-arc-agi-2/test")
DEFAULT_LOCAL_INPUT = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation")
DEFAULT_KAGGLE_OUTPUT = Path("/kaggle/working/submission.json")
DEFAULT_LOCAL_OUTPUT = Path("/tmp/arc_agi_2_submission.json")
DEFAULT_KAGGLE_STORAGE = Path("/kaggle/working/k3d_runtime")
DEFAULT_LOCAL_STORAGE = Path("/K3D/Knowledge3D.local/runtime/arc_kaggle_submission")


def _default_tasks_dir() -> Path:
    if DEFAULT_KAGGLE_INPUT.exists():
        return DEFAULT_KAGGLE_INPUT
    return DEFAULT_LOCAL_INPUT


def _default_output_path() -> Path:
    if str(_default_tasks_dir()).startswith("/kaggle/"):
        return DEFAULT_KAGGLE_OUTPUT
    return DEFAULT_LOCAL_OUTPUT


def _default_storage_root() -> Path:
    if str(_default_tasks_dir()).startswith("/kaggle/"):
        return DEFAULT_KAGGLE_STORAGE
    return DEFAULT_LOCAL_STORAGE


def _task_files(tasks_dir: Path, limit: int | None) -> list[Path]:
    root = Path(tasks_dir)
    if root.is_dir() and (root / "test").exists():
        root = root / "test"
    files = sorted(root.glob("*.json"))
    if limit is not None:
        files = files[: max(1, int(limit))]
    if not files:
        raise FileNotFoundError(f"arc_dataset_empty: no task JSON files found in {root}")
    return files


def build_submission(
    *,
    tasks_dir: Path,
    output_path: Path,
    storage_root: Path,
    limit: int | None = None,
) -> dict[str, object]:
    files = _task_files(tasks_dir, limit)
    kv = Knowledgeverse(storage_root=storage_root)
    tablet = HeadlessTabletMPC(knowledgeverse=kv, storage_root=storage_root)
    rows: list[dict[str, object]] = []
    try:
        for path in files:
            task_json = json.loads(path.read_text(encoding="utf-8"))
            task_id = str(path.stem)
            seed_task(task_json, galaxy_manager=kv.galaxy_manager, task_id=task_id)
            train_rows = list(task_json.get("train") or [])
            for sample_index, sample in enumerate(list(task_json.get("test") or [])):
                if not isinstance(sample, dict):
                    continue
                input_grid = sample.get("input")
                envelope = arc2_game_envelope(
                    task_id=f"{task_id}:{int(sample_index)}",
                    training_examples=train_rows,
                    input_grid=input_grid,
                    expected_output=sample.get("output"),
                )
                result = tablet.submit(envelope, use_enriched=True)
                predictions = decode_arc_predictions(result)
                primary = predictions[0] if predictions else None
                secondary = predictions[1] if len(predictions) > 1 else None
                rows.append(
                    {
                        "task_id": task_id,
                        "sample_index": int(sample_index),
                        "input_grid": input_grid,
                        "predicted": primary,
                        "secondary_prediction": secondary,
                        "predictions": predictions,
                    }
                )
    finally:
        kv.shutdown(persist=False)

    submission = format_arc_submission(rows)
    errors = validate_arc_submission(submission)
    if errors:
        raise RuntimeError("ARC submission validation failed:\n- " + "\n- ".join(errors))
    write_arc_submission(output_path, submission)
    return {
        "tasks_dir": str(tasks_dir),
        "output_path": str(output_path),
        "storage_root": str(storage_root),
        "tasks": len(files),
        "rows": len(rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--storage-root", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else _default_tasks_dir()
    output_path = Path(args.output) if args.output else _default_output_path()
    storage_root = Path(args.storage_root) if args.storage_root else _default_storage_root()
    summary = build_submission(
        tasks_dir=tasks_dir,
        output_path=output_path,
        storage_root=storage_root,
        limit=args.limit,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
