#!/usr/bin/env python3
"""Send ARC tasks to running K3D daemon via ROUTE command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from knowledge3d.tablet.wine.game2d_wine import arc2_game_envelope

try:
    from benchmarks.daemon_client import DaemonClient
except ModuleNotFoundError:  # Direct script execution.
    from daemon_client import DaemonClient


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    candidates = [
        Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
        Path("/K3D/Knowledge3D.local/datasets/arc_agi_2/evaluation"),
        Path("../Knowledge3D.local/datasets/exams/arc-src/data/evaluation"),
        Path("../Knowledge3D.local/datasets/arc_agi_2/evaluation"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("")


def _load_tasks(dataset_root: Path, max_tasks: int) -> list[dict[str, Any]]:
    if dataset_root.exists():
        tasks: list[dict[str, Any]] = []
        for file_path in sorted(dataset_root.glob("*.json"))[:max_tasks]:
            try:
                payload = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            train = payload.get("train")
            test = payload.get("test")
            if not isinstance(train, list) or not isinstance(test, list) or not test:
                continue
            input_grid = test[0].get("input")
            if input_grid is None:
                continue
            tasks.append(
                {
                    "task_id": file_path.stem,
                    "training_examples": train,
                    "input_grid": input_grid,
                    "expected_output": test[0].get("output"),
                }
            )
        if tasks:
            return tasks
    return [
        {
            "task_id": "synthetic_flip_h",
            "training_examples": [
                {"input": [[1, 2], [3, 4]], "output": [[2, 1], [4, 3]]},
                {"input": [[5, 6], [7, 8]], "output": [[6, 5], [8, 7]]},
            ],
            "input_grid": [[9, 0], [1, 2]],
            "expected_output": [[0, 9], [2, 1]],
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument(
        "--allow-zero-gpu",
        action="store_true",
        help="Disable sovereignty assertion (debug only).",
    )
    args = parser.parse_args()

    client = DaemonClient(host=args.host, port=args.port)
    tasks = _load_tasks(_resolve_dataset_path(args.dataset_path), max_tasks=max(1, int(args.max_tasks)))

    ok = 0
    use_enriched = True
    enforce_gpu = not bool(args.allow_zero_gpu)
    for task in tasks:
        envelope = arc2_game_envelope(
            task_id=str(task["task_id"]),
            training_examples=list(task["training_examples"]),
            input_grid=task["input_grid"],
            expected_output=task.get("expected_output"),
        )
        payload = envelope.to_route_payload(use_enriched=use_enriched)
        response = client.send(payload)
        if enforce_gpu:
            client.assert_gpu_for_solved_command(
                response,
                solved_key="task_result",
                context=f"arc_sender:{task['task_id']}",
            )
        task_result = response.get("task_result")
        if response.get("status") == "ok" and isinstance(task_result, dict) and task_result.get("status") == "success":
            ok += 1

    print(
        json.dumps(
            {
                "sender": "arc_sender",
                "total": len(tasks),
                "ok": ok,
                "failed": len(tasks) - ok,
            },
            indent=2,
        )
    )
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
