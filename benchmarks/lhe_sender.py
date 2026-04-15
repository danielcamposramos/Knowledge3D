#!/usr/bin/env python3
"""Send Last Humanity Exam questions to running K3D daemon via ROUTE command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from knowledge3d.tablet.wine.question_wine import build_question_session_tape

try:
    from benchmarks.daemon_client import DaemonClient
except ModuleNotFoundError:  # Direct script execution.
    from daemon_client import DaemonClient


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    candidates = [
        Path("/K3D/Knowledge3D.local/datasets/last_humanity_exam"),
        Path("../Knowledge3D.local/datasets/last_humanity_exam"),
        Path("/K3D/Knowledge3D.local/datasets/exams/hle-src"),
        Path("../Knowledge3D.local/datasets/exams/hle-src"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("")


def _load_questions(dataset_root: Path, max_questions: int) -> list[dict[str, Any]]:
    candidates = [
        dataset_root / "last_humanity_exam.json",
        dataset_root / "questions.json",
        dataset_root / "dataset.json",
    ]
    out: list[dict[str, Any]] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("questions", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            continue
        for idx, row in enumerate(rows):
            if len(out) >= max_questions:
                return out
            if not isinstance(row, dict):
                continue
            question = str(row.get("question_text") or row.get("question") or "").strip()
            if not question:
                continue
            out.append(
                {
                    "task_id": str(row.get("id") or f"lhe_{idx}"),
                    "question": question,
                    "options": row.get("options") if isinstance(row.get("options"), list) else [],
                    "domain": str(row.get("domain") or "multi"),
                }
            )
    if out:
        return out
    return [
        {
            "task_id": "lhe_synth_1",
            "question": "If all A are B and all B are C, what follows?",
            "options": ["Some A are not C", "All A are C", "No B are C", "All C are A"],
            "domain": "logic",
        }
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument(
        "--allow-zero-gpu",
        action="store_true",
        help="Disable sovereignty assertion (debug only).",
    )
    args = parser.parse_args()

    client = DaemonClient(host=args.host, port=args.port)
    rows = _load_questions(_resolve_dataset_path(args.dataset_path), max_questions=max(1, int(args.max_questions)))

    ok = 0
    tape = build_question_session_tape(
        session_id=f"lhe_sender_{int(time.time())}",
        suite_name="lhe_sender",
        rows=[
            {
                "id": str(row["task_id"]),
                "question": str(row["question"]),
                "options": list(row.get("options") or []),
                "domain": str(row.get("domain") or "multi"),
            }
            for row in rows
        ],
        use_enriched=True,
    )
    response = client.tablet_session_run_tape(tape)
    results = list(response.get("results") or []) if response.get("status") == "ok" else []
    for row, result_row in zip(rows, results):
        emitted = dict(result_row.get("emitted") or {})
        task_result = dict(emitted.get("task_result") or {})
        if not bool(args.allow_zero_gpu) and not bool(task_result.get("gpu_execution", False)):
            raise RuntimeError(f"lhe_sender:{row['task_id']}: missing_gpu_execution")
        if str(emitted.get("status") or "").lower() == "success":
            ok += 1

    print(
        json.dumps(
            {
                "sender": "lhe_sender",
                "total": len(rows),
                "ok": ok,
                "failed": len(rows) - ok,
            },
            indent=2,
        )
    )
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
