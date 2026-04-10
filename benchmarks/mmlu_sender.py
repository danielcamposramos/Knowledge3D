#!/usr/bin/env python3
"""Send MMLU questions to running K3D daemon via ROUTE command."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from knowledge3d.tablet.wine.question_wine import mmlu_question_envelope

try:
    from benchmarks.daemon_client import DaemonClient
except ModuleNotFoundError:  # Direct script execution.
    from daemon_client import DaemonClient


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    candidates = [
        Path("/K3D/K3D_llama_cpp/datasets/MMLU/data"),
        Path("../K3D_llama_cpp/datasets/MMLU/data"),
        Path("/K3D/Knowledge3D.local/datasets/MMLU/data"),
        Path("../Knowledge3D.local/datasets/MMLU/data"),
    ]
    for candidate in candidates:
        if candidate.exists() and (candidate / "test").exists():
            return candidate
    return Path("")


def _load_questions(dataset_root: Path, max_questions: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    test_dir = dataset_root / "test"
    if test_dir.exists():
        for csv_file in sorted(test_dir.glob("*_test.csv")):
            subject = csv_file.stem.replace("_test", "")
            try:
                with csv_file.open("r", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    for idx, row in enumerate(reader):
                        if len(out) >= max_questions:
                            return out
                        if len(row) < 5:
                            continue
                        q = row[0].strip()
                        if not q:
                            continue
                        options = [s.strip() for s in row[1:5]]
                        out.append(
                            {
                                "task_id": f"mmlu_{subject}_{idx}",
                                "subject": subject,
                                "question": q,
                                "options": options,
                            }
                        )
            except Exception:
                continue
    if out:
        return out
    return [
        {
            "task_id": "mmlu_synth_1",
            "subject": "formal_logic",
            "question": "If all A are B and all B are C, what follows?",
            "options": ["Some A are not C", "All A are C", "No B are C", "All C are A"],
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
    use_enriched = True
    enforce_gpu = not bool(args.allow_zero_gpu)
    for row in rows:
        envelope = mmlu_question_envelope(
            task_id=str(row["task_id"]),
            question=str(row["question"]),
            options=list(row.get("options") or []),
            subject=str(row.get("subject") or "general"),
        )
        payload = envelope.to_route_payload(use_enriched=use_enriched)
        response = client.send(payload)
        if enforce_gpu:
            client.assert_gpu_for_solved_command(
                response,
                solved_key="task_result",
                context=f"mmlu_sender:{row['task_id']}",
            )
        task_result = response.get("task_result")
        if response.get("status") == "ok" and isinstance(task_result, dict) and task_result.get("status") == "success":
            ok += 1

    print(
        json.dumps(
            {
                "sender": "mmlu_sender",
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
