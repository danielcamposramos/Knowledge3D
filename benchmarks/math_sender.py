#!/usr/bin/env python3
"""Send Math benchmark questions to running K3D daemon via ROUTE command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from knowledge3d.tablet.wine.math_wine import build_math_session_tape

try:
    from benchmarks.daemon_client import DaemonClient
except ModuleNotFoundError:  # Direct script execution.
    from daemon_client import DaemonClient


def _resolve_dataset_path(dataset_path: str | None) -> Path:
    if dataset_path:
        return Path(dataset_path)
    candidates = [
        Path("/K3D/K3D_llama_cpp/datasets"),
        Path("/K3D/K3D_llama_cpp/datasets/GSM8K"),
        Path("/K3D/K3D_llama_cpp/datasets/math"),
        Path("/K3D/Knowledge3D.local/datasets/math_competitions"),
        Path("../Knowledge3D.local/datasets/math_competitions"),
        Path("data"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("")


def _load_questions(dataset_root: Path, max_questions: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if dataset_root.exists():
        for comp, fname in (("AMC", "amc_problems.json"), ("AIME", "aime_problems.json"), ("IMO", "imo_problems.json")):
            path = dataset_root / fname
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            records = payload.get("problems", []) if isinstance(payload, dict) else payload
            if not isinstance(records, list):
                continue
            for idx, row in enumerate(records):
                if len(out) >= max_questions:
                    return out
                if not isinstance(row, dict):
                    continue
                text = str(row.get("problem_text") or row.get("question") or "").strip()
                if not text:
                    continue
                out.append(
                    {
                        "task_id": str(row.get("id") or f"{comp.lower()}_{idx}"),
                        "competition": comp,
                        "question": text,
                    }
                )
        for path in (
            dataset_root / "grade_school_math" / "data" / "test.jsonl",
            dataset_root / "GSM8K" / "grade_school_math" / "data" / "test.jsonl",
            dataset_root / "data" / "train.jsonl",
            dataset_root / "math" / "data" / "train.jsonl",
        ):
            if not path.exists():
                continue
            family = "GSM8K" if "grade_school_math" in str(path) else "MATH"
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if len(out) >= max_questions:
                        return out
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = str(row.get("question") or row.get("problem") or "").strip()
                    if not text:
                        continue
                    competition = family
                    if family == "MATH":
                        math_type = str(row.get("type") or "MATH").strip()
                        competition = f"MATH:{math_type}" if math_type else "MATH"
                    out.append(
                        {
                            "task_id": str(row.get("id") or f"{family.lower()}_{idx}"),
                            "competition": competition,
                            "question": text,
                        }
                    )
    if out:
        return out
    return [{"task_id": "synthetic_math_1", "competition": "AMC", "question": "If 2x + 3 = 11, what is x?"}]


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
    questions = _load_questions(_resolve_dataset_path(args.dataset_path), max_questions=max(1, int(args.max_questions)))

    ok = 0
    tape = build_math_session_tape(
        session_id=f"math_sender_{int(time.time())}",
        suite_name="math_sender",
        rows=[
            {
                "id": str(row["task_id"]),
                "problem_text": str(row["question"]),
                "competition": str(row.get("competition") or ""),
            }
            for row in questions
        ],
        use_enriched=True,
    )
    response = client.tablet_session_run_tape(tape)
    results = list(response.get("results") or []) if response.get("status") == "ok" else []
    for row, result_row in zip(questions, results):
        emitted = dict(result_row.get("emitted") or {})
        task_result = dict(emitted.get("task_result") or {})
        if not bool(args.allow_zero_gpu) and not bool(task_result.get("gpu_execution", False)):
            raise RuntimeError(f"math_sender:{row['task_id']}: missing_gpu_execution")
        if str(emitted.get("status") or "").lower() == "success":
            ok += 1

    print(
        json.dumps(
            {
                "sender": "math_sender",
                "total": len(questions),
                "ok": ok,
                "failed": len(questions) - ok,
            },
            indent=2,
        )
    )
    return 0 if ok == len(questions) else 1


if __name__ == "__main__":
    raise SystemExit(main())
