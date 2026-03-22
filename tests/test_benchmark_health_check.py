from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.tools.benchmark_health_check import evaluate_answer, load_questions, run_health_check


def test_run_health_check_logs_results(tmp_path: Path) -> None:
    log_path = tmp_path / "health_log.jsonl"

    def _query(row):
        return row["expected"]

    summary = run_health_check("gsm8k", 1, log_path, query_fn=_query)

    assert summary["suite"] == "gsm8k"
    assert summary["correct"] == 1
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["correct"] is True
    assert payload["suite"] == "gsm8k"


def test_load_questions_mmlu_uses_correct_letter_in_synthetic_fallback() -> None:
    rows = load_questions("mmlu", count=1)

    assert rows
    assert rows[0]["suite"] == "mmlu"
    assert rows[0]["expected"] in {"A", "B", "C", "D"}
    assert rows[0]["payload"]["correct_letter"] == rows[0]["expected"]


def test_evaluate_answer_accepts_mmlu_letter_and_arc_json_string() -> None:
    assert evaluate_answer("mmlu", "A", "A") is True
    assert evaluate_answer("arc", "[[1, 2], [3, 4]]", [[1, 2], [3, 4]]) is True


def test_run_health_check_resumes_with_session_rows(tmp_path: Path) -> None:
    log_path = tmp_path / "health_log.jsonl"

    def _query(row):
        return row["expected"]

    first = run_health_check("mmlu", 2, log_path, query_fn=_query, session_id="sess-1")
    second = run_health_check("mmlu", 2, log_path, query_fn=_query, session_id="sess-1")

    assert first["total"] == 2
    assert second["total"] == 2
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert all(line["session_id"] == "sess-1" for line in lines)
