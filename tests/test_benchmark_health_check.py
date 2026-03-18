from __future__ import annotations

import json
from pathlib import Path

from knowledge3d.tools.benchmark_health_check import run_health_check


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
