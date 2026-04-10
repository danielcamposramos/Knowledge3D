from __future__ import annotations

import json
from pathlib import Path

from scripts import run_arc3_agent, run_full_benchmark


def _fake_tablet_suite(args):
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    hardware = {"logical_cores": 8, "physical_cores": 4}
    summary = {
        "elapsed_seconds": 1.25,
        "log_dir": str(log_dir),
        "hardware_profile": hardware,
        "sleep_consolidation": {"status": "completed"},
        "execution_artifacts": {
            "summary": str(log_dir / "summary.execution.json"),
            "full_results": str(log_dir / "full_results.execution.json"),
        },
        "orchestrator": {"session_model": "one_live_knowledgeverse"},
    }
    results = {
        "mmlu": {"suite": "mmlu", "total": 4, "correct": 4, "accuracy": 1.0, "results": []},
        "imo": {"suite": "imo", "total": 3, "correct": 3, "accuracy": 1.0, "results": []},
        "gsm8k": {"suite": "gsm8k", "total": 2, "correct": 2, "accuracy": 1.0, "results": []},
        "lhe": {"suite": "lhe", "total": 2, "correct": 2, "accuracy": 1.0, "results": []},
        "arc2": {"suite": "arc2", "total": 2, "correct": 2, "accuracy": 1.0, "results": []},
        "math": {"status": "skipped", "reason": "math_count<=0", "results": []},
        "amc_aime": {"status": "skipped", "reason": "amc_aime_count<=0", "results": []},
        "omni_math": {"status": "skipped", "reason": "omni_math_count<=0", "results": []},
    }
    for artifact in summary["execution_artifacts"].values():
        Path(artifact).write_text("{}", encoding="utf-8")
    return {"summary": summary, "results": results}


def test_run_full_benchmark_writes_phase_e_logs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_full_benchmark, "run_tablet_benchmark_suite", _fake_tablet_suite)

    payload = run_full_benchmark.run_full_benchmark(
        mmlu_count=4,
        imo_count=3,
        gsm8k_count=2,
        lhe_count=2,
        arc2_count=2,
        arc3_count=5,
        storage_root=tmp_path / "storage",
        log_root=tmp_path / "logs",
    )

    summary = payload["summary"]
    log_dir = Path(summary["log_dir"])
    assert log_dir.exists()
    assert (log_dir / "summary.json").exists()
    assert (log_dir / "full_results.json").exists()

    decoded = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
    assert decoded["suites"]["mmlu"]["correct"] == 4
    assert decoded["suites"]["imo"]["correct"] == 3
    assert decoded["suites"]["gsm8k"]["correct"] == 2
    assert decoded["suites"]["lhe"]["correct"] == 2
    assert decoded["suites"]["arc2"]["correct"] == 2
    assert set(decoded["suites"]) == {"mmlu", "imo", "gsm8k", "lhe", "arc2", "math", "amc_aime", "omni_math"}
    assert decoded["archived_suites"]["arc3_local"]["requested_count"] == 5
    assert decoded["archived_suites"]["arc3_local"]["status"] == "archived"
    assert decoded["hardware_profile"]["logical_cores"] >= 1


def test_run_arc3_agent_helpers_point_to_project_logs():
    path = run_arc3_agent.default_live_log_path()
    assert str(path).startswith("/K3D/Knowledge3D.local/logs/arc3_live_")
    assert path.suffix == ".jsonl"
    assert run_arc3_agent.scorecard_url("https://three.arcprize.org", "abc123") == (
        "https://three.arcprize.org/scorecards/abc123"
    )


def test_run_arc3_agent_normalizes_wrapped_frame():
    wrapped = [[[1, 2], [3, 4]]]
    plain = [[1, 2], [3, 4]]
    assert run_arc3_agent.normalize_frame(wrapped) == plain
    assert run_arc3_agent.normalize_frame(plain) == plain
    nested = {"frame": {"layers": [wrapped]}}
    assert run_arc3_agent.normalize_frame(nested) == plain
