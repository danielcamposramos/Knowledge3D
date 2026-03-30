from __future__ import annotations

import json

from scripts import run_arc3_agent, run_full_benchmark


class _FakeKnowledgeverse:
    def __init__(self, storage_root=None) -> None:
        self.storage_root = storage_root


def _fake_native_suite(*, suite_name, count, knowledgeverse, **kwargs):
    del kwargs
    _fake_native_suite.seen_kv_ids.append(id(knowledgeverse))
    return {
        "suite": suite_name,
        "total": count,
        "correct": count,
        "accuracy": 1.0,
        "results": [{"suite": suite_name, "id": f"{suite_name}_{i}", "kv_id": id(knowledgeverse)} for i in range(count)],
    }


def test_run_full_benchmark_writes_phase_e_logs(tmp_path, monkeypatch):
    _fake_native_suite.seen_kv_ids = []
    monkeypatch.setattr(run_full_benchmark, "_run_native_suite", _fake_native_suite)
    monkeypatch.setattr(run_full_benchmark, "_write_jsonl", lambda path, rows: None)
    monkeypatch.setattr(run_full_benchmark, "Knowledgeverse", _FakeKnowledgeverse)
    monkeypatch.setattr(run_full_benchmark, "_ensure_full_benchmark_runtime", lambda: None)

    payload = run_full_benchmark.run_full_benchmark(
        mmlu_count=4,
        gsm8k_count=2,
        lhe_count=2,
        arc2_count=2,
        arc3_count=5,
        storage_root=tmp_path / "storage",
        log_root=tmp_path / "logs",
    )

    summary = payload["summary"]
    log_dir = tmp_path / "logs" / f"phase_e_{summary['timestamp']}"
    assert log_dir.exists()
    assert (log_dir / "summary.json").exists()
    assert (log_dir / "full_results.json").exists()

    decoded = json.loads((log_dir / "summary.json").read_text(encoding="utf-8"))
    assert decoded["suites"]["mmlu"]["correct"] == 4
    assert decoded["suites"]["gsm8k"]["correct"] == 2
    assert decoded["suites"]["lhe"]["correct"] == 2
    assert decoded["suites"]["arc2"]["correct"] == 2
    assert decoded["suites"]["arc3_local"]["correct"] == 5
    assert set(decoded["suites"]) == {"mmlu", "gsm8k", "lhe", "arc2", "arc3_local"}
    assert len(set(_fake_native_suite.seen_kv_ids)) == 1


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
