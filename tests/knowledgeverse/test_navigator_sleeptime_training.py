from __future__ import annotations

import json

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.navigator_specialist import NAVIGATOR_SWARM_NAME
from knowledge3d.knowledgeverse.sleeptime import SleepTimeConsolidation


class _FakeEmbedEngine:
    def embed_sentence_gpu(self, text: str):
        token = str(text or "")
        base = max(1, sum((idx + 1) * ord(ch) for idx, ch in enumerate(token)))
        return [float(((idx + 1) * base) % 997) / 997.0 for idx in range(16)]


def test_knowledgeverse_registers_navigator_lane_on_cold_boot(tmp_path) -> None:
    kv = Knowledgeverse(storage_root=tmp_path / "kv_nav_cold_boot", eager_load_default_galaxies=False)
    assert NAVIGATOR_SWARM_NAME in kv.adaptive_swarm.base.specialists


def test_sleeptime_trains_navigator_lane_from_trace_packets(tmp_path, monkeypatch) -> None:
    storage_root = tmp_path / "kv_nav_sleeptime"
    kv = Knowledgeverse(storage_root=storage_root, eager_load_default_galaxies=False)
    monkeypatch.setattr(kv, "get_gpu_query_embedding_engine", lambda: _FakeEmbedEngine())
    monkeypatch.setattr(kv, "_sovereign_hot_path", None)

    logs_dir = storage_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    session_id = "navigator-session"
    health_log_path = logs_dir / "health_log.jsonl"
    rows = [
        {
            "session_id": session_id,
            "suite": "mmlu",
            "question": "What is the capital of France?",
            "expected": "Paris",
            "correct": True,
            "retrieved_stars": [{"id": "reality_fact_seed", "metadata": {"reality_refs": ["capital_city"]}}],
        },
        {
            "session_id": session_id,
            "suite": "math",
            "question": "What is 17 * 9?",
            "answer": "100",
            "correct": False,
            "retrieved_stars": [{"id": "math_fact_seed", "metadata": {"math_refs": ["arithmetic_product"]}}],
        },
        {
            "session_id": session_id,
            "suite": "lhe",
            "question": "If all A are B and all B are C, what follows?",
            "expected": "All A are C",
            "correct": True,
            "retrieved_stars": [{"id": "meta_inference_seed", "metadata": {"meta_refs": ["syllogism"]}}],
        },
    ]
    with health_log_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    run_state_path = logs_dir / "health_log.full.run_state.json"
    run_state_path.write_text(json.dumps({"session_id": session_id}), encoding="utf-8")

    sleeptime = SleepTimeConsolidation(
        knowledgeverse=kv,
        journal_path=logs_dir / "sleeptime_journal.jsonl",
        health_log_path=health_log_path,
    )
    summary = sleeptime._run_contrastive_training()

    navigator_summary = summary["specialists_trained"]["navigator"]
    assert navigator_summary["trained"] is True
    assert navigator_summary["steps"] > 0
    assert kv.adaptive_swarm.specialist_steps[NAVIGATOR_SWARM_NAME] > 0
    state_payload = kv.navigator_specialist.weight_store.load()
    assert state_payload["navigator_recent_traces"]
    assert state_payload["navigator_training_state"]["last_training_stats"]["steps"] > 0
