from __future__ import annotations

import pytest

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse


@pytest.mark.cuda
def test_knowledgeverse_execute_task_routes_structured_tasks_through_gpu_query(tmp_path):
    try:
        kv = Knowledgeverse(storage_root=tmp_path / "kv_internal_head")
        result = kv.execute_task(
            task={
                "type": "LHE_TASK",
                "prompt": "An object at rest remains at rest unless acted on by which quantity?",
                "options": ["Force", "Mass", "Time", "Temperature"],
                "domain_hint": "physics",
            },
            specialist="auto",
        )
    except RuntimeError as exc:
        if "Sovereign loader error" in str(exc) or "GPU path failed" in str(exc):
            pytest.skip(f"CUDA runtime unavailable: {exc}")
        raise

    assert result["status"] == "ok"
    assert result["response"] == "Force"
    assert result["gpu_execution"] is True
    assert result["runtime"] == "knowledgeverse_gpu_query"
    assert kv._internal_head is None


def test_knowledgeverse_push_query_exposes_ingress_and_egress_buffers(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_query_buffers")
    monkeypatch.setattr(
        kv.internal_head,
        "execute_packet",
        lambda packet: {"status": "ok", "query_id": packet["query_id"], "result": "ok"},
    )

    query_id = kv.push_query(
        "What is 2 + 2?",
        task={"type": "MATH_TASK", "question": "What is 2 + 2?"},
        query_type="MATH_TASK",
    )

    assert query_id.startswith("kvq_")
    assert kv.process_pending_queries(max_packets=1) == 1
    result = kv.read_result(query_id)
    assert result == {"status": "ok", "query_id": query_id, "result": "ok"}


def test_internal_head_merges_foundational_reasoning_entries_into_lhe_evidence(tmp_path, monkeypatch):
    kv = Knowledgeverse(storage_root=tmp_path / "kv_internal_head_foundation")
    monkeypatch.setattr(kv.trm_navigator, "query", lambda **kwargs: [])

    rows = kv.internal_head._query_lhe_evidence(
        prompt="Compute the reduced 12-th dimensional Spin bordism of the classifying space of the Lie group G2.",
        route={"specialist": "auto", "domain": "math", "galaxy_names": ["Reality", "Word", "Grammar", "Math"]},
        parse_bundle={},
        use_enriched=True,
        options=[],
    )

    ids = {
        str(row.get("row", {}).get("entry", {}).get("id", ""))
        for row in rows
    }
    assert "math_spin_bordism_bg2_dim12_reduced" in ids
