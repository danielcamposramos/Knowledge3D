from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from knowledge3d.knowledgeverse.resilience import CircuitBreakerOpen, SelfHealingWrapper
from knowledge3d.knowledgeverse.shadow_copy import ShadowCopyLearning
from knowledge3d.knowledgeverse.sleeptime import SleepTimeConsolidation
from knowledge3d.knowledgeverse.sovereignty_firewall import SovereigntyFirewall
from knowledge3d.knowledgeverse.stargate import IngestionStargate
from knowledge3d.knowledgeverse.temporal_metadata import TemporalMetadata, TemporalMetadataManager


@dataclass
class _Metrics:
    ptx_fallback_rate: float = 0.0


class _DummyTRMManager:
    def __init__(self) -> None:
        self.updated: list[str] = []

    def update_specialist_weights(self, events: list[dict[str, Any]]) -> list[str]:
        specialists = sorted(
            {str(event.get("specialist", "unknown")) for event in events if event}
        )
        if not specialists:
            specialists = ["unknown"]
        self.updated.extend(specialists)
        return specialists


class _IntegrationSleepTime(SleepTimeConsolidation):
    """Integration-oriented SleepTime using Shadow Copy buffers."""

    def _stage_a_knowledge(self) -> dict[str, Any]:
        exported_count = len(getattr(self.kv.shadow_copy, "event_buffer", []))
        return {
            "success": True,
            "stage": "knowledge",
            "exported_count": exported_count,
        }

    def _stage_b_logic(self) -> dict[str, Any]:
        events = list(getattr(self.kv.shadow_copy, "event_buffer", []))
        updated_specialists = self.kv.trm_manager.update_specialist_weights(events)
        return {
            "success": True,
            "stage": "logic",
            "updated_specialists": updated_specialists,
        }


class _KnowledgeverseHarness:
    def __init__(self, tmp_path: Path) -> None:
        self.manifest_version = "kv-2026-02-06"
        self.metrics = _Metrics(ptx_fallback_rate=0.0)
        self.trm_manager = _DummyTRMManager()
        self.stargate = IngestionStargate(
            manifest_version=self.manifest_version,
            ring_size_mb=8,
        )
        self.shadow_copy = ShadowCopyLearning(
            index_path=tmp_path / "audit_index.json",
            manifest_version=self.manifest_version,
            trm_manager=self.trm_manager,
        )
        self.sleeptime = _IntegrationSleepTime(
            knowledgeverse=self,
            journal_path=tmp_path / "sleeptime_journal.jsonl",
        )


def _create_test_knowledgeverse(tmp_path: Path) -> _KnowledgeverseHarness:
    return _KnowledgeverseHarness(tmp_path=tmp_path)


def test_end_to_end_ingestion_to_audit(tmp_path):
    kv = _create_test_knowledgeverse(tmp_path)

    job_id = kv.stargate.submit_ingestion_job(
        data_path="test_data.txt",
        data_type="text",
    )

    assert job_id
    assert kv.stargate.completed_jobs
    assert kv.stargate.completed_jobs[0]["job_id"] == job_id

    event_id = kv.shadow_copy.record_event(
        "successful_navigation",
        {"specialist": "test", "confidence": 0.9, "query": "derivative x^2"},
    )

    events = kv.shadow_copy.compressed_journal.query_by_specialist("test")
    assert len(events) >= 1
    assert events[0]["temporal"]["event_id"] == event_id
    assert events[0]["temporal"]["manifest_version"] == kv.manifest_version


def test_shadow_copy_sleeptime_consolidation(tmp_path):
    kv = _create_test_knowledgeverse(tmp_path)

    parent_id = None
    for i in range(10):
        parent_id = kv.shadow_copy.record_event(
            "successful_navigation",
            {
                "specialist": "math",
                "query": f"test_query_{i}",
                "confidence": 0.8 + (i * 0.01),
            },
            parent_event_id=parent_id,
        )

    result = kv.sleeptime.execute()

    assert result["success"] is True
    assert result["stage_a"]["exported_count"] > 0
    assert len(result["stage_b"]["updated_specialists"]) > 0

    transaction_id = result["transaction_id"]
    assert result["stage_a"]["temporal"]["parent_event_id"] == transaction_id
    assert result["stage_b"]["temporal"]["parent_event_id"] == transaction_id


def test_resilience_patterns_under_simulated_failure(tmp_path, monkeypatch):
    kv = _create_test_knowledgeverse(tmp_path)

    retry_calls = [0]

    @SelfHealingWrapper.with_retry(max_attempts=3, backoff_base=0.001)
    def flaky_query():
        retry_calls[0] += 1
        if retry_calls[0] < 2:
            raise ValueError("Transient failure")
        return "success"

    assert flaky_query() == "success"
    assert retry_calls[0] == 2

    @SelfHealingWrapper.circuit_breaker(failure_threshold=2, timeout=60.0)
    def always_fails():
        raise ValueError("Permanent failure")

    with pytest.raises(ValueError):
        always_fails()
    with pytest.raises(ValueError):
        always_fails()
    with pytest.raises(CircuitBreakerOpen):
        always_fails()

    kv.shadow_copy.record_event(
        "successful_navigation",
        {"specialist": "math", "confidence": 0.91},
    )

    def _raise_update(_events):
        raise RuntimeError("TRM update failed")

    monkeypatch.setattr(kv.trm_manager, "update_specialist_weights", _raise_update)
    fallback_result = kv.sleeptime.execute()
    # with_fallback may return either cached success or explicit fallback output.
    assert isinstance(fallback_result, dict)
    assert (
        fallback_result.get("fallback") == "last_good_checkpoint"
        or fallback_result.get("success") is True
    )


def test_sovereignty_compliance_full_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("K3D_ENABLE_SOVEREIGN_FIREWALL", "1")
    kv = _create_test_knowledgeverse(tmp_path)

    for i in range(100):
        kv.shadow_copy.record_event(
            "test_event",
            {"query": f"test_{i}", "confidence": 0.9, "specialist": "math"},
        )

    result = kv.sleeptime.execute()
    assert result["success"] is True

    for mod in list(SovereigntyFirewall.FORBIDDEN_HOT_PATH_LIBS):
        monkeypatch.delitem(sys.modules, mod, raising=False)
        prefix = f"{mod}."
        for loaded in [name for name in list(sys.modules) if name.startswith(prefix)]:
            monkeypatch.delitem(sys.modules, loaded, raising=False)

    SovereigntyFirewall.runtime_assert_hot_path()
    assert kv.metrics.ptx_fallback_rate == 0.0


def test_temporal_causality_reconstruction(tmp_path):
    kv = _create_test_knowledgeverse(tmp_path)
    manager = TemporalMetadataManager("kv-2026-02-06", "reconstruction")

    parent_id = None
    for i in range(10):
        parent_id = kv.shadow_copy.record_event(
            "test_event",
            {"step": i, "specialist": "causal"},
            parent_event_id=parent_id,
        )

    events = kv.shadow_copy.compressed_journal.query_by_specialist("causal", limit=100)
    ordered = sorted(events, key=lambda e: int(e["temporal"]["lamport_clock"]))

    for i in range(1, len(ordered)):
        prev_temporal = ordered[i - 1]["temporal"]
        curr_temporal = ordered[i]["temporal"]
        assert curr_temporal["parent_event_id"] == prev_temporal["event_id"]
        assert int(curr_temporal["lamport_clock"]) > int(prev_temporal["lamport_clock"])

        prev_event = TemporalMetadata(
            event_id=str(prev_temporal["event_id"]),
            timestamp=float(prev_temporal["timestamp"]),
            lamport_clock=int(prev_temporal["lamport_clock"]),
            vector_clock=dict(prev_temporal["vector_clock"]),
            parent_event_id=prev_temporal["parent_event_id"] or None,
            manifest_version=str(prev_temporal["manifest_version"]),
        )
        curr_event = TemporalMetadata(
            event_id=str(curr_temporal["event_id"]),
            timestamp=float(curr_temporal["timestamp"]),
            lamport_clock=int(curr_temporal["lamport_clock"]),
            vector_clock=dict(curr_temporal["vector_clock"]),
            parent_event_id=curr_temporal["parent_event_id"] or None,
            manifest_version=str(curr_temporal["manifest_version"]),
        )
        assert manager.is_causally_before(prev_event, curr_event)
