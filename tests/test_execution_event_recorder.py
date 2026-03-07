from __future__ import annotations

import json
from types import SimpleNamespace

from knowledge3d.knowledgeverse.execution_events import ExecutionEvent, ExecutionEventRecorder


def _event(*, mode: str, stamp: int) -> ExecutionEvent:
    return ExecutionEvent(
        tool_id="tool_alpha",
        query_context="dispatch route benchmark",
        specialist_id="VisualSpecialist",
        math_core_tier=2,
        execution_us=1000,
        outcome=1,
        quality_signal=0.8,
        ternary_quality=1,
        timestamp_us=stamp,
        chain_depth=1,
        promotion_pressure=False,
        execution_mode=mode,
        runtime_status="ptx_bridge_available",
    )


def test_execution_event_recorder_buffers_chain_steps_but_flushes_top_level_immediately(tmp_path) -> None:
    recorder = ExecutionEventRecorder(storage_root=tmp_path / "kv_events", buffer_size=64, flush_interval_s=999.0)
    event_path = tmp_path / "kv_events" / "logs" / "execution_events.jsonl"

    recorder.append(_event(mode="tool_chain_step", stamp=1))
    assert not event_path.exists()

    recorder.append(_event(mode="tool_entrypoint_chain", stamp=2))
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(rows) == 2
    assert rows[0]["execution_mode"] == "tool_chain_step"
    assert rows[1]["execution_mode"] == "tool_entrypoint_chain"


def test_execution_event_recorder_flush_method_persists_buffer(tmp_path) -> None:
    recorder = ExecutionEventRecorder(storage_root=tmp_path / "kv_events_flush", buffer_size=64, flush_interval_s=999.0)
    event_path = tmp_path / "kv_events_flush" / "logs" / "execution_events.jsonl"

    recorder.append(_event(mode="tool_chain_step", stamp=1))
    recorder.flush()

    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["timestamp_us"] == 1


def test_execution_event_recorder_only_emits_top_level_events_to_shadow_copy(tmp_path) -> None:
    shadow_events: list[tuple[str, dict[str, object]]] = []

    class _ShadowCopy:
        def record_event(self, event_type: str, event_data: dict[str, object], parent_event_id=None) -> str:
            shadow_events.append((event_type, event_data))
            return "evt"

    knowledgeverse = SimpleNamespace(shadow_copy=_ShadowCopy())
    recorder = ExecutionEventRecorder(storage_root=tmp_path / "kv_events_shadow", buffer_size=64, flush_interval_s=999.0)

    recorder.append(_event(mode="tool_chain_step", stamp=1), knowledgeverse=knowledgeverse)
    assert shadow_events == []

    recorder.append(_event(mode="tool_entrypoint_chain", stamp=2), knowledgeverse=knowledgeverse)

    assert len(shadow_events) == 1
    assert shadow_events[0][0] == "tool_execution"
    assert shadow_events[0][1]["timestamp"] == 2 / 1_000_000.0
