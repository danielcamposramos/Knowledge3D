from __future__ import annotations

import json

from knowledge3d.knowledgeverse.execution_quality_tracker import ExecutionQualityTracker


def _event(timestamp_us: int, *, quality: float) -> dict[str, object]:
    return {
        "tool_id": "tool_alpha",
        "query_context": "dispatch route benchmark",
        "specialist_id": "VisualSpecialist",
        "execution_us": 1000,
        "outcome": 1,
        "quality_signal": quality,
        "timestamp_us": timestamp_us,
        "runtime_status": "ptx_bridge_available",
        "tool_kind": "scene_fusion",
    }


def test_execution_quality_tracker_debounces_state_writes(tmp_path) -> None:
    state_path = tmp_path / "execution_quality_tracker.json"
    gap_log_path = tmp_path / "specialist_gaps.jsonl"
    tracker = ExecutionQualityTracker(
        state_path=state_path,
        gap_log_path=gap_log_path,
        save_every=1000,
        save_interval_s=999.0,
    )

    tracker.observe_event(_event(1, quality=0.7), specialist_catalog=["VisualSpecialist"])
    saved_once = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_once["tools"]["tool_alpha"]["total_executions"] == 1

    tracker.observe_event(_event(2, quality=0.8), specialist_catalog=["VisualSpecialist"])
    still_saved_once = json.loads(state_path.read_text(encoding="utf-8"))
    assert still_saved_once["tools"]["tool_alpha"]["total_executions"] == 1

    tracker.flush()
    saved_twice = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved_twice["tools"]["tool_alpha"]["total_executions"] == 2
