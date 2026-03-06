from __future__ import annotations

from knowledge3d.cranium.sleep.memory_pressure_trigger import MemoryPressureTrigger


def test_memory_pressure_trigger_fires_on_ratio(monkeypatch):
    monkeypatch.setattr(
        "knowledge3d.cranium.sleep.memory_pressure_trigger.loader.get_vram_usage",
        lambda: (900, 1000),
    )
    trigger = MemoryPressureTrigger(threshold_ratio=0.82, reserve_bytes=50)
    snap = trigger.snapshot()

    assert snap.should_consolidate is True
    assert snap.reason == "threshold_ratio"
    assert abs(snap.usage_ratio - 0.9) < 1e-6


def test_memory_pressure_trigger_fires_on_reserve(monkeypatch):
    monkeypatch.setattr(
        "knowledge3d.cranium.sleep.memory_pressure_trigger.loader.get_vram_usage",
        lambda: (700, 1000),
    )
    trigger = MemoryPressureTrigger(threshold_ratio=0.9, reserve_bytes=400)
    snap = trigger.snapshot()

    assert snap.should_consolidate is True
    assert snap.reason == "reserve_bytes"
    assert snap.free_bytes == 300


def test_memory_pressure_trigger_stays_idle_when_vram_is_safe(monkeypatch):
    monkeypatch.setattr(
        "knowledge3d.cranium.sleep.memory_pressure_trigger.loader.get_vram_usage",
        lambda: (400, 1000),
    )
    trigger = MemoryPressureTrigger(threshold_ratio=0.85, reserve_bytes=200)
    snap = trigger.snapshot()

    assert snap.should_consolidate is False
    assert snap.reason == "ok"
