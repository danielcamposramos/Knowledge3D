from __future__ import annotations

from types import SimpleNamespace

from knowledge3d.cranium.trm_adapters import SelfUpdatingAdapter, loader


def _fake_buffers() -> SimpleNamespace:
    tensor = lambda value: SimpleNamespace(ptr=value)
    return SimpleNamespace(
        A_weights=tensor(1),
        B_weights=tensor(2),
        A_shadow_weights=tensor(3),
        B_shadow_weights=tensor(4),
        A_transposed=tensor(5),
        B_transposed=tensor(6),
        A_shadow_transposed=tensor(7),
        B_shadow_transposed=tensor(8),
    )


def test_navigator_shadow_promotion_logs_gate_outcome(monkeypatch) -> None:
    adapter = SelfUpdatingAdapter(shape=(4, 4), rank=2, specialist_name="navigator")
    adapter.validation_samples = [{"id": "gate_1"}]
    adapter.A_shadow.copy_from(adapter.A)
    adapter.B_shadow.copy_from(adapter.B)
    adapter.A_shadow._buffer[0] += 1.0
    adapter.B_shadow._buffer[0] += 1.0

    monkeypatch.setattr(adapter, "_ensure_device_weight_set", lambda shadow=False: (None, None, None, None))
    monkeypatch.setattr(adapter, "_ensure_device_buffers", lambda: _fake_buffers())
    monkeypatch.setattr(loader, "memcpy_dtod", lambda dst, src, size: None)

    primary_before = adapter.A.to_bytes()

    success, baseline, shadow = adapter.validate_and_commit(
        base_weights=0.0,
        eval_fn=lambda weights, _samples: float(weights[0][0]),
    )

    assert success is True
    assert shadow >= baseline
    assert adapter.performance_history[-1]["decision"] == "TRUE"
    assert adapter.A.to_bytes() != primary_before
