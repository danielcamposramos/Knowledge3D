from __future__ import annotations

from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


def test_gpu_counter_tracks_instance_and_global_calls() -> None:
    ModularRPNEngine.reset_global_gpu_call_count()
    engine = object.__new__(ModularRPNEngine)
    engine.gpu_call_count = 0

    engine._record_gpu_call(2)
    assert engine.get_gpu_call_count() == 2
    assert ModularRPNEngine.get_global_gpu_call_count() == 2

    engine.reset_gpu_call_count()
    assert engine.get_gpu_call_count() == 0
    assert ModularRPNEngine.get_global_gpu_call_count() == 2
