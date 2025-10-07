
"""End-to-end performance validation for the unified FSM pipeline."""

import numpy as np
import pytest


@pytest.mark.cuda
def test_unified_fsm_end_to_end_1k_nodes() -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - GPU unavailable
        pytest.skip("CUDA device not available")

    from knowledge3d.cranium.unified_fsm import UnifiedFSMContext

    fsm = UnifiedFSMContext()
    n_nodes = 1024

    unified_buffer = fsm.create_unified_buffer(n_nodes)
    random_payload = cupy.random.standard_normal(unified_buffer.shape, dtype=cupy.float32)
    unified_buffer[:] = random_payload

    query_embedding = np.random.standard_normal(512).astype(np.float32)

    start = cupy.cuda.Event()
    end = cupy.cuda.Event()

    start.record()
    from knowledge3d.cranium.actions.action_types import ActionBuffer

    output_action, state_trace, action_buffer = fsm.launch_fsm(
        unified_buffer, query_embedding, initial_state=3
    )
    end.record()
    end.synchronize()

    elapsed_ms = cupy.cuda.get_elapsed_time(start, end)

    assert output_action.shape == (512,)
    assert np.all(np.isfinite(output_action))

    assert isinstance(state_trace, list)
    assert state_trace, "FSM should record at least one state"

    assert elapsed_ms < 500.0, f"FSM dispatch took too long: {elapsed_ms:.2f}ms"

    assert isinstance(action_buffer, ActionBuffer)
