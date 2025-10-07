from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.cuda
def test_decode_actions_kernel_navigation() -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - GPU unavailable
        pytest.skip("CUDA device not available")

    from knowledge3d.cranium.actions.action_types import ActionBuffer, ActionType

    module = cupy.RawModule(path="knowledge3d/cranium/ptx/decode_actions.ptx")
    kernel = module.get_function("decode_actions_kernel")

    payload = cupy.zeros(160, dtype=cupy.float32)
    payload[0:3] = cupy.asarray([2.0, 0.0, 0.0], dtype=cupy.float32)  # nav position
    payload[3] = cupy.float32(0.85)  # nav confidence
    payload[4] = cupy.float32(1.25)  # velocity
    payload[5] = cupy.float32(4.0)   # room id
    payload[6] = cupy.float32(0.42)  # curiosity hint

    # Dialogue logits (index 5 highest)
    payload[32 + 5] = cupy.float32(2.0)
    payload[32 + 1] = cupy.float32(0.5)

    # Memory fallback
    payload[96:100] = cupy.asarray([0.1, 0.2, 0.3, 0.4], dtype=cupy.float32)
    payload[100] = cupy.float32(0.2)
    payload[101] = cupy.float32(3.0)
    payload[112] = cupy.float32(7.0)  # tablet mutation

    action_buffer = ActionBuffer()

    kernel(
        (1,),
        (32,),
        (
            payload,
            np.uint64(action_buffer.device_ptr),
            np.float32(0.5),   # nav threshold
            np.float32(0.4),   # mem threshold
            np.float32(0.85),  # dialogue temperature
        ),
    )
    cupy.cuda.runtime.deviceSynchronize()

    assert action_buffer.get_action_type() == ActionType.NAV_MOVE
    np.testing.assert_allclose(action_buffer.buffer["confidence"][0], 0.85, rtol=1e-5)
    np.testing.assert_allclose(action_buffer.buffer["curiosity"][0], 0.42, rtol=1e-5)

    position, nav_conf = action_buffer.extract_nav_move()
    np.testing.assert_allclose(position, [2.0, 0.0, 0.0], rtol=1e-5)
    assert nav_conf == pytest.approx(0.85, rel=1e-5)

    # Direction is normalised
    direction = np.asarray(action_buffer.buffer["nav_direction"][0])
    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], rtol=1e-6)

    # Tablet metadata mirrored from payload
    mutation_type, _ = action_buffer.extract_tablet_mutation()
    assert mutation_type == 7
