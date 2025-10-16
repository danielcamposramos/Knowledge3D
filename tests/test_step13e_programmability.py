import numpy as np
import pytest

from knowledge3d.cranium.bridges.thinking_tag_rpn import ThinkingTagRPNBridge
from knowledge3d.cranium.ptx_runtime import rpn_opcodes as ropc


@pytest.mark.gpu
def test_store_and_recall_roundtrip() -> None:
    bridge = ThinkingTagRPNBridge()

    program = [
        42.0,                       # value
        0.0,                        # slot id
        ropc.OP_STORE,              # store slot 0
        123.0,                      # distractor value
        0.0,                        # slot id to recall
        ropc.OP_RECALL,             # recall slot 0
    ]
    stack = bridge._execute_rpn_program(program)
    assert np.isclose(stack[-1], 42.0)

    bridge.cleanup()


@pytest.mark.gpu
def test_multiple_variable_slots() -> None:
    bridge = ThinkingTagRPNBridge()

    program = [
        1.0, 0.0, ropc.OP_STORE,
        2.0, 1.0, ropc.OP_STORE,
        3.0, 2.0, ropc.OP_STORE,
        0.0, ropc.OP_RECALL,
        1.0, ropc.OP_RECALL,
        2.0, ropc.OP_RECALL,
    ]
    stack = bridge._execute_rpn_program(program)
    assert np.isclose(stack[-3], 1.0)
    assert np.isclose(stack[-2], 2.0)
    assert np.isclose(stack[-1], 3.0)

    bridge.cleanup()
