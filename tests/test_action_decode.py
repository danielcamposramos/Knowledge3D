from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.actions.action_types import ActionBuffer, ActionType


def test_action_buffer_defaults():
    buf = ActionBuffer()
    assert buf.get_action_type() == ActionType.NO_ACTION
    assert buf.get_confidence() == 0.0
    tokens, thinking = buf.extract_dialogue_tokens()
    assert tokens.size == 0
    assert thinking == 0.0


@pytest.mark.parametrize("action_type", [ActionType.NAV_MOVE, ActionType.DIALOGUE, ActionType.WRITE_MEM])
def test_action_type_enum_roundtrip(action_type):
    buf = ActionBuffer()
    buf.buffer["action_type"][0] = np.uint32(action_type.value)
    assert buf.get_action_type() == action_type


def test_nav_helpers():
    buf = ActionBuffer()
    buf.buffer["nav_position"][0] = [1.0, 2.0, 3.0]
    buf.buffer["nav_confidence"][0] = 0.75
    pos, conf = buf.extract_nav_move()
    np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])
    assert conf == pytest.approx(0.75)
