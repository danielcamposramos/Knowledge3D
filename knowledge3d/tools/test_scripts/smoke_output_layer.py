# DEPRECATED: legacy smoke script maintained for quick manual runs.
"""
Minimal smoke test for the fused-head output layer.

This keeps a simple entry point that exercises the ActionRouter without diving
into the full training stack.  The script will be replaced by a richer demo
once the PTX output layer is fully production ready.
"""

from __future__ import annotations

import numpy as np

from knowledge3d.cranium.actions import ActionBuffer, ActionType
from knowledge3d.cranium.output_router import ActionRouter


def main() -> None:
    buffer = ActionBuffer()
    buffer.buffer["action_type"][0] = np.uint32(ActionType.UPDATE_TABLET.value)
    buffer.buffer["confidence"][0] = 0.9
    buffer.buffer["curiosity"][0] = 0.1
    buffer.buffer["tablet_mutation_type"][0] = np.uint32(1)
    buffer.buffer["tablet_data"][0][:] = 0

    router = ActionRouter()
    result = router.dispatch(buffer)
    print("Action result:", result)


if __name__ == "__main__":
    main()
