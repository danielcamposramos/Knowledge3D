"""ARC3 persistent brain state backed by sovereign VRAM allocations.

This reuses the existing K3D persistent-state pattern used in the RPN and swarm
paths: a small GPU buffer that survives across kernel launches and carries the
avatar's ongoing cognitive state between frames.
"""

from __future__ import annotations

import ctypes
import struct
from typing import Any

from knowledge3d.cranium.sovereign import loader


BRAIN_REASONING_STATE_BYTES = 32 * 4
BRAIN_CHAIN_STATES_BYTES = 9 * 32 * 4
BRAIN_PREV_FRAME_BYTES = 32 * 4
BRAIN_ACTION_RING_BYTES = 7
BRAIN_ACTION_RING_LEN_BYTES = 1
BRAIN_TERNARY_SIGNAL_BYTES = 1
BRAIN_ALIGN_PAD_BYTES = 3
BRAIN_FRAME_COUNT_BYTES = 4
BRAIN_SPECIALIST_TRACE_BYTES = 9 * 4
BRAIN_TRM_VECTOR_BYTES = 512 * 4

BRAIN_REASONING_OFFSET = 0
BRAIN_CHAINS_OFFSET = BRAIN_REASONING_OFFSET + BRAIN_REASONING_STATE_BYTES
BRAIN_PREV_FRAME_OFFSET = BRAIN_CHAINS_OFFSET + BRAIN_CHAIN_STATES_BYTES
BRAIN_ACTION_RING_OFFSET = BRAIN_PREV_FRAME_OFFSET + BRAIN_PREV_FRAME_BYTES
BRAIN_ACTION_RING_LEN_OFFSET = BRAIN_ACTION_RING_OFFSET + BRAIN_ACTION_RING_BYTES
BRAIN_TERNARY_OFFSET = BRAIN_ACTION_RING_LEN_OFFSET + BRAIN_ACTION_RING_LEN_BYTES
BRAIN_FRAME_COUNT_OFFSET = BRAIN_TERNARY_OFFSET + BRAIN_TERNARY_SIGNAL_BYTES + BRAIN_ALIGN_PAD_BYTES
BRAIN_SPECIALIST_TRACE_OFFSET = BRAIN_FRAME_COUNT_OFFSET + BRAIN_FRAME_COUNT_BYTES
BRAIN_TRM_Q_OFFSET = BRAIN_SPECIALIST_TRACE_OFFSET + BRAIN_SPECIALIST_TRACE_BYTES
BRAIN_TRM_Y_OFFSET = BRAIN_TRM_Q_OFFSET + BRAIN_TRM_VECTOR_BYTES
BRAIN_TRM_Z_OFFSET = BRAIN_TRM_Y_OFFSET + BRAIN_TRM_VECTOR_BYTES
BRAIN_TOTAL_BYTES = (
    BRAIN_REASONING_STATE_BYTES
    + BRAIN_CHAIN_STATES_BYTES
    + BRAIN_PREV_FRAME_BYTES
    + BRAIN_ACTION_RING_BYTES
    + BRAIN_ACTION_RING_LEN_BYTES
    + BRAIN_TERNARY_SIGNAL_BYTES
    + BRAIN_ALIGN_PAD_BYTES
    + BRAIN_FRAME_COUNT_BYTES
    + BRAIN_SPECIALIST_TRACE_BYTES
    + BRAIN_TRM_VECTOR_BYTES
    + BRAIN_TRM_VECTOR_BYTES
    + BRAIN_TRM_VECTOR_BYTES
)


def _bytes_ptr(payload: bytearray) -> ctypes.c_void_p:
    return ctypes.c_void_p(ctypes.addressof(ctypes.c_ubyte.from_buffer(payload)))


class PersistentBrainState:
    """VRAM-resident ARC3 brain state that survives across frames."""

    def __init__(self) -> None:
        self.gpu_ptr = loader.gpu_malloc(BRAIN_TOTAL_BYTES)
        self.reset()

    def _download(self) -> bytearray:
        payload = bytearray(BRAIN_TOTAL_BYTES)
        loader.memcpy_dtoh(_bytes_ptr(payload), self.gpu_ptr, len(payload))
        return payload

    def _upload(self, payload: bytearray) -> None:
        loader.memcpy_htod(self.gpu_ptr, _bytes_ptr(payload), len(payload))

    def reset(self) -> None:
        payload = bytearray(BRAIN_TOTAL_BYTES)
        self._upload(payload)

    def reset_chains(self) -> None:
        data = self._download()
        for index in range(BRAIN_CHAINS_OFFSET, BRAIN_CHAINS_OFFSET + BRAIN_CHAIN_STATES_BYTES):
            data[index] = 0
        self._upload(data)

    def close(self) -> None:
        if getattr(self, "gpu_ptr", None):
            loader.gpu_free(self.gpu_ptr)
            self.gpu_ptr = None

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def read_raw(self) -> bytearray:
        return self._download()

    def read_state(self) -> dict[str, Any]:
        data = self.read_raw()
        reasoning = list(struct.unpack_from("<32f", data, BRAIN_REASONING_OFFSET))
        chains = [
            list(struct.unpack_from("<32f", data, BRAIN_CHAINS_OFFSET + (chain * BRAIN_REASONING_STATE_BYTES)))
            for chain in range(9)
        ]
        prev_frame = list(struct.unpack_from("<32f", data, BRAIN_PREV_FRAME_OFFSET))
        specialist_trace = list(struct.unpack_from("<9f", data, BRAIN_SPECIALIST_TRACE_OFFSET))
        trm_q = list(struct.unpack_from("<512f", data, BRAIN_TRM_Q_OFFSET))
        trm_y = list(struct.unpack_from("<512f", data, BRAIN_TRM_Y_OFFSET))
        trm_z = list(struct.unpack_from("<512f", data, BRAIN_TRM_Z_OFFSET))
        ring_len = min(int(data[BRAIN_ACTION_RING_LEN_OFFSET]), 7)
        action_ring = [
            int(data[BRAIN_ACTION_RING_OFFSET + index])
            for index in range(ring_len)
        ]
        ternary_signal = int(struct.unpack_from("<b", data, BRAIN_TERNARY_OFFSET)[0])
        frame_count = int(struct.unpack_from("<I", data, BRAIN_FRAME_COUNT_OFFSET)[0])
        return {
            "reasoning": reasoning,
            "reasoning_norm": sum(value * value for value in reasoning) ** 0.5,
            "chains": chains,
            "prev_frame": prev_frame,
            "action_ring": action_ring,
            "ternary_signal": ternary_signal,
            "frame_count": frame_count,
            "specialist_trace": specialist_trace,
            "trm_q_norm": sum(value * value for value in trm_q) ** 0.5,
            "trm_y_norm": sum(value * value for value in trm_y) ** 0.5,
            "trm_z_norm": sum(value * value for value in trm_z) ** 0.5,
        }


__all__ = [
    "BRAIN_ACTION_RING_LEN_OFFSET",
    "BRAIN_ACTION_RING_OFFSET",
    "BRAIN_CHAINS_OFFSET",
    "BRAIN_FRAME_COUNT_OFFSET",
    "BRAIN_PREV_FRAME_OFFSET",
    "BRAIN_REASONING_OFFSET",
    "BRAIN_SPECIALIST_TRACE_OFFSET",
    "BRAIN_TERNARY_OFFSET",
    "BRAIN_TRM_Q_OFFSET",
    "BRAIN_TRM_Y_OFFSET",
    "BRAIN_TRM_Z_OFFSET",
    "BRAIN_TOTAL_BYTES",
    "PersistentBrainState",
]
