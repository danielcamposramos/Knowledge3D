from __future__ import annotations

"""
GPU action buffer contract for the fused head output layer.

This module defines the ActionBuffer structure used by the PTX output layer to
emit navigation/dialogue/memory/tablet actions.  The structure mirrors the
layout described in Step7.2 of the swarm chain: a compact 288-byte record
holding navigation vectors, dialogue token slots, memory-write metadata, and
tablet mutations.  Python utilities are provided so higher-level components can
inspect and dispatch actions without copying data to the CPU unless strictly
required.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Optional, Tuple, Union
import numpy as np

try:  # CuPy is optional; fall back to NumPy buffers during tests.
    import cupy as cp

    _HAS_CUPY = True
except Exception:  # pragma: no cover - CPU fallback when CuPy missing
    cp = None  # type: ignore
    _HAS_CUPY = False


class ActionType(IntEnum):
    """Action family identifiers shared with PTX kernels."""

    NAV_MOVE = 0x00
    NAV_LOOK = 0x01
    DIALOGUE = 0x02
    WRITE_MEM = 0x03
    UPDATE_TABLET = 0x04
    NO_ACTION = 0xFF


# The dtype mirrors the layout documented in Step7.2 but is now aligned to
# natural 4/8-byte boundaries so PTX kernels can compute byte offsets easily.
# The structure is 288 bytes wide (72 × 4-byte words).
ACTION_BUFFER_DTYPE = np.dtype(
    [
        # Header -----------------------------------------------------------------
        ("action_type", np.uint32),  # ActionType value (enum stored as u32)
        ("confidence", np.float32),
        ("curiosity", np.float32),  # Curiosity bias score
        ("flags", np.uint32),
        # Navigation --------------------------------------------------------------
        ("nav_position", np.float32, (3,)),
        ("nav_direction", np.float32, (3,)),
        ("nav_velocity", np.float32),
        ("nav_room_id", np.uint32),
        ("nav_confidence", np.float32),
        ("nav_reserved", np.uint32, (6,)),
        # Dialogue ----------------------------------------------------------------
        ("dialogue_token_ids", np.uint16, (32,)),  # Max 32 tokens
        ("dialogue_length", np.uint32),
        ("dialogue_temperature", np.float32),
        ("dialogue_thinking_score", np.float32),
        ("dialogue_reserved", np.uint32, (6,)),
        # Memory write ------------------------------------------------------------
        ("mem_summary_hash", np.uint64),
        ("mem_zone_id", np.uint32),
        ("mem_confidence", np.float32),
        ("mem_embedding", np.float32, (4,)),  # Compressed summary embedding
        ("mem_reserved", np.uint32, (8,)),
        # Tablet mutation ---------------------------------------------------------
        ("tablet_mutation_type", np.uint32),
        ("tablet_data", np.uint32, (6,)),
        ("tablet_reserved", np.uint32, (4,)),
    ],
    align=True,
)

# Sanity guard – the contract assumes 288-byte records so we assert it here.
assert ACTION_BUFFER_DTYPE.itemsize == 288, ACTION_BUFFER_DTYPE.itemsize


def _ensure_device_array(array: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
    """
    Ensure a buffer lives on the GPU when CuPy is available.

    When CuPy is missing (e.g., during CPU-only unit tests) we keep NumPy arrays
    so the rest of the code can still operate.
    """
    if _HAS_CUPY:
        return cp.asarray(array)
    return array


@dataclass
class ActionResult:
    """Outcome of dispatching an action to the surrounding systems."""

    action_type: ActionType
    confidence: float
    curiosity: float
    success: bool
    metadata: dict


class ActionBuffer:
    """
    Thin wrapper around the GPU action struct emitted by PTX kernels.

    The buffer always stores a single action (batch size 1) which is sufficient
    for the current MVP demo.  Future work can extend this to handle batches by
    storing multiple ActionBuffer entries in a single device array.
    """

    def __init__(self, buffer: Optional[Union[np.ndarray, "cp.ndarray"]] = None):
        if buffer is None:
            array = np.zeros(1, dtype=ACTION_BUFFER_DTYPE)
            self.buffer = _ensure_device_array(array)
            self.reset()
        else:
            if getattr(buffer, "dtype", None) != ACTION_BUFFER_DTYPE:
                raise TypeError("ActionBuffer requires dtype ACTION_BUFFER_DTYPE")
            self.buffer = buffer

    # ------------------------------------------------------------------
    # GPU helpers
    @property
    def device_ptr(self) -> int:
        """Return the raw device pointer (CUdeviceptr) for PTX kernel launches."""
        if _HAS_CUPY and isinstance(self.buffer, cp.ndarray):
            return self.buffer.data.ptr
        # NumPy fallback: expose underlying memory address (only for tests).
        return self.buffer.__array_interface__["data"][0]

    # ------------------------------------------------------------------
    # Header accessors
    def get_action_type(self) -> ActionType:
        value = int(self.buffer["action_type"][0])
        try:
            return ActionType(value)
        except ValueError:
            return ActionType.NO_ACTION

    def get_confidence(self) -> float:
        return float(self.buffer["confidence"][0])

    def get_curiosity(self) -> float:
        return float(self.buffer["curiosity"][0])

    # ------------------------------------------------------------------
    # Navigation helpers
    def extract_nav_move(self) -> Tuple[np.ndarray, float]:
        position = np.asarray(self.buffer["nav_position"][0]).astype(np.float32)
        confidence = float(self.buffer["nav_confidence"][0])
        return position, confidence

    def extract_nav_look(self) -> Tuple[np.ndarray, float]:
        direction = np.asarray(self.buffer["nav_direction"][0]).astype(np.float32)
        confidence = float(self.buffer["nav_confidence"][0])
        return direction, confidence

    # ------------------------------------------------------------------
    # Dialogue helpers
    def extract_dialogue_tokens(self) -> Tuple[np.ndarray, float]:
        length = int(self.buffer["dialogue_length"][0])
        length = max(0, min(length, 32))
        tokens = np.asarray(self.buffer["dialogue_token_ids"][0][:length], dtype=np.uint16)
        thinking_score = float(self.buffer["dialogue_thinking_score"][0])
        return tokens, thinking_score

    # ------------------------------------------------------------------
    # Memory helpers
    def extract_memory_write(self) -> Tuple[int, np.ndarray, float]:
        zone_id = int(self.buffer["mem_zone_id"][0])
        embedding = np.asarray(self.buffer["mem_embedding"][0], dtype=np.float32)
        confidence = float(self.buffer["mem_confidence"][0])
        return zone_id, embedding, confidence

    # ------------------------------------------------------------------
    # Tablet helpers
    def extract_tablet_mutation(self) -> Tuple[int, np.ndarray]:
        mutation_type = int(self.buffer["tablet_mutation_type"][0])
        payload = np.asarray(self.buffer["tablet_data"][0], dtype=np.uint32)
        return mutation_type, payload

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero the buffer contents in-place (preserves device residency)."""
        if _HAS_CUPY and isinstance(self.buffer, cp.ndarray):
            self.buffer.fill(0)
        else:
            self.buffer[...] = 0
        self.buffer["action_type"][0] = np.uint32(ActionType.NO_ACTION.value)
