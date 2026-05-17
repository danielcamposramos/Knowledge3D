"""Sovereign ctypes ActionBuffer — 288-byte binary contract for PTX output.

Sovereign successor to the numpy-dtype ``ActionBuffer`` archived in
``Old_Attempts/2026-04-18/knowledge3d/cranium/actions/action_types.py`` by
the Absolute Sovereignty Purge.

The layout mirrors Step 7.2 of the swarm chain spec exactly (288 bytes,
72 × 4-byte words). Offsets match the numpy dtype with ``align=True``:

    offset  0   u32    action_type
    offset  4   f32    confidence
    offset  8   f32    curiosity
    offset 12   u32    flags
    offset 16   f32[3] nav_position
    offset 28   f32[3] nav_direction
    offset 40   f32    nav_velocity
    offset 44   u32    nav_room_id
    offset 48   f32    nav_confidence
    offset 52   u32[6] nav_reserved
    offset 76   u16[32] dialogue_token_ids
    offset 140  u32    dialogue_length
    offset 144  f32    dialogue_temperature
    offset 148  f32    dialogue_thinking_score
    offset 152  u32[6] dialogue_reserved
    offset 176  u64    mem_summary_hash
    offset 184  u32    mem_zone_id
    offset 188  f32    mem_confidence
    offset 192  f32[4] mem_embedding
    offset 208  u32[8] mem_reserved
    offset 240  u32    tablet_mutation_type
    offset 244  u32[6] tablet_data
    offset 268  u32[4] tablet_reserved
    offset 284  u32    _trailing_pad   (mirror of numpy align=True tail pad)
    TOTAL      288 bytes

The structure is consumed by ``decode_actions.ptx`` and
``trm_step_fused.ptx``. Device residency is delegated to
``knowledge3d.cranium.sovereign.loader`` (``gpu_malloc`` +
``memcpy_htod`` / ``memcpy_dtoh``).

No numpy, no cupy, no torch, no fallbacks.
Per ``feedback_no_fallbacks_ever_including_sleeptime.md`` and
``TEMP/CLAUDE_TABLET_LIVE_LOOP_SPEC_04.18.2026.md`` §1.1.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Sequence


__all__ = [
    "ActionType",
    "ACTION_BUFFER_SIZE",
    "ActionBufferStruct",
    "ActionBuffer",
    "ActionResult",
]


ACTION_BUFFER_SIZE = 288  # bytes


class ActionType(IntEnum):
    """Action family identifiers shared with PTX kernels."""

    NAV_MOVE = 0x00
    NAV_LOOK = 0x01
    DIALOGUE = 0x02
    WRITE_MEM = 0x03
    UPDATE_TABLET = 0x04
    NO_ACTION = 0xFF


class ActionBufferStruct(ctypes.Structure):
    """Pure-ctypes mirror of the 288-byte PTX action buffer.

    Field order + types + alignment (8-byte, natural) match numpy
    ``dtype(..., align=True)``. Any change here MUST be mirrored in the
    PTX consumer kernels in ``knowledge3d/cranium/ptx/``.
    """

    _pack_ = 0  # natural alignment — matches numpy align=True
    _fields_ = [
        # Header -----------------------------------------------------------
        ("action_type", ctypes.c_uint32),
        ("confidence", ctypes.c_float),
        ("curiosity", ctypes.c_float),
        ("flags", ctypes.c_uint32),
        # Navigation -------------------------------------------------------
        ("nav_position", ctypes.c_float * 3),
        ("nav_direction", ctypes.c_float * 3),
        ("nav_velocity", ctypes.c_float),
        ("nav_room_id", ctypes.c_uint32),
        ("nav_confidence", ctypes.c_float),
        ("nav_reserved", ctypes.c_uint32 * 6),
        # Dialogue ---------------------------------------------------------
        ("dialogue_token_ids", ctypes.c_uint16 * 32),
        ("dialogue_length", ctypes.c_uint32),
        ("dialogue_temperature", ctypes.c_float),
        ("dialogue_thinking_score", ctypes.c_float),
        ("dialogue_reserved", ctypes.c_uint32 * 6),
        # Memory write -----------------------------------------------------
        ("mem_summary_hash", ctypes.c_uint64),
        ("mem_zone_id", ctypes.c_uint32),
        ("mem_confidence", ctypes.c_float),
        ("mem_embedding", ctypes.c_float * 4),
        ("mem_reserved", ctypes.c_uint32 * 8),
        # Tablet mutation --------------------------------------------------
        ("tablet_mutation_type", ctypes.c_uint32),
        ("tablet_data", ctypes.c_uint32 * 6),
        ("tablet_reserved", ctypes.c_uint32 * 4),
        # Trailing pad to hit 288 bytes (numpy align=True does the same) --
        ("_trailing_pad", ctypes.c_uint32),
    ]


# Static guard — the PTX contract assumes 288 bytes exactly.
assert ctypes.sizeof(ActionBufferStruct) == ACTION_BUFFER_SIZE, (
    f"ActionBufferStruct size {ctypes.sizeof(ActionBufferStruct)} != "
    f"{ACTION_BUFFER_SIZE} — contract violation vs PTX consumers."
)


_ARRAY_FIELDS = frozenset(
    {
        "nav_position",
        "nav_direction",
        "nav_reserved",
        "dialogue_token_ids",
        "dialogue_reserved",
        "mem_embedding",
        "mem_reserved",
        "tablet_data",
        "tablet_reserved",
    }
)


@dataclass
class ActionResult:
    """Outcome of dispatching an action to the surrounding systems."""

    action_type: ActionType
    confidence: float
    curiosity: float
    success: bool
    metadata: Dict[str, Any]


class ActionBuffer:
    """Host-side handle around a single :class:`ActionBufferStruct`.

    Contract:
      * ``buffer`` is the underlying ctypes Structure — callers may
        read/write fields with normal attribute access
        (``buf.buffer.action_type = 5``).
      * ``device_ptr()`` returns the raw address suitable for ctypes
        use (cast to ``c_void_p``). For device residency, stage via
        ``knowledge3d.cranium.sovereign.loader.memcpy_htod``.
      * ``as_bytes()`` / ``load_bytes()`` round-trip the 288-byte blob.
    """

    __slots__ = ("buffer",)

    def __init__(self, buffer: Optional[ActionBufferStruct] = None) -> None:
        if buffer is None:
            self.buffer = ActionBufferStruct()
            self.reset()
        else:
            if not isinstance(buffer, ActionBufferStruct):
                raise TypeError(
                    "ActionBuffer requires an ActionBufferStruct instance; "
                    f"got {type(buffer).__name__}"
                )
            self.buffer = buffer

    # ------------------------------------------------------------------ #
    # Raw address / byte access
    # ------------------------------------------------------------------ #
    def device_ptr(self) -> int:
        """Host address of the buffer as an integer.

        For GPU use, stage through ``loader.gpu_malloc`` + ``memcpy_htod``.
        The sovereign pattern never passes a host pointer to the GPU.
        """
        return ctypes.addressof(self.buffer)

    def as_bytes(self) -> bytes:
        return bytes(self.buffer)

    def load_bytes(self, raw: bytes) -> None:
        if len(raw) != ACTION_BUFFER_SIZE:
            raise ValueError(
                f"ActionBuffer.load_bytes: got {len(raw)} bytes, "
                f"expected {ACTION_BUFFER_SIZE}"
            )
        ctypes.memmove(ctypes.addressof(self.buffer), raw, ACTION_BUFFER_SIZE)

    # ------------------------------------------------------------------ #
    # Numpy-dtype-style accessors (shim for callers that haven't migrated
    # to plain attribute access yet). Supports:
    #     buf["action_type"][0] = 5
    #     buf["tablet_data"][0][:] = (a, b, c, d, e, f)
    # No numpy is used — returned rows are thin indexing proxies.
    # ------------------------------------------------------------------ #
    def __getitem__(self, key: str) -> "_ActionBufferFieldRow":
        if not isinstance(key, str):
            raise TypeError(
                f"ActionBuffer subscript must be a field name, got {type(key).__name__}"
            )
        return _ActionBufferFieldRow(self.buffer, key)

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            raise TypeError(
                f"ActionBuffer subscript must be a field name, got {type(key).__name__}"
            )
        _assign_field(self.buffer, key, value)

    # ------------------------------------------------------------------ #
    # Header accessors
    # ------------------------------------------------------------------ #
    def get_action_type(self) -> ActionType:
        value = int(self.buffer.action_type)
        try:
            return ActionType(value)
        except ValueError:
            return ActionType.NO_ACTION

    def get_confidence(self) -> float:
        return float(self.buffer.confidence)

    def get_curiosity(self) -> float:
        return float(self.buffer.curiosity)

    # ------------------------------------------------------------------ #
    # Navigation helpers
    # ------------------------------------------------------------------ #
    def extract_nav_move(self) -> "tuple[List[float], float]":
        position = [float(v) for v in self.buffer.nav_position]
        confidence = float(self.buffer.nav_confidence)
        return position, confidence

    def extract_nav_look(self) -> "tuple[List[float], float]":
        direction = [float(v) for v in self.buffer.nav_direction]
        confidence = float(self.buffer.nav_confidence)
        return direction, confidence

    # ------------------------------------------------------------------ #
    # Dialogue helpers
    # ------------------------------------------------------------------ #
    def extract_dialogue_tokens(self) -> "tuple[List[int], float]":
        length = int(self.buffer.dialogue_length)
        length = max(0, min(length, 32))
        tokens = [int(self.buffer.dialogue_token_ids[i]) for i in range(length)]
        thinking_score = float(self.buffer.dialogue_thinking_score)
        return tokens, thinking_score

    # ------------------------------------------------------------------ #
    # Memory helpers
    # ------------------------------------------------------------------ #
    def extract_memory_write(self) -> "tuple[int, List[float], float]":
        zone_id = int(self.buffer.mem_zone_id)
        embedding = [float(v) for v in self.buffer.mem_embedding]
        confidence = float(self.buffer.mem_confidence)
        return zone_id, embedding, confidence

    # ------------------------------------------------------------------ #
    # Tablet helpers
    # ------------------------------------------------------------------ #
    def extract_tablet_mutation(self) -> "tuple[int, List[int]]":
        mutation_type = int(self.buffer.tablet_mutation_type)
        payload = [int(v) for v in self.buffer.tablet_data]
        return mutation_type, payload

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Zero the buffer in-place and mark as NO_ACTION."""
        ctypes.memset(ctypes.addressof(self.buffer), 0, ACTION_BUFFER_SIZE)
        self.buffer.action_type = int(ActionType.NO_ACTION.value)


# ---------------------------------------------------------------------- #
# Numpy-dtype-style row proxy — emulates ``arr["field"][0]`` + slice set.
# ---------------------------------------------------------------------- #
class _ActionBufferFieldRow:
    """Row proxy that defers to ``[0]`` for scalars or array indexing.

    Supports two usage patterns from the legacy numpy API:
      * ``buf["action_type"][0] = 5``  (scalar write)
      * ``buf["tablet_data"][0][:] = (a, b, c, d, e, f)``  (array write)
    """

    __slots__ = ("_struct", "_field")

    def __init__(self, struct: ActionBufferStruct, field: str) -> None:
        if not hasattr(struct, field):
            raise KeyError(f"ActionBuffer has no field {field!r}")
        self._struct = struct
        self._field = field

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(
                "ActionBuffer row proxy is batch-size 1; only index 0 is valid"
            )
        if self._field in _ARRAY_FIELDS:
            return _ActionBufferArrayView(self._struct, self._field)
        return getattr(self._struct, self._field)

    def __setitem__(self, index: int, value: Any) -> None:
        if index != 0:
            raise IndexError(
                "ActionBuffer row proxy is batch-size 1; only index 0 is valid"
            )
        _assign_field(self._struct, self._field, value)


class _ActionBufferArrayView:
    """Slice-writable view over a fixed-size ctypes array field."""

    __slots__ = ("_struct", "_field")

    def __init__(self, struct: ActionBufferStruct, field: str) -> None:
        self._struct = struct
        self._field = field

    def _array(self):
        return getattr(self._struct, self._field)

    def __len__(self) -> int:
        return len(self._array())

    def __getitem__(self, index):
        arr = self._array()
        if isinstance(index, slice):
            return [arr[i] for i in range(*index.indices(len(arr)))]
        return arr[int(index)]

    def __setitem__(self, index, value) -> None:
        arr = self._array()
        n = len(arr)
        if isinstance(index, slice):
            indices = list(range(*index.indices(n)))
            values = _coerce_sequence(value, len(indices))
            for i, v in zip(indices, values):
                arr[i] = v
            return
        idx = int(index)
        arr[idx] = _coerce_scalar(value)

    def __iter__(self):
        return iter(list(self._array()))


def _assign_field(struct: ActionBufferStruct, field: str, value: Any) -> None:
    if not hasattr(struct, field):
        raise KeyError(f"ActionBuffer has no field {field!r}")
    if field in _ARRAY_FIELDS:
        arr = getattr(struct, field)
        n = len(arr)
        values = _coerce_sequence(value, n, allow_broadcast=True)
        for i in range(n):
            arr[i] = values[i]
    else:
        setattr(struct, field, _coerce_scalar(value))


def _coerce_scalar(value: Any):
    """Best-effort scalar coercion — matches numpy-dtype assignment semantics."""
    if isinstance(value, (bool, int)):
        return int(value)
    if isinstance(value, float):
        return float(value)
    # Accept objects that expose ``__int__`` / ``__float__`` / ``value``.
    value_attr = getattr(value, "value", None)
    if value_attr is not None and value_attr is not value:
        return _coerce_scalar(value_attr)
    if hasattr(value, "__int__"):
        return int(value)
    if hasattr(value, "__float__"):
        return float(value)
    return value


def _coerce_sequence(value: Any, length: int, *, allow_broadcast: bool = False) -> List[Any]:
    """Turn ``value`` into a length-``length`` list of coerced scalars.

    If ``value`` is a scalar and ``allow_broadcast`` is True, broadcast it
    across the full length (numpy ``[...] = 0`` semantics).
    """
    if isinstance(value, (bytes, bytearray, str)):
        raise TypeError(
            f"ActionBuffer array field cannot accept {type(value).__name__} assignment"
        )
    if isinstance(value, Sequence):
        items = list(value)
    elif hasattr(value, "__iter__") and not isinstance(value, (int, float, bool)):
        items = list(value)
    elif allow_broadcast:
        coerced = _coerce_scalar(value)
        return [coerced] * length
    else:
        raise TypeError(
            f"ActionBuffer array field requires a sequence, got {type(value).__name__}"
        )
    if len(items) != length:
        raise ValueError(
            f"ActionBuffer array field expected {length} values, got {len(items)}"
        )
    return [_coerce_scalar(v) for v in items]
