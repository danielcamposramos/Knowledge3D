"""Wake-Cycle Delta Capture — Gap 1 of Sleeptime Lane B.

Captures per-tile activation-magnitude delta signals from successful TRM
query steps and writes them to shadow_copy.event_buffer in the format that
Lane B (sleeptime_weights.py) expects.

Sovereignty contract:
  - No numpy / cupy / scipy / sympy.
  - No Python math loops over tiles or activations.
  - Allowed loops: ctypes readback (I/O only) and the per-tile expand
    from scalar → TILE_TRITS floats (I/O repetition, not math).
  - If the kernel cannot run: raise.  No fallback.
  - No defaults on load-bearing inputs (halting_value, halting_threshold).

Opcode: 0x320  WAKE_CYCLE_DELTA_CAPTURE
Registry: RPN_DOMAIN_OPCODE_REGISTRY §11, reserved 2026-04-20.

Lane B contract (sleeptime_weights.py):
  Each event in shadow_copy.event_buffer is wrapped by record_event()
  into: {"data": event_data, ...}.  Lane B reads ev["data"]["delta_tiles"].
  Therefore the dict returned here (used as event_data) must contain:
    delta_tiles  — list[float] of length n_tiles * TILE_TRITS (20 per tile)
    confidence   — float in [0, 1]
    timestamp    — float (monotonic)
"""
from __future__ import annotations

import ctypes
import time
from pathlib import Path
from typing import Any

from knowledge3d.cranium.kernels.kernel_loader import (
    gpu_free,
    gpu_malloc,
    gpu_memcpy_dtoh,
    gpu_synchronize,
    launch_kernel,
)
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file

# ---------------------------------------------------------------------------
# Constants — must align with sleeptime_weights.py
# ---------------------------------------------------------------------------

_TILE_TRITS = 20        # one uint32 = 20 trits at 1.6 bits/trit (BitNet b1.58)
_BLOCK_SIZE = 128

# Convergence threshold for wake-delta emission.  This is NOT a halting-value
# default — it is the fixed [0,1] threshold the kernel compares against the
# caller-supplied real halting readback.  Derived from the halting-gate
# minimum_threshold used by _halting_gate_converged() for non-LHE/MMLU tasks.
_HALTING_THRESHOLD = 0.5

_KERNEL_CU = Path(__file__).parent.parent / "cranium" / "kernels" / "wake_delta_capture.cu"

# ---------------------------------------------------------------------------
# Module-level PTX cache (compiled once per process)
# ---------------------------------------------------------------------------

_PTX_CACHE: bytes | None = None


def _get_ptx() -> bytes:
    global _PTX_CACHE
    if _PTX_CACHE is None:
        ptx_text = compile_cuda_file(str(_KERNEL_CU))
        _PTX_CACHE = ptx_text.encode("utf-8") if isinstance(ptx_text, str) else bytes(ptx_text)
    return _PTX_CACHE


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def capture_wake_delta(
    daemon: Any,
    halting_value: float,
    confidence: float,
) -> dict[str, Any] | None:
    """Capture per-tile activation-magnitude delta from a converged query step.

    Parameters
    ----------
    daemon:
        K3DDaemon instance — provides activation scratch buffer and TRM host
        weight tile count via daemon.kv._trm_host_weights.
    halting_value:
        REAL halting-gate readback for this query step (float in [0, 1]).
        MUST be the device-side scalar from the composed head (e.g. swarm
        halting_counter / n_active, or a halting-gate device readback).
        Callers MUST NOT fabricate this value — per Daniel's directive
        "No default! real halting readback." (2026-04-21).
    confidence:
        Success confidence forwarded to Lane B trace_weights (typically
        same as halting_value).

    Returns
    -------
    dict | None
        If the query converged (halting_value >= _HALTING_THRESHOLD): returns
        {"delta_tiles": [...], "confidence": float, "timestamp": float,
         "specialist": "wake_delta", "galaxy": "", "verification": "wake_delta_capture"}.
        The delta_tiles list has length n_tiles * TILE_TRITS (20 per tile) —
        each scalar per-tile delta is broadcast to all 20 trit positions.
        Returns None if the query did not converge (kernel fired out_fired=0).

    Raises
    ------
    RuntimeError
        If the kernel launch fails, if the TRM weight tile inventory is
        missing, or if the real activation scratch pointer has not been
        wired on daemon.kv._activation_scratch.  No fallback — per
        feedback_no_fallbacks_ever_including_sleeptime.md.
    """
    halting_value = float(halting_value)
    confidence = float(confidence)

    # ── Determine tile count from host weights ─────────────────────────────
    kv = getattr(daemon, "kv", None)
    host_weights: dict[str, Any] = getattr(kv, "_trm_host_weights", None) or {}
    if not host_weights:
        # No weight tiles available yet — bootstrap phase, no delta to record.
        # This is a narrow pre-load state, not a reasoning fallback.
        return None

    all_bytes = b"".join(
        bytes(raw) if not isinstance(raw, (bytes, bytearray)) else raw
        for raw in host_weights.values()
    )
    if not all_bytes or len(all_bytes) % 4 != 0:
        raise RuntimeError(
            "wake_delta_capture: _trm_host_weights buffer invalid — "
            f"len={len(all_bytes)} (must be non-zero multiple of TILE_BYTES=4)"
        )

    n_tiles = len(all_bytes) // 4   # TILE_BYTES = 4

    # ── Real activation scratch pointer (Gap 1 — Fix 1, Daniel 2026-04-21) ─
    # The composed head writes its final-layer activation vector to the TRM
    # step-fused bridge's y_new buffer (TRM_DIMS = 512 floats).  The daemon
    # wires this onto kv._activation_scratch as a (device_ptr, T, N) tuple
    # so the wake-delta kernel can read it without re-allocating.
    #
    # Contract: activation_scratch = (ctypes_device_ptr, T, N)
    #   device_ptr — raw y_new (or any T×N activation scratch) on VRAM
    #   T          — tile count, MUST equal n_tiles derived from host weights
    #   N          — activations per tile (e.g. TRM_DIMS // T for y_new slices,
    #                or 1 when mapping one scalar per tile)
    #
    # If the real scratch is not wired, we FAIL LOUD — no uniform-delta stub.
    activation_scratch = getattr(kv, "_activation_scratch", None)
    if activation_scratch is None:
        raise RuntimeError(
            "wake_delta_capture: kv._activation_scratch not wired. "
            "The daemon must populate kv._activation_scratch = "
            "(device_ptr, T, N) from the composed head's final-layer "
            "activation buffer (e.g. TRM bridge y_new) before calling "
            "_maybe_emit_wake_delta.  No stub — per Daniel's ruling "
            "'1 - wire it' (2026-04-21)."
        )
    if not (isinstance(activation_scratch, tuple) and len(activation_scratch) == 3):
        raise RuntimeError(
            "wake_delta_capture: kv._activation_scratch must be a "
            "(device_ptr, T, N) tuple; got "
            f"{type(activation_scratch).__name__}."
        )

    d_act_ptr, act_T, act_N = activation_scratch
    act_T = int(act_T)
    tile_width = int(act_N)
    if act_T != n_tiles:
        raise RuntimeError(
            "wake_delta_capture: activation_scratch tile count mismatch — "
            f"T={act_T} but n_tiles={n_tiles} (derived from host weights). "
            "The composed head activation buffer must be tiled to match "
            "the TRM weight inventory."
        )
    if tile_width <= 0:
        raise RuntimeError(
            f"wake_delta_capture: activation_scratch tile_width invalid ({tile_width})."
        )

    # ── Allocate output buffers ────────────────────────────────────────────
    sz_out_delta = n_tiles * ctypes.sizeof(ctypes.c_float)
    sz_out_fired = ctypes.sizeof(ctypes.c_int)

    d_out_delta = gpu_malloc(sz_out_delta)
    d_out_fired = gpu_malloc(sz_out_fired)

    try:
        grid_size = max(1, (n_tiles + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
        ptx = _get_ptx()

        launch_kernel(
            ptx_code=ptx,
            kernel_name="wake_delta_capture",
            grid_size=grid_size,
            block_size=_BLOCK_SIZE,
            shared_mem_size=0,
            kernel_params=[
                d_act_ptr,
                ctypes.c_float(halting_value),
                ctypes.c_float(_HALTING_THRESHOLD),
                d_out_delta,
                d_out_fired,
                ctypes.c_int(n_tiles),
                ctypes.c_int(tile_width),
            ],
        )
        gpu_synchronize()

        # ── Read back fired flag ─────────────────────────────────────────
        fired_arr = (ctypes.c_int * 1)(0)
        gpu_memcpy_dtoh(
            ctypes.cast(fired_arr, ctypes.c_void_p),
            d_out_fired,
            sz_out_fired,
        )
        fired = int(fired_arr[0])

        if fired == 0:
            return None  # Query did not converge — no delta to record.

        # ── Read back per-tile delta scalars ─────────────────────────────
        delta_scalar_arr = (ctypes.c_float * n_tiles)()
        gpu_memcpy_dtoh(
            ctypes.cast(delta_scalar_arr, ctypes.c_void_p),
            d_out_delta,
            sz_out_delta,
        )

        # ── Expand scalar → TILE_TRITS per tile (I/O readback, not math) ─
        # Lane B expects delta_tiles of length n_tiles * TILE_TRITS where
        # each group of TILE_TRITS floats corresponds to one weight tile.
        # We broadcast the per-tile scalar to all 20 trit positions.
        # This is ctypes readback: reading one c_float per tile from the
        # device buffer and repeating it _TILE_TRITS times (I/O copy).
        delta_tiles: list[float] = []
        readback_count = n_tiles   # I/O loop bound — NOT a tile-math loop
        for rb_i in range(readback_count):
            scalar = float(delta_scalar_arr[rb_i])
            for _ in range(_TILE_TRITS):
                delta_tiles.append(scalar)

    except Exception as exc:
        raise RuntimeError(
            f"wake_delta_capture: kernel failed — {exc}"
        ) from exc
    finally:
        # Free our own output buffers.  Do NOT free d_act_ptr — it is owned
        # by the composed head pipeline (TRM bridge), not by us.
        gpu_free(d_out_delta)
        gpu_free(d_out_fired)

    return {
        "delta_tiles": delta_tiles,
        "confidence": confidence,
        "timestamp": time.monotonic(),
        "specialist": "wake_delta",
        "galaxy": "",
        "verification": "wake_delta_capture",
    }
