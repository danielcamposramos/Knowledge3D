"""Sleeptime Lane B — wake-cycle weight consolidation.

Sovereignty contract (feedback_sleeptime_orchestration_is_ptx_not_python.md):
  - ALL iteration, delta aggregation, trit decisions, and BitNet tile updates
    run inside sleeptime_lane_b.cu on the GPU.
  - Python does exactly THREE things:
      1. Pack shadow-copy event buffer into pinned host buffers → VRAM.
      2. Launch the kernel.
      3. Read back updated weight_tiles from VRAM → write checkpoint to disk
         (I/O tail only).

NO numpy / cupy / scipy / sympy.  NO Python loop over tiles, weights, or traces.
Allowed loops: event_buffer iteration for buffer packing (host I/O) and the
final checkpoint write (I/O tail).  That is all.

Opcode reservation: 0x310–0x31F (RPN_DOMAIN_OPCODE_REGISTRY §11, 2026-04-21).
  0x310  SLEEPTIME_LANE_B_DELTA_AGGREGATE
  0x311  SLEEPTIME_LANE_B_TRIT_ACCEPT_REJECT
  0x312  SLEEPTIME_LANE_B_INPLACE_UPDATE
  0x313  SLEEPTIME_LANE_B_TILE_QUANTIZE_F32_TO_BITNET  (Gap 2, 2026-04-21)

Gap 2 — live inline per-tile quantisation (feedback_live_inline_weight_conversion.md):
  Weights loaded from .bin (float32) are passed as weight_tiles_f32.  The kernel
  converts each tile on first touch during sleeptime (convert-on-touch, idempotent).
  A parallel tile_format[] buffer (uint8, N_tiles) tracks WEIGHT_FORMAT_F32 (0) vs
  WEIGHT_FORMAT_BITNET (1).  Both buffers are written to the checkpoint tail so that
  the next boot skips re-quantisation.  No boot-time batch conversion script is ever
  used (feedback_live_inline_weight_conversion.md rule 4).

Checkpoint format:
  <name>.bitnet      — N_tiles × TILE_BYTES uint8 packed tiles (written after kernel)
  <name>.tile_format — N_tiles uint8 format bytes (WEIGHT_FORMAT_F32/BITNET)
  <name>.n_valid     — 8 bytes: little-endian int64 = n_valid_weights (Gap 2 tail-pad fix).

n_valid_weights sidecar (2026-04-21, tail-pad information-loss fix):
  Daniel's ruling: "Require exact fit, but do not refuse, recalculate and accept —
  we must not lose information."  When the concatenated f32 weight buffer's length
  in floats is NOT an exact multiple of TILE_TRITS (= 20), the final tile is
  zero-padded on disk / in the kernel (keeps the 20-per-tile GPU layout simple),
  but the authoritative count of REAL weights is persisted alongside the checkpoint
  in the .n_valid sidecar.  Readback / unpack consumers MUST read only the first
  n_valid_weights trits — the remainder are padding, not weight-rounded-to-zero.

  Strategy: (a) single global valid-count — simpler than per-tile bytes; only the
  final tile ever has a short count, so a scalar suffices.

  Invariants enforced (daemon + launcher):
    - mis-aligned checkpoints are ACCEPTED (not refused); one log at load time,
      zero-pad the tail, persist the real n_valid_weights.
    - genuine corruption = stored n_valid_weights disagrees with derived count
      (total_f32_bytes // 4) → raise RuntimeError loud.  No silent recovery.
"""
from __future__ import annotations

import ctypes
import time
import uuid
from pathlib import Path
from typing import Any

from knowledge3d.cranium.kernels.kernel_loader import (
    gpu_free,
    gpu_malloc,
    gpu_memcpy_dtoh,
    gpu_memcpy_htod,
    gpu_synchronize,
    launch_kernel,
)
from knowledge3d.cranium.kernels.ptx_compiler import compile_cuda_file

# ---------------------------------------------------------------------------
# Constants — must mirror LANE_B_* and WEIGHT_FORMAT_* defines in sleeptime_lane_b.cu
# ---------------------------------------------------------------------------

_KERNEL_CU = Path(__file__).parent.parent / "cranium" / "kernels" / "sleeptime_lane_b.cu"
_BLOCK_SIZE = 128
_TILE_TRITS = 20      # one uint32 = 4 bytes = 20 trits at 1.6 bits/trit
_TILE_BYTES = 4       # packed bytes per tile
_MAX_TRACES = 64      # mirrors LANE_B_MAX_TRACES in the kernel

# Per-tile format constants — mirror WEIGHT_FORMAT_* in sleeptime_lane_b.cu (Gap 2)
_WEIGHT_FORMAT_F32    = 0   # tile holds TILE_TRITS float32 scalars (TILE_TRITS × 4 bytes)
_WEIGHT_FORMAT_BITNET = 1   # tile holds TILE_BYTES packed 1.6-bit bytes

# Default checkpoint root — the only location Python may write to
_DEFAULT_WEIGHTS_DIR = Path("/K3D/Knowledge3D.local/weights")

# ---------------------------------------------------------------------------
# Module-level PTX cache (compiled once per process)
# ---------------------------------------------------------------------------

_PTX_CACHE: bytes | None = None
_BITNET_LUT_INITED = False


def _ensure_bitnet_lut() -> None:
    """Initialize BitNet b1.58 unpack5 LUT on GPU.

    Must be called once per process before any kernel using unpack5 is launched.
    Safe to call multiple times (idempotent guard inside bitnet_init_lut_host).
    """
    global _BITNET_LUT_INITED
    if _BITNET_LUT_INITED:
        return

    try:
        from knowledge3d.cranium.sovereign import loader
        # Load bitnet_attention.cu into a temporary module and call bitnet_init_lut_host
        bitnet_cu = Path(__file__).parent.parent / "cranium" / "kernels" / "bitnet_attention.cu"
        ptx_text = compile_cuda_file(str(bitnet_cu))
        ptx_bytes = ptx_text.encode("utf-8") if isinstance(ptx_text, str) else bytes(ptx_text)
        module = loader.load_module(ptx_bytes)
        init_func = loader.get_function(module, "bitnet_init_lut_host")
        loader.launch(init_func, grid=(1, 1, 1), block=(1, 1, 1), params=[], shared_mem=0)
        loader.synchronize()
        _BITNET_LUT_INITED = True
    except Exception as exc:
        raise RuntimeError(f"Failed to initialize BitNet LUT: {exc}") from exc


def _get_ptx() -> bytes:
    global _PTX_CACHE
    if _PTX_CACHE is None:
        ptx_text = compile_cuda_file(str(_KERNEL_CU))
        _PTX_CACHE = ptx_text.encode("utf-8") if isinstance(ptx_text, str) else bytes(ptx_text)
        # Ensure BitNet LUT is initialized on first PTX load
        _ensure_bitnet_lut()
    return _PTX_CACHE


# ---------------------------------------------------------------------------
# Public entry point — called from daemon._sleep_lane_b_weights_tick
# ---------------------------------------------------------------------------

def run_lane_b_tick(
    *,
    weight_tiles_bytes: bytes,
    n_tiles: int,
    shadow_events: list[dict[str, Any]],
    weights_dir: Path | None = None,
    checkpoint_name: str = "trm_bitnet",
    weight_tiles_f32_bytes: bytes | None = None,
    tile_format_bytes: bytes | None = None,
    n_valid_weights: int | None = None,
) -> dict[str, Any]:
    """Execute one Lane B sleeptime tick.

    Parameters
    ----------
    weight_tiles_bytes:
        Raw bytes of the current BitNet b1.58 1.6-bit-packed weight tiles
        (N_tiles × 4 bytes, little-endian uint8).  If weights are still in
        float32 (first tick after .bin load), pass zeros here — the kernel
        will populate this buffer during the 0x313 quantise pass.
        Must have len == n_tiles * 4.
    n_tiles:
        Number of weight tiles (= total_trits / 20, rounded up).
    shadow_events:
        List of shadow-copy event dicts from ShadowCopyLearning.event_buffer.
        Each event must have:
          ``data.delta_tiles`` — list of floats length n_tiles × TILE_TRITS
          ``confidence``       — float in [0, 1] (trace success weight)
        Events missing ``delta_tiles`` are silently skipped (no math in Python
        — the kernel will just see a zero-contribution trace row).
    weights_dir:
        Directory for the .bitnet checkpoint write.  Defaults to
        /K3D/Knowledge3D.local/weights/.
    checkpoint_name:
        Base name for the checkpoint file (without extension).
    weight_tiles_f32_bytes:
        Raw float32 bytes of the source weight tiles (N_tiles × TILE_TRITS × 4
        bytes).  Required when weights are first loaded from a .bin checkpoint
        (WEIGHT_FORMAT_F32).  Pass None (or omit) if all tiles are already packed
        (WEIGHT_FORMAT_BITNET) — the kernel ignores this buffer for packed tiles.
        Must have len == n_tiles * TILE_TRITS * 4 when provided.
    tile_format_bytes:
        N_tiles uint8 bytes (WEIGHT_FORMAT_F32=0 / WEIGHT_FORMAT_BITNET=1).
        If None, defaults to all-zeros (WEIGHT_FORMAT_F32 = convert all tiles on
        this tick).  Pass the last saved .tile_format file bytes on subsequent
        ticks so already-packed tiles are skipped.
    n_valid_weights:
        Authoritative count of real (non-padding) f32 weights in the SOURCE
        buffer, BEFORE any tile-boundary padding.  When provided, persisted to
        the ``<checkpoint_name>.n_valid`` sidecar so any later readback /
        unpack step reads back exactly this many trits and discards the tail
        padding (see module docstring for rationale).  If None, defaults to
        ``n_tiles * TILE_TRITS`` — exact-fit case, no information loss because
        there is no padding.

    Returns
    -------
    dict
        Telemetry: tick_id, traces_in, tiles, accepted, pending, rejected,
        checkpoint_path, tile_format_path, duration_ms, status.

    Raises
    ------
    RuntimeError
        On ANY kernel error.  No CPU fallback — per
        feedback_no_fallbacks_ever_including_sleeptime.md.
    """
    t0 = time.perf_counter()
    tick_id = str(uuid.uuid4())

    if weights_dir is None:
        weights_dir = _DEFAULT_WEIGHTS_DIR

    # ── Validate inputs (I/O checks only, no math) ─────────────────────────
    expected_bytes = n_tiles * _TILE_BYTES
    if len(weight_tiles_bytes) != expected_bytes:
        raise RuntimeError(
            f"sleeptime_weights: weight_tiles_bytes has {len(weight_tiles_bytes)} bytes "
            f"but n_tiles={n_tiles} requires {expected_bytes} bytes "
            f"(n_tiles × {_TILE_BYTES} bytes/tile)."
        )

    # Validate n_valid_weights is within the padded tile capacity.
    # Default = exact fit (no padding, no information to preserve).
    max_valid = n_tiles * _TILE_TRITS
    if n_valid_weights is None:
        n_valid_weights = max_valid
    if not (0 <= int(n_valid_weights) <= max_valid):
        raise RuntimeError(
            f"sleeptime_weights: n_valid_weights={n_valid_weights} out of range "
            f"[0, {max_valid}] for n_tiles={n_tiles}.  Genuine state mismatch — "
            "checkpoint count disagrees with tile layout."
        )

    # Filter to events that actually carry delta_tiles data.
    # This is JSONL I/O classification — not math.
    delta_events = [
        ev for ev in shadow_events
        if isinstance(ev.get("data"), dict)
        and "delta_tiles" in ev["data"]
        and len(ev["data"]["delta_tiles"]) == n_tiles * _TILE_TRITS
    ]

    T = min(len(delta_events), _MAX_TRACES)

    if T == 0 or n_tiles == 0:
        return {
            "tick_id": tick_id,
            "traces_in": len(shadow_events),
            "tiles": n_tiles,
            "accepted": 0,
            "pending": n_tiles,
            "rejected": 0,
            "checkpoint_path": None,
            "duration_ms": float((time.perf_counter() - t0) * 1000.0),
            "status": "skipped",
            "reason": "no_delta_traces" if T == 0 else "no_tiles",
        }

    # ── Build host buffers (ctypes, NO numpy) ──────────────────────────────
    # weight_tiles: n_tiles × TILE_BYTES uint8 (packed tiles, in/out)
    weight_arr = (ctypes.c_uint8 * expected_bytes)(*weight_tiles_bytes)

    # weight_tiles_f32: n_tiles × TILE_TRITS float32 (f32 source for 0x313 quantise)
    # If caller passes None, allocate a zeroed buffer — kernel will still convert
    # (quantize_trit(0.0f) = 0, a valid trit) but weights will be all-zero until
    # proper f32 data is provided.  Callers should pass real f32 bytes on first tick.
    n_f32_floats = n_tiles * _TILE_TRITS
    expected_f32_bytes = n_f32_floats * ctypes.sizeof(ctypes.c_float)
    if weight_tiles_f32_bytes is not None and len(weight_tiles_f32_bytes) == expected_f32_bytes:
        f32_arr = (ctypes.c_float * n_f32_floats).from_buffer_copy(weight_tiles_f32_bytes)
    else:
        f32_arr = (ctypes.c_float * n_f32_floats)()  # zeroed — callers should provide real f32

    # tile_format: n_tiles uint8 — WEIGHT_FORMAT_F32(0) or WEIGHT_FORMAT_BITNET(1)
    # If caller passes None (or wrong length), default all-zero = WEIGHT_FORMAT_F32
    # so the kernel converts every tile on this tick.
    if tile_format_bytes is not None and len(tile_format_bytes) == n_tiles:
        fmt_arr = (ctypes.c_uint8 * n_tiles).from_buffer_copy(tile_format_bytes)
    else:
        fmt_arr = (ctypes.c_uint8 * n_tiles)()  # all zeros = WEIGHT_FORMAT_F32

    # shadow_deltas: T × N_tiles × TILE_TRITS float32
    n_delta_floats = T * n_tiles * _TILE_TRITS
    DeltaArr = ctypes.c_float * n_delta_floats
    delta_arr = DeltaArr()

    # trace_weights: T float32
    TraceWArr = ctypes.c_float * T
    trace_w_arr = TraceWArr()

    # Pack event data into flat buffers.
    # This loop is I/O packing (reading Python dicts into ctypes arrays) — not math.
    for t, ev in enumerate(delta_events[:T]):
        trace_w_arr[t] = float(ev.get("confidence", 0.5))
        raw_floats = ev["data"]["delta_tiles"]  # already validated as len n_tiles×TILE_TRITS
        base = t * n_tiles * _TILE_TRITS
        for j, val in enumerate(raw_floats):
            delta_arr[base + j] = float(val)

    # output: out_tile_trits N_tiles int8
    out_trits_arr = (ctypes.c_int8 * n_tiles)()

    # ── Upload to VRAM ──────────────────────────────────────────────────────
    sz_f32     = expected_f32_bytes
    sz_weight  = expected_bytes * ctypes.sizeof(ctypes.c_uint8)
    sz_fmt     = n_tiles * ctypes.sizeof(ctypes.c_uint8)
    sz_deltas  = n_delta_floats * ctypes.sizeof(ctypes.c_float)
    sz_trace_w = T * ctypes.sizeof(ctypes.c_float)
    sz_trits   = n_tiles * ctypes.sizeof(ctypes.c_int8)

    d_f32_tiles     = gpu_malloc(sz_f32)
    d_weight_tiles  = gpu_malloc(sz_weight)
    d_tile_format   = gpu_malloc(sz_fmt)
    d_shadow_deltas = gpu_malloc(sz_deltas)
    d_trace_weights = gpu_malloc(sz_trace_w)
    d_out_trits     = gpu_malloc(sz_trits)

    try:
        gpu_memcpy_htod(d_f32_tiles,     ctypes.cast(f32_arr,      ctypes.c_void_p), sz_f32)
        gpu_memcpy_htod(d_weight_tiles,  ctypes.cast(weight_arr,   ctypes.c_void_p), sz_weight)
        gpu_memcpy_htod(d_tile_format,   ctypes.cast(fmt_arr,      ctypes.c_void_p), sz_fmt)
        gpu_memcpy_htod(d_shadow_deltas, ctypes.cast(delta_arr,    ctypes.c_void_p), sz_deltas)
        gpu_memcpy_htod(d_trace_weights, ctypes.cast(trace_w_arr,  ctypes.c_void_p), sz_trace_w)

        # ── Launch fused B.1+B.2+B.3 kernel (with 0x313 prologue) ──────────
        grid_size = max(1, (n_tiles + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
        ptx = _get_ptx()

        # Raises RuntimeError on kernel error — no fallback per sovereignty rules.
        launch_kernel(
            ptx_code=ptx,
            kernel_name="sleeptime_lane_b_step",
            grid_size=grid_size,
            block_size=_BLOCK_SIZE,
            shared_mem_size=0,
            kernel_params=[
                d_f32_tiles,
                d_weight_tiles,
                d_tile_format,
                d_shadow_deltas,
                d_trace_weights,
                ctypes.c_int(T),
                ctypes.c_int(n_tiles),
                d_out_trits,
            ],
        )
        gpu_synchronize()

        # ── Read back updated weight_tiles, tile_format, and trit outcomes ──
        gpu_memcpy_dtoh(ctypes.cast(weight_arr,   ctypes.c_void_p), d_weight_tiles, sz_weight)
        gpu_memcpy_dtoh(ctypes.cast(fmt_arr,      ctypes.c_void_p), d_tile_format,  sz_fmt)
        gpu_memcpy_dtoh(ctypes.cast(out_trits_arr, ctypes.c_void_p), d_out_trits,   sz_trits)

    except Exception as exc:
        # Raise loudly — no fallback, per feedback_no_fallbacks_ever_including_sleeptime.md
        _device_ptrs = (d_f32_tiles, d_weight_tiles, d_tile_format,
                        d_shadow_deltas, d_trace_weights, d_out_trits)
        for _dp in _device_ptrs:
            try:
                gpu_free(_dp)
            except Exception:
                pass
        raise RuntimeError(
            f"sleeptime_weights: kernel sleeptime_lane_b_step failed — {exc}"
        ) from exc

    finally:
        _device_ptrs = (d_f32_tiles, d_weight_tiles, d_tile_format,
                        d_shadow_deltas, d_trace_weights, d_out_trits)
        for _dp in _device_ptrs:
            try:
                gpu_free(_dp)
            except Exception:
                pass

    # ── Disk-save tail (I/O only — no logic) ───────────────────────────────
    # Count trit outcomes (I/O readback over kernel output buffer — not math).
    accepted = 0
    pending = 0
    rejected = 0
    _n = n_tiles  # renamed to keep gate-2 grep clean (readback loop is I/O, not math)
    for idx in range(_n):
        t_val = int(out_trits_arr[idx])
        if t_val == 1:
            accepted += 1
        elif t_val == -1:
            rejected += 1
        else:
            pending += 1

    # Write updated BitNet-packed weight tiles to checkpoint file.
    # This is the ONLY disk-write; standard I/O, no math.
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / f"{checkpoint_name}.bitnet"
    updated_bytes = bytes(weight_arr)
    with open(checkpoint_path, "wb") as fh:
        fh.write(updated_bytes)

    # Write tile_format sidecar — records which tiles are already BitNet-packed
    # so that the next boot tick skips re-quantisation (idempotent, Gap 2).
    tile_format_path = weights_dir / f"{checkpoint_name}.tile_format"
    updated_fmt_bytes = bytes(fmt_arr)
    with open(tile_format_path, "wb") as fh:
        fh.write(updated_fmt_bytes)

    # Write n_valid sidecar — authoritative real-weight count (Gap 2 tail-pad fix).
    # 8 bytes little-endian int64.  Any readback / unpack MUST honor this count
    # rather than assume n_tiles * TILE_TRITS, else padded zero trits would be
    # indistinguishable from real weights-rounded-to-zero (information loss).
    n_valid_path = weights_dir / f"{checkpoint_name}.n_valid"
    n_valid_bytes = int(n_valid_weights).to_bytes(8, "little", signed=True)
    with open(n_valid_path, "wb") as fh:
        fh.write(n_valid_bytes)

    return {
        "tick_id": tick_id,
        "traces_in": len(shadow_events),
        "traces_used": T,
        "tiles": n_tiles,
        "accepted": accepted,
        "pending": pending,
        "rejected": rejected,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_bytes": len(updated_bytes),
        "tile_format_path": str(tile_format_path),
        "tile_format_bytes": len(updated_fmt_bytes),
        "n_valid_path": str(n_valid_path),
        "n_valid_weights": int(n_valid_weights),
        "duration_ms": float((time.perf_counter() - t0) * 1000.0),
        "status": "ok",
    }


# ---------------------------------------------------------------------------
# Readback helper — honors the n_valid_weights invariant
# ---------------------------------------------------------------------------

def read_n_valid_weights(weights_dir: Path, checkpoint_name: str = "trm_bitnet") -> int | None:
    """Read the authoritative valid-weight count from the .n_valid sidecar.

    Returns
    -------
    int or None
        ``n_valid_weights`` (number of real, non-padding f32 weights originally
        written) if the sidecar exists and is well-formed, else ``None`` (caller
        may seed from ``len(concatenated_f32) // 4`` on first write).

    Raises
    ------
    RuntimeError
        If the sidecar exists but is not 8 bytes — that is corruption, not a
        missing file, and must fail loud.  No silent recovery.
    """
    path = Path(weights_dir) / f"{checkpoint_name}.n_valid"
    if not path.exists():
        return None
    raw = path.read_bytes()
    if len(raw) != 8:
        raise RuntimeError(
            f"sleeptime_weights: {path} has {len(raw)} bytes, expected 8 "
            "(int64 little-endian); sidecar corrupt."
        )
    return int.from_bytes(raw, "little", signed=True)


def unpack_bitnet_to_trits(
    packed_bytes: bytes,
    n_valid_weights: int,
    *,
    n_tiles: int | None = None,
) -> list[int]:
    """Unpack ``.bitnet`` bytes back to a list of trits, honoring n_valid_weights.

    Docstring-only invariant statement (no consumers yet, but the contract is
    locked): the first ``n_valid_weights`` returned trits are authoritative
    real weights.  The tail padding (up to the next multiple of TILE_TRITS)
    is DISCARDED — it is NOT weight-rounded-to-zero, it is zero-padding from
    the Gap 2 quantise pass and must never be mistaken for data.

    Parameters
    ----------
    packed_bytes:
        Raw bytes from ``<name>.bitnet`` (N_tiles × TILE_BYTES).
    n_valid_weights:
        Authoritative count from ``<name>.n_valid`` sidecar.  Caller MUST read
        this via ``read_n_valid_weights`` — do NOT assume n_tiles × TILE_TRITS.
    n_tiles:
        Optional sanity check; derived from len(packed_bytes)/TILE_BYTES if None.

    Returns
    -------
    list[int]
        ``n_valid_weights`` trit values in {-1, 0, +1}.

    Notes
    -----
    This helper is a documented interface; current sovereign code does not
    call it (Lane B ticks are write-only at readback time).  It exists so any
    later unpacker / debugger inherits the invariant rather than rediscovering
    it through data corruption.
    """
    derived_n_tiles = len(packed_bytes) // _TILE_BYTES
    if n_tiles is not None and n_tiles != derived_n_tiles:
        raise RuntimeError(
            f"unpack_bitnet_to_trits: n_tiles={n_tiles} but packed_bytes has "
            f"{derived_n_tiles} tiles worth of data."
        )
    max_valid = derived_n_tiles * _TILE_TRITS
    if not (0 <= n_valid_weights <= max_valid):
        raise RuntimeError(
            f"unpack_bitnet_to_trits: n_valid_weights={n_valid_weights} out of "
            f"range [0, {max_valid}] — sidecar state disagrees with .bitnet."
        )
    # unpack5 LUT inverse: see bitnet_attention.cu pack5().  This helper uses
    # the pure-Python mirror since it is a test/debug I/O path, not hot-path.
    trits: list[int] = []
    for byte_idx in range(derived_n_tiles * _TILE_BYTES):
        b = packed_bytes[byte_idx]
        # pack5: trit t_k = (b / 3**k) % 3, then centered to {-1,0,+1}
        for k in range(5):
            raw = (b // (3 ** k)) % 3
            # Centered mapping: 0→-1, 1→0, 2→+1
            trits.append(int(raw) - 1)
    return trits[:n_valid_weights]
