"""Sleeptime Lane A — temporary-star promote / merge / discard.

Sovereignty contract (feedback_sleeptime_orchestration_is_ptx_not_python.md):
  - ALL iteration, gravity computation, and decision logic runs inside
    sleeptime_lane_a.cu on the GPU.
  - Python does exactly THREE things:
      1. Load the temporary JSONL into a pinned host buffer → VRAM.
      2. Launch the kernel.
      3. Read back the outcome buffer → append JSONL (promote) or tombstone
         (discard).  Stars with trit=0 stay in _temporary/ untouched.

NO numpy / cupy / scipy / sympy.  NO Python for-loop over stars for math.
Allowed loops: JSONL readline (host-buffer upload) and JSONL writeline
(I/O tail).  That is all.
"""
from __future__ import annotations

import ctypes
import json
import os
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
# Constants
# ---------------------------------------------------------------------------

_KERNEL_CU = Path(__file__).parent.parent / "cranium" / "kernels" / "sleeptime_lane_a.cu"
_BLOCK_SIZE = 128
_DIM = 16          # Matryoshka tier-64 prefix (16 floats × 4 = 64 bytes ≈ tier_64)
_MAX_RULES = 64    # mirrors LANE_A_MAX_RULES in the kernel

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
# Public entry point — called from daemon._run_sleep_consolidation_tick
# ---------------------------------------------------------------------------

def run_lane_a_tick(
    *,
    storage_root: Path,
    house_stars: list[dict[str, Any]],
    grammar_rules: list[dict[str, Any]],
) -> dict[str, Any]:
    """Execute one Lane A sleeptime tick.

    Parameters
    ----------
    storage_root:
        Root of the K3D local storage tree.  Temporary stars live at
        ``storage_root/galaxies/_temporary/``.
    house_stars:
        Flat list of existing house/galaxy star dicts.  Each must have
        ``embedding16`` (list[float], len 16) and ``galaxy`` (str) fields.
    grammar_rules:
        List of grammar-galaxy rule dicts.  Each must have
        ``domain_hash`` (int) and ``rule_strength`` (int, one of -1/0/+1).

    Returns
    -------
    dict
        Telemetry: tick_id, pending_before, promoted, merged, discarded,
        pending_after, duration_ms.
    """
    t0 = time.perf_counter()
    tick_id = str(uuid.uuid4())

    temp_dir = storage_root / "galaxies" / "_temporary"
    galaxy_dir = storage_root / "galaxies"

    # ── Collect all pending (confidence_trit == 0) candidate stars ──────────
    # This is the ONLY Python loop over stars — it is JSONL I/O, not math.
    candidate_entries: list[tuple[Path, dict[str, Any]]] = []
    if temp_dir.exists():
        for jl in sorted(temp_dir.glob("*.jsonl")):
            with open(jl, encoding="utf-8") as fh:
                for raw in fh:
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("confidence_trit", 0) == 0:
                        candidate_entries.append((jl, entry))

    pending_before = len(candidate_entries)
    if pending_before == 0:
        return {
            "tick_id": tick_id,
            "pending_before": 0,
            "promoted": 0,
            "merged": 0,
            "discarded": 0,
            "pending_after": 0,
            "duration_ms": float((time.perf_counter() - t0) * 1000.0),
            "status": "skipped",
            "reason": "no_pending_candidates",
        }

    N = pending_before
    H = len(house_stars)
    R = min(len(grammar_rules), _MAX_RULES)

    # ── Build host buffers (ctypes, no numpy) ──────────────────────────────
    # cand_embeddings: N × DIM float32
    FloatN_DIM = ctypes.c_float * (N * _DIM)
    cand_emb_host = FloatN_DIM()
    # cand_domain_hash: N uint32
    Uint32N = ctypes.c_uint32 * N
    cand_dom_host = Uint32N()

    for i, (_, entry) in enumerate(candidate_entries):
        emb = (entry.get("embedding16") or ([0.0] * _DIM))[:_DIM]
        while len(emb) < _DIM:
            emb.append(0.0)
        base = i * _DIM
        for d in range(_DIM):
            cand_emb_host[base + d] = float(emb[d])
        # domain_hash: use last 32 bits of star_id hash or 0
        raw_hash = entry.get("domain_hash") or 0
        cand_dom_host[i] = ctypes.c_uint32(int(raw_hash) & 0xFFFFFFFF).value

    # house_embeddings: H × DIM float32
    FloatH_DIM = ctypes.c_float * max(1, H * _DIM)
    house_emb_host = FloatH_DIM()
    Uint32H = ctypes.c_uint32 * max(1, H)
    house_dom_host = Uint32H()
    Int32H = ctypes.c_int32 * max(1, H)
    house_gal_host = Int32H()

    # Build a stable galaxy→int mapping (host-buffer packing, not decision logic)
    gal_names: list[str] = []
    _gal_map: dict[str, int] = {}
    for hs in house_stars:
        gname = str(hs.get("galaxy") or "")
        if gname not in _gal_map:
            _gal_map[gname] = len(gal_names)
            gal_names.append(gname)

    for h, hs in enumerate(house_stars):
        emb = (hs.get("embedding16") or ([0.0] * _DIM))[:_DIM]
        while len(emb) < _DIM:
            emb.append(0.0)
        base = h * _DIM
        for d in range(_DIM):
            house_emb_host[base + d] = float(emb[d])
        raw_hash = hs.get("domain_hash") or 0
        house_dom_host[h] = ctypes.c_uint32(int(raw_hash) & 0xFFFFFFFF).value
        house_gal_host[h] = ctypes.c_int32(_gal_map.get(str(hs.get("galaxy") or ""), -1)).value

    # rule buffers: R entries
    RSize = max(1, R)
    Uint32R = ctypes.c_uint32 * RSize
    Int8R = ctypes.c_int8 * RSize
    rule_dom_host = Uint32R()
    rule_str_host = Int8R()
    for r, rule in enumerate(grammar_rules[:R]):
        rule_dom_host[r] = ctypes.c_uint32(int(rule.get("domain_hash") or 0) & 0xFFFFFFFF).value
        rule_str_host[r] = ctypes.c_int8(max(-1, min(1, int(rule.get("rule_strength") or 0)))).value

    # output buffers
    Int8N = ctypes.c_int8 * N
    Int32N = ctypes.c_int32 * N
    out_trits_host = Int8N()
    out_gal_host = Int32N()
    out_idx_host = Int32N()

    # ── Upload to VRAM ──────────────────────────────────────────────────────
    sz_float_N_DIM = N * _DIM * ctypes.sizeof(ctypes.c_float)
    sz_float_H_DIM = max(1, H) * _DIM * ctypes.sizeof(ctypes.c_float)
    sz_uint_N = N * ctypes.sizeof(ctypes.c_uint32)
    sz_uint_H = max(1, H) * ctypes.sizeof(ctypes.c_uint32)
    sz_int_H  = max(1, H) * ctypes.sizeof(ctypes.c_int32)
    sz_uint_R = RSize * ctypes.sizeof(ctypes.c_uint32)
    sz_int8_R = RSize * ctypes.sizeof(ctypes.c_int8)
    sz_int8_N = N * ctypes.sizeof(ctypes.c_int8)
    sz_int_N  = N * ctypes.sizeof(ctypes.c_int32)

    d_cand_emb  = gpu_malloc(sz_float_N_DIM)
    d_cand_dom  = gpu_malloc(sz_uint_N)
    d_house_emb = gpu_malloc(sz_float_H_DIM)
    d_house_dom = gpu_malloc(sz_uint_H)
    d_house_gal = gpu_malloc(sz_int_H)
    d_rule_dom  = gpu_malloc(sz_uint_R)
    d_rule_str  = gpu_malloc(sz_int8_R)
    d_out_trits = gpu_malloc(sz_int8_N)
    d_out_gal   = gpu_malloc(sz_int_N)
    d_out_idx   = gpu_malloc(sz_int_N)

    try:
        gpu_memcpy_htod(d_cand_emb,  ctypes.cast(cand_emb_host, ctypes.c_void_p),  sz_float_N_DIM)
        gpu_memcpy_htod(d_cand_dom,  ctypes.cast(cand_dom_host, ctypes.c_void_p),  sz_uint_N)
        gpu_memcpy_htod(d_house_emb, ctypes.cast(house_emb_host, ctypes.c_void_p), sz_float_H_DIM)
        gpu_memcpy_htod(d_house_dom, ctypes.cast(house_dom_host, ctypes.c_void_p), sz_uint_H)
        gpu_memcpy_htod(d_house_gal, ctypes.cast(house_gal_host, ctypes.c_void_p), sz_int_H)
        gpu_memcpy_htod(d_rule_dom,  ctypes.cast(rule_dom_host, ctypes.c_void_p),  sz_uint_R)
        gpu_memcpy_htod(d_rule_str,  ctypes.cast(rule_str_host, ctypes.c_void_p),  sz_int8_R)

        # ── Launch kernel ───────────────────────────────────────────────────
        grid_size = max(1, (N + _BLOCK_SIZE - 1) // _BLOCK_SIZE)
        ptx = _get_ptx()

        # Kernel args: all pointer/int params in order matching sleeptime_lane_a_eval signature
        launch_kernel(
            ptx_code=ptx,
            kernel_name="sleeptime_lane_a_eval",
            grid_size=grid_size,
            block_size=_BLOCK_SIZE,
            shared_mem_size=0,
            kernel_params=[
                d_cand_emb,  d_cand_dom,  ctypes.c_int(N),
                d_house_emb, d_house_dom, d_house_gal, ctypes.c_int(H),
                d_rule_dom,  d_rule_str,  ctypes.c_int(R),
                d_out_trits, d_out_gal,   d_out_idx,
            ],
        )
        gpu_synchronize()

        # ── Read back outcome buffers ───────────────────────────────────────
        gpu_memcpy_dtoh(ctypes.cast(out_trits_host, ctypes.c_void_p), d_out_trits, sz_int8_N)
        gpu_memcpy_dtoh(ctypes.cast(out_gal_host,   ctypes.c_void_p), d_out_gal,   sz_int_N)
        gpu_memcpy_dtoh(ctypes.cast(out_idx_host,   ctypes.c_void_p), d_out_idx,   sz_int_N)

    finally:
        for ptr in (d_cand_emb, d_cand_dom, d_house_emb, d_house_dom,
                    d_house_gal, d_rule_dom, d_rule_str,
                    d_out_trits, d_out_gal, d_out_idx):
            try:
                gpu_free(ptr)
            except Exception:
                pass

    # ── Disk-save tail (I/O only — no logic) ───────────────────────────────
    # This is the ONLY second loop over stars; it is JSONL writeline only.
    promoted = 0
    merged = 0
    discarded = 0
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for i, (jl_path, entry) in enumerate(candidate_entries):
        trit = int(out_trits_host[i])
        gal_id = int(out_gal_host[i])
        house_idx = int(out_idx_host[i])

        if trit == 1:
            # Promote: append to target galaxy JSONL
            gal_name = gal_names[gal_id] if 0 <= gal_id < len(gal_names) else "Reality"
            target_path = galaxy_dir / f"{gal_name}.jsonl"
            galaxy_dir.mkdir(parents=True, exist_ok=True)
            promoted_entry = dict(entry)
            promoted_entry["confidence_trit"] = 1
            promoted_entry["temporary"] = False
            promoted_entry["promoted_at"] = now_iso
            promoted_entry["source_ingest_id"] = entry.get("ingest_id", "")
            promoted_entry["best_house_match_idx"] = house_idx
            with open(target_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(promoted_entry, ensure_ascii=False) + "\n")
            # Check if this is a merge (best_house_match_idx != -1)
            if house_idx >= 0:
                merged += 1
            else:
                promoted += 1

        elif trit == -1:
            # Discard: append tombstone to the _temporary/ JSONL (append-only audit)
            tombstone = {
                "tombstone": True,
                "star_id": entry.get("star_id") or entry.get("ingest_id"),
                "ingest_id": entry.get("ingest_id"),
                "discarded_at": now_iso,
                "confidence_trit": -1,
            }
            with open(jl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(tombstone, ensure_ascii=False) + "\n")
            discarded += 1
        # trit == 0: leave untouched (stays pending for next tick)

    pending_after = pending_before - promoted - merged - discarded

    return {
        "tick_id": tick_id,
        "pending_before": pending_before,
        "promoted": promoted,
        "merged": merged,
        "discarded": discarded,
        "pending_after": pending_after,
        "duration_ms": float((time.perf_counter() - t0) * 1000.0),
        "status": "ok",
    }
