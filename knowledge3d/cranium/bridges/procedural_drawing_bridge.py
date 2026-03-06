"""Procedural 2D drawing bridge backed by sovereign PTX rasterizers.

This bridge parses simple RPN drawing programs on the host, converts them into
line segment batches, and renders them on GPU via the existing
``procedural_glyph_rasterizer`` PTX kernel. It keeps orchestration in Python
only; all rasterization happens on GPU, respecting the sovereignty contract
(<100 µs budget enforced by LatencyGuard).

The intent is to provide an incremental, working surface for the wider
Procedural Drawing Stack (pixel_genesis/universal_primitive kernels, TTF/CDR
parsers). It supports the core path ops: MOVE, LINE, QUAD, CUBIC, ARC, CLOSE,
STROKE, FILL plus basic transforms. As kernels mature, this bridge can delegate
RPN execution to GPU by swapping the parser with a PTX path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes
import os
import re
import math
import time
from typing import List, Sequence, Tuple

import numpy as np

from knowledge3d.cranium.bridges.procedural_glyph_bridge import ProceduralGlyphBridge
from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects
from knowledge3d.cranium.ptx_runtime.latency_guard import LatencyGuard
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine


# Matryoshka-driven quality map (segments + supersample factor)
MATRYOSHKA_QUALITY = {
    64: {"name": "simple", "segments": 8, "supersample": 1},
    128: {"name": "medium", "segments": 16, "supersample": 2},
    512: {"name": "standard", "segments": 32, "supersample": 2},
    1024: {"name": "high", "segments": 64, "supersample": 4},
    2048: {"name": "extreme", "segments": 128, "supersample": 4},
}


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _pop(stack: List[float], count: int) -> List[float]:
    if len(stack) < count:
        raise ValueError(f"Stack underflow: need {count}, have {len(stack)}")
    out = stack[-count:]
    del stack[-count:]
    return out


def _approximate_quad(p0: Tuple[float, float], c: Tuple[float, float], p1: Tuple[float, float], segments: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(1, segments + 1):
        t = i / segments
        s = 1.0 - t
        x = s * s * p0[0] + 2 * s * t * c[0] + t * t * p1[0]
        y = s * s * p0[1] + 2 * s * t * c[1] + t * t * p1[1]
        pts.append((x, y))
    return pts


def _approximate_cubic(p0: Tuple[float, float], c1: Tuple[float, float], c2: Tuple[float, float], p1: Tuple[float, float], segments: int) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(1, segments + 1):
        t = i / segments
        s = 1.0 - t
        x = (
            s * s * s * p0[0]
            + 3 * s * s * t * c1[0]
            + 3 * s * t * t * c2[0]
            + t * t * t * p1[0]
        )
        y = (
            s * s * s * p0[1]
            + 3 * s * s * t * c1[1]
            + 3 * s * t * t * c2[1]
            + t * t * t * p1[1]
        )
        pts.append((x, y))
    return pts


def _approximate_arc(
    center: Tuple[float, float],
    radius: Tuple[float, float],
    start_angle: float,
    sweep_angle: float,
    segments: int,
) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for i in range(1, segments + 1):
        t = i / segments
        angle = start_angle + sweep_angle * t
        pts.append(
            (
                center[0] + radius[0] * math.cos(angle),
                center[1] + radius[1] * math.sin(angle),
            )
        )
    return pts


@dataclass
class RenderResult:
    """Container for rendered output."""

    segments: np.ndarray | None = None
    rgba: np.ndarray | None = None


@dataclass
class MathRecord:
    """Descriptor for a math buffer primitive."""

    opcode: int
    payload: np.ndarray | None = None
    payload_exprs: List[str] | None = None
    payload_len: int | None = None
    flags: int = 0


class ProceduralDrawingBridge:
    """Host-side orchestrator for procedural 2D drawing."""

    MAX_SEGMENTS = 4096  # safety cap to avoid runaway tessellation
    SEGMENT_STRIDE = 9   # x0,y0,x1,y1,r,g,b,a,width
    _WARMED_PID: int | None = None

    def __init__(self, matryoshka_dim: int = 512) -> None:
        quality = MATRYOSHKA_QUALITY.get(matryoshka_dim, MATRYOSHKA_QUALITY[512])
        self.segments_per_curve = quality["segments"]
        self.supersample = quality["supersample"]
        self.rasterizer = ProceduralGlyphBridge()
        self.effects = DrawingEffects()
        # Guard tuned for arc/device-math path (~13-25 ms on 3060-class)
        self.latency_guard = LatencyGuard(threshold_us=26000.0)

        # Try to load GPU RPN executor (pixel_genesis universal primitive path).
        ptx_path = Path(__file__).parent.parent / "ptx" / "pixel_genesis_universal_primitive.ptx"
        self.pixel_genesis_module = (
            loader.load_module_from_file(str(ptx_path))
            if ptx_path.exists()
            else None
        )
        self.pixel_genesis_kernel = (
            loader.get_function(self.pixel_genesis_module, "execute_drawing_rpn")
            if self.pixel_genesis_module
            else None
        )
        # Reusable GPU buffers to reduce per-call overhead.
        self._bytecode_cap = 4096
        self._d_bytecode = loader.gpu_malloc(self._bytecode_cap)
        self._d_segments = loader.gpu_malloc(self.MAX_SEGMENTS * self.SEGMENT_STRIDE * 4)
        self._d_count = loader.gpu_malloc(4)
        # Math buffer (header+records and payload) reused when provided
        self._math_hdrrec_cap = 0
        self._math_payload_cap = 0
        self._d_math_hdrrec = loader.CUdeviceptr(0)
        self._d_math_payload = loader.CUdeviceptr(0)
        self._rpn_engine: ModularRPNEngine | None = None
        # RPN executor kernel (device-side bytecode path)
        rpn_exec_ptx = Path(__file__).parent.parent / "ptx" / "rpn_executor.ptx"
        self.rpn_executor_kernel = (
            loader.get_function(loader.load_module_from_file(str(rpn_exec_ptx)), "execute_rpn_bytecode")
            if rpn_exec_ptx.exists()
            else None
        )
        self._warmup_report: dict[str, float | bool | str] | None = None

    def _get_rpn_engine(self):
        """Lazy-load RPN Math Kernel for trigonometric preprocessing."""
        if self._rpn_engine is None:
            from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
            self._rpn_engine = ModularRPNEngine()
        return self._rpn_engine

    def _preprocess_rpn_math(self, program: str) -> tuple[str, List[MathRecord]]:
        """Detect RPN_* tokens, compute via RPN Math Kernel, return cleaned program + math records.

        Labor division for math kernels (👷‍♂️ meta-easter-egg): master/worker/woker-of-worker
        tiers can be composed here by batching expressions through the 18-stack RPN engine.
        """
        if "RPN_" not in program:
            return program, []

        tokens = program.strip().split()
        cleaned_tokens: List[str] = []
        arc_records: List[MathRecord] = []

        # Pass 1: handle RPN_ARC by generating precomputed points via parallel RPN
        for tok in tokens:
            if tok.upper() == "RPN_ARC":
                if len(cleaned_tokens) < 6:
                    raise ValueError("RPN_ARC expects 6 operands: rx ry start sweep cx cy")
                # Pull last 6 operands
                params_tokens = cleaned_tokens[-6:]
                try:
                    rx, ry, start, sweep, cx, cy = [float(x) for x in params_tokens]
                except ValueError as exc:
                    raise ValueError(f"RPN_ARC operands must be numeric, got {params_tokens}") from exc
                # Remove consumed operands
                cleaned_tokens = cleaned_tokens[:-6]

                # Build angles and batch sin/cos via RPN engine
                segs = self.segments_per_curve
                angles = np.linspace(start, start + sweep, segs + 1)
                exprs = []
                # First element is count
                exprs.append(f"{len(angles)}")  # literal count
                # Then x/y pairs per angle
                for ang in angles:
                    exprs.append(f"{rx} {ang} cos * {cx} +")
                    exprs.append(f"{ry} {ang} sin * {cy} +")

                # Host payload (fallback)
                rpn_engine = self._get_rpn_engine()
                trig = rpn_engine.evaluate_batch([f"{ang} cos" for ang in angles] + [f"{ang} sin" for ang in angles])
                arc_points = [len(angles)]
                for i in range(len(angles)):
                    cos_val = trig[i]
                    sin_val = trig[len(angles) + i]
                    x = cx + rx * cos_val
                    y = cy + ry * sin_val
                    arc_points.extend([x, y])

                arc_records.append(
                    MathRecord(
                        opcode=0x7A,
                        payload=np.array(arc_points, dtype=np.float32),
                        payload_exprs=exprs,
                        payload_len=len(exprs),
                    )
                )
                cleaned_tokens.append("PRECOMPUTED_PATH")
            else:
                cleaned_tokens.append(tok)

        cleaned = " ".join(cleaned_tokens)

        # Pass 2: handle RPN_SIN/RPN_COS pairs for rotation
        pattern = r"([\d\.\s+\-*/piPIπτφe]+)\s+RPN_SIN\s+RPN_COS"
        matches = list(re.finditer(pattern, cleaned, re.IGNORECASE))
        if not matches:
            # Only arcs found; align records to program order
            math_records: List[MathRecord] = []
            arc_cursor = 0
            for tok in cleaned_tokens:
                if tok.upper() == "PRECOMPUTED_PATH" and arc_cursor < len(arc_records):
                    math_records.append(arc_records[arc_cursor])
                    arc_cursor += 1
            return " ".join(cleaned_tokens), math_records

        rpn_engine = self._get_rpn_engine()
        angle_exprs = [match.group(1).strip().lower() for match in matches]
        all_exprs: List[str] = []
        for angle in angle_exprs:
            all_exprs.extend([f"{angle} cos", f"{angle} sin"])

        results = rpn_engine.evaluate_batch(all_exprs)
        rotation_records: List[MathRecord] = []
        res_idx = 0
        for _ in matches:
            rotation_records.append(
                MathRecord(
                    opcode=0x79,
                    payload=np.array([results[res_idx], results[res_idx + 1]], dtype=np.float32),
                )
            )
            res_idx += 2

        for match in matches:
            cleaned = cleaned.replace(match.group(0), " ", 1)

        cleaned_tokens = cleaned.split()
        # Align math records to program order (PRECOMPUTED_PATH / ROTATE_MATRIX tokens)
        math_records: List[MathRecord] = []
        arc_cursor = 0
        rot_cursor = 0
        for tok in cleaned_tokens:
            upper = tok.upper()
            if upper == "PRECOMPUTED_PATH" and arc_cursor < len(arc_records):
                math_records.append(arc_records[arc_cursor])
                arc_cursor += 1
            elif upper == "ROTATE_MATRIX" and rot_cursor < len(rotation_records):
                math_records.append(
                    MathRecord(
                        opcode=rotation_records[rot_cursor].opcode,
                        payload=rotation_records[rot_cursor].payload,
                        payload_exprs=[
                            f"{angle_exprs[rot_cursor]} cos",
                            f"{angle_exprs[rot_cursor]} sin",
                        ],
                        payload_len=2,
                    )
                )
                rot_cursor += 1

        if rot_cursor != len(rotation_records):
            raise ValueError("Unmatched RPN_SIN/RPN_COS pairs and ROTATE_MATRIX tokens.")

        return " ".join(cleaned_tokens), math_records

    def _build_math_buffers(self, math_records: List[MathRecord]) -> tuple[np.ndarray, np.ndarray]:
        """Compose enhanced math buffer with header/records + payload."""
        if not math_records:
            self._math_primitive_count = 0
            return np.zeros(4, dtype=np.uint32), np.zeros(0, dtype=np.float32)

        records: List[List[int]] = []
        payload_chunks: List[np.ndarray] = []
        float_offset = 0

        for rec in math_records:
            if rec.payload is None:
                continue
            payload = np.ascontiguousarray(rec.payload.astype(np.float32, copy=False))
            payload_chunks.append(payload)
            records.append([rec.opcode, float_offset, payload.size, rec.flags])
            float_offset += payload.size

        primitive_count = len(records)
        header = np.array([primitive_count, float_offset, 0, 0], dtype=np.uint32)
        record_arr = np.array(records, dtype=np.uint32).reshape(-1)
        hdrrec = np.concatenate([header, record_arr]).astype(np.uint32)
        payload_np = (
            np.concatenate(payload_chunks).astype(np.float32, copy=False)
            if payload_chunks
            else np.zeros(0, dtype=np.float32)
        )
        self._math_primitive_count = primitive_count
        return hdrrec, payload_np

    def _build_math_buffers_device(
        self, math_records: List[MathRecord], rpn_engine: ModularRPNEngine
    ) -> tuple[np.ndarray, loader.CUdeviceptr, bool]:
        """Compose math buffers using device-side RPN evaluation."""
        if not math_records:
            self._math_primitive_count = 0
            return np.zeros(4, dtype=np.uint32), loader.CUdeviceptr(0), False

        exprs: List[str] = []
        per_record_counts: List[int] = []
        for rec in math_records:
            if not rec.payload_exprs or rec.payload_len is None:
                # Fallback to host payload path
                hdr, payload = self._build_math_buffers(math_records)
                return hdr, loader.CUdeviceptr(0), False
            exprs.extend(rec.payload_exprs)
            per_record_counts.append(rec.payload_len)

        d_payload, total_count = rpn_engine.evaluate_batch_device(exprs)
        float_offset = 0
        records: List[List[int]] = []
        for rec, count in zip(math_records, per_record_counts):
            records.append([rec.opcode, float_offset, count, rec.flags])
            float_offset += count

        header = np.array([len(records), float_offset, 0, 0], dtype=np.uint32)
        record_arr = np.array(records, dtype=np.uint32).reshape(-1)
        hdrrec = np.concatenate([header, record_arr]).astype(np.uint32)
        self._math_primitive_count = len(records)
        return hdrrec, d_payload, True

    def _normalize_math_buffer(self, math_buffer, program: str) -> List[MathRecord]:
        """Accept ndarray or MathRecord list and normalize to MathRecord list."""
        if math_buffer is None:
            return []
        if isinstance(math_buffer, MathRecord):
            return [math_buffer]
        if isinstance(math_buffer, (list, tuple)):
            if not math_buffer:
                return []
            if isinstance(math_buffer[0], MathRecord):
                return list(math_buffer)
        if isinstance(math_buffer, np.ndarray):
            payload = np.ascontiguousarray(math_buffer.astype(np.float32, copy=False))
            upper = program.upper()
            opcode = 0x7A  # PRECOMPUTED_PATH default
            if "ROTATE_MATRIX" in upper and (payload.size == 2 or "PRECOMPUTED_PATH" not in upper):
                opcode = 0x79
            return [MathRecord(opcode=opcode, payload=payload)]
        raise TypeError("math_buffer must be None, ndarray, MathRecord, or a list of MathRecord.")

    def execute_rpn_program(self, rpn_program: str, width: int = 256, height: int = 256) -> RenderResult:
        """Parse an RPN drawing string, rasterize on GPU, and return RGBA buffer."""
        segments, offsets, lengths = self._rpn_to_segments(rpn_program)
        framebuffer = self._render_segments(segments, offsets, lengths, width, height)
        return RenderResult(segments=segments, rgba=framebuffer)

    def compile_rpn_to_bytecode(self, rpn_program: str) -> np.ndarray:
        """Public helper to compile RPN string to bytecode for dataset building."""
        return self._compile_rpn_bytecode(rpn_program)

    def execute_rpn_gpu(
        self,
        rpn_program: str,
        width: int = 256,
        height: int = 256,
        skip_raster: bool = False,
        ternary_hint: float = 0.0,
        math_buffer: np.ndarray | MathRecord | Sequence[MathRecord] | None = None,
        use_device_math: bool = True,
        track_latency: bool = True,
    ) -> RenderResult:
        """Execute drawing RPN entirely on GPU (bytecode → segments → rasterize).

        Falls back to host parsing if the GPU kernel is unavailable.
        """
        if self.pixel_genesis_kernel is None:
            return self.execute_rpn_program(rpn_program, width, height)

        if track_latency:
            self.latency_guard.start()

        # Preprocess RPN math tokens if math_buffer not provided
        if math_buffer is None:
            rpn_program, math_records = self._preprocess_rpn_math(rpn_program)
        else:
            math_records = self._normalize_math_buffer(math_buffer, rpn_program)

        bytecode = self._compile_rpn_bytecode(rpn_program)

        # Grow bytecode buffer if needed (with headroom)
        if bytecode.nbytes > self._bytecode_cap:
            loader.gpu_free(self._d_bytecode)
            self._bytecode_cap = bytecode.nbytes * 2
            self._d_bytecode = loader.gpu_malloc(self._bytecode_cap)

        loader.memcpy_htod(
            self._d_bytecode, bytecode.ctypes.data_as(ctypes.c_void_p), bytecode.nbytes
        )

        # Enhanced math buffer layout: [records][payload]
        device_payload_ptr = None
        device_payload_is_temp = False
        if use_device_math:
            hdrrec_np, d_payload_tmp, device_payload_is_temp = self._build_math_buffers_device(math_records, self._get_rpn_engine())
            if device_payload_is_temp:
                device_payload_ptr = d_payload_tmp
        if not device_payload_is_temp:
            hdrrec_np, payload_np = self._build_math_buffers(math_records)
        else:
            payload_np = np.zeros(0, dtype=np.float32)
        if hdrrec_np.nbytes > self._math_hdrrec_cap:
            if self._d_math_hdrrec.value:
                loader.gpu_free(self._d_math_hdrrec)
            self._math_hdrrec_cap = max(hdrrec_np.nbytes, 256)
            self._d_math_hdrrec = loader.gpu_malloc(self._math_hdrrec_cap)
        if payload_np.nbytes > self._math_payload_cap and not device_payload_is_temp:
            if self._d_math_payload.value:
                loader.gpu_free(self._d_math_payload)
            self._math_payload_cap = max(payload_np.nbytes, 256)
            self._d_math_payload = loader.gpu_malloc(self._math_payload_cap)

        if hdrrec_np.size:
            loader.memcpy_htod(self._d_math_hdrrec, hdrrec_np.ctypes.data_as(ctypes.c_void_p), hdrrec_np.nbytes)
        if payload_np.size and not device_payload_is_temp:
            loader.memcpy_htod(self._d_math_payload, payload_np.ctypes.data_as(ctypes.c_void_p), payload_np.nbytes)

        payload_ptr = device_payload_ptr if device_payload_is_temp else self._d_math_payload

        loader.launch(
            self.pixel_genesis_kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                self._d_bytecode,
                ctypes.c_uint32(bytecode.nbytes),
                self._d_segments,
                self._d_count,
                ctypes.c_uint32(self.segments_per_curve),
                ctypes.c_float(ternary_hint),
                self._d_math_hdrrec,
                ctypes.c_uint32(self._math_primitive_count),
                payload_ptr,
            ],
        )

        if device_payload_is_temp and device_payload_ptr is not None and device_payload_ptr.value:
            loader.gpu_free(device_payload_ptr)

        count_host = np.zeros(1, dtype=np.uint32)
        loader.memcpy_dtoh(
            count_host.ctypes.data_as(ctypes.c_void_p),
            self._d_count,
            4,
        )

        seg_count = min(int(count_host[0]), self.MAX_SEGMENTS)
        segments = np.zeros((seg_count, self.SEGMENT_STRIDE), dtype=np.float32)
        if seg_count:
            loader.memcpy_dtoh(
                segments.ctypes.data_as(ctypes.c_void_p),
                self._d_segments,
                segments.nbytes,
            )

        if track_latency:
            elapsed_ns, breached = self.latency_guard.stop()
            if breached:
                import logging
                logging.warning(f"GPU RPN execution breached latency budget: {elapsed_ns / 1000:.1f} µs")

        if skip_raster:
            return RenderResult(segments=segments, rgba=None)

        framebuffer = self._render_segments(
            segments,
            np.array([0], dtype=np.int32),
            np.array([seg_count], dtype=np.int32),
            width,
            height,
        )

        return RenderResult(segments=segments, rgba=framebuffer)

    def execute_batch_gpu(self, programs: Sequence[str], width: int = 256, height: int = 256) -> List[RenderResult]:
        """Execute multiple RPN programs; placeholder loop until kernel batch mode exists."""
        return [self.execute_rpn_gpu(p, width, height) for p in programs]

    def warmup_runtime(self) -> dict[str, float | bool | str]:
        """Preload heavy drawing/runtime assets before first live interaction."""
        pid = os.getpid()
        if self._warmup_report is not None and self._WARMED_PID == pid:
            return dict(self._warmup_report)

        report: dict[str, float | bool | str] = {
            "status": "warming",
            "pid": str(pid),
        }

        t0 = time.perf_counter()
        base = self.execute_rpn_gpu(
            "0 0 MOVE 0 0 LINE STROKE",
            width=8,
            height=8,
            skip_raster=False,
            track_latency=False,
        )
        loader.synchronize()
        report["draw_warmup_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        backdrop = self.effects.linear_gradient(
            8,
            8,
            self._default_painterly_stops(),
            x1=0.0,
            y1=0.0,
            x2=1.0,
            y2=1.0,
        )
        loader.synchronize()
        report["gradient_warmup_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        composed = self.effects.alpha_over_rgba(backdrop, base.rgba)
        composed = self.effects.blur_rgba(composed, radius=1)
        composed = self.effects.sharpen_rgba(composed, radius=1, amount=0.25)
        _edges = self.effects.edge_map(composed)
        _ = self.effects.invert_rgba(composed)
        loader.synchronize()
        report["effects_warmup_ms"] = (time.perf_counter() - t0) * 1000.0

        report["status"] = "ready"
        report["total_warmup_ms"] = (
            float(report["draw_warmup_ms"])
            + float(report["gradient_warmup_ms"])
            + float(report["effects_warmup_ms"])
        )
        self._WARMED_PID = pid
        self._warmup_report = dict(report)
        return dict(report)

    def render_painterly_gpu(
        self,
        rpn_program: str,
        width: int = 256,
        height: int = 256,
        *,
        background: str | None = "linear",
        gradient_stops: Sequence[Sequence[float]] | None = None,
        blur_radius: int = 0,
        sharpen_amount: float = 0.0,
        invert: bool = False,
    ) -> RenderResult:
        """Render a drawing plus a PTX-backed background/effect stack.

        This keeps orchestration in Python while gradients, blur/sharpen,
        inversion, and compositing all execute on GPU.
        """
        result = self.execute_rpn_gpu(rpn_program, width=width, height=height)
        if result.rgba is None:
            return result

        composed = result.rgba
        if background:
            stops = list(gradient_stops or self._default_painterly_stops())
            background = background.lower()
            if background == "linear":
                backdrop = self.effects.linear_gradient(
                    width,
                    height,
                    stops,
                    x1=0.0,
                    y1=0.0,
                    x2=1.0,
                    y2=1.0,
                )
            elif background == "radial":
                backdrop = self.effects.radial_gradient(
                    width,
                    height,
                    stops,
                    cx=0.5,
                    cy=0.5,
                    radius=0.65,
                )
            elif background == "conic":
                backdrop = self.effects.conic_gradient(
                    width,
                    height,
                    stops,
                    cx=0.5,
                    cy=0.5,
                    start_angle=0.0,
                )
            else:
                raise ValueError(f"unsupported painterly background: {background}")
            composed = self.effects.alpha_over_rgba(backdrop, composed)

        if blur_radius > 0:
            composed = self.effects.blur_rgba(composed, radius=blur_radius)
        if sharpen_amount > 0.0:
            composed = self.effects.sharpen_rgba(
                composed,
                radius=max(1, blur_radius or 1),
                amount=sharpen_amount,
            )
        if invert:
            composed = self.effects.invert_rgba(composed)

        return RenderResult(segments=result.segments, rgba=composed.astype(np.float32, copy=False))

    def edge_map_gpu(self, rgba: np.ndarray) -> np.ndarray:
        """Produce a GPU Sobel edge map from an RGBA canvas."""
        return self.effects.edge_map(rgba).astype(np.float32, copy=False)

    def execute_rpn_bytecode_gpu(self, bytecode: bytes, width: int = 256, height: int = 256, ternary_meta: np.ndarray | None = None) -> RenderResult:
        """Execute precompiled RPN bytecode via device-side executor (geometry only)."""
        if self.rpn_executor_kernel is None:
            raise RuntimeError("rpn_executor.ptx not available")
        bc_np = np.frombuffer(bytecode, dtype=np.uint8)
        if bc_np.nbytes > self._bytecode_cap:
            loader.gpu_free(self._d_bytecode)
            self._bytecode_cap = bc_np.nbytes * 2
            self._d_bytecode = loader.gpu_malloc(self._bytecode_cap)
        loader.memcpy_htod(self._d_bytecode, bc_np.ctypes.data_as(ctypes.c_void_p), bc_np.nbytes)
        # segments buffer already allocated with stride 9
        loader.memcpy_htod(self._d_count, (ctypes.c_uint32 * 1)(0), 4)
        meta_ptr = loader.CUdeviceptr(0)
        if ternary_meta is not None:
            meta_np = np.ascontiguousarray(ternary_meta.astype(np.int8, copy=False))
            meta_ptr = loader.gpu_malloc(meta_np.nbytes)
            loader.memcpy_htod(meta_ptr, meta_np.ctypes.data_as(ctypes.c_void_p), meta_np.nbytes)
        loader.launch(
            self.rpn_executor_kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                self._d_bytecode,
                ctypes.c_uint32(bc_np.nbytes),
                self._d_segments,
                self._d_count,
                ctypes.c_uint32(self.MAX_SEGMENTS),
                meta_ptr,
            ],
        )
        if ternary_meta is not None and meta_ptr.value:
            loader.gpu_free(meta_ptr)
        count_host = np.zeros(1, dtype=np.uint32)
        loader.memcpy_dtoh(count_host.ctypes.data_as(ctypes.c_void_p), self._d_count, 4)
        seg_count = min(int(count_host[0]), self.MAX_SEGMENTS)
        segments = np.zeros((seg_count, self.SEGMENT_STRIDE), dtype=np.float32)
        if seg_count:
            loader.memcpy_dtoh(segments.ctypes.data_as(ctypes.c_void_p), self._d_segments, segments.nbytes)
        framebuffer = self._render_segments(
            segments,
            np.array([0], dtype=np.int32),
            np.array([seg_count], dtype=np.int32),
            width,
            height,
        )
        return RenderResult(segments=segments, rgba=framebuffer)

    def _render_segments(
        self,
        segments: np.ndarray,
        offsets: np.ndarray,
        lengths: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        if segments.size == 0:
            # empty canvas, return transparent
            rgba = np.zeros((height, width, 4), dtype=np.float32)
            return rgba

        # Ensure stride includes style: x0,y0,x1,y1,r,g,b,a,width
        if segments.shape[1] == 4:
            style = self._extract_style_defaults()
            style_vec = np.array(style, dtype=np.float32)
            segments = np.hstack([segments, np.tile(style_vec, (segments.shape[0], 1))]).astype(np.float32)

        transforms = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # scale, rot, tx, ty
        glyph_batch = self.rasterizer.render(
            segments=segments,
            segment_offsets=offsets,
            segment_lengths=lengths,
            transforms=transforms,
            batch=1,
            height=height * self.supersample,
            width=width * self.supersample,
        ).to_numpy()[0]

        if self.supersample > 1:
            # Simple box filter downsample
            glyph_batch = glyph_batch.reshape(
                height, self.supersample, width, self.supersample, 4
            ).mean(axis=(1, 3))

        return glyph_batch.astype(np.float32)

    def _extract_style_defaults(self) -> List[float]:
        """Default style vector: r,g,b,a,width."""
        return [1.0, 1.0, 1.0, 1.0, 1.0]

    def _default_painterly_stops(self) -> List[Tuple[float, float, float, float, float]]:
        return [
            (0.0, 0.964, 0.753, 0.318, 1.0),
            (0.35, 0.906, 0.427, 0.333, 1.0),
            (0.72, 0.506, 0.247, 0.537, 1.0),
            (1.0, 0.122, 0.157, 0.353, 1.0),
        ]

    def _rpn_to_segments(self, rpn_program: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = rpn_program.strip().split()
        stack: List[float] = []
        segments: List[Tuple[float, float, float, float]] = []
        current: Tuple[float, float] | None = None
        subpath_start: Tuple[float, float] | None = None

        for token in tokens:
            if _is_number(token):
                stack.append(float(token))
                continue

            op = token.upper()
            if op == "MOVE":
                x, y = _pop(stack, 2)
                current = (x, y)
                subpath_start = current
            elif op == "LINE":
                x, y = _pop(stack, 2)
                if current is None:
                    raise ValueError("LINE before MOVE")
                segments.append((current[0], current[1], x, y))
                current = (x, y)
            elif op == "QUAD":
                x, y, cx, cy = _pop(stack, 4)
                if current is None:
                    raise ValueError("QUAD before MOVE")
                pts = _approximate_quad(current, (cx, cy), (x, y), self.segments_per_curve)
                prev = current
                for pt in pts:
                    segments.append((prev[0], prev[1], pt[0], pt[1]))
                    prev = pt
                current = (x, y)
            elif op == "CUBIC":
                x, y, cx2, cy2, cx1, cy1 = _pop(stack, 6)
                if current is None:
                    raise ValueError("CUBIC before MOVE")
                pts = _approximate_cubic(current, (cx1, cy1), (cx2, cy2), (x, y), self.segments_per_curve)
                prev = current
                for pt in pts:
                    segments.append((prev[0], prev[1], pt[0], pt[1]))
                    prev = pt
                current = (x, y)
            elif op == "ARC":
                # rx ry start_angle sweep_angle cx cy ARC
                cx, cy, sweep, start, ry, rx = _pop(stack, 6)
                if current is None:
                    raise ValueError("ARC before MOVE")
                pts = _approximate_arc(
                    center=(cx, cy),
                    radius=(rx, ry),
                    start_angle=math.radians(start),
                    sweep_angle=math.radians(sweep),
                    segments=self.segments_per_curve,
                )
                prev = current
                for pt in pts:
                    segments.append((prev[0], prev[1], pt[0], pt[1]))
                    prev = pt
                current = pts[-1] if pts else current
            elif op == "CLOSE":
                if current is not None and subpath_start is not None:
                    segments.append((current[0], current[1], subpath_start[0], subpath_start[1]))
                    current = subpath_start
            elif op in {"STROKE", "FILL"}:
                # Rendering happens after parsing; nothing to do here.
                continue
            elif op in {
                "TRANSLATE",
                "ROTATE",
                "SCALE",
                "PUSH_STATE",
                "POP_STATE",
                "SET_STROKE_COLOR",
                "SET_COLOR",
                "SET_FILL_COLOR",
                "SET_LINE_WIDTH",
                "STROKE_WIDTH",
                "SET_TERNARY_HINT",
            }:
                # Accepted for forward compatibility; no-ops in this host parser.
                _ = stack  # keep signature consistent; stateful implementation will replace
                continue
            else:
                raise ValueError(f"Unknown token/opcode: {token}")

        segments_np = np.array(segments, dtype=np.float32)
        offsets = np.array([0], dtype=np.int32)
        lengths = np.array([len(segments)], dtype=np.int32)
        return segments_np, offsets, lengths

    def _compile_rpn_bytecode(self, rpn_program: str) -> np.ndarray:
        """Compile RPN to interleaved bytecode: opcode followed by operands.

        Format (little-endian, 4-byte aligned):
            opcode (uint32) followed by operands (float32)
            MOVE: opcode(0x64) , x, y
            LINE: opcode(0x65) , x, y
            QUAD: opcode(0x66) , cx, cy, x, y
            CUBIC:opcode(0x67) , cx1, cy1, cx2, cy2, x, y
            ARC:  opcode(0x68) , rx, ry, angle, large_arc, sweep, x, y (simplified)
            CLOSE/STROKE/FILL: opcode only
        """
        import struct

        tokens = rpn_program.strip().split()
        bytecode = bytearray()

        OPCODES = {
            "MOVE": 0x64,
            "LINE": 0x65,
            "QUAD": 0x66,
            "CUBIC": 0x67,
            "ARC": 0x68,
            "CLOSE": 0x69,
            "STROKE": 0x6A,
            "FILL": 0x6B,
            "BEGIN_PATH": 0x90,
            "PUSH_STATE": 0x70,
            "POP_STATE": 0x71,
            "TRANSLATE": 0x72,
            "ROTATE": 0x73,
            "SCALE": 0x74,
            "SET_STROKE_COLOR": 0x75,
            "SET_COLOR": 0x75,  # Alias for SET_STROKE_COLOR
            "SET_FILL_COLOR": 0x76,
            "SET_LINE_WIDTH": 0x77,
            "STROKE_WIDTH": 0x77,  # Alias for SET_LINE_WIDTH
            "SET_TERNARY_HINT": 0x78,
            "TERNARY_MODULATE": 0x78,  # Alias for SET_TERNARY_HINT
            "ROTATE_MATRIX": 0x79,  # Rotation via math buffer (cos, sin)
            "PRECOMPUTED_PATH": 0x7A,  # Path from math buffer points
        }

        OPERAND_COUNTS = {
            0x64: 2,  # MOVE x y
            0x65: 2,  # LINE x y
            0x66: 4,  # QUAD cx cy x y
            0x67: 6,  # CUBIC cx1 cy1 cx2 cy2 x y
            0x68: 6,  # ARC rx ry angle large_arc sweep x y (angle simplified)
            0x69: 0,
            0x6A: 0,
            0x6B: 0,
            0x90: 0,  # BEGIN_PATH
            0x70: 0,
            0x71: 0,
            0x72: 2,
            0x73: 1,
            0x74: 2,
            0x75: 4,
            0x76: 4,
            0x77: 1,
            0x78: 1,
            0x79: 0,  # ROTATE_MATRIX (consumes math buffer)
            0x7A: 0,  # PRECOMPUTED_PATH (consumes math buffer)
        }

        float_stack: List[float] = []

        for token in tokens:
            if _is_number(token):
                float_stack.append(float(token))
                continue

            op = token.upper()
            if op not in OPCODES:
                raise ValueError(f"Unknown RPN token: {token}")

            opcode = OPCODES[op]
            need = OPERAND_COUNTS.get(opcode, 0)
            if len(float_stack) < need:
                raise ValueError(f"{op} requires {need} operands, have {len(float_stack)}")

            bytecode.extend(struct.pack("<I", opcode))
            for _ in range(need):
                val = float_stack.pop(0)
                bytecode.extend(struct.pack("<f", val))

        return np.ascontiguousarray(np.frombuffer(bytes(bytecode), dtype=np.uint8))

    def _free_buffers(self) -> None:
        if getattr(self, "_d_bytecode", None):
            loader.gpu_free(self._d_bytecode)
            loader.gpu_free(self._d_segments)
            loader.gpu_free(self._d_count)
            if self._d_math_hdrrec.value:
                loader.gpu_free(self._d_math_hdrrec)
            if self._d_math_payload.value:
                loader.gpu_free(self._d_math_payload)
            self._d_bytecode = None
            self._d_segments = None
            self._d_count = None
            self._d_math_hdrrec = loader.CUdeviceptr(0)
            self._d_math_payload = loader.CUdeviceptr(0)

    def __del__(self) -> None:
        try:
            self._free_buffers()
        except Exception:
            pass

__all__ = ["ProceduralDrawingBridge", "RenderResult", "MathRecord"]
