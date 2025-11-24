"""
Lightweight RPN bridge for the sovereign three-tier architecture.

Tier 1 focuses on ultra-fast execution of the most common operations
(arithmetic, elementary math, comparisons, and basic stack modifiers).
The bridge attempts to load the dedicated lightweight PTX kernel. When a
CUDA context is not available (e.g., in CPU-only CI), it gracefully
falls back to a minimal CPU interpreter that mirrors the Tier‑1 opcode
set so unit tests can still exercise the public API.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Iterable, Optional

import math
import struct

from knowledge3d.cranium.sovereign import loader
from .rpn_config import RPN_GRID_DIM, TIER1_BLOCK_DIM


class LightweightRPNEngine:
    """Tier‑1 RPN engine supporting the 20-op lightweight instruction set."""

    MAX_INSTANCES = 18  # Tesla 3-6-9: 18/3=6 (ternary resonance)
    STACK_DEPTH = 69    # Tesla 6-9: literal 6&9, Yin-Yang mirror symmetry
    INSTANCE_STRIDE = 1040
    SUPPORTED_OPS = {
        0, 1,
        10, 11, 12, 13, 15,
        20, 21, 22, 24, 25, 26,
        40, 42, 44, 46, 47,
        50, 51, 52,
    }

    def __init__(self):
        self._kernel = None
        self._device_state = None
        self._gpu_enabled = False
        self._scratch_codes: Optional[loader.CUdeviceptr] = None
        self._scratch_scalars: Optional[loader.CUdeviceptr] = None
        self._scratch_vectors: Optional[loader.CUdeviceptr] = None
        self._scratch_codes_capacity = 0
        self._scratch_scalars_capacity = 0
        self._scratch_vectors_capacity = 0
        self._cached_codes_bytes: Optional[bytes] = None
        self._cached_scalars_bytes: Optional[bytes] = None
        self._cached_vectors_bytes: Optional[bytes] = None
        self._cached_result: Optional[float] = None
        self._cached_codes_obj: Optional[bytes] = None
        self._cached_scalars_obj: Optional[bytes] = None
        self._cached_vectors_obj: Optional[bytes] = None

        ptx_path = Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel_lite.ptx"
        if ptx_path.exists():
            try:
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[LightweightRPN] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
                self._kernel = loader.load_ptx_file(str(ptx_path), "modular_rpn_geometric_kernel")
                if os.environ.get("K3D_RPN_DEBUG"):
                    print("[LightweightRPN] Loaded PTX")
                self._device_state = loader.gpu_malloc(self.MAX_INSTANCES * self.INSTANCE_STRIDE)
                if os.environ.get("K3D_RPN_DEBUG"):
                    print("[LightweightRPN] Allocated device state")
                zeros = (ctypes.c_uint8 * (self.MAX_INSTANCES * self.INSTANCE_STRIDE))()
                loader.memcpy_htod(self._device_state, ctypes.cast(zeros, ctypes.c_void_p), ctypes.sizeof(zeros))
                if os.environ.get("K3D_RPN_DEBUG"):
                    print("[LightweightRPN] GPU path enabled")
            except RuntimeError as exc:
                # GPU unavailable – fall back to CPU interpreter.
                self._kernel = None
                self._device_state = None
                if os.environ.get("K3D_RPN_DEBUG"):
                    print(f"[LightweightRPN] GPU path disabled (RuntimeError): {exc}")

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

    def _ensure_scratch_buffer(self, kind: str, required_bytes: int) -> loader.CUdeviceptr:
        if required_bytes <= 0:
            required_bytes = 1
        ptr_attr = f"_scratch_{kind}"
        cap_attr = f"_scratch_{kind}_capacity"
        ptr: Optional[loader.CUdeviceptr] = getattr(self, ptr_attr)
        capacity: int = getattr(self, cap_attr)
        if ptr is None or capacity < required_bytes:
            if ptr is not None:
                loader.gpu_free(ptr)
            new_capacity = max(required_bytes, capacity * 2 if capacity else required_bytes)
            ptr = loader.gpu_malloc(new_capacity)
            setattr(self, ptr_attr, ptr)
            setattr(self, cap_attr, new_capacity)
        return ptr

    def cleanup(self) -> None:
        """Release GPU resources associated with the Tier-1 engine."""
        if self._scratch_codes is not None:
            loader.gpu_free(self._scratch_codes)
            self._scratch_codes = None
            self._scratch_codes_capacity = 0
        if self._scratch_scalars is not None:
            loader.gpu_free(self._scratch_scalars)
            self._scratch_scalars = None
            self._scratch_scalars_capacity = 0
        if self._scratch_vectors is not None:
            loader.gpu_free(self._scratch_vectors)
            self._scratch_vectors = None
            self._scratch_vectors_capacity = 0
        if self._device_state is not None:
            loader.gpu_free(self._device_state)
            self._device_state = None
        self._gpu_enabled = False

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    @property
    def gpu_enabled(self) -> bool:
        """Return True when the dedicated Tier‑1 PTX kernel is active."""
        return self._gpu_enabled and self._kernel is not None

    def execute_single(
        self,
        instance_id: int,
        op_codes: Iterable[int],
        scalars: Iterable[float],
        vectors: Iterable[Iterable[float]],
    ) -> float:
        """Execute a single RPN program on Tier‑1."""
        op_codes_list = [int(o) for o in op_codes]
        scalars_list = [float(s) for s in scalars]
        vectors_list = [[float(c) for c in v] for v in vectors]

        cached = (
            self._cached_result is not None
            and self._cached_codes_bytes == bytes(self._encode_uint16(op_codes_list))
            and self._cached_scalars_bytes == self._encode_f32(scalars_list)
            and self._cached_vectors_bytes == self._encode_f32_flat(vectors_list)
        )

        if cached and self._gpu_enabled:
            return self._cached_result

        unsupported = [op for op in op_codes_list if op not in self.SUPPORTED_OPS]
        if unsupported:
            raise ValueError(
                f"Unsupported ops for Tier 1: {unsupported}. "
                "Use the Tier 2 (standard) or Tier 3 (advanced) engine."
            )

        if self._gpu_enabled:
            return self._execute_gpu(instance_id, op_codes_list, scalars_list, vectors_list)

        result = self._execute_cpu(op_codes_list, scalars_list, vectors_list)
        self._cached_result = result
        self._cached_codes_obj = None
        self._cached_scalars_obj = None
        self._cached_vectors_obj = None
        self._cached_codes_bytes = bytes(self._encode_uint16(op_codes_list))
        self._cached_scalars_bytes = self._encode_f32(scalars_list)
        self._cached_vectors_bytes = self._encode_f32_flat(vectors_list)
        return result

    # ------------------------------------------------------------------ #
    # GPU path (shares structure with Tier‑2 engine but trimmed ops)
    # ------------------------------------------------------------------ #
    def _execute_gpu(
        self,
        instance_id: int,
        op_codes: list[int],
        scalars: list[float],
        vectors: list[list[float]],
    ) -> float:
        codes_bytes = bytes(self._encode_uint16(op_codes))
        scalars_bytes = self._encode_f32(scalars)
        vectors_bytes = self._encode_f32_flat(vectors)

        d_codes = self._ensure_scratch_buffer("codes", len(codes_bytes))
        d_scalars = self._ensure_scratch_buffer("scalars", len(scalars_bytes))
        d_vectors = self._ensure_scratch_buffer("vectors", len(vectors_bytes))

        loader.memcpy_htod(d_codes, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(codes_bytes))), len(codes_bytes))
        loader.memcpy_htod(d_scalars, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(scalars_bytes))), len(scalars_bytes))
        loader.memcpy_htod(d_vectors, ctypes.c_void_p(ctypes.addressof(ctypes.create_string_buffer(vectors_bytes))), len(vectors_bytes))

        loader.launch(
            self._kernel,
            grid=(RPN_GRID_DIM, 1, 1),
            block=(TIER1_BLOCK_DIM, 1, 1),
            params=[
                ctypes.c_uint32(instance_id),
                ctypes.c_uint64(d_codes.value),
                ctypes.c_uint64(d_scalars.value),
                ctypes.c_uint64(d_vectors.value),
                ctypes.c_uint64(self._device_state.value),
                ctypes.c_uint32(len(op_codes)),
            ],
        )

        # Read stack header to locate top value
        header = (ctypes.c_uint32 * 4)()
        loader.memcpy_dtoh(
            ctypes.cast(header, ctypes.c_void_p),
            loader.CUdeviceptr(int(self._device_state.value + instance_id * self.INSTANCE_STRIDE)),
            ctypes.sizeof(header),
        )
        size = int(header[1])
        if size == 0:
            raise RuntimeError("Tier‑1 GPU execution produced empty stack")

        stack_top = (header[0] + size - 1) & 63
        element_offset = instance_id * self.INSTANCE_STRIDE + 16 + stack_top * 16
        result_vec = (ctypes.c_float * 4)()
        loader.memcpy_dtoh(
            ctypes.cast(result_vec, ctypes.c_void_p),
            loader.CUdeviceptr(int(self._device_state.value + element_offset)),
            ctypes.sizeof(result_vec),
        )
        result = float(result_vec[0])
        self._cached_result = result
        self._cached_codes_obj = op_codes
        self._cached_scalars_obj = scalars
        self._cached_vectors_obj = vectors
        self._cached_codes_bytes = codes_bytes
        self._cached_scalars_bytes = scalars_bytes
        self._cached_vectors_bytes = vectors_bytes
        return result

    def reset_instance(self, instance_id: int) -> None:
        """Reset the circular stack metadata for a given instance."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id {instance_id} (expected 0-{self.MAX_INSTANCES - 1})")
        if not self._gpu_enabled or self._device_state is None:
            return

        header_zero = (ctypes.c_uint32 * 4)()
        offset = instance_id * self.INSTANCE_STRIDE
        loader.memcpy_htod(
            loader.CUdeviceptr(int(self._device_state.value + offset)),
            header_zero.ctypes.data_as(ctypes.c_void_p),
            header_zero.nbytes,
        )

    # ------------------------------------------------------------------ #
    # CPU interpreter (fallback)
    # ------------------------------------------------------------------ #
    def _execute_cpu(
        self,
        op_codes: Iterable[int],
        scalars: Iterable[float],
        vectors: Iterable[Iterable[float]],
    ) -> float:
        stack: list[list[float]] = []
        scalar_index = 0
        vector_index = 0

        def pop_scalar() -> float:
            if not stack:
                raise RuntimeError("Tier‑1 CPU fallback stack underflow")
            value = stack.pop()
            return float(value[0])

        def push_scalar(value: float) -> None:
            stack.append([value, 0.0, 0.0, 0.0])

        for op in op_codes:
            if op == 0:  # literal scalar
                push_scalar(float(scalars[scalar_index]))
                scalar_index += 1
            elif op == 1:  # literal vector
                vec = [0.0, 0.0, 0.0, 0.0]
                cur_vec = list(vectors[vector_index])
                for idx in range(min(3, len(cur_vec))):
                    vec[idx] = cur_vec[idx]
                vector_index += 1
                stack.append(vec)
            elif op in (10, 11, 12, 13):  # add/sub/mul/div
                b = pop_scalar()
                a = pop_scalar()
                if op == 10:
                    push_scalar(a + b)
                elif op == 11:
                    push_scalar(a - b)
                elif op == 12:
                    push_scalar(a * b)
                else:
                    push_scalar(a / b)
            elif op == 15:  # neg
                push_scalar(-pop_scalar())
            elif op in (20, 21, 22, 24, 25, 26):
                a = pop_scalar()
                if op == 20:
                    push_scalar(math.sqrt(a))
                elif op == 21:
                    push_scalar(math.exp(a))
                elif op == 22:
                    push_scalar(math.log(a))
                elif op == 24:
                    push_scalar(math.sin(a))
                elif op == 25:
                    push_scalar(math.cos(a))
                else:
                    push_scalar(math.tan(a))
            elif op in (40, 42, 44, 46, 47):
                b = pop_scalar()
                a = pop_scalar()
                if op == 40:
                    push_scalar(1.0 if a > b else 0.0)
                elif op == 42:
                    push_scalar(1.0 if a < b else 0.0)
                elif op == 44:
                    push_scalar(1.0 if abs(a - b) <= 1e-6 else 0.0)
                elif op == 46:
                    push_scalar(max(a, b))
                else:
                    push_scalar(min(a, b))
            elif op == 50:  # dup
                if not stack:
                    raise RuntimeError("Tier‑1 CPU fallback stack underflow")
                stack.append(list(stack[-1]))
            elif op == 51:  # swap
                if len(stack) < 2:
                    raise RuntimeError("Tier‑1 CPU fallback stack underflow")
                stack[-1], stack[-2] = stack[-2], stack[-1]
            elif op == 52:  # drop
                if not stack:
                    raise RuntimeError("Tier‑1 CPU fallback stack underflow")
                stack.pop()
            else:
                raise RuntimeError(f"Unexpected opcode {op} in Tier‑1 CPU interpreter")

        if not stack:
            raise RuntimeError("Tier‑1 CPU fallback produced empty stack")
        return float(stack[-1][0])

    # ------------------------------------------------------------------ #
    # Encoding helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _encode_uint16(values: Iterable[int]) -> list[int]:
        out: list[int] = []
        for v in values:
            out.append(int(v) & 0xFFFF)
        return out

    @staticmethod
    def _encode_f32(values: Iterable[float]) -> bytes:
        return b"".join(struct.pack("<f", float(v)) for v in values)

    @staticmethod
    def _encode_f32_flat(vectors: Iterable[Iterable[float]]) -> bytes:
        flat: list[float] = []
        for vec in vectors:
            flat.extend(float(x) for x in vec)
        return b"".join(struct.pack("<f", v) for v in flat)

__all__ = ["LightweightRPNEngine"]
