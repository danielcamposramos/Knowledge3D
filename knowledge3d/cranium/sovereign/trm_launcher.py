"""TRM (Tiny Recursive Model) Launcher - Sovereign GPU Execution

Supports both the original PTX micro-kernels and the new Tier‑3 RPN backend
activated via `K3D_USE_RPN_TRM=1` (or the `use_rpn` constructor flag).
"""

from __future__ import annotations

import os
import ctypes
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

from .loader import (
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)


def _ptr_value(ptr) -> int:
    return int(ptr.value) if hasattr(ptr, "value") else int(ptr)


def _encode_pointer_literal(
    ptr,
    rows: int,
    cols: int,
) -> List[float]:
    raw = _ptr_value(ptr)
    lo = raw & 0xFFFFFFFF
    hi = (raw >> 32) & 0xFFFFFFFF
    lo_f = np.array([lo], dtype=np.uint32).view(np.float32)[0]
    hi_f = np.array([hi], dtype=np.uint32).view(np.float32)[0]
    return [float(rows), float(cols), float(lo_f), float(hi_f)]


class TRMLauncher:
    """Sovereign TRM launcher with optional Tier‑3 RPN backend."""

    def __init__(self, ptx_path: Optional[str] = None, use_rpn: Optional[bool] = None):
        if use_rpn is None:
            self.use_rpn = os.getenv("K3D_USE_RPN_TRM", "0").lower() in {"1", "true", "yes"}
        else:
            self.use_rpn = bool(use_rpn)

        self.ptx_path: Optional[str] = None
        self.kernels: dict[str, ctypes.c_void_p] = {}

        if not self.use_rpn:
            if ptx_path is None:
                ptx_path = str(Path(__file__).parent.parent / "ptx" / "trm_extensions.ptx")
            self.ptx_path = ptx_path
            print(f"🔥 TRM Launcher: Loading sovereign PTX kernels from {ptx_path}")
            self._load_kernels()

        # Shared workspace buffers
        self.d_temp = gpu_malloc(512 * 4)
        self.d_hidden = gpu_malloc(1024 * 4)
        self.d_temp2 = gpu_malloc(512 * 4)
        self.d_hidden2 = gpu_malloc(1024 * 4)

        # RPN backend wiring
        self._advanced_rpn = None
        self.d_zero_512 = None
        if self.use_rpn:
            from knowledge3d.cranium.bridges.advanced_rpn import AdvancedRPNEngine
            from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
                OP_POINTER_LITERAL,
                OP_TRM_MATVEC_1024x512,
                OP_TRM_MATVEC_512x1024,
                OP_TRM_SWIGLU_1024,
                OP_TRM_VEC_ADD3_512,
            )

            self._advanced_rpn = AdvancedRPNEngine()
            self._op_pointer_literal = OP_POINTER_LITERAL
            self._op_matvec_512x1024 = OP_TRM_MATVEC_512x1024
            self._op_matvec_1024x512 = OP_TRM_MATVEC_1024x512
            self._op_vec_add3_512 = OP_TRM_VEC_ADD3_512
            self._op_swiglu_1024 = OP_TRM_SWIGLU_1024

            zero_host = np.zeros(512, dtype=np.float32)
            self.d_zero_512 = gpu_malloc(zero_host.nbytes)
            memcpy_htod(self.d_zero_512, zero_host.ctypes.data_as(ctypes.c_void_p), zero_host.nbytes)
        else:
            self._advanced_rpn = None
            self._op_pointer_literal = None
            self._op_matvec_512x1024 = None
            self._op_matvec_1024x512 = None
            self._op_vec_add3_512 = None
            self._op_swiglu_1024 = None

        print("✅ TRM Launcher initialized (backend: {})".format("RPN" if self.use_rpn else "PTX"))

    # ------------------------------------------------------------------ #
    # PTX helpers
    # ------------------------------------------------------------------ #

    def _load_kernels(self) -> None:
        assert self.ptx_path is not None
        kernel_names = [
            "swiglu_vec_512",
            "swiglu_vec_1024",
            "vec_add_512",
            "vec_add3_512",
            "matvec_512x1024",
            "matvec_1024x512",
        ]
        for name in kernel_names:
            self.kernels[name] = load_ptx_file(self.ptx_path, name)
            print(f"   ✓ Loaded {name}")

    def refine_step(
        self,
        d_q: int,
        d_y: int,
        d_z: int,
        d_W1: int,
        d_W2: int,
        d_W3: int,
        d_W4: int,
        d_z_new: int,
        d_y_new: int,
    ) -> None:
        launch(
            self.kernels["vec_add3_512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_q),
                ctypes.c_uint64(d_y),
                ctypes.c_uint64(d_z),
                ctypes.c_uint64(self.d_temp.value),
            ],
        )

        launch(
            self.kernels["matvec_512x1024"],
            grid=(1, 1, 1),
            block=(1024, 1, 1),
            params=[
                ctypes.c_uint64(d_W1),
                ctypes.c_uint64(self.d_temp.value),
                ctypes.c_uint64(self.d_hidden.value),
            ],
        )

        launch(
            self.kernels["swiglu_vec_1024"],
            grid=(4, 1, 1),
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(self.d_hidden.value),
                ctypes.c_uint64(self.d_hidden.value),
            ],
        )

        launch(
            self.kernels["matvec_1024x512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_W2),
                ctypes.c_uint64(self.d_hidden.value),
                ctypes.c_uint64(d_z_new),
            ],
        )

        launch(
            self.kernels["vec_add_512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_y),
                ctypes.c_uint64(d_z_new),
                ctypes.c_uint64(self.d_temp2.value),
            ],
        )

        launch(
            self.kernels["matvec_512x1024"],
            grid=(1, 1, 1),
            block=(1024, 1, 1),
            params=[
                ctypes.c_uint64(d_W3),
                ctypes.c_uint64(self.d_temp2.value),
                ctypes.c_uint64(self.d_hidden2.value),
            ],
        )

        launch(
            self.kernels["swiglu_vec_1024"],
            grid=(4, 1, 1),
            block=(256, 1, 1),
            params=[
                ctypes.c_uint64(self.d_hidden2.value),
                ctypes.c_uint64(self.d_hidden2.value),
            ],
        )

        launch(
            self.kernels["matvec_1024x512"],
            grid=(1, 1, 1),
            block=(512, 1, 1),
            params=[
                ctypes.c_uint64(d_W4),
                ctypes.c_uint64(self.d_hidden2.value),
                ctypes.c_uint64(d_y_new),
            ],
        )

        synchronize()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def refine(
        self,
        q: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        W1: np.ndarray,
        W2: np.ndarray,
        W3: np.ndarray,
        W4: np.ndarray,
        n_steps: int = 6,
        eps: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert q.dtype == y.dtype == z.dtype == np.float32
        assert q.shape == y.shape == z.shape == (512,)
        assert W1.shape == (1024, 512) and W1.dtype == np.float32
        assert W2.shape == (512, 1024) and W2.dtype == np.float32
        assert W3.shape == (1024, 512) and W3.dtype == np.float32
        assert W4.shape == (512, 1024) and W4.dtype == np.float32

        d_q = gpu_malloc(q.nbytes)
        d_y = gpu_malloc(y.nbytes)
        d_z = gpu_malloc(z.nbytes)
        d_W1 = gpu_malloc(W1.nbytes)
        d_W2 = gpu_malloc(W2.nbytes)
        d_W3 = gpu_malloc(W3.nbytes)
        d_W4 = gpu_malloc(W4.nbytes)
        d_z_new = gpu_malloc(512 * 4)
        d_y_new = gpu_malloc(512 * 4)

        try:
            memcpy_htod(d_q, q.ctypes.data_as(ctypes.c_void_p), q.nbytes)
            memcpy_htod(d_y, y.ctypes.data_as(ctypes.c_void_p), y.nbytes)
            memcpy_htod(d_z, z.ctypes.data_as(ctypes.c_void_p), z.nbytes)
            memcpy_htod(d_W1, W1.ctypes.data_as(ctypes.c_void_p), W1.nbytes)
            memcpy_htod(d_W2, W2.ctypes.data_as(ctypes.c_void_p), W2.nbytes)
            memcpy_htod(d_W3, W3.ctypes.data_as(ctypes.c_void_p), W3.nbytes)
            memcpy_htod(d_W4, W4.ctypes.data_as(ctypes.c_void_p), W4.nbytes)

            if self.use_rpn:
                result = self._refine_rpn(
                    d_q,
                    d_y,
                    d_z,
                    d_W1,
                    d_W2,
                    d_W3,
                    d_W4,
                    d_z_new,
                    d_y_new,
                    n_steps,
                    eps,
                )
            else:
                result = self._refine_ptx(
                    d_q,
                    d_y,
                    d_z,
                    d_W1,
                    d_W2,
                    d_W3,
                    d_W4,
                    d_z_new,
                    d_y_new,
                    n_steps,
                    eps,
                )

            return result

        finally:
            gpu_free(d_q)
            gpu_free(d_y)
            gpu_free(d_z)
            gpu_free(d_W1)
            gpu_free(d_W2)
            gpu_free(d_W3)
            gpu_free(d_W4)
            gpu_free(d_z_new)
            gpu_free(d_y_new)

    # ------------------------------------------------------------------ #
    # Internal refinement paths
    # ------------------------------------------------------------------ #

    def _refine_ptx(
        self,
        d_q,
        d_y,
        d_z,
        d_W1,
        d_W2,
        d_W3,
        d_W4,
        d_z_new,
        d_y_new,
        n_steps: int,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        z_old = np.zeros(512, dtype=np.float32)

        for step in range(n_steps):
            memcpy_dtoh(z_old.ctypes.data_as(ctypes.c_void_p), d_z, z_old.nbytes)

            self.refine_step(
                d_q.value,
                d_y.value,
                d_z.value,
                d_W1.value,
                d_W2.value,
                d_W3.value,
                d_W4.value,
                d_z_new.value,
                d_y_new.value,
            )

            z_new = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(z_new.ctypes.data_as(ctypes.c_void_p), d_z_new, z_new.nbytes)

            drift = np.max(np.abs(z_new - z_old))
            if drift < eps:
                print(f"   🛑 TRM halted at step {step + 1}/{n_steps} (drift={drift:.6f} < {eps})")
                y_final = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
                return y_final, z_new

            memcpy_htod(d_z, z_new.ctypes.data_as(ctypes.c_void_p), z_new.nbytes)
            y_tmp = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(y_tmp.ctypes.data_as(ctypes.c_void_p), d_y_new, y_tmp.nbytes)
            memcpy_htod(d_y, y_tmp.ctypes.data_as(ctypes.c_void_p), y_tmp.nbytes)

        y_final = np.zeros(512, dtype=np.float32)
        z_final = np.zeros(512, dtype=np.float32)
        memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
        memcpy_dtoh(z_final.ctypes.data_as(ctypes.c_void_p), d_z_new, z_final.nbytes)
        return y_final, z_final

    def _refine_rpn(
        self,
        d_q,
        d_y,
        d_z,
        d_W1,
        d_W2,
        d_W3,
        d_W4,
        d_z_new,
        d_y_new,
        n_steps: int,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._advanced_rpn is None:
            raise RuntimeError("RPN backend not initialised")

        self._advanced_rpn.reset_instance(0)
        z_old = np.zeros(512, dtype=np.float32)

        for step in range(n_steps):
            memcpy_dtoh(z_old.ctypes.data_as(ctypes.c_void_p), d_z, z_old.nbytes)

            op_codes: List[int] = []
            scalars: List[float] = []

            scalars.extend(_encode_pointer_literal(d_q, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_y, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_z, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_temp, 512, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_vec_add3_512)

            scalars.extend(_encode_pointer_literal(self.d_temp, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_W1, 1024, 512))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_hidden, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_matvec_512x1024)

            scalars.extend(_encode_pointer_literal(self.d_hidden, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_hidden, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_swiglu_1024)

            scalars.extend(_encode_pointer_literal(self.d_hidden, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_W2, 512, 1024))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_z_new, 512, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_matvec_1024x512)

            scalars.extend(_encode_pointer_literal(d_y, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_z_new, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_zero_512, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_temp2, 512, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_vec_add3_512)

            scalars.extend(_encode_pointer_literal(self.d_temp2, 512, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_W3, 1024, 512))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_hidden2, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_matvec_512x1024)

            scalars.extend(_encode_pointer_literal(self.d_hidden2, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(self.d_hidden2, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_swiglu_1024)

            scalars.extend(_encode_pointer_literal(self.d_hidden2, 1024, 1))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_W4, 512, 1024))
            op_codes.append(self._op_pointer_literal)
            scalars.extend(_encode_pointer_literal(d_y_new, 512, 1))
            op_codes.append(self._op_pointer_literal)
            op_codes.append(self._op_matvec_1024x512)

            op_codes_np = np.asarray(op_codes, dtype=np.uint16)
            scalars_np = np.asarray(scalars, dtype=np.float32)
            self._advanced_rpn.execute_program(0, op_codes_np, scalars=scalars_np)

            z_new = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(z_new.ctypes.data_as(ctypes.c_void_p), d_z_new, z_new.nbytes)

            drift = np.max(np.abs(z_new - z_old))
            if drift < eps:
                print(f"   🛑 TRM (RPN) halted at step {step + 1}/{n_steps} (drift={drift:.6f} < {eps})")
                y_final = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
                return y_final, z_new

            memcpy_htod(d_z, z_new.ctypes.data_as(ctypes.c_void_p), z_new.nbytes)
            y_tmp = np.zeros(512, dtype=np.float32)
            memcpy_dtoh(y_tmp.ctypes.data_as(ctypes.c_void_p), d_y_new, y_tmp.nbytes)
            memcpy_htod(d_y, y_tmp.ctypes.data_as(ctypes.c_void_p), y_tmp.nbytes)

        y_final = np.zeros(512, dtype=np.float32)
        z_final = np.zeros(512, dtype=np.float32)
        memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
        memcpy_dtoh(z_final.ctypes.data_as(ctypes.c_void_p), d_z_new, z_final.nbytes)
        return y_final, z_final

    # ------------------------------------------------------------------ #
    # Resource management
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        gpu_free(self.d_temp)
        gpu_free(self.d_hidden)
        gpu_free(self.d_temp2)
        gpu_free(self.d_hidden2)
        if self.d_zero_512 is not None:
            gpu_free(self.d_zero_512)
            self.d_zero_512 = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass


__all__ = ["TRMLauncher"]
