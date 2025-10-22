"""TRM (Tiny Recursive Model) Launcher - Sovereign GPU Execution

Supports three execution paths:
1. Legacy PTX kernels (default, fastest today)
2. Tier-3 RPN backend (`K3D_USE_RPN_TRM=1` or `use_rpn=True`)
3. Fused PTX kernel (`K3D_USE_FUSED_TRM=1` or `use_fused=True`)
"""

from __future__ import annotations

import os
import ctypes
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

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


def _encode_pointer_literal(ptr, rows: int, cols: int) -> List[float]:
    raw = _ptr_value(ptr)
    lo = raw & 0xFFFFFFFF
    hi = (raw >> 32) & 0xFFFFFFFF
    lo_f = np.array([lo], dtype=np.uint32).view(np.float32)[0]
    hi_f = np.array([hi], dtype=np.uint32).view(np.float32)[0]
    return [float(rows), float(cols), float(lo_f), float(hi_f)]


class TRMLauncher:
    """Sovereign TRM launcher with optional RPN or fused execution."""

    def __init__(
        self,
        ptx_path: Optional[str] = None,
        use_rpn: Optional[bool] = None,
        use_fused: Optional[bool] = None,
    ):
        if use_rpn is None:
            self.use_rpn = os.getenv("K3D_USE_RPN_TRM", "0").lower() in {"1", "true", "yes"}
        else:
            self.use_rpn = bool(use_rpn)

        if use_fused is None:
            self.use_fused = os.getenv("K3D_USE_FUSED_TRM", "0").lower() in {"1", "true", "yes"}
        else:
            self.use_fused = bool(use_fused)

        if self.use_fused:
            # Fused path owns the math, no need to run the slower Tier-3 interpreter.
            self.use_rpn = False

        self.ptx_path: Optional[str] = None
        self.kernels: dict[str, ctypes.c_void_p] = {}

        if not self.use_rpn and not self.use_fused:
            if ptx_path is None:
                ptx_path = str(Path(__file__).parent.parent / "ptx" / "trm_extensions.ptx")
            self.ptx_path = ptx_path
            print(f"🔥 TRM Launcher: Loading sovereign PTX kernels from {ptx_path}")
            self._load_ptx_kernels()

        # Shared buffers for PTX and RPN paths
        self.d_temp = gpu_malloc(512 * 4)
        self.d_hidden = gpu_malloc(1024 * 4)
        self.d_temp2 = gpu_malloc(512 * 4)
        self.d_hidden2 = gpu_malloc(1024 * 4)

        # RPN backend wiring (Tier-3)
        self._advanced_rpn = None
        self._op_pointer_literal = None
        self._op_matvec_512x1024 = None
        self._op_matvec_1024x512 = None
        self._op_vec_add3_512 = None
        self._op_swiglu_1024 = None
        self.d_zero_512 = None
        self._rpn_pointer_layout: List[tuple[str, int, int]] = []
        self._rpn_opcodes: Optional[np.ndarray] = None
        self._rpn_scalars_host: Optional[np.ndarray] = None
        self._d_rpn_opcodes = None
        self._d_rpn_scalars = None

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
            self._init_rpn_buffers()

        # Fused kernel
        self.kernel_fused = None
        self.d_workspace = None
        if self.use_fused:
            fused_ptx = Path(__file__).parent.parent / "ptx" / "trm_step_fused.ptx"
            if not fused_ptx.exists():
                raise FileNotFoundError(f"Fused TRM PTX not found: {fused_ptx}")
            self.kernel_fused = load_ptx_file(str(fused_ptx), "trm_step_fused")
            # Workspace = 512 + 1024 + 512 + 1024 floats
            self.d_workspace = gpu_malloc((512 + 1024 + 512 + 1024) * 4)

        backend = "FUSED" if self.use_fused else ("RPN" if self.use_rpn else "PTX")
        print(f"✅ TRM Launcher initialized (backend: {backend})")

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
        if n_steps != 6:
            warnings.warn(
                f"⚠️  Using n_steps={n_steps} breaks Tesla 3/6/9 resonance. Recommended: n_steps=6",
                RuntimeWarning,
            )
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

            if self.use_fused:
                result = self._refine_fused(
                    d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, d_z_new, d_y_new, n_steps, eps
                )
            elif self.use_rpn:
                result = self._refine_rpn(
                    d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, d_z_new, d_y_new, n_steps, eps
                )
            else:
                result = self._refine_ptx(
                    d_q, d_y, d_z, d_W1, d_W2, d_W3, d_W4, d_z_new, d_y_new, n_steps, eps
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

    def cleanup(self) -> None:
        gpu_free(self.d_temp)
        gpu_free(self.d_hidden)
        gpu_free(self.d_temp2)
        gpu_free(self.d_hidden2)
        if self.d_zero_512 is not None:
            gpu_free(self.d_zero_512)
            self.d_zero_512 = None
        if self.d_workspace is not None:
            gpu_free(self.d_workspace)
            self.d_workspace = None
        if self._d_rpn_opcodes is not None:
            gpu_free(self._d_rpn_opcodes)
            self._d_rpn_opcodes = None
        if self._d_rpn_scalars is not None:
            gpu_free(self._d_rpn_scalars)
            self._d_rpn_scalars = None
        self._rpn_scalars_host = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Backend implementations
    # ------------------------------------------------------------------ #

    def _load_ptx_kernels(self) -> None:
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
        z_new_host = np.zeros(512, dtype=np.float32)
        y_host = np.zeros(512, dtype=np.float32)

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

            memcpy_dtoh(z_new_host.ctypes.data_as(ctypes.c_void_p), d_z_new, z_new_host.nbytes)
            drift = np.max(np.abs(z_new_host - z_old))
            if drift < eps:
                y_final = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
                return y_final, z_new_host.copy()

            memcpy_htod(d_z, z_new_host.ctypes.data_as(ctypes.c_void_p), z_new_host.nbytes)

            memcpy_dtoh(y_host.ctypes.data_as(ctypes.c_void_p), d_y_new, y_host.nbytes)
            memcpy_htod(d_y, y_host.ctypes.data_as(ctypes.c_void_p), y_host.nbytes)

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
        if self._advanced_rpn is None or self._rpn_opcodes is None:
            raise RuntimeError("RPN backend not initialised")

        self._advanced_rpn.reset_instance(0)

        import time  # Local import to avoid global dependency when unused

        timing = {"build": 0.0, "execute": 0.0, "memcpy": 0.0}
        z_old = np.zeros(512, dtype=np.float32)
        z_new_host = np.zeros(512, dtype=np.float32)
        y_host = np.zeros(512, dtype=np.float32)

        pointer_map = {
            "q": d_q,
            "y": d_y,
            "z": d_z,
            "temp": self.d_temp,
            "W1": d_W1,
            "hidden": self.d_hidden,
            "W2": d_W2,
            "z_new": d_z_new,
            "zero": self.d_zero_512,
            "temp2": self.d_temp2,
            "W3": d_W3,
            "hidden2": self.d_hidden2,
            "W4": d_W4,
            "y_new": d_y_new,
        }

        for step in range(n_steps):
            memcpy_dtoh(z_old.ctypes.data_as(ctypes.c_void_p), d_z, z_old.nbytes)

            t0 = time.perf_counter()
            self._update_rpn_scalars(pointer_map)
            timing["build"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            self._advanced_rpn.execute_prebuilt(
                0,
                self._d_rpn_opcodes,
                self._d_rpn_scalars,
                len(self._rpn_opcodes),
            )
            timing["execute"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            memcpy_dtoh(z_new_host.ctypes.data_as(ctypes.c_void_p), d_z_new, z_new_host.nbytes)
            timing["memcpy"] += time.perf_counter() - t0

            drift = np.max(np.abs(z_new_host - z_old))
            if drift < eps:
                y_final = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
                total = sum(timing.values()) or 1e-12
                print("\nRPN Timing Breakdown (steps={}):".format(n_steps))
                print(
                    f"  Build:   {timing['build'] * 1e3:6.2f} ms "
                    f"({timing['build']/total*100:.1f}%)"
                )
                print(
                    f"  Execute: {timing['execute'] * 1e3:6.2f} ms "
                    f"({timing['execute']/total*100:.1f}%)"
                )
                print(
                    f"  Memcpy:  {timing['memcpy'] * 1e3:6.2f} ms "
                    f"({timing['memcpy']/total*100:.1f}%)"
                )
                return y_final, z_new_host.copy()

            memcpy_htod(d_z, z_new_host.ctypes.data_as(ctypes.c_void_p), z_new_host.nbytes)

            memcpy_dtoh(y_host.ctypes.data_as(ctypes.c_void_p), d_y_new, y_host.nbytes)
            memcpy_htod(d_y, y_host.ctypes.data_as(ctypes.c_void_p), y_host.nbytes)

        y_final = np.zeros(512, dtype=np.float32)
        z_final = np.zeros(512, dtype=np.float32)
        memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
        memcpy_dtoh(z_final.ctypes.data_as(ctypes.c_void_p), d_z_new, z_final.nbytes)

        total = sum(timing.values()) or 1e-12
        print("\nRPN Timing Breakdown (steps={}):".format(n_steps))
        print(
            f"  Build:   {timing['build'] * 1e3:6.2f} ms "
            f"({timing['build']/total*100:.1f}%)"
        )
        print(
            f"  Execute: {timing['execute'] * 1e3:6.2f} ms "
            f"({timing['execute']/total*100:.1f}%)"
        )
        print(
            f"  Memcpy:  {timing['memcpy'] * 1e3:6.2f} ms "
            f"({timing['memcpy']/total*100:.1f}%)"
        )
        return y_final, z_final

    def _refine_fused(
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
        if self.kernel_fused is None or self.d_workspace is None:
            raise RuntimeError("Fused backend not initialized")

        z_old = np.zeros(512, dtype=np.float32)
        z_new_host = np.zeros(512, dtype=np.float32)
        y_host = np.zeros(512, dtype=np.float32)

        for step in range(n_steps):
            memcpy_dtoh(z_old.ctypes.data_as(ctypes.c_void_p), d_z, z_old.nbytes)

            launch(
                self.kernel_fused,
                grid=(1, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_y.value),
                    ctypes.c_uint64(d_z.value),
                    ctypes.c_uint64(d_W1.value),
                    ctypes.c_uint64(d_W2.value),
                    ctypes.c_uint64(d_W3.value),
                    ctypes.c_uint64(d_W4.value),
                    ctypes.c_uint64(d_z_new.value),
                    ctypes.c_uint64(d_y_new.value),
                    ctypes.c_uint64(self.d_workspace.value),
                ],
            )
            synchronize()

            memcpy_dtoh(z_new_host.ctypes.data_as(ctypes.c_void_p), d_z_new, z_new_host.nbytes)
            drift = np.max(np.abs(z_new_host - z_old))
            if drift < eps:
                y_final = np.zeros(512, dtype=np.float32)
                memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
                return y_final, z_new_host.copy()

            memcpy_htod(d_z, z_new_host.ctypes.data_as(ctypes.c_void_p), z_new_host.nbytes)

            memcpy_dtoh(y_host.ctypes.data_as(ctypes.c_void_p), d_y_new, y_host.nbytes)
            memcpy_htod(d_y, y_host.ctypes.data_as(ctypes.c_void_p), y_host.nbytes)

        y_final = np.zeros(512, dtype=np.float32)
        z_final = np.zeros(512, dtype=np.float32)
        memcpy_dtoh(y_final.ctypes.data_as(ctypes.c_void_p), d_y_new, y_final.nbytes)
        memcpy_dtoh(z_final.ctypes.data_as(ctypes.c_void_p), d_z_new, z_final.nbytes)
        return y_final, z_final

    def _init_rpn_buffers(self) -> None:
        pointer_layout: List[tuple[str, int, int]] = []
        op_codes: List[int] = []

        def push(name: str, rows: int, cols: int) -> None:
            pointer_layout.append((name, rows, cols))
            op_codes.append(self._op_pointer_literal)

        def emit(opcode: int) -> None:
            op_codes.append(opcode)

        push("q", 512, 1)
        push("y", 512, 1)
        push("z", 512, 1)
        push("temp", 512, 1)
        emit(self._op_vec_add3_512)

        push("temp", 512, 1)
        push("W1", 1024, 512)
        push("hidden", 1024, 1)
        emit(self._op_matvec_512x1024)

        push("hidden", 1024, 1)
        push("hidden", 1024, 1)
        emit(self._op_swiglu_1024)

        push("hidden", 1024, 1)
        push("W2", 512, 1024)
        push("z_new", 512, 1)
        emit(self._op_matvec_1024x512)

        push("y", 512, 1)
        push("z_new", 512, 1)
        push("zero", 512, 1)
        push("temp2", 512, 1)
        emit(self._op_vec_add3_512)

        push("temp2", 512, 1)
        push("W3", 1024, 512)
        push("hidden2", 1024, 1)
        emit(self._op_matvec_512x1024)

        push("hidden2", 1024, 1)
        push("hidden2", 1024, 1)
        emit(self._op_swiglu_1024)

        push("hidden2", 1024, 1)
        push("W4", 512, 1024)
        push("y_new", 512, 1)
        emit(self._op_matvec_1024x512)
        self._rpn_pointer_layout = pointer_layout
        self._rpn_opcodes = np.asarray(op_codes, dtype=np.uint16)
        self._d_rpn_opcodes = gpu_malloc(self._rpn_opcodes.nbytes)
        memcpy_htod(
            self._d_rpn_opcodes,
            self._rpn_opcodes.ctypes.data_as(ctypes.c_void_p),
            self._rpn_opcodes.nbytes,
        )
        self._rpn_scalars_host = np.zeros(len(pointer_layout) * 4, dtype=np.float32)
        self._d_rpn_scalars = gpu_malloc(self._rpn_scalars_host.nbytes)

    def _update_rpn_scalars(self, pointer_map: dict[str, ctypes.c_void_p]) -> None:
        host = self._rpn_scalars_host
        assert host is not None
        for idx, (name, rows, cols) in enumerate(self._rpn_pointer_layout):
            ptr = pointer_map[name]
            assert ptr is not None, f"Pointer '{name}' not initialised"
            host[idx * 4 : idx * 4 + 4] = _encode_pointer_literal(ptr, rows, cols)
        memcpy_htod(
            self._d_rpn_scalars,
            host.ctypes.data_as(ctypes.c_void_p),
            host.nbytes,
        )

__all__ = ["TRMLauncher"]
