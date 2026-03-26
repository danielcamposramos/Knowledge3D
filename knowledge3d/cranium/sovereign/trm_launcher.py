"""TRM (Tiny Recursive Model) Launcher - Sovereign GPU Execution

Supports three execution paths:
1. Legacy PTX kernels (default, fastest today)
2. Tier-3 RPN backend (`K3D_USE_RPN_TRM=1` or `use_rpn=True`)
3. Fused PTX kernel (`K3D_USE_FUSED_TRM=1` or `use_fused=True`)
"""

from __future__ import annotations

import ctypes
import os
import struct
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .loader import (
    load_ptx_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
)
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32


def _ptr_value(ptr) -> int:
    return int(ptr.value) if hasattr(ptr, "value") else int(ptr)


def _encode_pointer_literal(ptr, rows: int, cols: int) -> List[float]:
    raw = _ptr_value(ptr)
    lo_f = struct.unpack("<f", struct.pack("<I", raw & 0xFFFFFFFF))[0]
    hi_f = struct.unpack("<f", struct.pack("<I", (raw >> 32) & 0xFFFFFFFF))[0]
    return [float(rows), float(cols), float(lo_f), float(hi_f)]


def _void_p(address: int) -> ctypes.c_void_p:
    return ctypes.c_void_p(int(address))


def _copy_host_to_device(dst_device, tensor: HostTensorF32) -> None:
    memcpy_htod(dst_device, _void_p(tensor.data_ptr), tensor.nbytes)


def _copy_device_to_host(tensor: HostTensorF32, src_device) -> None:
    memcpy_dtoh(_void_p(tensor.data_ptr), src_device, tensor.nbytes)


def _coerce_vector_tensor(name: str, value: object, size: int = 512) -> HostTensorF32:
    tensor = HostTensorF32.from_array_like(value)
    if tensor.shape != (int(size), 1):
        raise ValueError(f"{name} must have shape ({int(size)},), got {tensor.shape}")
    return tensor


def _coerce_matrix_tensor(name: str, value: object, rows: int, cols: int) -> HostTensorF32:
    tensor = HostTensorF32.from_array_like(value, rows=rows, cols=cols)
    if tensor.shape != (int(rows), int(cols)):
        raise ValueError(f"{name} must have shape ({int(rows)}, {int(cols)}), got {tensor.shape}")
    return tensor


def _max_abs_diff(lhs: HostTensorF32, rhs: HostTensorF32) -> float:
    if lhs.shape != rhs.shape:
        raise ValueError(f"Shape mismatch: {lhs.shape} != {rhs.shape}")
    max_value = 0.0
    for idx in range(lhs.size):
        diff = abs(float(lhs._buffer[idx]) - float(rhs._buffer[idx]))
        if diff > max_value:
            max_value = diff
    return float(max_value)


class TRMVector(list):
    """Sequence-compatible float vector without a NumPy dependency."""

    def __init__(self, values: Iterable[float] = ()) -> None:
        super().__init__(float(value) for value in values)

    @classmethod
    def from_tensor(cls, tensor: HostTensorF32) -> "TRMVector":
        return cls(tensor.to_flat_list())

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def dtype(self) -> str:
        return "float32"

    def tolist(self) -> list[float]:
        return list(self)

    def copy(self) -> "TRMVector":
        return TRMVector(self)

    def _binary(self, other: object, op) -> "TRMVector":
        if isinstance(other, (int, float)):
            scalar = float(other)
            return TRMVector(op(float(value), scalar) for value in self)
        if not isinstance(other, Sequence) and not hasattr(other, "__iter__"):
            raise TypeError(f"Unsupported operand type: {type(other)!r}")
        other_values = list(other)
        if len(other_values) != len(self):
            raise ValueError(f"Length mismatch: {len(self)} != {len(other_values)}")
        return TRMVector(op(float(lhs), float(rhs)) for lhs, rhs in zip(self, other_values))

    def __mul__(self, other: object) -> "TRMVector":
        return self._binary(other, lambda lhs, rhs: lhs * rhs)

    def __rmul__(self, other: object) -> "TRMVector":
        return self.__mul__(other)

    def __add__(self, other: object) -> "TRMVector":
        return self._binary(other, lambda lhs, rhs: lhs + rhs)

    def __sub__(self, other: object) -> "TRMVector":
        return self._binary(other, lambda lhs, rhs: lhs - rhs)

    def __rsub__(self, other: object) -> "TRMVector":
        if isinstance(other, (int, float)):
            scalar = float(other)
            return TRMVector(scalar - float(value) for value in self)
        if not isinstance(other, Sequence) and not hasattr(other, "__iter__"):
            raise TypeError(f"Unsupported operand type: {type(other)!r}")
        other_values = list(other)
        if len(other_values) != len(self):
            raise ValueError(f"Length mismatch: {len(other_values)} != {len(self)}")
        return TRMVector(float(lhs) - float(rhs) for lhs, rhs in zip(other_values, self))


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
            self.use_fused = os.getenv("K3D_USE_FUSED_TRM", "1").lower() in {"1", "true", "yes"}
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
        self._rpn_opcodes = None
        self._rpn_scalars_host: Optional[HostTensorF32] = None
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

            zero_host = HostTensorF32.zeros(512, 1)
            self.d_zero_512 = gpu_malloc(zero_host.nbytes)
            _copy_host_to_device(self.d_zero_512, zero_host)
            self._init_rpn_buffers()

        # Fused kernel
        self.kernel_fused = None
        self.kernel_recursive_fused = None
        self.d_workspace = None
        if self.use_fused:
            fused_ptx = Path(__file__).parent.parent / "ptx" / "trm_recursive_fused.ptx"
            if not fused_ptx.exists():
                raise FileNotFoundError(f"Recursive fused TRM PTX not found: {fused_ptx}")
            self.kernel_recursive_fused = load_ptx_file(str(fused_ptx), "trm_recursive_fused")
            # Workspace = temp(512) + hidden(1024) + temp2(512) + hidden2(1024)
            #           + z_new(512) + y_new(512) = 4096 floats
            self.d_workspace = gpu_malloc(4096 * 4)

        backend = "FUSED" if self.use_fused else ("RPN" if self.use_rpn else "PTX")
        print(f"✅ TRM Launcher initialized (backend: {backend})")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def refine(
        self,
        q: object,
        y: object,
        z: object,
        W1: object,
        W2: object,
        W3: object,
        W4: object,
        n_steps: int = 6,
        eps: float = 1e-4,
    ) -> Tuple[TRMVector, TRMVector]:
        if n_steps != 6:
            warnings.warn(
                f"⚠️  Using n_steps={n_steps} breaks Tesla 3/6/9 resonance. Recommended: n_steps=6",
                RuntimeWarning,
            )
        q_tensor = _coerce_vector_tensor("q", q)
        y_tensor = _coerce_vector_tensor("y", y)
        z_tensor = _coerce_vector_tensor("z", z)
        W1_tensor = _coerce_matrix_tensor("W1", W1, 1024, 512)
        W2_tensor = _coerce_matrix_tensor("W2", W2, 512, 1024)
        W3_tensor = _coerce_matrix_tensor("W3", W3, 1024, 512)
        W4_tensor = _coerce_matrix_tensor("W4", W4, 512, 1024)

        d_q = gpu_malloc(q_tensor.nbytes)
        d_y = gpu_malloc(y_tensor.nbytes)
        d_z = gpu_malloc(z_tensor.nbytes)
        d_W1 = gpu_malloc(W1_tensor.nbytes)
        d_W2 = gpu_malloc(W2_tensor.nbytes)
        d_W3 = gpu_malloc(W3_tensor.nbytes)
        d_W4 = gpu_malloc(W4_tensor.nbytes)
        d_z_new = gpu_malloc(512 * 4)
        d_y_new = gpu_malloc(512 * 4)

        try:
            _copy_host_to_device(d_q, q_tensor)
            _copy_host_to_device(d_y, y_tensor)
            _copy_host_to_device(d_z, z_tensor)
            _copy_host_to_device(d_W1, W1_tensor)
            _copy_host_to_device(d_W2, W2_tensor)
            _copy_host_to_device(d_W3, W3_tensor)
            _copy_host_to_device(d_W4, W4_tensor)

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
    ) -> Tuple[TRMVector, TRMVector]:
        z_old = HostTensorF32.zeros(512, 1)
        z_new_host = HostTensorF32.zeros(512, 1)
        y_host = HostTensorF32.zeros(512, 1)

        for step in range(n_steps):
            _copy_device_to_host(z_old, d_z)

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

            _copy_device_to_host(z_new_host, d_z_new)
            drift = _max_abs_diff(z_new_host, z_old)
            if drift < eps:
                y_final = HostTensorF32.zeros(512, 1)
                _copy_device_to_host(y_final, d_y_new)
                return TRMVector.from_tensor(y_final), TRMVector.from_tensor(z_new_host)

            _copy_host_to_device(d_z, z_new_host)

            _copy_device_to_host(y_host, d_y_new)
            _copy_host_to_device(d_y, y_host)

        y_final = HostTensorF32.zeros(512, 1)
        z_final = HostTensorF32.zeros(512, 1)
        _copy_device_to_host(y_final, d_y_new)
        _copy_device_to_host(z_final, d_z_new)
        return TRMVector.from_tensor(y_final), TRMVector.from_tensor(z_final)

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
    ) -> Tuple[TRMVector, TRMVector]:
        if self._advanced_rpn is None or self._rpn_opcodes is None:
            raise RuntimeError("RPN backend not initialised")

        self._advanced_rpn.reset_instance(0)

        import time  # Local import to avoid global dependency when unused

        timing = {"build": 0.0, "execute": 0.0, "memcpy": 0.0}
        z_old = HostTensorF32.zeros(512, 1)
        z_new_host = HostTensorF32.zeros(512, 1)
        y_host = HostTensorF32.zeros(512, 1)

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
            _copy_device_to_host(z_old, d_z)

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
            _copy_device_to_host(z_new_host, d_z_new)
            timing["memcpy"] += time.perf_counter() - t0

            drift = _max_abs_diff(z_new_host, z_old)
            if drift < eps:
                y_final = HostTensorF32.zeros(512, 1)
                _copy_device_to_host(y_final, d_y_new)
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
                return TRMVector.from_tensor(y_final), TRMVector.from_tensor(z_new_host)

            _copy_host_to_device(d_z, z_new_host)

            _copy_device_to_host(y_host, d_y_new)
            _copy_host_to_device(d_y, y_host)

        y_final = HostTensorF32.zeros(512, 1)
        z_final = HostTensorF32.zeros(512, 1)
        _copy_device_to_host(y_final, d_y_new)
        _copy_device_to_host(z_final, d_z_new)

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
        return TRMVector.from_tensor(y_final), TRMVector.from_tensor(z_final)

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
    ) -> Tuple[TRMVector, TRMVector]:
        if self.kernel_recursive_fused is None or self.d_workspace is None:
            raise RuntimeError("Fused backend not initialized")

        d_steps = gpu_malloc(ctypes.sizeof(ctypes.c_int32))
        d_drift = gpu_malloc(ctypes.sizeof(ctypes.c_float))

        try:
            launch(
                self.kernel_recursive_fused,
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
                    ctypes.c_uint64(self.d_workspace.value),
                    ctypes.c_uint64(d_steps.value),
                    ctypes.c_uint64(d_drift.value),
                    ctypes.c_int32(int(n_steps)),
                    ctypes.c_float(float(eps)),
                ],
            )
            synchronize()

            y_final = HostTensorF32.zeros(512, 1)
            z_final = HostTensorF32.zeros(512, 1)
            _copy_device_to_host(y_final, d_y)
            _copy_device_to_host(z_final, d_z)
            return TRMVector.from_tensor(y_final), TRMVector.from_tensor(z_final)
        finally:
            gpu_free(d_steps)
            gpu_free(d_drift)

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
        self._rpn_opcodes = (ctypes.c_uint16 * len(op_codes))(*op_codes)
        opcodes_nbytes = ctypes.sizeof(self._rpn_opcodes)
        self._d_rpn_opcodes = gpu_malloc(opcodes_nbytes)
        memcpy_htod(
            self._d_rpn_opcodes,
            ctypes.cast(self._rpn_opcodes, ctypes.c_void_p),
            opcodes_nbytes,
        )
        self._rpn_scalars_host = HostTensorF32.zeros(len(pointer_layout) * 4, 1)
        self._d_rpn_scalars = gpu_malloc(self._rpn_scalars_host.nbytes)

    def _update_rpn_scalars(self, pointer_map: dict[str, ctypes.c_void_p]) -> None:
        host = self._rpn_scalars_host
        assert host is not None
        flat_values: List[float] = []
        for name, rows, cols in self._rpn_pointer_layout:
            ptr = pointer_map[name]
            assert ptr is not None, f"Pointer '{name}' not initialised"
            flat_values.extend(_encode_pointer_literal(ptr, rows, cols))
        host.set_flat(flat_values)
        memcpy_htod(
            self._d_rpn_scalars,
            _void_p(host.data_ptr),
            host.nbytes,
        )

__all__ = ["TRMLauncher", "TRMVector"]
