"""
Sovereign Bridges - Pure ctypes + libcuda.so bridges for all Step8 kernels

This module provides Python bridges for all 15 Step8 kernels using the sovereign
loader (pure ctypes + CUDA Driver API). For the RPN bridge used in the hot
path we avoid NumPy entirely; other bridges may lazily import NumPy inside
helper functions when used outside the sovereignty‑critical loop.

Usage:
    from knowledge3d.cranium.bridges.sovereign_bridges import LatencyGuard, ARCReasoner, ...

    guard = LatencyGuard(threshold_us=95.0)
    guard.start()
    # ... GPU work ...
    elapsed_ns, breached = guard.stop()

Architecture:
    - All bridges use knowledge3d.cranium.sovereign.loader
    - All memory management via gpu_malloc/gpu_free
    - All kernel launches via sovereign launch()
    - No CuPy, no cuda-python in hot path
"""

import ctypes
import math
from pathlib import Path
import random
from typing import Tuple, Optional, Iterable, Sequence, List

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.sovereign.loader import (
    get_function,
    get_global,
    load_ptx_file,
    load_module_from_file,
    gpu_malloc,
    gpu_free,
    memcpy_htod,
    memcpy_dtoh,
    launch,
    synchronize,
    CUdeviceptr,
)
from knowledge3d.cranium.bridges.rpn_config import RPN_GRID_DIM, TIER2_BLOCK_DIM


class _HostVector:
    """Small ctypes-backed vector view for scalar staging/readback."""

    CTYPE = ctypes.c_float
    PYTHON_TYPE = float

    def __init__(self, values: Iterable[float] = ()):
        self._values = [self.PYTHON_TYPE(value) for value in values]
        self._shape = (len(self._values),)
        self._sync_buffer()

    @classmethod
    def zeros(cls, size: int):
        return cls(cls.PYTHON_TYPE(0) for _ in range(int(size)))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._shape)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self._values)

    @property
    def nbytes(self) -> int:
        return ctypes.sizeof(self._buffer)

    @property
    def data_ptr(self) -> int:
        return ctypes.addressof(self._buffer)

    def tolist(self) -> list:
        if len(self._shape) == 2:
            rows, cols = self._shape
            return [
                [self._values[row * cols + col] for col in range(cols)]
                for row in range(rows)
            ]
        return list(self._values)

    def copy(self):
        clone = self.__class__(self._values)
        clone._shape = tuple(self._shape)
        return clone

    def fill(self, value) -> None:
        scalar = self.PYTHON_TYPE(value)
        for idx in range(len(self._values)):
            self._values[idx] = scalar
        self._sync_buffer()

    def reshape(self, rows: int, cols: int | None = None):
        if cols is None:
            cols = max(1, len(self._values) // max(int(rows), 1))
        rows_i = int(rows)
        cols_i = int(cols)
        if rows_i * cols_i != len(self._values):
            raise ValueError(f"Cannot reshape {len(self._values)} values into ({rows_i}, {cols_i})")
        clone = self.copy()
        clone._shape = (rows_i, cols_i)
        return clone

    def mean(self) -> float:
        return float(sum(float(value) for value in self._values) / max(len(self._values), 1))

    def min(self):
        return self.PYTHON_TYPE(min(self._values))

    def max(self):
        return self.PYTHON_TYPE(max(self._values))

    def sum(self):
        return self.PYTHON_TYPE(sum(self._values))

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            if len(self._shape) != 2:
                raise TypeError("Tuple indexing requires a 2D sovereign vector view")
            row, col = index
            rows, cols = self._shape
            row_i = int(row)
            col_i = int(col)
            if row_i < 0 or row_i >= rows or col_i < 0 or col_i >= cols:
                raise IndexError(index)
            return self._values[row_i * cols + col_i]
        return self._values[index]

    def __setitem__(self, index, value) -> None:
        self._values[index] = self.PYTHON_TYPE(value)
        self._sync_buffer()

    def _sync_buffer(self) -> None:
        self._buffer = (self.CTYPE * len(self._values))(*self._values)

    def _load_from_device(self, ptr, nbytes: int | None = None) -> None:
        size_bytes = self.nbytes if nbytes is None else int(nbytes)
        memcpy_dtoh(ctypes.c_void_p(self.data_ptr), ptr, size_bytes)
        self._values = [self.PYTHON_TYPE(self._buffer[idx]) for idx in range(len(self._values))]


class Float32Vector(_HostVector):
    CTYPE = ctypes.c_float
    PYTHON_TYPE = float


class Int32Vector(_HostVector):
    CTYPE = ctypes.c_int32
    PYTHON_TYPE = int


class UInt32Vector(_HostVector):
    CTYPE = ctypes.c_uint32
    PYTHON_TYPE = int


class UInt64Vector(_HostVector):
    CTYPE = ctypes.c_uint64
    PYTHON_TYPE = int


class UInt8Vector(_HostVector):
    CTYPE = ctypes.c_uint8
    PYTHON_TYPE = int


class Int8Vector(_HostVector):
    CTYPE = ctypes.c_int8
    PYTHON_TYPE = int


class TritInspectionBatch:
    """Structured ternary inspection result without NumPy dependency."""

    def __init__(
        self,
        counts: Sequence[int],
        sums: Sequence[int],
        means: Sequence[float],
        variances: Sequence[float],
        bottlenecks: Sequence[int],
    ) -> None:
        self._fields = {
            "count": Int32Vector(counts),
            "sum": Int32Vector(sums),
            "mean": Float32Vector(means),
            "var": Float32Vector(variances),
            "bottlenecks": Int32Vector(bottlenecks),
        }

    @property
    def shape(self) -> tuple[int]:
        return self._fields["count"].shape

    def __len__(self) -> int:
        return len(self._fields["count"])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._fields[key]
        return {
            field: values[key]
            for field, values in self._fields.items()
        }


class _TritInspectorRecord(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int32),
        ("sum", ctypes.c_int32),
        ("mean", ctypes.c_float),
        ("var", ctypes.c_float),
        ("bottlenecks", ctypes.c_int32),
    ]


def _shape_of(values: object) -> tuple[int, ...]:
    shape = getattr(values, "shape", None)
    if shape is not None:
        return tuple(int(dim) for dim in shape)
    if isinstance(values, HostTensorF32):
        return values.shape
    if isinstance(values, (Float32Vector, Int32Vector, UInt32Vector, UInt64Vector, UInt8Vector, Int8Vector)):
        return values.shape
    if isinstance(values, (list, tuple)):
        if values and isinstance(values[0], (list, tuple)):
            return (len(values), len(values[0]))
        return (len(values),)
    raise TypeError(f"Unsupported shape source: {type(values).__name__}")


def _float_list(values: object) -> list[float]:
    return HostTensorF32.from_array_like(values).to_flat_list()


def _f32_vector(values: object) -> Float32Vector:
    return Float32Vector(_float_list(values))


def _f32_matrix(values: object, rows: int | None = None, cols: int | None = None) -> HostTensorF32:
    if rows is None or cols is None:
        shape = _shape_of(values)
        if len(shape) == 1:
            rows = shape[0]
            cols = 1
        elif len(shape) == 2:
            rows, cols = shape
        else:
            raise ValueError(f"Unsupported rank for float32 matrix coercion: {shape}")
    return HostTensorF32.from_array_like(values, rows=int(rows), cols=int(cols))


def _i32_vector(values: object) -> Int32Vector:
    if isinstance(values, Int32Vector):
        return values.copy()
    if isinstance(values, (list, tuple)):
        return Int32Vector(int(value) for value in values)
    shape = getattr(values, "shape", None)
    flat = getattr(values, "flat", None)
    if flat is not None and shape is not None:
        return Int32Vector(int(value) for value in flat)
    return Int32Vector(int(value) for value in values)


def _u32_vector(values: object) -> UInt32Vector:
    if isinstance(values, UInt32Vector):
        return values.copy()
    if isinstance(values, (list, tuple)):
        return UInt32Vector(int(value) for value in values)
    shape = getattr(values, "shape", None)
    flat = getattr(values, "flat", None)
    if flat is not None and shape is not None:
        return UInt32Vector(int(value) for value in flat)
    return UInt32Vector(int(value) for value in values)


def _u64_vector(values: object) -> UInt64Vector:
    if isinstance(values, UInt64Vector):
        return values.copy()
    return UInt64Vector(int(value) for value in values)


def _copy_htod(device_ptr, host) -> None:
    if isinstance(host, HostTensorF32):
        memcpy_htod(device_ptr, ctypes.c_void_p(host.data_ptr), host.nbytes)
        return
    if isinstance(host, _HostVector):
        memcpy_htod(device_ptr, ctypes.c_void_p(host.data_ptr), host.nbytes)
        return
    memcpy_htod(device_ptr, ctypes.cast(host, ctypes.c_void_p), ctypes.sizeof(host))


def _copy_dtoh(host, device_ptr) -> None:
    if isinstance(host, HostTensorF32):
        memcpy_dtoh(ctypes.c_void_p(host.data_ptr), device_ptr, host.nbytes)
        return
    if isinstance(host, _HostVector):
        host._load_from_device(device_ptr)
        return
    memcpy_dtoh(ctypes.cast(host, ctypes.c_void_p), device_ptr, ctypes.sizeof(host))


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [float(start)]
    step = (float(stop) - float(start)) / float(count - 1)
    return [float(start) + (step * idx) for idx in range(count)]


def _arange(count: int) -> list[float]:
    return [float(idx) for idx in range(max(int(count), 0))]


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / max(len(values), 1))


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return float(sum((float(value) - mean) ** 2 for value in values) / len(values))


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(max(_dot(values, values), 0.0))


def _max_abs(values: Sequence[float]) -> float:
    return max((abs(float(value)) for value in values), default=0.0)


def _clip_int(value: int, low: int, high: int) -> int:
    return max(int(low), min(int(high), int(value)))


def _sigmoid_list(values: Sequence[float]) -> list[float]:
    return [1.0 / (1.0 + math.exp(-float(value))) for value in values]


def _percentile_abs(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(abs(float(value)) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(100.0, float(percentile))) / 100.0 * float(len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _stack_rows(rows: Sequence[Sequence[float]]) -> HostTensorF32:
    if not rows:
        return HostTensorF32.zeros(0, 0)
    return HostTensorF32.from_array_like([list(row) for row in rows], rows=len(rows), cols=len(rows[0]))

# Base paths
KERNELS_DIR = Path(__file__).parent.parent / "kernels"


# ============================================================================
# Kimi's Kernels
# ============================================================================

class LatencyGuard:
    """Sovereign Latency Guard - Records GPU timing with %globaltimer

    Uses gre_sub100micro_gate.ptx to measure kernel execution time directly
    on GPU, avoiding CPU timer overhead.

    Args:
        threshold_us: Maximum allowed latency in microseconds (default: 100.0)
    """

    def __init__(self, threshold_us: float = 100.0):
        self.threshold_us = float(threshold_us)
        self.threshold_ns = int(threshold_us * 1_000.0)

        # Load PTX kernel
        ptx_path = KERNELS_DIR / "gre_sub100micro_gate.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_sub100micro_gate")

        # Allocate device buffers (reused across calls)
        self.d_timestamps = gpu_malloc(2 * 8)  # 2 x uint64
        self.d_flag = gpu_malloc(4)            # 1 x uint32

        # Host buffers for readback
        self.timestamps = UInt64Vector.zeros(2)
        self.flag = UInt32Vector.zeros(1)

    def start(self):
        """Record start timestamp on GPU"""
        launch(
            self.kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                ctypes.c_uint64(self.d_timestamps.value),
                ctypes.c_uint64(self.d_flag.value),
                ctypes.c_uint64(self.threshold_ns),
                ctypes.c_uint32(0),  # mode=0 (start)
            ],
        )
        synchronize()

    def stop(self) -> Tuple[int, bool]:
        """Record stop timestamp and check threshold

        Returns:
            (elapsed_ns, breached): Elapsed time in ns and whether threshold was exceeded
        """
        launch(
            self.kernel,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params=[
                ctypes.c_uint64(self.d_timestamps.value),
                ctypes.c_uint64(self.d_flag.value),
                ctypes.c_uint64(self.threshold_ns),
                ctypes.c_uint32(1),  # mode=1 (stop)
            ],
        )
        synchronize()

        # Copy results back
        _copy_dtoh(self.timestamps, self.d_timestamps)
        _copy_dtoh(self.flag, self.d_flag)

        elapsed_ns = int(self.timestamps[1] - self.timestamps[0])
        breached = bool(self.flag[0] == 0xDEADBEEF)

        return elapsed_ns, breached

    def cleanup(self):
        """Free GPU memory"""
        gpu_free(self.d_timestamps)
        gpu_free(self.d_flag)

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


class ARCReasoner:
    """Sovereign ARC Reasoner - Extracts rules from ARC grids

    Uses gre_arc_reasoner.ptx to analyze ARC-AGI grids and extract
    compact rule representations.

    Example:
        reasoner = ARCReasoner()
        grid = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        rule_id, rotation, color_checksum = reasoner.extract_rules(grid)
    """

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_arc_reasoner.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_arc_reasoner")

    def extract_rules(self, grid) -> Tuple[int, int, int]:
        """Extract rules from ARC grid

        Args:
            grid: 2D int32 array (will be flattened)

        Returns:
            (rule_id, rotation_count, color_checksum): Extracted rule parameters
        """
        grid_tensor = _i32_vector(
            value
            for row in HostTensorF32.from_array_like(grid).to_nested_list()
            for value in row
        )
        grid_size = len(grid_tensor)

        # Allocate GPU memory
        d_grid = gpu_malloc(grid_tensor.nbytes)
        d_output = gpu_malloc(3 * 4)  # 3 x int32

        try:
            # Copy grid to GPU
            _copy_htod(d_grid, grid_tensor)

            # Launch kernel
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(d_grid.value),
                    ctypes.c_uint32(grid_size),
                    ctypes.c_uint64(d_output.value),
                ],
            )
            synchronize()

            # Copy results back
            output = Int32Vector.zeros(3)
            _copy_dtoh(output, d_output)

            return int(output[0]), int(output[1]), int(output[2])

        finally:
            gpu_free(d_grid)
            gpu_free(d_output)


class OOMSpillManager:
    """Sovereign OOM Spill Manager - Memory overflow protection

    Uses gre_oom_spill.ptx to compute spill plans when GPU memory is low.
    """

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_oom_spill.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_oom_spill")

    def compute_spill_plan(
        self,
        oldest_index: int,
        atom_size_bytes: int,
        available_bytes: int,
        request_count: int
    ) -> Tuple[int, int]:
        """Compute how many atoms to spill given available memory

        Args:
            oldest_index: Index of oldest atom in memory
            atom_size_bytes: Size of each atom in bytes
            available_bytes: Available GPU memory
            request_count: Number of atoms requested

        Returns:
            (atoms_to_spill, bytes_required): Spill plan
        """
        # Prepare stats input (uint64[2])
        StatsArray = ctypes.c_uint64 * 2
        stats = StatsArray(ctypes.c_uint64(oldest_index), ctypes.c_uint64(atom_size_bytes))
        OutputArray = ctypes.c_uint64 * 2
        output = OutputArray()

        # Allocate GPU memory
        d_stats = gpu_malloc(ctypes.sizeof(stats))
        d_output = gpu_malloc(ctypes.sizeof(output))

        try:
            # Copy stats to GPU
            memcpy_htod(d_stats, ctypes.cast(stats, ctypes.c_void_p), ctypes.sizeof(stats))

            # Launch kernel
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(d_stats.value),
                    ctypes.c_uint64(available_bytes),
                    ctypes.c_uint32(request_count),
                    ctypes.c_uint64(d_output.value),
                ],
            )
            synchronize()

            # Copy results back
            memcpy_dtoh(ctypes.cast(output, ctypes.c_void_p), d_output, ctypes.sizeof(output))

            return int(output[0]), int(output[1])

        finally:
            gpu_free(d_stats)
            gpu_free(d_output)


# ============================================================================
# Qwen's Kernel
# ============================================================================

class GalaxyResonanceEngine:
    """Sovereign Galaxy Resonance Engine - Recursive core blending

    Uses galaxy_resonance_engine.ptx to blend embeddings with latent state
    using alpha-weighted combination (RPN-style lerp).
    """

    def __init__(self):
        ptx_path = KERNELS_DIR / "galaxy_resonance_engine.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "galaxy_resonance_engine")

    def resonate(
        self,
        embeddings,
        latent,
        alpha: float = 0.5
    ):
        """Blend embeddings with latent state

        Args:
            embeddings: Input embeddings [batch_size, vector_dim]
            latent: Latent state [batch_size, vector_dim]
            alpha: Blend factor (0.0 to 1.0)

        Returns:
            output: Blended result [batch_size, vector_dim]
        """
        embeddings_host = _f32_matrix(embeddings)
        latent_host = _f32_matrix(latent)
        assert embeddings_host.shape == latent_host.shape
        batch_size, vector_dim = embeddings_host.shape

        # Compute element counts in bytes assuming float32 inputs
        elem_count = batch_size * vector_dim
        byte_count = elem_count * 4

        # Allocate GPU memory
        d_embeddings = gpu_malloc(byte_count)
        d_latent = gpu_malloc(byte_count)
        d_output = gpu_malloc(byte_count)

        try:
            # Copy inputs to GPU
            _copy_htod(d_embeddings, embeddings_host)
            _copy_htod(d_latent, latent_host)

            # Launch kernel (one block per batch element)
            launch(
                self.kernel,
                grid=(batch_size, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_embeddings.value),
                    ctypes.c_uint64(d_latent.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_uint32(vector_dim),
                    ctypes.c_uint32(batch_size),
                    ctypes.c_float(alpha),
                ],
            )
            synchronize()

            # Copy result back
            # Allocate host buffer and copy result back
            OutArray = ctypes.c_float * elem_count
            out_host = OutArray()
            memcpy_dtoh(ctypes.cast(out_host, ctypes.c_void_p), d_output, byte_count)

            flat = [float(out_host[i]) for i in range(elem_count)]
            rows: List[List[float]] = []
            for b in range(batch_size):
                start = b * vector_dim
                rows.append(flat[start:start + vector_dim])
            return _stack_rows(rows)

        finally:
            gpu_free(d_embeddings)
            gpu_free(d_latent)
            gpu_free(d_output)

    def resonate_list(
        self,
        embeddings: list[list[float]],
        latent: list[list[float]] | list[float],
        alpha: float = 0.5,
    ) -> list[list[float]]:
        embedding_arr = _f32_matrix(embeddings)
        latent_shape = _shape_of(latent)
        if embedding_arr.ndim != 2:
            raise ValueError("embeddings_must_be_rank2")
        if len(latent_shape) == 1:
            latent_values = _float_list(latent)
            latent_rows = [list(latent_values) for _ in range(embedding_arr.shape[0])]
            latent_arr = _stack_rows(latent_rows)
        else:
            latent_arr = _f32_matrix(latent)
        if latent_arr.shape != embedding_arr.shape:
            raise ValueError("latent_shape_mismatch")
        resonated = self.resonate(embedding_arr, latent_arr, alpha=alpha)
        return HostTensorF32.from_array_like(resonated).to_nested_list()


__all__ = [
    "LatencyGuard",
    "ARCReasoner",
    "OOMSpillManager",
    "GalaxyResonanceEngine",
    "DefeasibleResolver",
]


# ============================================================================
# Deep Seek's Kernels
# ============================================================================

class GeometryRouter:
    """Sovereign Geometry Router - spatial relationship features."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_geometry_router.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_geometry_router")

    def compute_relations(self, embeddings_a, embeddings_b):
        """Compute 16 spatial relationship features for aligned embedding pairs."""
        shape_a = _shape_of(embeddings_a)
        shape_b = _shape_of(embeddings_b)
        if len(shape_a) == 1:
            arr_a = _f32_matrix(embeddings_a, rows=1, cols=shape_a[0])
        else:
            arr_a = _f32_matrix(embeddings_a)
        if len(shape_b) == 1:
            arr_b = _f32_matrix(embeddings_b, rows=1, cols=shape_b[0])
        else:
            arr_b = _f32_matrix(embeddings_b)
        if arr_a.shape != arr_b.shape:
            raise ValueError(f"embeddings_a and embeddings_b must match; got {arr_a.shape} vs {arr_b.shape}")
        if arr_a.ndim != 2 or arr_a.shape[0] <= 0 or arr_a.shape[1] <= 0:
            raise ValueError("embedding pairs must be non-empty [N x D]")

        pair_count, dims = arr_a.shape
        feature_count = 16
        relations = HostTensorF32.zeros(pair_count, feature_count)

        d_a = gpu_malloc(arr_a.nbytes)
        d_b = gpu_malloc(arr_b.nbytes)
        d_relations = gpu_malloc(relations.nbytes)
        try:
            _copy_htod(d_a, arr_a)
            _copy_htod(d_b, arr_b)
            _copy_htod(d_relations, relations)
            launch(
                self.kernel,
                grid=((pair_count + 127) // 128, 1, 1),
                block=(128, 1, 1),
                params=[
                    ctypes.c_uint64(d_a.value),
                    ctypes.c_uint64(d_b.value),
                    ctypes.c_uint64(d_relations.value),
                    ctypes.c_int(pair_count),
                    ctypes.c_int(dims),
                    ctypes.c_int(feature_count),
                ],
            )
            synchronize()
            _copy_dtoh(relations, d_relations)
            return relations
        finally:
            gpu_free(d_a)
            gpu_free(d_b)
            gpu_free(d_relations)

    def route(self, input_data, shape_id: int):
        """Compatibility wrapper returning relation features to a shape prototype."""
        vector = _float_list(input_data)
        dims = len(vector)
        if dims <= 0:
            return [0.0 for _ in range(16)]
        prototype_row = [0.0 for _ in range(dims)]
        if int(shape_id) == 0:
            prototype_row = _linspace(1.0, 0.2, dims)
        elif int(shape_id) == 1:
            half = max(1, dims // 2)
            prototype_row = [1.0 if idx < half else 0.5 for idx in range(dims)]
        elif int(shape_id) == 2:
            prototype_row = [math.sin(value) for value in _linspace(0.0, math.pi * 2.0, dims)]
        elif int(shape_id) == 3:
            prototype_row = _linspace(0.1, 1.0, dims)
        else:
            prototype_row = [1.0 for _ in range(dims)]
        relations = self.compute_relations([vector], [prototype_row])
        return relations[0]


class FractalEmitter:
    """Sovereign Fractal Emitter - multi-scale self-similarity scoring."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_fractal_emitter.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_fractal_emitter")

    def compute_self_similarity(self, features, num_scales: int = 3):
        """Compute multi-scale self-similarity scores for [N x D] features."""
        shape = _shape_of(features)
        if len(shape) == 1:
            arr = _f32_matrix(features, rows=1, cols=shape[0])
        else:
            arr = _f32_matrix(features)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("features must be a non-empty [N x D] array")
        scores = Float32Vector.zeros(arr.shape[0])

        d_features = gpu_malloc(arr.nbytes)
        d_scores = gpu_malloc(scores.nbytes)

        try:
            _copy_htod(d_features, arr)
            _copy_htod(d_scores, scores)
            launch(
                self.kernel,
                grid=((arr.shape[0] + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_features.value),
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_int(arr.shape[0]),
                    ctypes.c_int(arr.shape[1]),
                    ctypes.c_int(max(1, int(num_scales))),
                ],
            )
            synchronize()
            _copy_dtoh(scores, d_scores)
            return scores
        finally:
            gpu_free(d_features)
            gpu_free(d_scores)

    def emit(self, atoms, base_scale: float = 1.0):
        """Compatibility wrapper returning simple coordinates from self-similarity."""
        arr = _float_list(atoms)
        if not arr:
            return HostTensorF32.zeros(0, 3)
        scores = self.compute_self_similarity([[value] for value in arr], num_scales=1)
        denom = max(float(len(arr) - 1), 1.0)
        rows = [
            [
                float(value) * float(base_scale),
                float(scores[idx]) * float(base_scale),
                (float(idx) / denom) * float(base_scale),
            ]
            for idx, value in enumerate(arr)
        ]
        return _stack_rows(rows)


class CognitiveExecutive:
    """Sovereign Cognitive Executive - trust weighting for swarm chains."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_cognitive_executive.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_cognitive_executive")

    def compute_trust_weights(self, resonance_matrix, chain_norms):
        """Return (trust_weights[8], coherence_score) from swarm diagnostics."""
        matrix = _f32_matrix(resonance_matrix, rows=8, cols=8)
        norms = _f32_vector(chain_norms)
        if matrix.shape != (8, 8):
            raise ValueError(f"resonance_matrix must be (8, 8), got {matrix.shape}")
        if norms.shape != (8,):
            raise ValueError(f"chain_norms must be (8,), got {norms.shape}")
        trust = Float32Vector.zeros(8)
        coherence = Float32Vector.zeros(1)

        d_matrix = gpu_malloc(matrix.nbytes)
        d_norms = gpu_malloc(norms.nbytes)
        d_trust = gpu_malloc(trust.nbytes)
        d_coherence = gpu_malloc(coherence.nbytes)

        try:
            _copy_htod(d_matrix, matrix)
            _copy_htod(d_norms, norms)
            _copy_htod(d_trust, trust)
            _copy_htod(d_coherence, coherence)
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(8, 1, 1),
                params=[
                    ctypes.c_uint64(d_matrix.value),
                    ctypes.c_uint64(d_norms.value),
                    ctypes.c_uint64(d_trust.value),
                    ctypes.c_uint64(d_coherence.value),
                ],
            )
            synchronize()
            _copy_dtoh(trust, d_trust)
            _copy_dtoh(coherence, d_coherence)
            return trust, float(coherence[0])
        finally:
            gpu_free(d_matrix)
            gpu_free(d_norms)
            gpu_free(d_trust)
            gpu_free(d_coherence)


# ============================================================================
# GLM's Kernels
# ============================================================================

class ResonanceField:
    """Sovereign Resonance Field - cross-galaxy interference scoring"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_resonance_field.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_resonance_field")

    def compute_resonance(self, candidate_embeddings, galaxy_ids, base_scores):
        """Cross-galaxy interference scoring."""
        shape = _shape_of(candidate_embeddings)
        if len(shape) == 1:
            embeddings = _f32_matrix(candidate_embeddings, rows=shape[0], cols=1)
        else:
            embeddings = _f32_matrix(candidate_embeddings)
        if embeddings.ndim != 2:
            raise ValueError("candidate_embeddings_must_be_rank1_or_rank2")
        galaxy_arr = _i32_vector(galaxy_ids)
        score_arr = _f32_vector(base_scores)
        count = int(embeddings.shape[0])
        dim = int(embeddings.shape[1])
        if galaxy_arr.shape[0] != count:
            raise ValueError("galaxy_ids_length_mismatch")
        if score_arr.shape[0] != count:
            raise ValueError("base_scores_length_mismatch")

        embedding_bytes = int(embeddings.nbytes)
        galaxy_bytes = int(galaxy_arr.nbytes)
        score_bytes = int(score_arr.nbytes)
        output_bytes = int(score_arr.nbytes)

        d_embeddings = gpu_malloc(embedding_bytes)
        d_galaxy_ids = gpu_malloc(galaxy_bytes)
        d_scores = gpu_malloc(score_bytes)
        d_output = gpu_malloc(output_bytes)
        try:
            _copy_htod(d_embeddings, embeddings)
            _copy_htod(d_galaxy_ids, galaxy_arr)
            _copy_htod(d_scores, score_arr)

            launch(
                self.kernel,
                grid=((count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_embeddings.value),
                    ctypes.c_uint64(d_galaxy_ids.value),
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_int32(count),
                    ctypes.c_int32(dim),
                ],
            )
            synchronize()

            out_host = Float32Vector.zeros(count)
            _copy_dtoh(out_host, d_output)
            return out_host
        finally:
            gpu_free(d_embeddings)
            gpu_free(d_galaxy_ids)
            gpu_free(d_scores)
            gpu_free(d_output)

    def compute(self, positions, density):
        """Compatibility wrapper for legacy tests/non-hot-path usage."""
        embeddings = _f32_matrix(positions)
        if embeddings.ndim != 2:
            raise ValueError("positions_must_be_rank2")
        count = int(embeddings.shape[0])
        galaxy_ids = Int32Vector(int(idx) for idx in range(count))
        return self.compute_resonance(embeddings, galaxy_ids, density)


class AtomicFissionFusion:
    """Sovereign Atomic Fission/Fusion - compositional consistency operations."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_atomic_fission_fusion.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_atomic_fission_fusion")

    def _run_compose_decompose(self, compound, atoms, *, mode: int):
        compound_arr = _f32_vector(compound)
        atom_shape = _shape_of(atoms)
        if len(atom_shape) == 1:
            atom_arr = _f32_matrix(atoms, rows=1, cols=atom_shape[0])
        else:
            atom_arr = _f32_matrix(atoms)
        if atom_arr.ndim != 2 or atom_arr.shape[0] <= 0 or atom_arr.shape[1] <= 0:
            raise ValueError("atoms must be a non-empty 2D array-like [K x D]")
        if compound_arr.size <= 0:
            raise ValueError("compound must be non-empty")
        if atom_arr.shape[1] != compound_arr.size:
            raise ValueError("compound and atoms must share the same feature dimension")

        d_compound = gpu_malloc(compound_arr.nbytes)
        d_atoms = gpu_malloc(atom_arr.nbytes)
        d_result = gpu_malloc(compound_arr.nbytes)
        d_consistency = gpu_malloc(4)

        try:
            _copy_htod(d_compound, compound_arr)
            _copy_htod(d_atoms, atom_arr)

            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_uint64(d_compound.value),
                    ctypes.c_uint64(d_atoms.value),
                    ctypes.c_uint64(d_result.value),
                    ctypes.c_uint64(d_consistency.value),
                    ctypes.c_int(int(atom_arr.shape[0])),
                    ctypes.c_int(int(atom_arr.shape[1])),
                    ctypes.c_int(int(mode)),
                ],
            )
            synchronize()

            result_host = (ctypes.c_float * int(compound_arr.size))()
            consistency_host = ctypes.c_float()
            memcpy_dtoh(ctypes.cast(result_host, ctypes.c_void_p), d_result, compound_arr.nbytes)
            memcpy_dtoh(ctypes.cast(ctypes.pointer(consistency_host), ctypes.c_void_p), d_consistency, 4)

            result = Float32Vector(float(result_host[idx]) for idx in range(compound_arr.size))
            return result, float(consistency_host.value)
        finally:
            gpu_free(d_compound)
            gpu_free(d_atoms)
            gpu_free(d_result)
            gpu_free(d_consistency)

    def decompose(self, compound, atoms):
        """Project a compound embedding onto atom directions."""
        return self._run_compose_decompose(compound, atoms, mode=0)

    def compose(self, atoms):
        """Fuse a small atom set into a weighted centroid and report agreement."""
        atom_shape = _shape_of(atoms)
        if len(atom_shape) == 1:
            atom_arr = _f32_matrix(atoms, rows=1, cols=atom_shape[0])
        else:
            atom_arr = _f32_matrix(atoms)
        if atom_arr.ndim != 2 or atom_arr.shape[0] <= 0:
            raise ValueError("atoms must be a non-empty 2D array-like [K x D]")
        compound = Float32Vector(
            _mean([float(atom_arr[row, col]) for row in range(atom_arr.shape[0])])
            for col in range(atom_arr.shape[1])
        )
        return self._run_compose_decompose(compound, atom_arr, mode=1)

    def transform(self, atoms, mode: int, ratio: float):
        """Compatibility scalar transform kept for legacy callers."""
        arr = _float_list(atoms)
        factor = float(ratio)
        if abs(factor) < 1e-6:
            factor = 1e-6
        if int(mode) == 0:
            return [float(value) * factor for value in arr]
        return [float(value) / factor for value in arr]

    def transform_list(
        self,
        atoms: Sequence[float],
        mode: int,
        ratio: float,
    ) -> list[float]:
        """List-friendly compatibility transform without exposing NumPy to the caller."""
        values = [float(value) for value in atoms]
        if not values:
            return []
        transformed = self.transform(values, mode=mode, ratio=ratio)
        return [float(value) for value in transformed]

    def create_sparse(self, weights, sparsity_level: float, preserve_important: bool = True) -> dict:
        """Create sparse weight representation for efficient GPU computation.

        This method converts dense weights into sparse format, keeping only the most
        important values based on magnitude. Used for adaptive sparsity in thinking tags.

        Args:
            weights: Weight arrays (can be dict or ndarray)
            sparsity_level: Target sparsity (0.0 = dense, 1.0 = maximally sparse)
            preserve_important: If True, keep high-magnitude values

        Returns:
            Sparse weight dictionary with same keys as input
        """
        if isinstance(weights, dict):
            # Process each weight matrix
            sparse_dict = {}
            for key, W in weights.items():
                shape = None
                try:
                    shape = _shape_of(W)
                except Exception:
                    sparse_dict[key] = W
                    continue
                tensor = _f32_matrix(W) if len(shape) == 2 else _f32_vector(W)
                flat = _float_list(tensor)

                if preserve_important:
                    # Keep top-k values by magnitude
                    threshold_percentile = sparsity_level * 100.0
                    threshold = _percentile_abs(flat, threshold_percentile)
                    sparse_flat = [value if abs(value) >= threshold else 0.0 for value in flat]
                else:
                    # Random sparsification
                    sparse_flat = [
                        value if random.random() > float(sparsity_level) else 0.0
                        for value in flat
                    ]

                if len(shape) == 2:
                    sparse_dict[key] = HostTensorF32.from_array_like(
                        sparse_flat,
                        rows=shape[0],
                        cols=shape[1],
                    )
                else:
                    sparse_dict[key] = Float32Vector(sparse_flat)
            return sparse_dict

        else:
            try:
                shape = _shape_of(weights)
            except Exception:
                return weights
            flat = _float_list(weights)
            if preserve_important:
                threshold_percentile = sparsity_level * 100.0
                threshold = _percentile_abs(flat, threshold_percentile)
                sparse_flat = [value if abs(value) >= threshold else 0.0 for value in flat]
            else:
                sparse_flat = [
                    value if random.random() > float(sparsity_level) else 0.0
                    for value in flat
                ]
            if len(shape) == 2:
                return HostTensorF32.from_array_like(sparse_flat, rows=shape[0], cols=shape[1])
            return Float32Vector(sparse_flat)


class DefeasibleResolver:
    """Sovereign Defeasible Resolver - non-monotonic conflict resolution."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_defeasible_resolver.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_defeasible_resolver")

    def resolve(
        self,
        conclusions,
        rule_strengths,
        superiority,
        *,
        num_workers: int | None = None,
        num_candidates: int | None = None,
        max_superiors: int | None = None,
    ):
        conclusion_shape = _shape_of(conclusions)
        if len(conclusion_shape) == 1:
            if num_workers is None or num_candidates is None:
                raise ValueError("num_workers_and_num_candidates_required_for_flat_conclusions")
            conclusion_arr = _f32_matrix(conclusions, rows=int(num_workers), cols=int(num_candidates))
        else:
            conclusion_arr = _f32_matrix(conclusions)
        if conclusion_arr.ndim != 2:
            raise ValueError("conclusions_must_be_rank1_or_rank2")

        worker_count = int(num_workers or conclusion_arr.shape[0])
        candidate_count = int(num_candidates or conclusion_arr.shape[1])
        if conclusion_arr.shape != (worker_count, candidate_count):
            raise ValueError(
                f"conclusion_shape_mismatch: expected {(worker_count, candidate_count)}, got {conclusion_arr.shape}"
            )

        strength_arr = Int8Vector(int(value) for value in rule_strengths)
        if strength_arr.shape[0] != worker_count:
            raise ValueError("rule_strengths_length_mismatch")

        superiority_shape = _shape_of(superiority)
        if len(superiority_shape) == 1:
            if max_superiors is None:
                raise ValueError("max_superiors_required_for_flat_superiority")
            superiority_arr = HostTensorF32.from_array_like(
                [int(value) for value in superiority],
                rows=worker_count,
                cols=int(max_superiors),
            )
        else:
            superiority_arr = _f32_matrix(superiority)
        if superiority_arr.ndim != 2 or superiority_arr.shape[0] != worker_count:
            raise ValueError("superiority_shape_mismatch")
        superiority_cap = int(max_superiors or superiority_arr.shape[1])
        if superiority_arr.shape[1] != superiority_cap:
            raise ValueError("max_superiors_shape_mismatch")

        verdicts = Float32Vector.zeros(candidate_count)
        proof_tags = UInt32Vector.zeros(candidate_count)

        d_conclusions = gpu_malloc(conclusion_arr.nbytes)
        d_strengths = gpu_malloc(strength_arr.nbytes)
        d_superiority = gpu_malloc(superiority_arr.nbytes)
        d_verdicts = gpu_malloc(verdicts.nbytes)
        d_proof_tags = gpu_malloc(proof_tags.nbytes)
        try:
            _copy_htod(d_conclusions, conclusion_arr)
            _copy_htod(d_strengths, strength_arr)
            # superiority uses uint32 semantics, so stage via ctypes directly
            superiority_values = [int(value) for value in _float_list(superiority_arr)]
            superiority_buf = (ctypes.c_uint32 * len(superiority_values))(*superiority_values)
            _copy_htod(d_superiority, superiority_buf)
            _copy_htod(d_verdicts, verdicts)
            _copy_htod(d_proof_tags, proof_tags)

            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(min(max(candidate_count, 1), 128), 1, 1),
                params=[
                    ctypes.c_uint64(d_conclusions.value),
                    ctypes.c_uint64(d_strengths.value),
                    ctypes.c_uint64(d_superiority.value),
                    ctypes.c_uint64(d_verdicts.value),
                    ctypes.c_uint64(d_proof_tags.value),
                    ctypes.c_int(worker_count),
                    ctypes.c_int(candidate_count),
                    ctypes.c_int(superiority_cap),
                ],
            )
            synchronize()
            _copy_dtoh(verdicts, d_verdicts)
            _copy_dtoh(proof_tags, d_proof_tags)
            return verdicts, proof_tags
        finally:
            gpu_free(d_conclusions)
            gpu_free(d_strengths)
            gpu_free(d_superiority)
            gpu_free(d_verdicts)
            gpu_free(d_proof_tags)


class TemporalReasoning:
    """Sovereign Temporal Reasoning - ordered sequence pattern extraction."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_temporal_reasoning.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_temporal_reasoning")

    def compute_patterns(self, sequence):
        """Compute 24 temporal pattern features for an ordered [T x D] sequence."""
        arr = _f32_matrix(sequence)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("sequence must be a non-empty [T x D] array")
        seq_length, feat_dim = arr.shape
        pattern_count = 24
        patterns = Float32Vector.zeros(pattern_count)

        d_sequence = gpu_malloc(arr.nbytes)
        d_output = gpu_malloc(patterns.nbytes)

        try:
            _copy_htod(d_sequence, arr)
            _copy_htod(d_output, patterns)
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(d_sequence.value),
                    ctypes.c_uint64(d_output.value),
                    ctypes.c_int(seq_length),
                    ctypes.c_int(feat_dim),
                ],
            )
            synchronize()
            _copy_dtoh(patterns, d_output)
            return patterns
        finally:
            gpu_free(d_sequence)
            gpu_free(d_output)

    def compute_deltas(self, sequence):
        """Compatibility helper preserving the legacy frame-difference surface."""
        arr = _f32_matrix(sequence)
        if arr.ndim != 2:
            raise ValueError("sequence must be rank-2")
        delta_rows: list[list[float]] = [[0.0 for _ in range(arr.shape[1])] for _ in range(arr.shape[0])]
        if arr.shape[0] > 1:
            for row in range(arr.shape[0] - 1):
                delta_rows[row] = [
                    float(arr[row + 1, col]) - float(arr[row, col])
                    for col in range(arr.shape[1])
                ]
        return _stack_rows(delta_rows)

    def compute_coherence(self, crystallized, temporal_context):
        """Compute temporal coherence scores.

        Measures how well the crystallized output aligns with temporal context.
        Used in thinking tag inference for coherence scoring.

        Args:
            crystallized: Crystallized output vector
            temporal_context: Temporal context vector

        Returns:
            Coherence scores (per dimension)
        """
        crystallized_flat = _float_list(crystallized)
        context_flat = _float_list(temporal_context)

        # Ensure same shape for comparison
        if len(crystallized_flat) != len(context_flat):
            min_len = min(len(crystallized_flat), len(context_flat))
            crystallized_flat = crystallized_flat[:min_len]
            context_flat = context_flat[:min_len]

        # Compute element-wise coherence (similarity measure)
        # High coherence when values are similar
        diff = [abs(c - t) for c, t in zip(crystallized_flat, context_flat)]
        max_diff = max(diff) if diff and max(diff) > 0 else 1.0
        coherence = [1.0 - (value / max_diff) for value in diff]

        return Float32Vector(coherence)

    def estimate_coherence(self, context):
        """Estimate coherence from temporal context alone.

        Simplified version that estimates coherence without comparing to output.
        Useful for fallback paths.

        Args:
            context: Temporal context vector

        Returns:
            Estimated coherence scores
        """
        context_flat = _float_list(context)

        # Use temporal stability (low variance = high coherence)
        if len(context_flat) > 1:
            variance = _variance(context_flat)
            # Normalize variance to 0-1 range (assuming typical variance < 1.0)
            normalized_var = min(variance, 1.0)
            coherence_score = 1.0 - normalized_var
        else:
            coherence_score = 1.0

        # Return uniform coherence scores
        return Float32Vector([coherence_score for _ in context_flat])


# ============================================================================
# Grok's Kernels
# ============================================================================

class VectorResonator:
    """Sovereign Vector Resonator - attention-weighted vector blending."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_vector_resonator.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_vector_resonator")

    def resonate_attention(self, vectors):
        """Blend a small set of vectors with content-dependent attention."""
        arr = _f32_matrix(vectors)
        if arr.ndim != 2:
            raise ValueError("vectors must be a 2D array-like [K x D]")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("vectors must be non-empty")

        k_count, dims = arr.shape
        blended_bytes = dims * 4
        weights_bytes = k_count * 4
        d_vectors = gpu_malloc(arr.nbytes)
        d_blended = gpu_malloc(blended_bytes)
        d_weights = gpu_malloc(weights_bytes)

        try:
            _copy_htod(d_vectors, arr)

            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_uint64(d_vectors.value),
                    ctypes.c_uint64(d_blended.value),
                    ctypes.c_uint64(d_weights.value),
                    ctypes.c_int(k_count),
                    ctypes.c_int(dims),
                ],
            )
            synchronize()

            blended_host = (ctypes.c_float * dims)()
            weights_host = (ctypes.c_float * k_count)()
            memcpy_dtoh(ctypes.cast(blended_host, ctypes.c_void_p), d_blended, blended_bytes)
            memcpy_dtoh(ctypes.cast(weights_host, ctypes.c_void_p), d_weights, weights_bytes)

            blended = Float32Vector(float(blended_host[idx]) for idx in range(dims))
            weights = Float32Vector(float(weights_host[idx]) for idx in range(k_count))
            return blended, weights
        finally:
            gpu_free(d_vectors)
            gpu_free(d_blended)
            gpu_free(d_weights)

    def resonate(self, vec_a, vec_b, alpha: float):
        """Blend two vectors using content attention with an alpha prior."""
        arr_a = _float_list(vec_a)
        arr_b = _float_list(vec_b)
        if len(arr_a) != len(arr_b):
            raise AssertionError("input_vectors_must_match")
        _, weights = self.resonate_attention([arr_a, arr_b])

        prior = [
            min(max(float(alpha), 1e-4), 1.0 - 1e-4),
            min(max(1.0 - float(alpha), 1e-4), 1.0 - 1e-4),
        ]
        adjusted = [float(weights[idx]) * prior[idx] for idx in range(2)]
        total = float(sum(adjusted))
        if total <= 0.0:
            prior_total = sum(prior)
            adjusted = [value / prior_total for value in prior]
        else:
            adjusted = [value / total for value in adjusted]
        blended = [
            adjusted[0] * arr_a[idx] + adjusted[1] * arr_b[idx]
            for idx in range(len(arr_a))
        ]
        return Float32Vector(blended)

    def resonate_list(self, vec_a, vec_b, alpha: float) -> list[float]:
        blended = self.resonate(list(vec_a), list(vec_b), alpha)
        return blended.tolist()

    def resonate_attention_list(self, vectors) -> tuple[list[float], list[float]]:
        blended, weights = self.resonate_attention(vectors)
        return (blended.tolist(), weights.tolist())

    def calculate_complexity(self, input_embedding, modal_signature: list) -> float:
        """Calculate input complexity for adaptive sparsity decisions.

        Uses vector magnitude and modal diversity as complexity indicators.
        This is a heuristic for determining whether to use sparse or dense operations.

        Args:
            input_embedding: Input vector (float32)
            modal_signature: List of modality names (e.g., ['text', 'image'])

        Returns:
            Complexity score between 0.0 and 1.0
        """
        # Normalize input embedding if needed
        input_values = _float_list(input_embedding)

        # Calculate vector magnitude (normalized)
        magnitude = _norm(input_values)
        max_magnitude = math.sqrt(len(input_values))  # Maximum possible for unit components
        normalized_magnitude = min(magnitude / max_magnitude, 1.0)

        # Calculate modal diversity score (more modalities = more complex)
        modal_diversity = len(set(modal_signature)) / 3.0  # Normalize by max 3 modalities
        modal_diversity = min(modal_diversity, 1.0)

        # Combine factors (weighted average)
        complexity = 0.7 * normalized_magnitude + 0.3 * modal_diversity

        return float(complexity)

    def cosine_similarity(self, vec_a, vec_b) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            vec_a, vec_b: Input vectors

        Returns:
            Cosine similarity (-1.0 to 1.0)
        """
        flat_a = _float_list(vec_a)
        flat_b = _float_list(vec_b)
        dot_product = _dot(flat_a, flat_b)
        norm_a = _norm(flat_a)
        norm_b = _norm(flat_b)

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def compute(self, confidence_vector):
        """Compute confidence rays from crystallized output.

        This is used in thinking tag inference to generate per-tag confidence scores.

        Args:
            confidence_vector: Crystallized output vector

        Returns:
            Confidence scores (one per dimension)
        """
        # Sigmoid activation for confidence scores
        return Float32Vector(_sigmoid_list(_float_list(confidence_vector)))


class GraphCrystallizer:
    """Sovereign Graph Crystallizer - graph message passing with compatibility path."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_graph_crystallizer.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_graph_crystallizer")

    @staticmethod
    def _as_rank2_float32(values):
        shape = _shape_of(values)
        squeezed = len(shape) == 1
        if squeezed:
            array = _f32_matrix(values, rows=shape[0], cols=1)
        else:
            array = _f32_matrix(values)
        if array.ndim != 2:
            raise ValueError("node_features_must_be_rank1_or_rank2")
        return array, squeezed

    @staticmethod
    def _neighbor_weight_from_ema(ema_rate: float) -> float:
        # Preserve the intent of "more neighbor trust" for larger EMA values,
        # but keep the update bounded so message passing stays stable.
        clamped = min(max(float(ema_rate), 0.0), 1.0)
        return 0.25 + 0.25 * clamped

    def crystallize_graph(
        self,
        node_features,
        adjacency,
        neighbor_counts=None,
        rounds: int = 2,
        self_weight: float = 0.6,
        neighbor_weight: float = 0.4,
    ):
        node_arr, squeezed = self._as_rank2_float32(node_features)
        adjacency_shape = _shape_of(adjacency)
        if len(adjacency_shape) != 2:
            raise ValueError("adjacency_must_be_rank2")
        adjacency_arr = [
            [int(value) for value in row]
            for row in HostTensorF32.from_array_like(adjacency, rows=adjacency_shape[0], cols=adjacency_shape[1]).to_nested_list()
        ]
        if adjacency_arr.shape[0] != node_arr.shape[0]:
            raise ValueError("adjacency_row_mismatch")

        if neighbor_counts is None:
            neighbor_counts_list = [sum(1 for value in row if int(value) >= 0) for row in adjacency_arr]
        else:
            neighbor_counts_list = [int(value) for value in neighbor_counts]
            if len(neighbor_counts_list) != node_arr.shape[0]:
                raise ValueError("neighbor_counts_row_mismatch")
        max_neighbors = int(len(adjacency_arr[0]) if adjacency_arr else 0)
        node_count = int(node_arr.shape[0])
        feature_dim = int(node_arr.shape[1])
        rounds = max(int(rounds), 0)
        adjacency_host = HostTensorF32.from_array_like(adjacency_arr, rows=node_count, cols=max_neighbors)
        adjacency_values = [int(value) for value in adjacency_host.to_flat_list()]
        adjacency_buf = (ctypes.c_int32 * len(adjacency_values))(*adjacency_values)
        neighbor_counts_arr = Int32Vector(
            _clip_int(value, 0, max_neighbors) for value in neighbor_counts_list
        )

        if rounds == 0 or node_count == 0 or feature_dim == 0:
            result = node_arr
            return Float32Vector(result.to_flat_list()) if squeezed else result

        current_host = node_arr.copy()
        byte_count = int(current_host.nbytes)
        adjacency_bytes = int(ctypes.sizeof(adjacency_buf))
        counts_bytes = int(neighbor_counts_arr.nbytes)

        d_current = gpu_malloc(byte_count)
        d_output = gpu_malloc(byte_count)
        d_adjacency = gpu_malloc(adjacency_bytes)
        d_counts = gpu_malloc(counts_bytes)
        try:
            _copy_htod(d_current, current_host)
            _copy_htod(d_adjacency, adjacency_buf)
            _copy_htod(d_counts, neighbor_counts_arr)

            grid_x = max((node_count + 255) // 256, 1)
            for _ in range(rounds):
                launch(
                    self.kernel,
                    grid=(grid_x, 1, 1),
                    block=(256, 1, 1),
                    params=[
                        ctypes.c_uint64(d_current.value),
                        ctypes.c_uint64(d_adjacency.value),
                        ctypes.c_uint64(d_counts.value),
                        ctypes.c_uint64(d_output.value),
                        ctypes.c_int32(node_count),
                        ctypes.c_int32(feature_dim),
                        ctypes.c_int32(max_neighbors),
                        ctypes.c_float(float(self_weight)),
                        ctypes.c_float(float(neighbor_weight)),
                    ],
                )
                synchronize()
                d_current, d_output = d_output, d_current

            out_host = HostTensorF32.zeros(node_count, feature_dim)
            _copy_dtoh(out_host, d_current)
            return Float32Vector(out_host.to_flat_list()) if squeezed else out_host
        finally:
            gpu_free(d_current)
            gpu_free(d_output)
            gpu_free(d_adjacency)
            gpu_free(d_counts)

    def _build_compatibility_graph(self, node_arr, neighbor_arr):
        node_count = int(node_arr.shape[0])
        feature_dim = int(node_arr.shape[1])
        total_nodes = node_count * 2
        features = HostTensorF32.zeros(total_nodes, feature_dim)
        for row in range(node_count):
            for col in range(feature_dim):
                features._buffer[row * feature_dim + col] = float(node_arr[row, col])
                features._buffer[(node_count + row) * feature_dim + col] = float(neighbor_arr[row, col])

        max_neighbors = 3 if node_count > 1 else 1
        adjacency = [[-1 for _ in range(max_neighbors)] for _ in range(total_nodes)]
        neighbor_counts = Int32Vector.zeros(total_nodes)

        for idx in range(node_count):
            cursor = 0
            adjacency[idx, cursor] = node_count + idx
            cursor += 1
            if idx > 0:
                adjacency[idx, cursor] = idx - 1
                cursor += 1
            if idx + 1 < node_count and cursor < max_neighbors:
                adjacency[idx, cursor] = idx + 1
                cursor += 1
            neighbor_counts[idx] = cursor

        return features, adjacency, neighbor_counts

    def crystallize(self, nodes, neighbors, ema_rate: float = 0.999):
        """Compatibility API backed by the real graph message-passing kernel."""
        node_arr, squeezed = self._as_rank2_float32(nodes)
        neighbor_arr, _ = self._as_rank2_float32(neighbors)
        if neighbor_arr.shape != node_arr.shape:
            raise ValueError("neighbor_shape_mismatch")

        features, adjacency, neighbor_counts = self._build_compatibility_graph(node_arr, neighbor_arr)
        result = self.crystallize_graph(
            features,
            adjacency,
            neighbor_counts,
            rounds=1,
            self_weight=1.0 - self._neighbor_weight_from_ema(ema_rate),
            neighbor_weight=self._neighbor_weight_from_ema(ema_rate),
        )
        if isinstance(result, Float32Vector):
            result_rows = HostTensorF32.from_array_like(result.tolist(), rows=node_arr.shape[0], cols=1)
        else:
            result_rows = result
        sliced = HostTensorF32.zeros(node_arr.shape[0], node_arr.shape[1])
        for row in range(node_arr.shape[0]):
            for col in range(node_arr.shape[1]):
                sliced._buffer[row * node_arr.shape[1] + col] = float(result_rows[row, col])
        return Float32Vector(sliced.to_flat_list()) if squeezed else sliced

    def crystallize_list(
        self,
        nodes: list[list[float]],
        neighbors: list[list[float]] | list[float],
        ema_rate: float = 0.999,
    ) -> list[list[float]]:
        node_arr = _f32_matrix(nodes)
        if node_arr.ndim != 2:
            raise ValueError("nodes_must_be_rank2")
        neighbor_shape = _shape_of(neighbors)
        if len(neighbor_shape) == 1:
            row = _float_list(neighbors)
            neighbor_arr = _stack_rows([row for _ in range(node_arr.shape[0])])
        else:
            neighbor_arr = _f32_matrix(neighbors)
        if neighbor_arr.shape != node_arr.shape:
            raise ValueError("neighbor_shape_mismatch")
        crystallized = self.crystallize(node_arr, neighbor_arr, ema_rate=ema_rate)
        return HostTensorF32.from_array_like(crystallized).to_nested_list()

    def smooth_intermediate(self, output, ema_buffer, warp_level: bool = True):
        """Smooth intermediate outputs using EMA buffer.

        This is used in thinking tag inference for dynamic crystallization.
        Applies EMA-based smoothing to reduce high-frequency noise.

        Args:
            output: Intermediate output vector
            ema_buffer: GPU buffer containing EMA state
            warp_level: If True, use warp-level synchronization

        Returns:
            Smoothed output vector
        """
        output_vec = _f32_vector(output)

        # For now, use simple EMA on CPU (can be optimized with GPU kernel later)
        # This maintains the interface while providing functional smoothing
        alpha = 0.999 if warp_level else 0.99

        # Read current EMA state from GPU buffer
        ema_state = Float32Vector.zeros(output_vec.size)
        if ema_buffer is not None and hasattr(ema_buffer, 'value'):
            try:
                _copy_dtoh(ema_state, ema_buffer)
            except Exception:
                pass  # First call, EMA state is zeros

        # Apply EMA: new_state = alpha * old_state + (1 - alpha) * new_value
        smoothed = Float32Vector(
            alpha * float(ema_state[idx]) + (1.0 - alpha) * float(output_vec[idx])
            for idx in range(output_vec.size)
        )

        # Write updated EMA state back to GPU buffer
        if ema_buffer is not None and hasattr(ema_buffer, 'value'):
            try:
                _copy_htod(ema_buffer, smoothed)
            except Exception:
                pass

        return smoothed

    def apply(self, output, ema_buffer):
        """Alias for smooth_intermediate() with default parameters.

        Args:
            output: Intermediate output vector
            ema_buffer: GPU buffer containing EMA state

        Returns:
            Smoothed output vector
        """
        return self.smooth_intermediate(output, ema_buffer, warp_level=True)


class SleepClusterRefiner:
    """Sovereign sleep-time cluster refiner backed by sleep_cluster_refiner.ptx.

    This bridge is intentionally thin: orchestration stays in Python, while
    centroid assignment, refinement, and silhouette scoring run on GPU via the
    canonical PTX runtime wrapper.
    """

    def __init__(self):
        from knowledge3d.cranium.ptx_runtime.sleep_cluster_kernels import SleepClusterKernels

        self._kernels = SleepClusterKernels()

    def refine_clusters(
        self,
        embeddings,
        n_clusters: int,
        n_iterations: int = 4,
        learning_rate: float = 0.2,
    ) -> dict:
        emb = _f32_matrix(embeddings)
        if emb.ndim != 2:
            raise ValueError(f"sleep_cluster_refiner expects rank-2 embeddings, got {emb.shape}")
        rows, dims = emb.shape
        if rows == 0:
            return {
                "assignments": Int32Vector.zeros(0),
                "centroids": HostTensorF32.zeros(0, dims),
                "cluster_counts": Int32Vector.zeros(0),
                "silhouette_scores": Float32Vector.zeros(0),
                "mean_silhouette": 0.0,
                "refined_embeddings": HostTensorF32.zeros(0, dims),
            }
        cluster_target = max(1, min(int(n_clusters), rows))
        if cluster_target == 1:
            assignments = Int32Vector.zeros(rows)
            centroid_row = [
                _mean([float(emb[row, col]) for row in range(rows)])
                for col in range(dims)
            ]
            centroids = _stack_rows([centroid_row])
            refined = self._kernels.refine_embeddings(centroids.tolist() and emb.to_nested_list(), centroids.to_nested_list(), assignments.tolist(), float(learning_rate))
            return {
                "assignments": assignments,
                "centroids": centroids,
                "cluster_counts": Int32Vector([rows]),
                "silhouette_scores": Float32Vector.zeros(rows),
                "mean_silhouette": 0.0,
                "refined_embeddings": HostTensorF32.from_array_like(refined),
            }

        rng = random.Random(0)
        seed_indices = rng.sample(list(range(rows)), k=cluster_target)
        centroids = _stack_rows([emb[idx] for idx in seed_indices])
        assignments = Int32Vector.zeros(rows)

        for _ in range(max(1, int(n_iterations))):
            similarity_rows = []
            for row in range(rows):
                emb_row = emb[row]
                similarity_rows.append([
                    _dot(emb_row, centroids[col]) for col in range(cluster_target)
                ])
            similarity = _stack_rows(similarity_rows)
            assignments_arr = self._kernels.assign_to_best_centroid(similarity.to_nested_list())
            assignments = Int32Vector(int(value) for value in assignments_arr)
            centroids_arr, counts_arr = self._kernels.accumulate_centroids(
                emb.to_nested_list(), assignments.tolist(), cluster_target
            )
            centroids = HostTensorF32.from_array_like(centroids_arr)
            counts = Int32Vector(int(value) for value in counts_arr)
            empty = [idx for idx, value in enumerate(counts) if int(value) <= 0]
            for idx in empty:
                replacement = emb[rng.randrange(rows)]
                for col in range(dims):
                    centroids._buffer[idx * dims + col] = float(replacement[col])

        refined_arr = self._kernels.refine_embeddings(
            emb.to_nested_list(), centroids.to_nested_list(), assignments.tolist(), float(learning_rate)
        )
        refined = HostTensorF32.from_array_like(refined_arr)
        silhouettes_arr = self._kernels.compute_silhouette_scores(
            refined.to_nested_list(), assignments.tolist(), cluster_target
        )
        silhouettes = Float32Vector(float(value) for value in silhouettes_arr)
        _, counts_arr = self._kernels.accumulate_centroids(
            refined.to_nested_list(), assignments.tolist(), cluster_target
        )
        counts = Int32Vector(int(value) for value in counts_arr)
        mean_silhouette = silhouettes.mean() if silhouettes.size else 0.0
        return {
            "assignments": assignments,
            "centroids": centroids,
            "cluster_counts": counts,
            "silhouette_scores": silhouettes,
            "mean_silhouette": mean_silhouette,
            "refined_embeddings": refined,
        }


class SleepGlyphConsolidator:
    """Sovereign glyph consolidator backed by sleep_glyph_consolidator.ptx."""

    def __init__(self):
        from knowledge3d.cranium.ptx_runtime.sleep_glyph_kernels import SleepGlyphKernels

        self._kernels = SleepGlyphKernels()

    def consolidate_glyphs(
        self,
        glyph_embeddings,
        similarity_threshold: float = 0.92,
    ) -> dict:
        emb = _f32_matrix(glyph_embeddings)
        if emb.ndim != 2:
            raise ValueError(f"sleep_glyph_consolidator expects rank-2 embeddings, got {emb.shape}")
        rows, dims = emb.shape
        if rows == 0:
            return {
                "assignments": Int32Vector.zeros(0),
                "group_count": 0,
                "group_sizes": [],
                "embeddings_shape": [0, dims],
            }
        assignments_arr = self._kernels.cluster_by_similarity(emb.to_nested_list(), float(similarity_threshold))
        assignments = Int32Vector(int(value) for value in assignments_arr)
        if assignments.size == 0:
            return {
                "assignments": assignments,
                "group_count": 0,
                "group_sizes": [],
                "embeddings_shape": [rows, dims],
            }
        counts_map: dict[int, int] = {}
        for value in assignments:
            counts_map[int(value)] = counts_map.get(int(value), 0) + 1
        return {
            "assignments": assignments,
            "group_count": len(counts_map),
            "group_sizes": [int(value) for _, value in sorted(counts_map.items())],
            "embeddings_shape": [rows, dims],
        }


class MultimodalHaltingGate:
    """Sovereign Multimodal Halting Gate - Geometry-aware halting"""

    def __init__(self):
        ptx_path = KERNELS_DIR / "gre_multimodal_halting_gate.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "gre_multimodal_halting_gate")

    def check_halt(self, logits, masks, threshold: float = 0.5):
        """Check halting conditions with modality masks
        
        Args:
            logits: Halting logits (float32)
            masks: Modality bitmasks (uint32, 0=inactive)
            threshold: Halting threshold
        
        Returns:
            Halt flags (uint32: 1=continue, 0=halt)
        """
        logits_arr = _f32_vector(logits)
        masks_arr = _u32_vector(masks)
        assert logits_arr.shape == masks_arr.shape
        return UInt32Vector(
            [
                0 if mask == 0 else (1 if float(logit) >= float(threshold) else 0)
                for logit, mask in zip(logits_arr.tolist(), masks_arr.tolist())
            ]
        )

    def analyze_scores(
        self,
        scores,
        candidate_hashes,
        *,
        minimum_threshold: float,
        gap_threshold: float,
        agreement_threshold: float,
    ):
        """Compute halting flags directly from raw path scores and candidate hashes.

        Args:
            scores: Raw path scores (float32)
            candidate_hashes: Candidate identity hashes (uint32)
            minimum_threshold: Minimum top-score threshold. <= 0 disables.
            gap_threshold: Winner-gap threshold. <= 0 disables.
            agreement_threshold: Agreement-count threshold. <= 0 disables.

        Returns:
            tuple[array_like, array_like]: (flags[4], metrics[3]) where metrics are
            [top_score, score_gap, agreement_count].
        """
        score_arr = _f32_vector(scores)
        hash_arr = _u32_vector(candidate_hashes)
        if score_arr.shape != hash_arr.shape:
            raise ValueError(
                f"score/hash shape mismatch: {score_arr.shape} != {hash_arr.shape}"
            )

        flags = UInt32Vector.zeros(4)
        metrics = Float32Vector.zeros(3)
        if score_arr.size == 0:
            return flags, metrics

        d_scores = gpu_malloc(score_arr.nbytes)
        d_hashes = gpu_malloc(hash_arr.nbytes)
        d_flags = gpu_malloc(flags.nbytes)
        d_metrics = gpu_malloc(metrics.nbytes)
        try:
            _copy_htod(d_scores, score_arr)
            _copy_htod(d_hashes, hash_arr)
            launch(
                self.kernel,
                grid=(1, 1, 1),
                block=(1, 1, 1),
                params=[
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_uint64(d_hashes.value),
                    ctypes.c_uint64(d_flags.value),
                    ctypes.c_uint64(d_metrics.value),
                    ctypes.c_uint32(int(score_arr.size)),
                    ctypes.c_float(float(minimum_threshold)),
                    ctypes.c_float(float(gap_threshold)),
                    ctypes.c_float(float(agreement_threshold)),
                ],
            )
            synchronize()
            _copy_dtoh(flags, d_flags)
            _copy_dtoh(metrics, d_metrics)
            return flags, metrics
        finally:
            gpu_free(d_scores)
            gpu_free(d_hashes)
            gpu_free(d_flags)
            gpu_free(d_metrics)


class ModularRPNEngine:
    """Sovereign Modular RPN Engine - GPU-native RPN execution

    Uses modular_rpn_kernel.ptx for geometric and semantic computations.
    Supports 15 parallel instances with 64-deep stacks (float4 elements).

    Operations:
        - Literals: scalar (op 0), vector (op 1)
        - Arithmetic: add(10), sub(11), mul(12), div(13), pow(14), neg(15)
        - Advanced: sqrt(20), exp(21), log(22), sin(24), cos(25), tan(26)
        - Comparison: gt(40), lt(42), eq(44), max(46), min(47)
        - Stack: dup(50), swap(51), drop(52), over(53), rot(54), clear(55)
        - Vector: dot(60), cross(61), mag(62), norm(63), rotate(70), scale(71), translate(72)
        - Conditional: ifelse(80)

    Example:
        engine = ModularRPNEngine()
        result = engine.execute_single(instance_id=0, op_codes=[0, 0, 10], scalars=[2.0, 3.0, 0.0], vectors=[[0.0, 0.0, 0.0]] * 3)
        # result = 5.0
    """

    MAX_INSTANCES = 18  # Tesla 3-6-9: 18/3=6 (ternary resonance)
    STACK_DEPTH = 69    # Tesla 6-9: 6+9=15→6, 6×9=54→9, Yin-Yang balance
    INSTANCE_STRIDE = 1040  # bytes per instance state

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "modular_rpn_kernel.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"RPN PTX kernel not found: {ptx_path}")

        self.module = load_module_from_file(str(ptx_path))
        self.kernel = get_function(self.module, "modular_rpn_geometric_kernel")
        self.extract_kernel = get_function(self.module, "modular_rpn_extract_top")

        # Allocate persistent state buffer (18 instances × 1040 bytes, Tesla 3-6-9 resonance)
        total_bytes = self.MAX_INSTANCES * self.INSTANCE_STRIDE
        self.d_state = gpu_malloc(total_bytes)
        self.d_galaxy_entries: Optional[CUdeviceptr] = None
        self.d_query_embeddings = gpu_malloc(self.MAX_INSTANCES * 16 * 4)
        self._galaxy_entry_count = 0
        self._galaxy_entry_stride = 19
        self._galaxy_embedding_dim = 16
        self._galaxy_embedding_offset = 3

        # Zero-initialize state buffer using ctypes (no NumPy)
        ZerosArray = ctypes.c_uint8 * total_bytes
        zeros = ZerosArray()
        memcpy_htod(self.d_state, ctypes.cast(zeros, ctypes.c_void_p), total_bytes)
        QueryZerosArray = ctypes.c_float * (self.MAX_INSTANCES * 16)
        query_zeros = QueryZerosArray()
        memcpy_htod(self.d_query_embeddings, ctypes.cast(query_zeros, ctypes.c_void_p), ctypes.sizeof(query_zeros))
        self._bind_runtime_globals()

    def _set_module_global(self, symbol_name: str, value, ctype) -> None:
        symbol_ptr, symbol_size = get_global(self.module, symbol_name)
        payload = ctype(value)
        copy_size = min(symbol_size, ctypes.sizeof(payload))
        memcpy_htod(symbol_ptr, ctypes.cast(ctypes.byref(payload), ctypes.c_void_p), copy_size)

    def _bind_runtime_globals(self) -> None:
        galaxy_ptr = int(self.d_galaxy_entries.value) if self.d_galaxy_entries is not None else 0
        self._set_module_global("g_galaxy_entries_ptr", galaxy_ptr, ctypes.c_uint64)
        self._set_module_global("g_galaxy_entry_count", int(self._galaxy_entry_count), ctypes.c_uint32)
        self._set_module_global("g_galaxy_entry_stride", int(self._galaxy_entry_stride), ctypes.c_uint32)
        self._set_module_global("g_galaxy_embedding_dim", int(self._galaxy_embedding_dim), ctypes.c_uint32)
        self._set_module_global("g_galaxy_embedding_offset", int(self._galaxy_embedding_offset), ctypes.c_uint32)
        self._set_module_global("g_query_embedding_ptr", int(self.d_query_embeddings.value), ctypes.c_uint64)
        self._set_module_global("g_query_embedding_stride", 16, ctypes.c_uint32)

    def bind_galaxy_buffer(
        self,
        flat_entries: Sequence[float],
        *,
        entry_count: int,
        entry_stride: int = 19,
        embedding_offset: int = 3,
        embedding_dim: int = 16,
    ) -> dict[str, int]:
        if entry_count < 0:
            raise ValueError("entry_count must be non-negative")
        expected = int(entry_count) * int(entry_stride)
        host_ptr: ctypes.c_void_p | None = None
        host_nbytes = 0
        data_len = 0
        if (
            hasattr(flat_entries, "dtype")
            and hasattr(flat_entries, "ctypes")
            and hasattr(flat_entries, "reshape")
            and str(getattr(flat_entries, "dtype", "")) == "float32"
        ):
            try:
                flat_array = flat_entries.reshape(-1)
                data_len = int(flat_array.size)
                host_nbytes = int(flat_array.nbytes)
                host_ptr = flat_array.ctypes.data_as(ctypes.c_void_p)
            except Exception:
                host_ptr = None
                data_len = 0
                host_nbytes = 0
        if host_ptr is None:
            data = [float(value) for value in flat_entries]
            data_len = len(data)
            if data:
                FloatArray = ctypes.c_float * len(data)
                host = FloatArray(*data)
                host_ptr = ctypes.cast(host, ctypes.c_void_p)
                host_nbytes = ctypes.sizeof(host)
            else:
                host_nbytes = 0
        if expected != data_len:
            raise ValueError(f"Galaxy buffer length mismatch: expected {expected}, got {data_len}")

        if self.d_galaxy_entries is not None:
            gpu_free(self.d_galaxy_entries)
            self.d_galaxy_entries = None
        if data_len > 0 and host_ptr is not None and host_nbytes > 0:
            self.d_galaxy_entries = gpu_malloc(host_nbytes)
            memcpy_htod(self.d_galaxy_entries, host_ptr, host_nbytes)
        self._galaxy_entry_count = int(entry_count)
        self._galaxy_entry_stride = int(entry_stride)
        self._galaxy_embedding_offset = int(embedding_offset)
        self._galaxy_embedding_dim = int(embedding_dim)
        self._bind_runtime_globals()
        return {
            "entry_count": self._galaxy_entry_count,
            "entry_stride": self._galaxy_entry_stride,
            "embedding_offset": self._galaxy_embedding_offset,
            "embedding_dim": self._galaxy_embedding_dim,
        }

    def store_embedding(
        self,
        *,
        instance_id: int,
        embedding: Sequence[float],
        slot: int = 0,
    ) -> None:
        if slot != 0:
            raise ValueError("Only query embedding slot 0 is currently supported")
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id}")
        values = [float(value) for value in embedding]
        if len(values) != 16:
            raise ValueError(f"Query embedding must have 16 values, got {len(values)}")
        FloatArray = ctypes.c_float * len(values)
        host = FloatArray(*values)
        offset = instance_id * len(values) * 4
        memcpy_htod(
            ctypes.c_void_p(self.d_query_embeddings.value + offset),
            ctypes.cast(host, ctypes.c_void_p),
            ctypes.sizeof(host),
        )

    def execute_single(
        self,
        instance_id: int,
        op_codes: Sequence[int],
        scalars: Sequence[float],
        vectors: Sequence[Sequence[float]],
    ) -> float:
        """Execute single RPN program on specified instance

        Args:
            instance_id: Instance slot (0-14)
            op_codes: RPN operation codes (uint16 array)
            scalars: Scalar literal pool (float32 array)
            vectors: Vector literal pool (float32 array, shape N×3)

        Returns:
            Result from top of stack (float32 scalar)
        """
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id} (must be 0-14)")

        # Prepare inputs as ctypes arrays (no NumPy on hot path)
        op_list = [int(o) for o in op_codes]
        OpArray = ctypes.c_uint16 * len(op_list)
        op_arr = OpArray(*op_list)

        scalar_list = [float(s) for s in scalars]
        ScalarArray = ctypes.c_float * len(scalar_list) if scalar_list else ctypes.c_float * 1
        scalar_arr = ScalarArray(*scalar_list) if scalar_list else None

        flat_vec: List[float] = [float(c) for vec in vectors for c in vec]
        VecArray = ctypes.c_float * len(flat_vec) if flat_vec else ctypes.c_float * 1
        vec_arr = VecArray(*flat_vec) if flat_vec else None

        # Allocate GPU memory
        d_op_codes = gpu_malloc(ctypes.sizeof(op_arr))
        d_scalars = gpu_malloc(ctypes.sizeof(scalar_arr)) if scalar_arr is not None else None
        d_vectors = gpu_malloc(ctypes.sizeof(vec_arr)) if vec_arr is not None else None

        try:
            # Copy inputs to GPU
            memcpy_htod(d_op_codes, ctypes.cast(op_arr, ctypes.c_void_p), ctypes.sizeof(op_arr))
            if d_scalars is not None and scalar_arr is not None:
                memcpy_htod(d_scalars, ctypes.cast(scalar_arr, ctypes.c_void_p), ctypes.sizeof(scalar_arr))
            if d_vectors is not None and vec_arr is not None and len(flat_vec):
                memcpy_htod(d_vectors, ctypes.cast(vec_arr, ctypes.c_void_p), ctypes.sizeof(vec_arr))

            # Launch kernel
            launch(
                self.kernel,
                grid=(RPN_GRID_DIM, 1, 1),
                block=(TIER2_BLOCK_DIM, 1, 1),
                params=[
                    ctypes.c_uint32(instance_id),
                    ctypes.c_uint64(d_op_codes.value),
                    ctypes.c_uint64(d_scalars.value if d_scalars is not None else 0),
                    ctypes.c_uint64(d_vectors.value if d_vectors is not None else 0),
                    ctypes.c_uint64(self.d_state.value),
                    ctypes.c_uint32(len(op_codes)),
                ],
            )
            synchronize()

            # Read result from instance stack (top element)
            # Stack layout: header (16 bytes: head, size, error, reserved) + stack[64] (64 × 16 bytes of float4)
            instance_offset = instance_id * self.INSTANCE_STRIDE

            # First, read head and size to find stack top
            HeaderArray = ctypes.c_uint32 * 4
            header_bytes = HeaderArray()
            memcpy_dtoh(
                ctypes.cast(header_bytes, ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + instance_offset),
                16,
            )

            head = int(header_bytes[0])
            size = int(header_bytes[1])
            error_code = int(header_bytes[2])

            if error_code != 0:
                raise RuntimeError(f"RPN execution error: code {error_code}")

            if size == 0:
                raise RuntimeError("RPN stack underflow - no result available")

            # Calculate position of top element
            # Stack top is at (head + size - 1) & 63
            stack_top_index = (head + size - 1) & 63

            # Read float4 from stack[stack_top_index]
            stack_base_offset = instance_offset + 16
            element_offset = stack_base_offset + (stack_top_index * 16)  # 16 bytes per float4

            ResultArray = ctypes.c_float * 4
            result_bytes = ResultArray()
            memcpy_dtoh(
                ctypes.cast(result_bytes, ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + element_offset),
                16,
            )

            return float(result_bytes[0])

        finally:
            gpu_free(d_op_codes)
            if d_scalars is not None:
                gpu_free(d_scalars)
            if d_vectors is not None:
                gpu_free(d_vectors)

    def execute_batch(
        self,
        programs: List[dict],
        max_instances: int = 15
    ) -> List[float]:
        """Execute batch of RPN programs in parallel across instances

        Args:
            programs: List of dicts with keys 'op_codes', 'scalars', 'vectors'
            max_instances: Max parallel instances (default 15)

        Returns:
            List of results (length = len(programs))
        """
        results: List[float] = []

        # Process in batches of max_instances
        for batch_start in range(0, len(programs), max_instances):
            batch = programs[batch_start:batch_start + max_instances]

            # Execute programs sequentially (kernel is single-threaded per instance)
            for i, program in enumerate(batch):
                result = self.execute_single(
                    instance_id=i,
                    op_codes=program["op_codes"],
                    scalars=program["scalars"],
                    vectors=program["vectors"],
                )
                results.append(result)

        return results

    def execute_batch_device(
        self,
        programs: List[dict],
    ) -> tuple[CUdeviceptr, int]:
        """Execute batch of RPN programs and write results to a device buffer.

        Returns (device_pointer, count). Caller owns the device buffer and must free it.
        """
        count = len(programs)
        if count == 0:
            return CUdeviceptr(0), 0

        d_out = gpu_malloc(count * 4)

        # Process sequentially per instance slot (reusing instance 0..MAX_INSTANCES-1)
        for i, program in enumerate(programs):
            instance_id = i % self.MAX_INSTANCES
            self.reset_instance(instance_id)
            op_list = [int(o) for o in program["op_codes"]]
            OpArray = ctypes.c_uint16 * len(op_list)
            op_arr = OpArray(*op_list)

            scalar_list = [float(s) for s in program["scalars"]]
            ScalarArray = ctypes.c_float * len(scalar_list) if scalar_list else ctypes.c_float * 1
            scalar_arr = ScalarArray(*scalar_list) if scalar_list else None

            flat_vec: List[float] = [float(c) for vec in program["vectors"] for c in vec]
            VecArray = ctypes.c_float * len(flat_vec) if flat_vec else ctypes.c_float * 1
            vec_arr = VecArray(*flat_vec) if flat_vec else None

            d_op_codes = gpu_malloc(ctypes.sizeof(op_arr))
            d_scalars = gpu_malloc(ctypes.sizeof(scalar_arr)) if scalar_arr is not None else None
            d_vectors = gpu_malloc(ctypes.sizeof(vec_arr)) if vec_arr is not None else None

            try:
                memcpy_htod(d_op_codes, ctypes.cast(op_arr, ctypes.c_void_p), ctypes.sizeof(op_arr))
                if d_scalars is not None and scalar_arr is not None:
                    memcpy_htod(d_scalars, ctypes.cast(scalar_arr, ctypes.c_void_p), ctypes.sizeof(scalar_arr))
                if d_vectors is not None and vec_arr is not None and flat_vec:
                    memcpy_htod(d_vectors, ctypes.cast(vec_arr, ctypes.c_void_p), ctypes.sizeof(vec_arr))

                launch(
                    self.kernel,
                    grid=(RPN_GRID_DIM, 1, 1),
                    block=(TIER2_BLOCK_DIM, 1, 1),
                    params=[
                        ctypes.c_uint32(instance_id),
                        ctypes.c_uint64(d_op_codes.value),
                        ctypes.c_uint64(d_scalars.value if d_scalars is not None else 0),
                        ctypes.c_uint64(d_vectors.value if d_vectors is not None else 0),
                        ctypes.c_uint64(self.d_state.value),
                        ctypes.c_uint32(len(op_list)),
                    ],
                )
                launch(
                    self.extract_kernel,
                    grid=(1, 1, 1),
                    block=(1, 1, 1),
                    params=[
                        ctypes.c_uint32(instance_id),
                        ctypes.c_uint64(self.d_state.value),
                        ctypes.c_uint64(d_out.value),
                        ctypes.c_uint32(i),
                    ],
                )
            finally:
                gpu_free(d_op_codes)
                if d_scalars is not None:
                    gpu_free(d_scalars)
                if d_vectors is not None:
                    gpu_free(d_vectors)

        synchronize()
        return d_out, count

    def read_instance_stack_scalars(self, instance_id: int) -> List[float]:
        """Return scalar values from one instance stack in logical bottom->top order."""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id}")

        instance_offset = instance_id * self.INSTANCE_STRIDE
        HeaderArray = ctypes.c_uint32 * 4
        header_bytes = HeaderArray()
        memcpy_dtoh(
            ctypes.cast(header_bytes, ctypes.c_void_p),
            ctypes.c_void_p(self.d_state.value + instance_offset),
            16,
        )

        head = int(header_bytes[0])
        size = int(header_bytes[1])
        error_code = int(header_bytes[2])

        if error_code != 0:
            raise RuntimeError(f"RPN execution error: code {error_code}")
        if size <= 0:
            return []

        stack_base_offset = instance_offset + 16
        ResultArray = ctypes.c_float * 4
        scalars: List[float] = []
        for logical_index in range(size):
            stack_index = (head + logical_index) & 63
            element_offset = stack_base_offset + (stack_index * 16)
            result_bytes = ResultArray()
            memcpy_dtoh(
                ctypes.cast(result_bytes, ctypes.c_void_p),
                ctypes.c_void_p(self.d_state.value + element_offset),
                16,
            )
            scalars.append(float(result_bytes[0]))
        return scalars

    def reset_instance(self, instance_id: int):
        """Reset instance state (clear stack, reset head/size)"""
        if not (0 <= instance_id < self.MAX_INSTANCES):
            raise ValueError(f"Invalid instance_id: {instance_id}")

        # Zero out instance state
        instance_offset = instance_id * self.INSTANCE_STRIDE
        ZerosArray = ctypes.c_uint8 * self.INSTANCE_STRIDE
        zeros = ZerosArray()
        memcpy_htod(
            ctypes.c_void_p(self.d_state.value + instance_offset),
            ctypes.cast(zeros, ctypes.c_void_p),
            self.INSTANCE_STRIDE,
        )

    def cleanup(self):
        """Free GPU memory"""
        if self.d_galaxy_entries is not None:
            gpu_free(self.d_galaxy_entries)
            self.d_galaxy_entries = None
        gpu_free(self.d_query_embeddings)
        gpu_free(self.d_state)

    def __del__(self):
        try:
            self.cleanup()
        except:
            pass


class GalaxyMemoryUpdater:
    """Sovereign Galaxy Memory Updater - Blend embeddings on GPU

    Uses galaxy_memory_updater.ptx to blend old and teacher embeddings
    with exponential moving average (EMA) on GPU.

    Formula: new = old * (1 - blend_factor) + teacher * blend_factor

    Example:
        updater = GalaxyMemoryUpdater()
        old_emb = [1.0, 2.0, 3.0]
        teacher_emb = [4.0, 5.0, 6.0]
        new_emb = updater.blend(old_emb, teacher_emb, blend_factor=0.3)
        # new_emb ≈ [1.9, 2.9, 3.9]
    """

    def __init__(self):
        ptx_path = Path(__file__).parent.parent / "ptx" / "galaxy_memory_updater.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(f"Galaxy memory PTX kernel not found: {ptx_path}")

        self.kernel = load_ptx_file(str(ptx_path), "update_star_embedding_kernel")

    def blend(self, old, teacher, blend_factor: float):
        """Blend old and teacher embeddings with GPU acceleration.

        Args:
            old: Old embedding (float32 array)
            teacher: Teacher embedding (float32 array, same shape as old)
            blend_factor: Blend factor (0.0 = keep old, 1.0 = use teacher)

        Returns:
            Blended embedding (float32 array, same shape as inputs)
        """
        # Prepare inputs
        old_arr = _f32_vector(old)
        teacher_arr = _f32_vector(teacher)

        if old_arr.shape != teacher_arr.shape:
            raise ValueError(f"Shape mismatch: old {old_arr.shape} vs teacher {teacher_arr.shape}")

        dim = int(old_arr.size)
        if dim == 0:
            return Float32Vector.zeros(0)

        # Allocate GPU memory
        d_old = gpu_malloc(old_arr.nbytes)
        d_teacher = gpu_malloc(teacher_arr.nbytes)
        d_out = gpu_malloc(old_arr.nbytes)

        try:
            # Copy inputs to GPU
            _copy_htod(d_old, old_arr)
            _copy_htod(d_teacher, teacher_arr)

            # Launch kernel
            threads = 256
            blocks = (dim + threads - 1) // threads

            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_old.value),
                    ctypes.c_uint64(d_teacher.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_float(blend_factor),
                    ctypes.c_uint32(dim),
                ],
            )
            synchronize()

            # Copy result back
            output = Float32Vector.zeros(dim)
            _copy_dtoh(output, d_out)

            return output

        finally:
            gpu_free(d_old)
            gpu_free(d_teacher)
            gpu_free(d_out)

    def blend_sequence(
        self,
        base,
        teachers: list,
        blend_factor: float = 0.3
    ):
        """Blend base embedding with sequence of teacher embeddings.

        Args:
            base: Base embedding (float32 array)
            teachers: List of teacher embeddings
            blend_factor: Blend factor for each step

        Returns:
            Final blended embedding
        """
        out = _f32_vector(base)
        if not teachers:
            return out

        for teacher in teachers:
            out = self.blend(out, teacher, blend_factor)

        return out


# Update __all__
__all__ = [
    # Kimi's
    "LatencyGuard",
    "ARCReasoner",
    "OOMSpillManager",
    # Qwen's
    "GalaxyResonanceEngine",
    # Deep Seek's
    "GeometryRouter",
    "FractalEmitter",
    "CognitiveExecutive",
    # GLM's
    "ResonanceField",
    "AtomicFissionFusion",
    "DefeasibleResolver",
    "TemporalReasoning",
    # Grok's
    "VectorResonator",
    "GraphCrystallizer",
    "MultimodalHaltingGate",
    # Runtime Engines
    "ModularRPNEngine",
    "GalaxyMemoryUpdater",
    # GLM's World Model
    "WorldModelBridge",
]


class WorldModelBridge:
    """
    Sovereign bridge for world model operations.
    Enables temporal coherence, multi-modal fusion, and dynamic mesh generation.
    
    GLM's World Model Integration - Multi-modal temporal generation.
    """
    def __init__(self):
        from pathlib import Path
        ptx_dir = Path(__file__).parent.parent / "ptx"
        
        # Load world model kernels
        self.temporal_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "compute_temporal_coherence"
        )
        self.fusion_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "fuse_multimodal_features"
        )
        self.prediction_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "predict_world_state"
        )
        self.dynamic_mesh_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "generate_dynamic_mesh"
        )
        self.resonance_kernel = load_ptx_file(
            str(ptx_dir / "gre_world_model.ptx"),
            "enhance_galaxy_resonance"
        )
    
    def compute_temporal_coherence(
        self,
        frame_features,  # (N_frames * feature_dim,) flattened
        n_frames: int,
        feature_dim: int
    ):
        """Compute temporal coherence scores across video frames."""
        # Allocate GPU memory
        features = _f32_vector(frame_features)
        d_features = gpu_malloc(features.nbytes)
        d_coherence = gpu_malloc(feature_dim * 4)  # float32
        
        try:
            # Copy to GPU
            _copy_htod(d_features, features)
            
            # Launch kernel
            threads = 256
            blocks = (feature_dim + threads - 1) // threads
            
            launch(
                self.temporal_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_features.value),
                    ctypes.c_uint64(d_coherence.value),
                    ctypes.c_int32(n_frames),
                    ctypes.c_int32(feature_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            coherence = Float32Vector.zeros(feature_dim)
            _copy_dtoh(coherence, d_coherence)
            
            return coherence
        
        finally:
            gpu_free(d_features)
            gpu_free(d_coherence)
    
    def fuse_multimodal_features(
        self,
        text_features,
        visual_features,
        text_weight: float = 0.5
    ):
        """Fuse text and visual features with attention weighting."""
        text = _f32_vector(text_features)
        visual = _f32_vector(visual_features)
        feature_dim = len(text)
        visual_weight = 1.0 - text_weight
        
        # Allocate GPU memory
        d_text = gpu_malloc(text.nbytes)
        d_visual = gpu_malloc(visual.nbytes)
        d_weights = gpu_malloc(8)  # 2 floats
        d_fused = gpu_malloc(text.nbytes)
        
        try:
            # Copy to GPU
            _copy_htod(d_text, text)
            _copy_htod(d_visual, visual)
            weights = Float32Vector([text_weight, visual_weight])
            _copy_htod(d_weights, weights)
            
            # Launch kernel
            threads = 256
            blocks = (feature_dim + threads - 1) // threads
            
            launch(
                self.fusion_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_text.value),
                    ctypes.c_uint64(d_visual.value),
                    ctypes.c_uint64(d_weights.value),
                    ctypes.c_uint64(d_fused.value),
                    ctypes.c_int32(feature_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            fused = Float32Vector.zeros(feature_dim)
            _copy_dtoh(fused, d_fused)
            
            return fused
        
        finally:
            gpu_free(d_text)
            gpu_free(d_visual)
            gpu_free(d_weights)
            gpu_free(d_fused)
    
    def predict_world_state(
        self,
        current_state,
        action_vector
    ):
        """Predict next world state given current state and action."""
        current = _f32_vector(current_state)
        action = _f32_vector(action_vector)
        state_dim = len(current)
        action_dim = len(action)
        
        # Allocate GPU memory
        d_current = gpu_malloc(current.nbytes)
        d_action = gpu_malloc(action.nbytes)
        d_predicted = gpu_malloc(current.nbytes)
        
        try:
            # Copy to GPU
            _copy_htod(d_current, current)
            _copy_htod(d_action, action)
            
            # Launch kernel
            threads = 256
            blocks = (state_dim + threads - 1) // threads
            
            launch(
                self.prediction_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_current.value),
                    ctypes.c_uint64(d_action.value),
                    ctypes.c_uint64(d_predicted.value),
                    ctypes.c_int32(state_dim),
                    ctypes.c_int32(action_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            predicted = Float32Vector.zeros(state_dim)
            _copy_dtoh(predicted, d_predicted)
            
            return predicted
        
        finally:
            gpu_free(d_current)
            gpu_free(d_action)
            gpu_free(d_predicted)
    
    def generate_dynamic_mesh(
        self,
        world_state,
        base_vertices  # (N, 3)
    ):
        """Generate dynamic mesh based on world state."""
        base_shape = _shape_of(base_vertices)
        state = _f32_vector(world_state)
        vertex_count = int(base_shape[0])
        state_dim = len(state)
        vertices_flat = _f32_vector(base_vertices)
        
        # Allocate GPU memory
        d_state = gpu_malloc(state.nbytes)
        d_base = gpu_malloc(vertices_flat.nbytes)
        d_dynamic = gpu_malloc(vertices_flat.nbytes)
        
        try:
            # Copy to GPU
            _copy_htod(d_state, state)
            _copy_htod(d_base, vertices_flat)
            
            # Launch kernel
            threads = 256
            blocks = (vertex_count + threads - 1) // threads
            
            launch(
                self.dynamic_mesh_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_state.value),
                    ctypes.c_uint64(d_base.value),
                    ctypes.c_uint64(d_dynamic.value),
                    ctypes.c_int32(vertex_count),
                    ctypes.c_int32(state_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            dynamic_flat = Float32Vector.zeros(vertices_flat.size)
            _copy_dtoh(dynamic_flat, d_dynamic)

            return HostTensorF32.from_array_like(dynamic_flat.tolist(), rows=base_shape[0], cols=base_shape[1])
        
        finally:
            gpu_free(d_state)
            gpu_free(d_base)
            gpu_free(d_dynamic)
    
    def enhance_galaxy_resonance(
        self,
        query_embedding,
        galaxy_embeddings  # (N, embedding_dim)
    ):
        """Enhance galaxy query with temperature-scaled similarity."""
        query_vec = _f32_vector(query_embedding)
        galaxy_shape = _shape_of(galaxy_embeddings)
        n_embeddings = galaxy_shape[0]
        embedding_dim = galaxy_shape[1]
        galaxy_flat = _f32_vector(galaxy_embeddings)
        
        # Allocate GPU memory
        d_query = gpu_malloc(query_vec.nbytes)
        d_galaxy = gpu_malloc(galaxy_flat.nbytes)
        d_resonance = gpu_malloc(n_embeddings * 4)  # float32
        
        try:
            # Copy to GPU
            _copy_htod(d_query, query_vec)
            _copy_htod(d_galaxy, galaxy_flat)
            
            # Launch kernel
            threads = 256
            blocks = (n_embeddings + threads - 1) // threads
            
            launch(
                self.resonance_kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_query.value),
                    ctypes.c_uint64(d_galaxy.value),
                    ctypes.c_uint64(d_resonance.value),
                    ctypes.c_int32(n_embeddings),
                    ctypes.c_int32(embedding_dim),
                ],
            )
            synchronize()
            
            # Copy result back
            resonance = Float32Vector.zeros(n_embeddings)
            _copy_dtoh(resonance, d_resonance)
            
            return resonance
        
        finally:
            gpu_free(d_query)
            gpu_free(d_galaxy)
            gpu_free(d_resonance)


# ============================================================================
# Trit Overlay + Inspector (Balanced Ternary Diagnostics)
# ============================================================================


class TritOverlayGenerator:
    """Generate RGBA8 overlays from packed ternary fields."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "trit_overlay_generator.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "trit_overlay_generator")
        self.guard = LatencyGuard(threshold_us=500.0)

    def generate(
        self,
        trits_packed,
        grid_shape: Tuple[int, int, int],
        field_stride: int,
        field_type: int = 0,
        threshold: float = 0.0,
    ):
        """Render ternary field overlay to RGBA8."""
        gx, gy, gz = (int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2]))
        trits = _u32_vector(trits_packed)
        rgba = UInt8Vector.zeros(gx * gy * gz * 4)

        d_trits = gpu_malloc(trits.nbytes)
        d_rgba = gpu_malloc(rgba.nbytes)
        try:
            _copy_htod(d_trits, trits)
            self.guard.start()
            launch(
                self.kernel,
                grid=(
                    (gx + 7) // 8,
                    (gy + 7) // 8,
                    (gz + 7) // 8,
                ),
                block=(8, 8, 8),
                params=[
                    ctypes.c_uint64(d_trits.value),
                    ctypes.c_uint64(d_rgba.value),
                    ctypes.c_int32(gx),
                    ctypes.c_int32(gy),
                    ctypes.c_int32(gz),
                    ctypes.c_int32(int(field_stride)),
                    ctypes.c_int32(int(field_type)),
                    ctypes.c_float(float(threshold)),
                ],
            )
            synchronize()
            self.guard.stop()
            _copy_dtoh(rgba, d_rgba)
            return rgba
        finally:
            gpu_free(d_trits)
            gpu_free(d_rgba)


class TritInspectorBridge:
    """Inspect packed ternary fields for specific nodes."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "trit_inspector.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "trit_inspector")
        self.guard = LatencyGuard(threshold_us=500.0)

    def inspect(
        self,
        trits_packed,
        node_indices,
        field_stride: int,
    ):
        """Inspect ternary fields at node_indices."""
        trits = _u32_vector(trits_packed)
        nodes = _i32_vector(node_indices)
        n = int(nodes.shape[0])
        RecordArray = _TritInspectorRecord * max(n, 1)
        out = RecordArray()

        d_trits = gpu_malloc(trits.nbytes)
        d_nodes = gpu_malloc(nodes.nbytes)
        d_out = gpu_malloc(ctypes.sizeof(out))
        try:
            _copy_htod(d_trits, trits)
            _copy_htod(d_nodes, nodes)
            self.guard.start()
            threads = 128
            blocks = (n + threads - 1) // threads
            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_trits.value),
                    ctypes.c_uint64(d_nodes.value),
                    ctypes.c_int32(n),
                    ctypes.c_int32(int(field_stride)),
                    ctypes.c_uint64(d_out.value),
                ],
            )
            synchronize()
            self.guard.stop()
            _copy_dtoh(out, d_out)
            counts: list[int] = []
            sums: list[int] = []
            means: list[float] = []
            variances: list[float] = []
            bottlenecks: list[int] = []
            for idx in range(n):
                counts.append(int(out[idx].count))
                sums.append(int(out[idx].sum))
                means.append(float(out[idx].mean))
                variances.append(float(out[idx].var))
                bottlenecks.append(int(out[idx].bottlenecks))
            return TritInspectionBatch(counts, sums, means, variances, bottlenecks)
        finally:
            gpu_free(d_trits)
            gpu_free(d_nodes)
            gpu_free(d_out)


class TernaryDepthField:
    """Compute ternary attract/neutral/repel field for a query embedding."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "ternary_depth_field.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "ternary_depth_field")
        self.guard = LatencyGuard(threshold_us=500.0)

    def compute(
        self,
        embeddings,
        query,
        attract_thresh: float = 0.35,
        repel_thresh: float = -0.05,
    ):
        """Return packed 2-bit trits indicating near/neutral/far (per node)."""
        emb = _f32_matrix(embeddings)
        q = _f32_vector(query)
        n_nodes, dim = emb.shape
        assert q.shape[0] == dim, "query dim mismatch"
        n_words = (n_nodes + 15) // 16
        out_host = UInt32Vector.zeros(n_words)

        d_emb = gpu_malloc(emb.nbytes)
        d_query = gpu_malloc(q.nbytes)
        d_out = gpu_malloc(out_host.nbytes)
        try:
            _copy_htod(d_emb, emb)
            _copy_htod(d_query, q)
            # zero output buffer (host prepared zeros)
            _copy_htod(d_out, out_host)
            self.guard.start()
            threads = 128
            blocks = (n_nodes + threads - 1) // threads
            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_emb.value),
                    ctypes.c_uint64(d_query.value),
                    ctypes.c_int32(int(n_nodes)),
                    ctypes.c_int32(int(dim)),
                    ctypes.c_float(float(attract_thresh)),
                    ctypes.c_float(float(repel_thresh)),
                    ctypes.c_uint64(d_out.value),
                ],
            )
            synchronize()
            self.guard.stop()
            _copy_dtoh(out_host, d_out)
            return out_host
        finally:
            gpu_free(d_emb)
            gpu_free(d_query)
            gpu_free(d_out)


class TernaryPruneDecision:
    """Map scores to ternary keep/discard signals on GPU."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "ternary_prune_decision.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "ternary_prune_decision")
        self.guard = LatencyGuard(threshold_us=500.0)

    def decide(
        self,
        scores,
        keep_thresh: float = 0.5,
        drop_thresh: float = 0.05,
    ):
        scores_host = _f32_vector(scores)
        n = int(scores_host.shape[0])
        out = Int8Vector.zeros(n)
        d_scores = gpu_malloc(scores_host.nbytes)
        d_out = gpu_malloc(out.nbytes)
        try:
            _copy_htod(d_scores, scores_host)
            self.guard.start()
            threads = 256
            blocks = (n + threads - 1) // threads
            launch(
                self.kernel,
                grid=(blocks, 1, 1),
                block=(threads, 1, 1),
                params=[
                    ctypes.c_uint64(d_scores.value),
                    ctypes.c_uint64(d_out.value),
                    ctypes.c_int32(n),
                    ctypes.c_float(float(keep_thresh)),
                    ctypes.c_float(float(drop_thresh)),
                ],
            )
            synchronize()
            self.guard.stop()
            _copy_dtoh(out, d_out)
            return out
        finally:
            gpu_free(d_scores)
            gpu_free(d_out)


class TernaryAttentionMask:
    """Compute ternary attention masks (packed 2-bit trits) from Q·K."""

    def __init__(self):
        ptx_path = KERNELS_DIR / "ternary_attention_mask.ptx"
        self.kernel = load_ptx_file(str(ptx_path), "ternary_attention_mask")
        self.threshold_kernel = load_ptx_file(str(ptx_path), "compute_adaptive_thresholds")
        self.guard = LatencyGuard(threshold_us=500.0)

    def compute(
        self,
        Q,
        K,
        attract_thresh: float,
        repel_thresh: float,
    ):
        """Return packed ternary masks for Q·K."""
        q_shape = _shape_of(Q)
        k_shape = _shape_of(K)
        if q_shape != k_shape:
            raise ValueError(f"Q and K must match; got {q_shape} vs {k_shape}")
        batch_size, seq_len, embed_dim = q_shape
        n_words = (seq_len * seq_len + 15) // 16

        q = _f32_matrix(Q, rows=batch_size * seq_len, cols=embed_dim)
        k = _f32_matrix(K, rows=batch_size * seq_len, cols=embed_dim)
        masks = UInt32Vector.zeros(batch_size * n_words)

        d_q = gpu_malloc(q.nbytes)
        d_k = gpu_malloc(k.nbytes)
        d_masks = gpu_malloc(masks.nbytes)
        try:
            _copy_htod(d_q, q)
            _copy_htod(d_k, k)
            _copy_htod(d_masks, masks)  # zero out

            block = (1, 1, 1)
            grid = (seq_len, seq_len, batch_size)
            self.guard.start()
            launch(
                self.kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_k.value),
                    ctypes.c_uint64(d_masks.value),
                    ctypes.c_float(float(attract_thresh)),
                    ctypes.c_float(float(repel_thresh)),
                    ctypes.c_int32(int(batch_size)),
                    ctypes.c_int32(int(seq_len)),
                    ctypes.c_int32(int(embed_dim)),
                ],
            )
            synchronize()
            self.guard.stop()
            _copy_dtoh(masks, d_masks)
            return masks.reshape(batch_size, n_words)
        finally:
            gpu_free(d_q)
            gpu_free(d_k)
            gpu_free(d_masks)

    def compute_adaptive_thresholds(
        self,
        Q,
        K,
        percentile_attract: float = 75.0,
        percentile_repel: float = 25.0,
    ) -> tuple[float, float]:
        """Compute approximate thresholds per batch, return averaged attract/repel."""
        q_shape = _shape_of(Q)
        k_shape = _shape_of(K)
        if q_shape != k_shape:
            raise ValueError(f"Q and K must match; got {q_shape} vs {k_shape}")
        batch_size, seq_len, embed_dim = q_shape
        q = _f32_matrix(Q, rows=batch_size * seq_len, cols=embed_dim)
        k = _f32_matrix(K, rows=batch_size * seq_len, cols=embed_dim)
        thresholds = HostTensorF32.zeros(batch_size, 2)

        d_q = gpu_malloc(q.nbytes)
        d_k = gpu_malloc(k.nbytes)
        d_thr = gpu_malloc(thresholds.nbytes)
        try:
            _copy_htod(d_q, q)
            _copy_htod(d_k, k)
            _copy_htod(d_thr, thresholds)

            block = (256, 1, 1)
            grid = (batch_size, 1, 1)
            self.guard.start()
            launch(
                self.threshold_kernel,
                grid=grid,
                block=block,
                params=[
                    ctypes.c_uint64(d_q.value),
                    ctypes.c_uint64(d_k.value),
                    ctypes.c_uint64(d_thr.value),
                    ctypes.c_float(float(percentile_attract)),
                    ctypes.c_float(float(percentile_repel)),
                    ctypes.c_int32(int(batch_size)),
                    ctypes.c_int32(int(seq_len)),
                    ctypes.c_int32(int(embed_dim)),
                ],
            )
            synchronize()
            self.guard.stop()
            _copy_dtoh(thresholds, d_thr)
            attract = _mean([float(thresholds[row, 0]) for row in range(batch_size)])
            repel = _mean([float(thresholds[row, 1]) for row in range(batch_size)])
            return attract, repel
        finally:
            gpu_free(d_q)
            gpu_free(d_k)
            gpu_free(d_thr)
