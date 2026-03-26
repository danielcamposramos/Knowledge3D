"""
Step 14 – Specialized Nine-Chain Swarm Bridge.

Provides a high-performance Python orchestrator for the specialised chain
kernels authored in `nine_chain_specialized.cu`. The bridge keeps all heavy
math on the GPU via the sovereign loader and exposes a compact API that mirrors
the prototype bridge while returning richer diagnostics (resonance matrix,
normalised weights, per-chain norms).
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32
from knowledge3d.cranium.sovereign import loader


class Float32Vector:
    """ctypes-backed float32 vector with a NumPy-free sovereign API."""

    def __init__(self, values=(), *, size: int | None = None) -> None:
        flat = [float(value) for value in values]
        if size is None:
            size = len(flat)
        size = int(size)
        if len(flat) != size:
            raise ValueError(f"Expected {size} values, received {len(flat)}")
        self._buffer = (ctypes.c_float * size)()
        for idx, value in enumerate(flat):
            self._buffer[idx] = value

    @classmethod
    def zeros(cls, size: int) -> "Float32Vector":
        return cls(size=int(size))

    @property
    def shape(self) -> tuple[int]:
        return (len(self),)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self)

    @property
    def nbytes(self) -> int:
        return len(self) * ctypes.sizeof(ctypes.c_float)

    @property
    def data_ptr(self) -> int:
        return ctypes.addressof(self._buffer)

    @property
    def flat(self) -> list[float]:
        return self.tolist()

    def copy(self) -> "Float32Vector":
        clone = Float32Vector.zeros(self.size)
        ctypes.memmove(clone.data_ptr, self.data_ptr, self.nbytes)
        return clone

    def fill(self, value: float) -> None:
        scalar = float(value)
        for idx in range(len(self)):
            self._buffer[idx] = scalar

    def set_flat(self, values) -> None:
        flat = [float(value) for value in values]
        if len(flat) != len(self):
            raise ValueError(f"Expected {len(self)} values, received {len(flat)}")
        for idx, value in enumerate(flat):
            self._buffer[idx] = value

    def tolist(self) -> list[float]:
        return [float(self._buffer[idx]) for idx in range(len(self))]

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        for idx in range(len(self)):
            yield float(self._buffer[idx])

    def __getitem__(self, index):
        return float(self._buffer[int(index)])


@dataclass(frozen=True)
class SwarmDiagnostics:
    """Container for the latest swarm diagnostics."""

    resonance_matrix: HostTensorF32
    resonance_raw: Float32Vector
    resonance_weights: Float32Vector
    chain_states: HostTensorF32
    chain_norms: Float32Vector


class NineChainSpecializedBridge:
    """GPU-native orchestrator for the Step 14 specialised swarm kernels."""

    NUM_CHAINS = 9
    NUM_ACTIVE_CHAINS = 8
    CHAIN_DIM = 128

    _KERNEL_NAMES = {
        "ingest": "chain_ingest_kernel",
        "fuse_a": "chain_fuse_a_kernel",
        "fuse_b": "chain_fuse_b_kernel",
        "spatial_a": "chain_spatial_a_kernel",
        "spatial_b": "chain_spatial_b_kernel",
        "spatial_c": "chain_spatial_c_kernel",
        "reason_reduction": "chain_reason_reductionist_kernel",
        "reason_creative": "chain_reason_creative_kernel",
        "resonance": "compute_resonance_optimized",
        "synthesis": "chain_synthesis_kernel",
        "iteration": "swarm_iteration_kernel",
    }

    def __init__(
        self,
        chain_dim: int = CHAIN_DIM,
        resonance_strategy: str = "mean",
        normalize_weights: bool = True,
        persistent_state: bool = True,
    ) -> None:
        if chain_dim != self.CHAIN_DIM:
            raise ValueError(
                f"Specialised swarm currently requires chain_dim {self.CHAIN_DIM}, "
                f"received {chain_dim}"
            )

        self.dim = chain_dim
        self.resonance_strategy = resonance_strategy
        self.normalize_weights = normalize_weights
        self.persistent_state = persistent_state

        ptx_path = Path(__file__).parent.parent / "ptx" / "nine_chain_specialized.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(
                "PTX for specialised nine-chain swarm not found.\n"
                "Rebuild with:\n"
                "  cd knowledge3d/cranium/kernels &&\n"
                "  nvcc -ptx -arch=sm_86 -O3 --use_fast_math nine_chain_specialized.cu \\\n"
                "      -o ../ptx/nine_chain_specialized.ptx"
            )

        self._module = loader.load_module_from_file(str(ptx_path))
        self._kernels: Dict[str, loader.CUfunction] = {
            key: loader.get_function(self._module, name)
            for key, name in self._KERNEL_NAMES.items()
        }

        self._block_dim = min(256, self.dim)

        self._d_input = loader.gpu_malloc(self.dim * 4)
        self._d_active_base = loader.gpu_malloc(self.NUM_ACTIVE_CHAINS * self.dim * 4)
        self._d_chain9 = loader.gpu_malloc(self.dim * 4)

        self._chain_buffers = [
            self._offset_ptr(self._d_active_base, idx * self.dim * 4)
            for idx in range(self.NUM_ACTIVE_CHAINS)
        ] + [self._d_chain9]

        self._d_resonance_matrix = loader.gpu_malloc(self.NUM_ACTIVE_CHAINS * self.NUM_ACTIVE_CHAINS * 4)

        self._host_zero = HostTensorF32.zeros(self.dim, 1)
        self._host_active = HostTensorF32.zeros(self.NUM_ACTIVE_CHAINS, self.dim)
        self._host_chain9 = Float32Vector.zeros(self.dim)
        self._host_resonance_matrix = HostTensorF32.zeros(
            self.NUM_ACTIVE_CHAINS,
            self.NUM_ACTIVE_CHAINS,
        )
        self._host_resonance_raw = Float32Vector.zeros(self.NUM_ACTIVE_CHAINS)
        self._host_resonance_weights = Float32Vector.zeros(self.NUM_ACTIVE_CHAINS)

        self._last_diag: Optional[SwarmDiagnostics] = None
        self._diagnostics_dirty = True

        self.reset_states()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_swarm(
        self,
        input_embedding,
        num_iterations: int = 1,
        reset_state: bool = False,
        readback_mode: str = "full",
    ) -> Tuple[Float32Vector, Optional[HostTensorF32], Optional[Float32Vector]]:
        """
        Run the specialised swarm.

        Args:
            input_embedding: Vector of length 128 (ThinkingTag fusion output).
            num_iterations: Optional refinement iterations (reuses persistent state).
            reset_state: When True, zero out all chain buffers before executing.
            readback_mode: 'full' (default) copies chain states + diagnostics,
                'output' returns only the synthesised vector, and 'diagnostics'
                refreshes diagnostics without returning states.

        Returns:
            Tuple of (synthesised_output, chain_states[9, D] or None, resonance_weights[8] or None)
        """
        vec = HostTensorF32.from_array_like(input_embedding, rows=self.dim, cols=1)
        if vec.shape != (self.dim, 1):
            raise ValueError(f"input_embedding must have shape ({self.dim},), got {vec.shape}")

        if reset_state or not self.persistent_state:
            self.reset_states()

        mode = readback_mode.lower()
        if mode not in {"full", "output", "diagnostics"}:
            raise ValueError("readback_mode must be one of {'full', 'output', 'diagnostics'}")

        loader.memcpy_htod(
            self._d_input,
            ctypes.c_void_p(vec.data_ptr),
            self.dim * 4,
        )

        iterations = max(1, int(num_iterations))
        for _ in range(iterations):
            self._run_once()

        loader.memcpy_dtoh(
            ctypes.c_void_p(self._host_chain9.data_ptr),
            self._d_chain9,
            self.dim * 4,
        )

        output = self._host_chain9.copy()
        chain_states: Optional[HostTensorF32] = None
        resonance_weights: Optional[Float32Vector] = None

        if mode in {"full", "diagnostics"}:
            self._refresh_diagnostics()
            chain_states = self._last_diag.chain_states.copy()
            resonance_weights = self._last_diag.resonance_weights.copy()
            if mode == "diagnostics":
                chain_states = None
        else:
            self._diagnostics_dirty = True

        return output, chain_states, resonance_weights

    def execute_swarm_device(
        self,
        input_embedding,
        *,
        d_candidate_indices: int | None = None,
        candidate_count: int = 0,
        num_iterations: int = 1,
        reset_state: bool = False,
    ) -> tuple[int, int, int]:
        """Run the swarm while keeping outputs resident on device."""
        vec = HostTensorF32.from_array_like(input_embedding, rows=self.dim, cols=1)
        if vec.shape != (self.dim, 1):
            raise ValueError(f"input_embedding must have shape ({self.dim},), got {vec.shape}")
        if reset_state or not self.persistent_state:
            self.reset_states()
        loader.memcpy_htod(
            self._d_input,
            ctypes.c_void_p(vec.data_ptr),
            self.dim * 4,
        )
        iterations = max(1, int(num_iterations))
        for _ in range(iterations):
            self._run_once()
        loader.synchronize()
        self._diagnostics_dirty = True
        return int(self._d_chain9.value), int(self.dim), int(self._d_resonance_matrix.value)

    def get_chain_diagnostics(self) -> SwarmDiagnostics:
        """Return the most recent diagnostic snapshot."""
        if self._last_diag is None or self._diagnostics_dirty:
            self._refresh_diagnostics()
        return self._last_diag

    def reset_states(self) -> None:
        """Zero all chain buffers to clear temporal state."""
        zero_ptr = ctypes.c_void_p(self._host_zero.data_ptr)
        for ptr in self._chain_buffers:
            loader.memcpy_htod(ptr, zero_ptr, self.dim * 4)
        self._host_resonance_matrix.fill(0.0)
        loader.memcpy_htod(
            self._d_resonance_matrix,
            ctypes.c_void_p(self._host_resonance_matrix.data_ptr),
            self.NUM_ACTIVE_CHAINS * self.NUM_ACTIVE_CHAINS * 4,
        )
        self._host_resonance_raw.fill(0.0)
        self._host_resonance_weights.fill(0.0)
        self._diagnostics_dirty = True

    def cleanup(self) -> None:
        """Release GPU resources."""
        loader.gpu_free(self._d_input)
        loader.gpu_free(self._d_active_base)
        loader.gpu_free(self._d_chain9)
        loader.gpu_free(self._d_resonance_matrix)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _run_once(self) -> None:
        block = (self._block_dim, 1, 1)
        self._diagnostics_dirty = True
        loader.launch(
            self._kernels["iteration"],
            grid=(1, 1, 1),
            block=block,
            params=[
                self._as_uint64(self._d_input),
                self._as_uint64(self._d_active_base),
                self._as_uint64(self._d_chain9),
                self._as_uint64(self._d_resonance_matrix),
                ctypes.c_int32(self.dim),
            ],
        )

    def _refresh_diagnostics(self) -> None:
        """Read back chain states and resonance diagnostics into host memory."""
        loader.synchronize()

        loader.memcpy_dtoh(
            ctypes.c_void_p(self._host_active.data_ptr),
            self._d_active_base,
            self.NUM_ACTIVE_CHAINS * self.dim * 4,
        )
        loader.memcpy_dtoh(
            ctypes.c_void_p(self._host_chain9.data_ptr),
            self._d_chain9,
            self.dim * 4,
        )
        loader.memcpy_dtoh(
            ctypes.c_void_p(self._host_resonance_matrix.data_ptr),
            self._d_resonance_matrix,
            self.NUM_ACTIVE_CHAINS * self.NUM_ACTIVE_CHAINS * 4,
        )

        weights = self._compute_resonance_weights()

        chain_rows = self._host_active.to_nested_list()
        chain_rows.append(self._host_chain9.tolist())
        chain_states = HostTensorF32.from_array_like(
            chain_rows,
            rows=self.NUM_CHAINS,
            cols=self.dim,
        )

        chain_norms = Float32Vector(
            (
                math.sqrt(sum(float(value) * float(value) for value in chain_states[row]))
                for row in range(self.NUM_CHAINS)
            ),
            size=self.NUM_CHAINS,
        )
        self._last_diag = SwarmDiagnostics(
            resonance_matrix=self._host_resonance_matrix.copy(),
            resonance_raw=self._host_resonance_raw.copy(),
            resonance_weights=weights.copy(),
            chain_states=chain_states,
            chain_norms=chain_norms,
        )
        self._diagnostics_dirty = False

    def _compute_resonance_weights(self) -> Float32Vector:
        """Derive resonance weights from the latest matrix."""
        rows = self._host_resonance_matrix.to_nested_list()
        if self.resonance_strategy == "max":
            raw_values = [max(row) if row else 0.0 for row in rows]
        elif self.resonance_strategy == "median":
            raw_values = []
            for row in rows:
                if not row:
                    raw_values.append(0.0)
                    continue
                sorted_row = sorted(float(value) for value in row)
                mid = len(sorted_row) // 2
                if len(sorted_row) % 2 == 0:
                    raw_values.append((sorted_row[mid - 1] + sorted_row[mid]) / 2.0)
                else:
                    raw_values.append(sorted_row[mid])
        else:
            raw_values = [
                sum(float(value) for value in row) / float(len(row) or 1)
                for row in rows
            ]

        self._host_resonance_raw.set_flat(raw_values)

        weights = [abs(value) if self.normalize_weights else float(value) for value in raw_values]
        total = float(sum(weights))
        if total < 1e-6:
            weights = [1.0 / self.NUM_ACTIVE_CHAINS for _ in range(self.NUM_ACTIVE_CHAINS)]
        else:
            if self.normalize_weights:
                weights = [float(value) / total for value in weights]

        self._host_resonance_weights.set_flat(weights)
        return self._host_resonance_weights

    def _launch(self, kernel_key: str, params: list, block: Tuple[int, int, int]) -> None:
        loader.launch(
            self._kernels[kernel_key],
            grid=(1, 1, 1),
            block=block,
            params=params,
        )

    @staticmethod
    def _offset_ptr(base: loader.CUdeviceptr, offset_bytes: int) -> loader.CUdeviceptr:
        return loader.CUdeviceptr(int(base.value) + offset_bytes)

    @staticmethod
    def _as_uint64(ptr: loader.CUdeviceptr) -> ctypes.c_uint64:
        return ctypes.c_uint64(int(ptr.value))
