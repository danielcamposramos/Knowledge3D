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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from knowledge3d.cranium.sovereign import loader


@dataclass(frozen=True)
class SwarmDiagnostics:
    """Container for the latest swarm diagnostics."""

    resonance_matrix: np.ndarray
    resonance_raw: np.ndarray
    resonance_weights: np.ndarray
    chain_states: np.ndarray
    chain_norms: np.ndarray


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

        self._host_zero = np.zeros(self.dim, dtype=np.float32)
        self._host_active = np.zeros(
            (self.NUM_ACTIVE_CHAINS, self.dim),
            dtype=np.float32,
        )
        self._host_chain9 = np.zeros(self.dim, dtype=np.float32)
        self._host_resonance_matrix = np.zeros(
            (self.NUM_ACTIVE_CHAINS, self.NUM_ACTIVE_CHAINS),
            dtype=np.float32,
        )
        self._host_resonance_raw = np.zeros(self.NUM_ACTIVE_CHAINS, dtype=np.float32)
        self._host_resonance_weights = np.zeros(self.NUM_ACTIVE_CHAINS, dtype=np.float32)

        self._last_diag: Optional[SwarmDiagnostics] = None
        self._diagnostics_dirty = True

        self.reset_states()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def execute_swarm(
        self,
        input_embedding: np.ndarray,
        num_iterations: int = 1,
        reset_state: bool = False,
        readback_mode: str = "full",
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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
        vec = np.asarray(input_embedding, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"input_embedding must have shape ({self.dim},), got {vec.shape}")

        if reset_state or not self.persistent_state:
            self.reset_states()

        mode = readback_mode.lower()
        if mode not in {"full", "output", "diagnostics"}:
            raise ValueError("readback_mode must be one of {'full', 'output', 'diagnostics'}")

        loader.memcpy_htod(
            self._d_input,
            vec.ctypes.data_as(ctypes.c_void_p),
            self.dim * 4,
        )

        iterations = max(1, int(num_iterations))
        for _ in range(iterations):
            self._run_once()

        loader.memcpy_dtoh(
            self._host_chain9.ctypes.data_as(ctypes.c_void_p),
            self._d_chain9,
            self.dim * 4,
        )

        output = self._host_chain9.copy()
        chain_states: Optional[np.ndarray] = None
        resonance_weights: Optional[np.ndarray] = None

        if mode in {"full", "diagnostics"}:
            self._refresh_diagnostics()
            chain_states = self._last_diag.chain_states.copy()
            resonance_weights = self._last_diag.resonance_weights.copy()
            if mode == "diagnostics":
                chain_states = None
        else:
            self._diagnostics_dirty = True

        return output, chain_states, resonance_weights

    def get_chain_diagnostics(self) -> SwarmDiagnostics:
        """Return the most recent diagnostic snapshot."""
        if self._last_diag is None or self._diagnostics_dirty:
            self._refresh_diagnostics()
        return self._last_diag

    def reset_states(self) -> None:
        """Zero all chain buffers to clear temporal state."""
        zero_ptr = self._host_zero.ctypes.data_as(ctypes.c_void_p)
        for ptr in self._chain_buffers:
            loader.memcpy_htod(ptr, zero_ptr, self.dim * 4)
        self._host_resonance_matrix.fill(0.0)
        loader.memcpy_htod(
            self._d_resonance_matrix,
            self._host_resonance_matrix.ctypes.data_as(ctypes.c_void_p),
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
            self._host_active.ctypes.data_as(ctypes.c_void_p),
            self._d_active_base,
            self.NUM_ACTIVE_CHAINS * self.dim * 4,
        )
        loader.memcpy_dtoh(
            self._host_chain9.ctypes.data_as(ctypes.c_void_p),
            self._d_chain9,
            self.dim * 4,
        )
        loader.memcpy_dtoh(
            self._host_resonance_matrix.ctypes.data_as(ctypes.c_void_p),
            self._d_resonance_matrix,
            self.NUM_ACTIVE_CHAINS * self.NUM_ACTIVE_CHAINS * 4,
        )

        weights = self._compute_resonance_weights()

        chain_states = np.zeros((self.NUM_CHAINS, self.dim), dtype=np.float32)
        chain_states[: self.NUM_ACTIVE_CHAINS] = self._host_active
        chain_states[-1] = self._host_chain9

        chain_norms = np.linalg.norm(chain_states, axis=1)
        self._last_diag = SwarmDiagnostics(
            resonance_matrix=self._host_resonance_matrix.copy(),
            resonance_raw=self._host_resonance_raw.copy(),
            resonance_weights=weights.copy(),
            chain_states=chain_states,
            chain_norms=chain_norms,
        )
        self._diagnostics_dirty = False

    def _compute_resonance_weights(self) -> np.ndarray:
        """Derive resonance weights from the latest matrix."""
        if self.resonance_strategy == "max":
            raw = np.max(self._host_resonance_matrix, axis=1)
        elif self.resonance_strategy == "median":
            raw = np.median(self._host_resonance_matrix, axis=1)
        else:
            raw = np.mean(self._host_resonance_matrix, axis=1)

        raw = raw.astype(np.float32, copy=False)
        self._host_resonance_raw[:] = raw

        weights = np.abs(raw) if self.normalize_weights else raw.copy()
        total = float(np.sum(weights))
        if total < 1e-6:
            weights = np.full(self.NUM_ACTIVE_CHAINS, 1.0 / self.NUM_ACTIVE_CHAINS, dtype=np.float32)
        else:
            if self.normalize_weights:
                weights /= total
            if weights.dtype != np.float32:
                weights = weights.astype(np.float32)

        self._host_resonance_weights[:] = weights
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
