"""
Matryoshka TRM: Bi-Directional Variable Dimensionality

Inspired by Matryoshka Representation Learning (Qwen-style embeddings).

Key Innovation: Single weight matrix supports MULTIPLE dimension levels:
- Downward: Shrink to 64 dims for efficiency (1024× faster!)
- Upward: Expand to 16K dims for capacity (research-level reasoning)

Properties:
- Prefix dimensions are self-contained
- W[:256,:256] works independently
- W[:512,:512] works independently
- W[:1024,:1024] works independently
- No retraining needed for any dimension level

Reasoning Capacity:
- Each dimension = one RPN stack line
- More dims = deeper reasoning chains
- 64 dims: Trivial tasks (single char OCR)
- 128 dims: Simple tasks (word recognition)
- 512 dims: Medium tasks (sentence understanding)
- 1024 dims: Complex tasks (multi-hop reasoning)
- 4096 dims: Research tasks (meta-analysis)

Usage:
    # Create Matryoshka base
    mat_trm = MatryoshkaTRM(max_dims=2048, min_dims=64)

    # Downward: Efficiency
    W_fast = mat_trm.get_base_at_dim(128)  # 256× faster than full!

    # Upward: Capacity
    mat_trm.expand_base_dimensions(4096)  # Existing knowledge preserved
"""

from __future__ import annotations

import ctypes
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

from knowledge3d.cranium.trm_adapters import (
    SelfUpdatingAdapter,
    AdapterConfig,
    _to_serializable,
)
from knowledge3d.cranium.ptx_runtime.rpn_math_core import HostTensorF32, RPNMathCore
from knowledge3d.cranium.sovereign import loader
from knowledge3d.cranium.bridges.matryoshka_bridge import MatryoshkaProjectionBridge


_MATH_CORE: Optional[RPNMathCore] = None


def _get_math_core() -> RPNMathCore:
    global _MATH_CORE
    if _MATH_CORE is None:
        _MATH_CORE = RPNMathCore()
    return _MATH_CORE


class _CtypesView:
    def __init__(self, tensor: "MatryoshkaTensor") -> None:
        self.data = tensor.data_ptr


class MatryoshkaTensor(HostTensorF32):
    """Sovereign float32 tensor wrapper with compatibility helpers."""

    @classmethod
    def zeros(cls, rows: int, cols: int = 1, on_mutate=None) -> "MatryoshkaTensor":
        tensor = cls(rows, cols, on_mutate=on_mutate)
        return tensor

    @classmethod
    def random_normal(
        cls,
        rows: int,
        cols: int,
        std: float = 0.01,
        on_mutate=None,
    ) -> "MatryoshkaTensor":
        base = HostTensorF32.random_normal(rows, cols, std, on_mutate=on_mutate)
        return cls.from_host_tensor(base)

    @classmethod
    def from_host_tensor(cls, tensor: HostTensorF32) -> "MatryoshkaTensor":
        clone = cls(tensor.rows, tensor.cols)
        clone.load_bytes(tensor.to_bytes())
        return clone

    @classmethod
    def from_bytes(cls, rows: int, cols: int, payload: bytes) -> "MatryoshkaTensor":
        tensor = cls(rows, cols)
        tensor.load_bytes(payload)
        return tensor

    @classmethod
    def from_array_like(
        cls,
        array: object,
        *,
        rows: int | None = None,
        cols: int | None = None,
        on_mutate=None,
    ) -> "MatryoshkaTensor":
        base = HostTensorF32.from_array_like(array, rows=rows, cols=cols, on_mutate=on_mutate)
        return cls.from_host_tensor(base)

    @property
    def ctypes(self) -> _CtypesView:
        return _CtypesView(self)

    def copy(self) -> "MatryoshkaTensor":
        return MatryoshkaTensor.from_bytes(self.rows, self.cols, self.to_bytes())

    def astype(self, *_args, copy: bool = True, **_kwargs) -> "MatryoshkaTensor":
        return self.copy() if copy else self

    def tolist(self) -> List[float] | List[List[float]]:
        if self.cols == 1:
            return self.to_flat_list()
        return self.to_nested_list()

    def tobytes(self) -> bytes:
        return self.to_bytes()

    def transpose(self) -> "MatryoshkaTensor":
        return MatryoshkaTensor.from_host_tensor(super().transpose())

    def prefix_square(self, dim: int) -> "MatryoshkaTensor":
        if dim < 0 or dim > self.rows or dim > self.cols:
            raise ValueError(f"Prefix dimension {dim} incompatible with tensor shape {self.shape}")
        out = MatryoshkaTensor.zeros(dim, dim)
        for row in range(dim):
            for col in range(dim):
                out._buffer[row * dim + col] = self._buffer[row * self.cols + col]
        return out

    def copy_prefix_from(self, other: object) -> None:
        source = MatryoshkaTensor.from_array_like(other)
        if source.rows > self.rows or source.cols > self.cols:
            raise ValueError(f"Source shape {source.shape} does not fit into destination {self.shape}")
        for row in range(source.rows):
            for col in range(source.cols):
                self._buffer[row * self.cols + col] = source._buffer[row * source.cols + col]
        self._notify_mutation()

    def __iter__(self):
        if self.cols == 1:
            for idx in range(self.rows):
                yield float(self._buffer[idx])
            return
        yield from super().__iter__()

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return float(self._buffer[int(index[0]) * self.cols + int(index[1])])
        if self.cols == 1:
            return float(self._buffer[int(index)])
        return super().__getitem__(index)

    def _binary_op(self, other: object, op) -> "MatryoshkaTensor":
        if isinstance(other, (int, float)):
            out = self.copy()
            scalar = float(other)
            for idx in range(out.size):
                out._buffer[idx] = op(float(out._buffer[idx]), scalar)
            return out
        rhs = MatryoshkaTensor.from_array_like(other, rows=self.rows, cols=self.cols)
        if rhs.shape != self.shape:
            raise ValueError(f"Shape mismatch: {self.shape} != {rhs.shape}")
        out = MatryoshkaTensor.zeros(self.rows, self.cols)
        for idx in range(self.size):
            out._buffer[idx] = op(float(self._buffer[idx]), float(rhs._buffer[idx]))
        return out

    def __add__(self, other: object) -> "MatryoshkaTensor":
        return self._binary_op(other, lambda a, b: a + b)

    def __radd__(self, other: object) -> "MatryoshkaTensor":
        return self.__add__(other)

    def __iadd__(self, other: object):
        result = self.__add__(other)
        self.load_bytes(result.to_bytes())
        return self

    def __mul__(self, other: object) -> "MatryoshkaTensor":
        if isinstance(other, (int, float)):
            out = self.copy()
            scalar = float(other)
            for idx in range(out.size):
                out._buffer[idx] = float(out._buffer[idx]) * scalar
            return out
        return self._binary_op(other, lambda a, b: a * b)

    def __rmul__(self, other: object) -> "MatryoshkaTensor":
        return self.__mul__(other)

    def __matmul__(self, other: object) -> "MatryoshkaTensor":
        result = _get_math_core().matmul_host(self, other)
        return MatryoshkaTensor.from_host_tensor(result)


class MatryoshkaTRM:
    """
    TRM with Matryoshka-style variable dimensionality.

    Supports bi-directional scaling:
    - Shrink for efficiency (64, 128, 256...)
    - Expand for capacity (4096, 8192, 16384...)

    Single weight matrix, variable compute!
    """

    # Standard dimension levels (geometric progression)
    STANDARD_DIM_LEVELS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    def __init__(self, max_dims: int = 2048, min_dims: int = 64,
                 init_std: float = 0.01):
        """
        Initialize Matryoshka TRM.

        Args:
            max_dims: Maximum dimension (current capacity)
            min_dims: Minimum dimension (for efficiency mode)
            init_std: Weight initialization standard deviation
        """
        self.max_dims = max_dims
        self.min_dims = min_dims

        # Full-capacity base weights
        self.W_base_full = MatryoshkaTensor.random_normal(max_dims, max_dims, init_std)

        # Supported dimension levels (within current range)
        self.dim_levels = [d for d in self.STANDARD_DIM_LEVELS
                          if min_dims <= d <= max_dims]

        # Specialists registry (by name)
        self.specialists: Dict[str, Dict[str, Any]] = {}

        # Current working dimension (for efficiency mode)
        self.current_working_dim = max_dims

        # GPU resources
        self._bridge: Optional[MatryoshkaProjectionBridge] = None
        self._gpu_weights: Optional[loader.CUdeviceptr] = None
        self._initialise_gpu_resources()

        print(f"[MatryoshkaTRM] Initialized")
        print(f"  Dimension range: {min_dims} - {max_dims}")
        print(f"  Supported levels: {self.dim_levels}")
        print(f"  Memory: {self._get_memory_mb(max_dims):.1f} MB (full capacity)")

    def _initialise_gpu_resources(self):
        """Allocate and populate GPU buffers for the base weight matrix."""
        self._bridge = MatryoshkaProjectionBridge()
        self._gpu_weights = loader.gpu_malloc(self.W_base_full.nbytes)
        loader.memcpy_htod(
            self._gpu_weights,
            ctypes.c_void_p(self.W_base_full.data_ptr),
            self.W_base_full.nbytes,
        )

    def get_base_at_dim(self, dim: int):
        """
        Get base weights at specific dimension.

        Matryoshka property: ALL prefix dimensions are valid.

        Args:
            dim: Desired dimension

        Returns:
            Truncated base weights [dim×dim]
        """
        if dim < self.min_dims:
            raise ValueError(f"Requested dim {dim} < minimum {self.min_dims}")

        if dim > self.max_dims:
            raise ValueError(f"Requested dim {dim} > maximum {self.max_dims}")

        # Return prefix submatrix
        return self.W_base_full.prefix_square(dim)

    def project_vector(self, vector: object, target_dim: int):
        """
        Project `vector` using GPU-native path (sovereign only).
        """
        if self._bridge is None or self._gpu_weights is None:
            raise RuntimeError(
                "Matryoshka GPU resources not initialized. Sovereign path requires CUDA + PTX loader."
            )
        vec = MatryoshkaTensor.from_array_like(vector)
        projected = self._bridge.project_host(self._gpu_weights, vec.to_flat_list(), target_dim, self.max_dims)
        return MatryoshkaTensor.from_array_like(projected, rows=target_dim, cols=1)

    def register_specialist(self, name: str, required_dims: int,
                          rank: Optional[int] = None,
                          config: Optional[AdapterConfig] = None):
        """
        Register specialist with dimension requirement.

        Args:
            name: Specialist identifier
            required_dims: How many dimensions this specialist needs
            rank: Adapter rank (default: dims//32)
            config: Adapter configuration
        """
        # Snap to nearest supported dimension level
        required_dims = self._snap_to_nearest_dim(required_dims)

        # Default rank: 1/32 of dimension (good compression ratio)
        if rank is None:
            rank = max(16, required_dims // 32)

        # Create self-updating adapter
        adapter = SelfUpdatingAdapter(
            shape=(required_dims, required_dims),
            rank=rank,
            specialist_name=name,
            config=config
        )

        # Store specialist info
        self.specialists[name] = {
            'adapter': adapter,
            'dims': required_dims,
            'rank': rank,
            'params': adapter.get_num_params(),
            'memory_mb': adapter.get_memory_mb()
        }

        print(f"[MatryoshkaTRM] Registered specialist '{name}':")
        print(f"  Dimensions: {required_dims} (RPN stack lines)")
        print(f"  Rank: {rank}")
        print(f"  Parameters: {adapter.get_num_params()/1e3:.1f}K")
        print(f"  Memory: {adapter.get_memory_mb():.2f} MB")

    def remove_specialist(self, name: str):
        """Remove specialist from registry."""
        if name in self.specialists:
            del self.specialists[name]
            print(f"[MatryoshkaTRM] Removed specialist '{name}'")

    def get_specialist_weights(self, name: str, include_base: bool = True):
        """
        Get specialist's active weights.

        Args:
            name: Specialist name
            include_base: If True, return base + adapter; if False, just adapter delta

        Returns:
            Active weights [dims×dims]
        """
        if name not in self.specialists:
            raise ValueError(f"Unknown specialist: {name}")

        specialist = self.specialists[name]
        dims = specialist['dims']
        adapter = specialist['adapter']

        if include_base:
            # Base + adapter delta
            W_base = self.get_base_at_dim(dims)
            return W_base + MatryoshkaTensor.from_array_like(adapter.get_delta(), rows=dims, cols=dims)
        else:
            # Just adapter delta
            return MatryoshkaTensor.from_array_like(adapter.get_delta(), rows=dims, cols=dims)

    def compute_with_specialist(self, input_data: object,
                               specialist_name: str):
        """
        Forward pass using specific specialist.

        Automatically uses correct dimension level.

        Args:
            input_data: Input vector
            specialist_name: Which specialist to use

        Returns:
            Output vector
        """
        if specialist_name not in self.specialists:
            raise ValueError(f"Unknown specialist: {specialist_name}")

        # Get specialist's active weights (base + adapter)
        W_active = self.get_specialist_weights(specialist_name, include_base=True)
        dims = W_active.shape[0]

        # Ensure input matches dimension
        input_resized = self._resize_input(input_data, dims)

        # Forward pass
        output = W_active @ input_resized

        return output

    def compute_with_moe(self, input_data: object,
                        specialist_weights: Dict[str, float]):
        """
        Compute with MoE (weighted combination of specialists).

        Handles specialists at different dimension levels.

        Args:
            input_data: Input vector
            specialist_weights: Dict mapping specialist_name → weight [0,1]

        Returns:
            Blended output
        """
        if len(specialist_weights) == 0:
            raise ValueError("No specialists selected for MoE")

        # Find maximum dimension among active specialists
        max_dim = max(self.specialists[name]['dims']
                     for name in specialist_weights.keys())

        # Get base at max dimension
        W_base = self.get_base_at_dim(max_dim)
        W_active = W_base.copy()

        # Blend specialist adapters
        for name, weight in specialist_weights.items():
            if weight < 0.01:  # Skip negligible contributions
                continue

            specialist = self.specialists[name]
            dims = specialist['dims']
            adapter = specialist['adapter']

            # Get adapter delta
            delta = MatryoshkaTensor.from_array_like(adapter.get_delta(), rows=dims, cols=dims)

            # Pad to max_dim if needed
            if dims < max_dim:
                delta_padded = MatryoshkaTensor.zeros(max_dim, max_dim)
                for row in range(dims):
                    for col in range(dims):
                        delta_padded._buffer[row * max_dim + col] = delta._buffer[row * dims + col]
                delta = delta_padded

            # Add weighted contribution
            W_active += weight * delta

        # Resize input to match max_dim
        input_resized = self._resize_input(input_data, max_dim)

        # Forward pass
        output = W_active @ input_resized

        return output

    def expand_base_dimensions(self, new_max_dims: int):
        """
        Expand base to support higher dimensions.

        Existing weights preserved in upper-left corner (Matryoshka property).
        New capacity added in bottom-right region.

        Args:
            new_max_dims: New maximum dimension
        """
        if new_max_dims <= self.max_dims:
            print(f"[MatryoshkaTRM] Already at {self.max_dims} dims, no expansion needed")
            return

        print(f"[MatryoshkaTRM] Expanding: {self.max_dims} → {new_max_dims} dims")

        # Create expanded weight matrix
        W_base_new = MatryoshkaTensor.random_normal(new_max_dims, new_max_dims, 0.01)

        # Copy existing weights to upper-left (preserve learned knowledge)
        W_base_new.copy_prefix_from(self.W_base_full)

        # Update
        self.W_base_full = W_base_new
        self.max_dims = new_max_dims

        # Update dimension levels
        self.dim_levels = [d for d in self.STANDARD_DIM_LEVELS
                          if self.min_dims <= d <= self.max_dims]

        print(f"  ✓ Base expanded, existing knowledge preserved")
        print(f"  ✓ New dimension levels: {self.dim_levels}")
        print(f"  ✓ New memory: {self._get_memory_mb(new_max_dims):.1f} MB")

    def shrink_for_efficiency(self, target_dims: int):
        """
        Set working dimension to lower value for efficiency.

        Useful for batch processing or latency-critical tasks.

        Args:
            target_dims: Target working dimension
        """
        target_dims = self._snap_to_nearest_dim(target_dims)

        if target_dims > self.max_dims:
            print(f"[MatryoshkaTRM] Cannot shrink to {target_dims} (> max {self.max_dims})")
            return

        self.current_working_dim = target_dims

        speedup = (self.max_dims / target_dims) ** 2
        memory_full = self._get_memory_mb(self.max_dims)
        memory_shrunk = self._get_memory_mb(target_dims)

        print(f"[MatryoshkaTRM] Efficiency mode: {target_dims} dims")
        print(f"  Speedup: {speedup:.0f}× faster than full {self.max_dims}")
        print(f"  Memory: {memory_shrunk:.1f} MB (vs {memory_full:.1f} MB full)")

    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        # Specialist stats
        specialist_stats = []
        total_specialist_params = 0

        for name, spec in self.specialists.items():
            adapter = spec['adapter']
            stats = adapter.get_stats()
            specialist_stats.append(stats)
            total_specialist_params += spec['params']

        # Base stats
        base_params = self.max_dims ** 2

        # Fix key names for consistency
        return {
            'base_model': {
                'max_dims': self.max_dims,
                'min_dims': self.min_dims,
                'current_working_dim': self.current_working_dim,
                'dim_levels': self.dim_levels,
                'params': base_params,
                'memory_mb': self._get_memory_mb(self.max_dims)
            },
            'specialists': specialist_stats,
            'num_specialists': len(specialist_stats),
            'total_specialist_params': total_specialist_params,
            'specialist_params': total_specialist_params,  # Alias for compatibility
            'total_params': base_params + total_specialist_params,
            'total_memory_mb': (base_params + total_specialist_params) * 4 / (1024**2)
        }

    def save_base(self, path: Path):
        """Save base weights."""
        with open(path, "wb") as handle:
            pickle.dump(
                {
                    "rows": self.W_base_full.rows,
                    "cols": self.W_base_full.cols,
                    "payload": self.W_base_full.to_bytes(),
                },
                handle,
            )
        print(f"[MatryoshkaTRM] Base weights saved to {path}")

    def load_base(self, path: Path):
        """Load base weights."""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        self.W_base_full = MatryoshkaTensor.from_bytes(
            int(payload["rows"]),
            int(payload["cols"]),
            payload["payload"],
        )
        self.max_dims = self.W_base_full.shape[0]
        self.dim_levels = [d for d in self.STANDARD_DIM_LEVELS
                          if self.min_dims <= d <= self.max_dims]
        print(f"[MatryoshkaTRM] Base weights loaded from {path}")
        print(f"  Dimensions: {self.max_dims}")

    def save_all(self, checkpoint_dir: Path):
        """Save complete system (base + all specialists)."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save base
        self.save_base(checkpoint_dir / 'base_weights.pkl')

        # Save each specialist
        for name, spec in self.specialists.items():
            adapter = spec['adapter']
            adapter.save_checkpoint(checkpoint_dir)

        # Save system metadata
        metadata = {
            'max_dims': self.max_dims,
            'min_dims': self.min_dims,
            'dim_levels': self.dim_levels,
            'specialist_names': list(self.specialists.keys()),
            'stats': self.get_system_stats()
        }

        serializable_metadata = _to_serializable(metadata)

        with open(checkpoint_dir / 'system_metadata.json', 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

        print(f"[MatryoshkaTRM] Complete system saved to {checkpoint_dir}")

    def load_all(self, checkpoint_dir: Path):
        """Load complete system."""
        # Load base
        self.load_base(checkpoint_dir / 'base_weights.pkl')

        # Load metadata
        with open(checkpoint_dir / 'system_metadata.json', 'r') as f:
            metadata = json.load(f)

        # Load specialists
        for name in metadata['specialist_names']:
            # Find adapter file
            adapter_path = checkpoint_dir / f'{name}_adapter.npz'
            if adapter_path.exists():
                adapter_meta = SelfUpdatingAdapter.peek_saved_metadata(adapter_path)
                dims = int(adapter_meta['shape'][0])
                rank = int(adapter_meta['rank'])

                # Register specialist
                self.register_specialist(name, dims, rank)

                # Load checkpoint
                self.specialists[name]['adapter'].load_checkpoint(checkpoint_dir)

        print(f"[MatryoshkaTRM] Complete system loaded from {checkpoint_dir}")

    def _snap_to_nearest_dim(self, dim: int) -> int:
        """Snap requested dimension to nearest supported level."""
        return min(self.dim_levels, key=lambda x: abs(x - dim))

    def __del__(self):
        """Release GPU buffers when the Matryoshka instance is collected."""
        try:
            if self._gpu_weights is not None:
                loader.gpu_free(self._gpu_weights)
                self._gpu_weights = None
        except Exception:
            pass

    def _resize_input(self, input_data: object, target_dim: int):
        """Resize input vector to match target dimension."""
        vector = MatryoshkaTensor.from_array_like(input_data)
        if vector.size == target_dim:
            return vector
        if vector.size > target_dim:
            return MatryoshkaTensor.from_array_like(vector.to_flat_list()[:target_dim], rows=target_dim, cols=1)
        padded = MatryoshkaTensor.zeros(target_dim, 1)
        for idx, value in enumerate(vector.to_flat_list()):
            padded._buffer[idx] = value
        return padded

    @staticmethod
    def _get_memory_mb(dims: int) -> float:
        """Calculate memory footprint for given dims (fp32)."""
        return dims ** 2 * 4 / (1024**2)


class DimensionSelector:
    """
    Intelligent dimension selection for optimal efficiency.

    Selects appropriate dimension level based on task complexity.
    """

    # Complexity thresholds for each dimension
    DIM_THRESHOLDS = {
        64: 0.1,      # Trivial (single char OCR, basic arithmetic)
        128: 0.3,     # Simple (word recognition, simple math)
        256: 0.5,     # Medium-low (sentence parsing, basic reasoning)
        512: 0.7,     # Medium (paragraph understanding, moderate reasoning)
        1024: 0.85,   # Complex (multi-paragraph, multi-hop reasoning)
        2048: 0.95,   # Very complex (document analysis, deep reasoning)
        4096: 1.0     # Maximum (corpus analysis, meta-reasoning)
    }

    @staticmethod
    def select_dim(complexity: float) -> int:
        """
        Select optimal dimension for given task complexity.

        Args:
            complexity: Task complexity [0.0, 1.0]

        Returns:
            Recommended dimension
        """
        for dim, threshold in sorted(DimensionSelector.DIM_THRESHOLDS.items()):
            if complexity <= threshold:
                return dim
        return 2048  # Default to high capacity

    @staticmethod
    def estimate_speedup(from_dim: int, to_dim: int) -> float:
        """Estimate speedup from dimension reduction."""
        return (from_dim / to_dim) ** 2

    @staticmethod
    def estimate_memory_savings(from_dim: int, to_dim: int) -> float:
        """Estimate memory savings in MB."""
        from_mem = MatryoshkaTRM._get_memory_mb(from_dim)
        to_mem = MatryoshkaTRM._get_memory_mb(to_dim)
        return from_mem - to_mem
