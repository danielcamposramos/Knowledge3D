"""
TRM Adapters: Self-Updating Low-Rank Specialist Modules

Implements LoRA-style adapters with independent self-updating capability.

Key Features:
- Low-rank decomposition: ΔW = A @ B (memory efficient)
- Shadow weights: Safe testing before committing
- Validation gating: Only accept improvements
- Independent evolution: Each specialist updates separately

Architecture:
    Base Model (W_base) + Adapter (A @ B) = Active Weights

Memory Efficiency:
    Full specialist: 2048×2048 = 16.8M params
    Adapter (rank-64): 2×(2048×64) = 262K params
    Reduction: 64× smaller!

Usage:
    # Create adapter
    adapter = SelfUpdatingAdapter(
        shape=(512, 512),
        rank=64,
        specialist_name='ocr'
    )

    # Train with self-updating
    adapter.fork_to_shadow()
    adapter.apply_gradient_to_shadow(gradient)
    success = adapter.validate_and_commit(base_weights, eval_fn)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import zipfile

from knowledge3d.cranium.ptx_runtime.rpn_math_core import (
    DeviceTensor,
    HostTensorF32,
    RPNMathCore,
)
from knowledge3d.cranium.sovereign import loader


def _to_serializable(obj: Any) -> Any:
    """Recursively convert array-like types to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, HostTensorF32):
        return obj.to_nested_list()
    if isinstance(obj, (int,)):
        return int(obj)
    if isinstance(obj, (float,)):
        return float(obj)
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    return obj


@dataclass
class AdapterConfig:
    """Adapter configuration."""
    rank: int = 64                    # Bottleneck dimension
    alpha: float = 1.0                # Scaling factor
    learning_rate: float = 0.001      # Update learning rate
    gradient_clip: float = 1.0        # Gradient clipping threshold
    min_improvement: float = 0.001    # Minimum improvement to commit (0.1%)
    max_degradation: float = 0.05     # Maximum allowed degradation (5%)


@dataclass
class AdapterDeviceBuffers:
    """GPU-side buffers for sovereign adapter updates."""

    dims: int
    rank: int
    gradient: DeviceTensor
    gradient_transposed: DeviceTensor
    grad_a: DeviceTensor
    grad_b: DeviceTensor
    grad_a_transposed: DeviceTensor
    grad_b_transposed: DeviceTensor
    A_weights: DeviceTensor
    B_weights: DeviceTensor
    A_shadow_weights: DeviceTensor
    B_shadow_weights: DeviceTensor
    A_transposed: DeviceTensor
    B_transposed: DeviceTensor
    A_shadow_transposed: DeviceTensor
    B_shadow_transposed: DeviceTensor
    grad_scale: DeviceTensor
    a_scale: DeviceTensor
    b_scale: DeviceTensor
    a_zero: DeviceTensor
    b_zero: DeviceTensor
    grad_scale_value: Optional[float] = None
    a_scale_value: Optional[float] = None
    b_scale_value: Optional[float] = None


class AdapterWeights:
    """
    Low-rank adapter using LoRA-style decomposition.

    Instead of storing full ΔW [D×D], decompose as:
        ΔW = α × (A @ B)
        where A [D×r], B [r×D], r << D

    Memory savings: O(D²) → O(2Dr)

    Example:
        Full: 2048×2048 = 4.2M params (16.8 MB)
        Rank-64: 2×(2048×64) = 262K params (1.05 MB)
        Reduction: 16× smaller
    """

    def __init__(self, shape: Tuple[int, int], rank: int = 64,
                 alpha: float = 1.0, init_std: float = 0.01):
        """
        Initialize low-rank adapter.

        Args:
            shape: Full weight shape [D, D]
            rank: Bottleneck dimension (r)
            alpha: Scaling factor for adapter strength
            init_std: Initialization standard deviation
        """
        if shape[0] != shape[1]:
            raise ValueError(f"Adapter requires square shape, got {shape}")

        self.shape = shape
        self.rank = min(rank, shape[0])  # Ensure rank <= dimension
        self.alpha = alpha

        # Low-rank decomposition: ΔW = A @ B
        self.A = HostTensorF32.random_normal(shape[0], self.rank, init_std)
        self.B = HostTensorF32.zeros(self.rank, shape[1])

    def _require_math_core(self) -> RPNMathCore:
        ensure = getattr(self, "_ensure_math_core", None)
        if callable(ensure):
            if not ensure():
                raise RuntimeError("GPU math core unavailable. Sovereign path requires CUDA context.")
            math_core = getattr(self, "_math_core", None)
            if math_core is None:
                raise RuntimeError("GPU math core missing after successful initialization.")
            return math_core
        return RPNMathCore()

    def _delta_from(self, left: HostTensorF32, right: HostTensorF32) -> List[List[float]]:
        math_core = self._require_math_core()
        delta = math_core.matmul_host(left, right)
        if self.alpha != 1.0:
            delta.scale_inplace(self.alpha)
        return delta.to_nested_list()

    def get_delta(self) -> List[List[float]]:
        """
        Reconstruct full adapter delta.

        Returns: ΔW = α × (A @ B)  [D×D]
        """
        device_delta = getattr(self, "_delta_from_device_weights", None)
        if callable(device_delta):
            maybe_delta = device_delta(primary=True)
            if maybe_delta is not None:
                return maybe_delta
        return self._delta_from(self.A, self.B)

    def apply_gradient(self, gradient: Any, lr: float = 0.001):
        """Sovereign GPU gradient application. No CPU fallback exists."""
        if not self._ensure_math_core():
            raise RuntimeError(
                f"[{self.specialist_name}] GPU math core unavailable. "
                "Sovereign path requires CUDA context. Fix the GPU path."
            )
        self.apply_gradient_rpn(gradient, lr)

    def apply_gradient_rpn(self, gradient: Any, lr: float = 0.001) -> float:
        """
        Sovereign RPN-based gradient application.

        Args:
            gradient: Full ΔW gradient matrix
            lr: Learning rate

        Returns:
            Gradient norm after clipping
        """
        apply_device = getattr(self, "_apply_gradient_device", None)
        if callable(apply_device):
            return float(apply_device(gradient, lr=lr, shadow=False))
        raise RuntimeError("Adapter missing sovereign device gradient implementation.")

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return self.A.size + self.B.size

    def get_memory_mb(self) -> float:
        """Get memory footprint in MB (fp32)."""
        return self.get_num_params() * 4 / (1024**2)

    def save(self, path: Path):
        """Save adapter to disk."""
        sync = getattr(self, "sync_weights_to_host", None)
        if callable(sync):
            sync()
        payload = {
            "alpha": self.alpha,
            "rank": self.rank,
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "A_shape": [int(self.A.rows), int(self.A.cols)],
            "B_shape": [int(self.B.rows), int(self.B.cols)],
        }
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("metadata.json", json.dumps(payload))
            archive.writestr("A.bin", self.A.to_bytes())
            archive.writestr("B.bin", self.B.to_bytes())

    def load(self, path: Path):
        """Load adapter from disk."""
        release = getattr(self, "_release_device_buffers", None)
        if callable(release):
            release()
        metadata = self.peek_saved_metadata(path)
        self.alpha = float(metadata["alpha"])
        self.rank = int(metadata["rank"])
        self.shape = (int(metadata["shape"][0]), int(metadata["shape"][1]))
        self.A = HostTensorF32.zeros(int(metadata["A_shape"][0]), int(metadata["A_shape"][1]))
        self.B = HostTensorF32.zeros(int(metadata["B_shape"][0]), int(metadata["B_shape"][1]))
        with zipfile.ZipFile(path, "r") as archive:
            self.A.load_bytes(archive.read("A.bin"))
            self.B.load_bytes(archive.read("B.bin"))
        hook = getattr(self, "_after_primary_host_reload", None)
        if callable(hook):
            hook()

    @staticmethod
    def peek_saved_metadata(path: Path) -> Dict[str, Any]:
        with zipfile.ZipFile(path, "r") as archive:
            return json.loads(archive.read("metadata.json").decode("utf-8"))


class SelfUpdatingAdapter(AdapterWeights):
    """
    Adapter with shadow weights and validation gating.

    Enables safe self-updating:
    1. Fork primary → shadow
    2. Apply gradient to shadow
    3. Validate shadow on holdout set
    4. Commit if improved, reject otherwise

    Prevents catastrophic forgetting through validation gate.
    """

    def __init__(self, shape: Tuple[int, int], rank: int = 64,
                 specialist_name: str = "specialist",
                 config: Optional[AdapterConfig] = None):
        """
        Initialize self-updating adapter.

        Args:
            shape: Weight shape [D, D]
            rank: Bottleneck dimension
            specialist_name: Identifier for this specialist
            config: Adapter configuration
        """
        super().__init__(shape, rank)

        self.specialist_name = specialist_name
        self.config = config or AdapterConfig()

        # Shadow weights (candidate updates)
        self.A_shadow = HostTensorF32.zeros(self.A.rows, self.A.cols)
        self.B_shadow = HostTensorF32.zeros(self.B.rows, self.B.cols)

        # Validation tracking
        self.validation_samples = []
        self.baseline_performance = 0.0

        # Update statistics
        self.update_count = 0
        self.accepted_count = 0
        self.rejected_count = 0
        self.performance_history = []

        print(f"[{specialist_name}] Self-updating adapter initialized")
        print(f"  Shape: {shape}, Rank: {rank}")
        print(f"  Parameters: {self.get_num_params()/1e3:.1f}K ({self.get_memory_mb():.2f} MB)")

        self._math_core: Optional[RPNMathCore] = None
        self._device_buffers: Optional[AdapterDeviceBuffers] = None
        self._rpn_available: bool = True
        self._primary_host_dirty: bool = False
        self._primary_device_dirty: bool = False
        self._shadow_host_dirty: bool = False
        self._shadow_device_dirty: bool = False
        self._bind_host_callbacks()

    def set_validation_samples(self, samples: List[Dict]):
        """Set specialist-specific validation set."""
        self.validation_samples = samples
        print(f"[{self.specialist_name}] Validation set: {len(samples)} samples")

    def _ensure_math_core(self) -> bool:
        """Initialize Tier-3 RPN math core if possible."""
        if not self._rpn_available:
            return False

        if self._math_core is None:
            try:
                self._math_core = RPNMathCore()
            except Exception as exc:  # pragma: no cover - GPU init failures
                print(f"[{self.specialist_name}] ⚠️ RPN math core unavailable: {exc}")
                self._rpn_available = False
                return False
        return True

    @staticmethod
    def _vector_view(tensor: DeviceTensor) -> DeviceTensor:
        """Return a vector view (rows*cols, 1) for element-wise ops."""
        return DeviceTensor(tensor.ptr, tensor.rows * tensor.cols, 1)

    def _bind_host_callbacks(self) -> None:
        self.A.set_mutation_callback(self._mark_primary_host_dirty)
        self.B.set_mutation_callback(self._mark_primary_host_dirty)
        self.A_shadow.set_mutation_callback(self._mark_shadow_host_dirty)
        self.B_shadow.set_mutation_callback(self._mark_shadow_host_dirty)

    def _mark_primary_host_dirty(self) -> None:
        self._primary_host_dirty = True
        self._primary_device_dirty = False

    def _mark_shadow_host_dirty(self) -> None:
        self._shadow_host_dirty = True
        self._shadow_device_dirty = False

    def _scale_vector(self, tensor: DeviceTensor, scale_buffer: DeviceTensor,
                      attr_name: str, value: float) -> None:
        """Scale tensor by value using cached buffer fills to avoid rewrites."""
        buffers = self._device_buffers
        if buffers is None or self._math_core is None:
            raise RuntimeError("RPN math core unavailable for scaling")

        current = getattr(buffers, attr_name)
        if current is None or not math.isclose(current, value, rel_tol=1e-9, abs_tol=1e-12):
            self._math_core.fill(scale_buffer, value)
            setattr(buffers, attr_name, value)

        self._math_core.vector_multiply(tensor, scale_buffer)

    def _ensure_device_buffers(self) -> Optional[AdapterDeviceBuffers]:
        if self._device_buffers is not None:
            return self._device_buffers

        if not self._ensure_math_core():
            return None

        dims = self.shape[0]
        rank = self.rank
        grad_len = dims * dims
        a_len = dims * rank
        b_len = rank * dims

        def alloc(size: int) -> loader.CUdeviceptr:
            return loader.gpu_malloc(size * 4)

        gradient = DeviceTensor(alloc(grad_len), dims, dims)
        gradient_transposed = DeviceTensor(alloc(grad_len), dims, dims)
        grad_a = DeviceTensor(alloc(a_len), dims, rank)
        grad_b = DeviceTensor(alloc(b_len), rank, dims)
        grad_a_transposed = DeviceTensor(alloc(a_len), rank, dims)
        grad_b_transposed = DeviceTensor(alloc(b_len), dims, rank)
        A_dev = DeviceTensor(alloc(a_len), dims, rank)
        B_dev = DeviceTensor(alloc(b_len), rank, dims)
        A_shadow_dev = DeviceTensor(alloc(a_len), dims, rank)
        B_shadow_dev = DeviceTensor(alloc(b_len), rank, dims)
        A_transposed = DeviceTensor(alloc(a_len), rank, dims)
        B_transposed = DeviceTensor(alloc(b_len), dims, rank)
        A_shadow_transposed = DeviceTensor(alloc(a_len), rank, dims)
        B_shadow_transposed = DeviceTensor(alloc(b_len), dims, rank)
        grad_scale = DeviceTensor(alloc(grad_len), grad_len, 1)
        a_scale = DeviceTensor(alloc(a_len), a_len, 1)
        b_scale = DeviceTensor(alloc(b_len), b_len, 1)
        a_zero = DeviceTensor(alloc(a_len), a_len, 1)
        b_zero = DeviceTensor(alloc(b_len), b_len, 1)

        buffers = AdapterDeviceBuffers(
            dims=dims,
            rank=rank,
            gradient=gradient,
            gradient_transposed=gradient_transposed,
            grad_a=grad_a,
            grad_b=grad_b,
            grad_a_transposed=grad_a_transposed,
            grad_b_transposed=grad_b_transposed,
            A_weights=A_dev,
            B_weights=B_dev,
            A_shadow_weights=A_shadow_dev,
            B_shadow_weights=B_shadow_dev,
            A_transposed=A_transposed,
            B_transposed=B_transposed,
            A_shadow_transposed=A_shadow_transposed,
            B_shadow_transposed=B_shadow_transposed,
            grad_scale=grad_scale,
            a_scale=a_scale,
            b_scale=b_scale,
            a_zero=a_zero,
            b_zero=b_zero,
        )

        # Initialize zero buffers once
        self._math_core.fill(a_zero, 0.0)
        self._math_core.fill(b_zero, 0.0)
        self._device_buffers = buffers
        self._upload_host_weight_pair(self.A, self.B, buffers.A_weights, buffers.B_weights, buffers.A_transposed, buffers.B_transposed)
        self._upload_host_weight_pair(
            self.A_shadow,
            self.B_shadow,
            buffers.A_shadow_weights,
            buffers.B_shadow_weights,
            buffers.A_shadow_transposed,
            buffers.B_shadow_transposed,
        )
        self._primary_host_dirty = False
        self._primary_device_dirty = False
        self._shadow_host_dirty = False
        self._shadow_device_dirty = False
        return buffers

    def _release_device_buffers(self) -> None:
        buffers = self._device_buffers
        if buffers is None:
            return
        for value in buffers.__dict__.values():
            if isinstance(value, DeviceTensor):
                loader.gpu_free(value.ptr)
        self._device_buffers = None

    def _after_primary_host_reload(self) -> None:
        self.A_shadow = HostTensorF32.zeros(self.A.rows, self.A.cols)
        self.B_shadow = HostTensorF32.zeros(self.B.rows, self.B.cols)
        self._bind_host_callbacks()
        self._primary_host_dirty = False
        self._primary_device_dirty = False
        self._shadow_host_dirty = False
        self._shadow_device_dirty = False

    def _upload_host_weight_pair(
        self,
        host_a: HostTensorF32,
        host_b: HostTensorF32,
        dev_a: DeviceTensor,
        dev_b: DeviceTensor,
        dev_a_t: DeviceTensor,
        dev_b_t: DeviceTensor,
    ) -> None:
        RPNMathCore.copy_to_device(host_a, dev_a.ptr)
        RPNMathCore.copy_to_device(host_b, dev_b.ptr)
        RPNMathCore.copy_to_device(host_a.transpose(), dev_a_t.ptr)
        RPNMathCore.copy_to_device(host_b.transpose(), dev_b_t.ptr)

    def _select_weight_buffers(self, shadow: bool = False) -> Tuple[DeviceTensor, DeviceTensor, DeviceTensor, DeviceTensor]:
        buffers = self._ensure_device_buffers()
        if buffers is None:
            raise RuntimeError("Adapter device buffers unavailable.")
        if shadow:
            return (
                buffers.A_shadow_weights,
                buffers.B_shadow_weights,
                buffers.A_shadow_transposed,
                buffers.B_shadow_transposed,
            )
        return (
            buffers.A_weights,
            buffers.B_weights,
            buffers.A_transposed,
            buffers.B_transposed,
        )

    def _ensure_device_weight_set(self, shadow: bool = False) -> Tuple[DeviceTensor, DeviceTensor, DeviceTensor, DeviceTensor]:
        buffers = self._ensure_device_buffers()
        if buffers is None:
            raise RuntimeError("Adapter device buffers unavailable.")
        if shadow:
            if self._shadow_host_dirty:
                self._upload_host_weight_pair(
                    self.A_shadow,
                    self.B_shadow,
                    buffers.A_shadow_weights,
                    buffers.B_shadow_weights,
                    buffers.A_shadow_transposed,
                    buffers.B_shadow_transposed,
                )
                self._shadow_host_dirty = False
                self._shadow_device_dirty = False
            return (
                buffers.A_shadow_weights,
                buffers.B_shadow_weights,
                buffers.A_shadow_transposed,
                buffers.B_shadow_transposed,
            )
        if self._primary_host_dirty:
            self._upload_host_weight_pair(
                self.A,
                self.B,
                buffers.A_weights,
                buffers.B_weights,
                buffers.A_transposed,
                buffers.B_transposed,
            )
            self._primary_host_dirty = False
            self._primary_device_dirty = False
        return (
            buffers.A_weights,
            buffers.B_weights,
            buffers.A_transposed,
            buffers.B_transposed,
        )

    def _sync_weight_pair_to_host(
        self,
        dev_a: DeviceTensor,
        dev_b: DeviceTensor,
        host_a: HostTensorF32,
        host_b: HostTensorF32,
    ) -> None:
        host_a.set_mutation_callback(None)
        host_b.set_mutation_callback(None)
        try:
            RPNMathCore.copy_to_host(dev_a.ptr, host_a)
            RPNMathCore.copy_to_host(dev_b.ptr, host_b)
        finally:
            self._bind_host_callbacks()

    def sync_weights_to_host(self) -> None:
        buffers = self._device_buffers
        if buffers is None or not self._primary_device_dirty:
            return
        self._sync_weight_pair_to_host(buffers.A_weights, buffers.B_weights, self.A, self.B)
        self._primary_device_dirty = False
        self._primary_host_dirty = False

    def sync_shadow_weights_to_host(self) -> None:
        buffers = self._device_buffers
        if buffers is None or not self._shadow_device_dirty:
            return
        self._sync_weight_pair_to_host(buffers.A_shadow_weights, buffers.B_shadow_weights, self.A_shadow, self.B_shadow)
        self._shadow_device_dirty = False
        self._shadow_host_dirty = False

    def _delta_from_device_weights(self, primary: bool = True) -> Optional[List[List[float]]]:
        if self._device_buffers is None:
            return None
        if primary and self._primary_host_dirty:
            return None
        if (not primary) and self._shadow_host_dirty:
            return None

        if not self._ensure_math_core() or self._math_core is None:
            return None

        left, right, _, _ = self._select_weight_buffers(shadow=not primary)
        dest_ptr = loader.gpu_malloc(self.shape[0] * self.shape[1] * 4)
        dest = DeviceTensor(dest_ptr, self.shape[0], self.shape[1])
        try:
            self._math_core.matmul(dest, left, right)
            host = HostTensorF32.zeros(self.shape[0], self.shape[1])
            RPNMathCore.copy_to_host(dest.ptr, host)
            if self.alpha != 1.0:
                host.scale_inplace(self.alpha)
            return host.to_nested_list()
        finally:
            loader.gpu_free(dest_ptr)

    def _apply_gradient_device(self, gradient: Any, lr: float = 0.001, shadow: bool = False) -> float:
        buffers = self._ensure_device_buffers()
        if buffers is None or self._math_core is None:
            raise RuntimeError(
                f"[{self.specialist_name}] GPU device buffers unavailable. "
                "Sovereign path requires allocated VRAM buffers. Fix gpu_malloc."
            )

        dims = buffers.dims
        grad_rows, grad_cols = RPNMathCore.shape_of(gradient)
        if (grad_rows, grad_cols) != (dims, dims):
            raise ValueError(f"Gradient shape {(grad_rows, grad_cols)} != adapter shape {(dims, dims)}")

        gradient_host = HostTensorF32.from_array_like(gradient, rows=dims, cols=dims)
        gradient_transposed = gradient_host.transpose()
        RPNMathCore.copy_to_device(gradient_host, buffers.gradient.ptr)
        RPNMathCore.copy_to_device(gradient_transposed, buffers.gradient_transposed.ptr)

        a_weights, b_weights, a_transposed, b_transposed = self._ensure_device_weight_set(shadow=shadow)
        grad_vec = self._vector_view(buffers.gradient)
        grad_vec_t = self._vector_view(buffers.gradient_transposed)
        grad_norm = self._math_core.vector_norm(grad_vec)

        clip = self.config.gradient_clip
        if clip > 0.0 and grad_norm > clip:
            scale = clip / max(grad_norm, 1e-6)
            self._scale_vector(grad_vec, buffers.grad_scale, "grad_scale_value", scale)
            self._scale_vector(grad_vec_t, buffers.grad_scale, "grad_scale_value", scale)
            grad_norm = clip

        self._math_core.matmul(buffers.grad_a, buffers.gradient, b_transposed)
        self._math_core.matmul(buffers.grad_a_transposed, b_weights, buffers.gradient_transposed)
        self._math_core.matmul(buffers.grad_b, a_transposed, buffers.gradient)
        self._math_core.matmul(buffers.grad_b_transposed, buffers.gradient_transposed, a_weights)

        a_vec = self._vector_view(buffers.grad_a)
        a_vec_t = self._vector_view(buffers.grad_a_transposed)
        a_dest = self._vector_view(a_weights)
        a_dest_t = self._vector_view(a_transposed)
        self._scale_vector(a_vec, buffers.a_scale, "a_scale_value", -lr)
        self._scale_vector(a_vec_t, buffers.a_scale, "a_scale_value", -lr)
        self._math_core.vec_add3(a_dest, a_dest, a_vec, buffers.a_zero)
        self._math_core.vec_add3(a_dest_t, a_dest_t, a_vec_t, buffers.a_zero)

        b_vec = self._vector_view(buffers.grad_b)
        b_vec_t = self._vector_view(buffers.grad_b_transposed)
        b_dest = self._vector_view(b_weights)
        b_dest_t = self._vector_view(b_transposed)
        self._scale_vector(b_vec, buffers.b_scale, "b_scale_value", -lr)
        self._scale_vector(b_vec_t, buffers.b_scale, "b_scale_value", -lr)
        self._math_core.vec_add3(b_dest, b_dest, b_vec, buffers.b_zero)
        self._math_core.vec_add3(b_dest_t, b_dest_t, b_vec_t, buffers.b_zero)

        if shadow:
            self._shadow_device_dirty = True
            self._shadow_host_dirty = False
        else:
            self._primary_device_dirty = True
            self._primary_host_dirty = False

        return float(grad_norm)

    def fork_to_shadow(self):
        """Copy primary weights → shadow for testing."""
        buffers = self._ensure_device_buffers()
        if buffers is None:
            raise RuntimeError("Cannot fork adapter weights without sovereign device buffers.")
        self._ensure_device_weight_set(shadow=False)
        size_a = self.A.nbytes
        size_b = self.B.nbytes
        loader.memcpy_dtod(buffers.A_shadow_weights.ptr, buffers.A_weights.ptr, size_a)
        loader.memcpy_dtod(buffers.B_shadow_weights.ptr, buffers.B_weights.ptr, size_b)
        loader.memcpy_dtod(buffers.A_shadow_transposed.ptr, buffers.A_transposed.ptr, size_a)
        loader.memcpy_dtod(buffers.B_shadow_transposed.ptr, buffers.B_transposed.ptr, size_b)
        self._shadow_host_dirty = False
        self._shadow_device_dirty = True

    def get_delta_shadow(self) -> List[List[float]]:
        """Get shadow delta: ΔW_shadow = α × (A_shadow @ B_shadow)"""
        maybe_delta = self._delta_from_device_weights(primary=False)
        if maybe_delta is not None:
            return maybe_delta
        return self._delta_from(self.A_shadow, self.B_shadow)

    def apply_gradient_to_shadow(self, gradient: Any,
                                 lr: Optional[float] = None):
        """
        Apply gradient to shadow weights.

        Primary weights unchanged - testing update safely.
        """
        lr = lr or self.config.learning_rate

        if not self._ensure_math_core():
            raise RuntimeError(
                f"[{self.specialist_name}] GPU math core unavailable for shadow update. "
                "Sovereign path requires CUDA context."
            )
        self._apply_gradient_device(gradient, lr=lr, shadow=True)

    def validate_and_commit(self, base_weights: Any,
                           eval_fn: Callable[[Any, List], float]) -> Tuple[bool, float, float]:
        """
        Validate shadow adapter and commit if performance improves.

        Args:
            base_weights: Current base model weights [D×D]
            eval_fn: Function that evaluates (weights, samples) → performance

        Returns:
            (success, baseline_perf, shadow_perf)
        """
        if len(self.validation_samples) == 0:
            print(f"[{self.specialist_name}] Warning: No validation samples, skipping validation")
            return False, 0.0, 0.0

        # Evaluate baseline (primary adapter + base)
        W_baseline = base_weights + self.get_delta()
        baseline_perf = eval_fn(W_baseline, self.validation_samples)

        # Evaluate shadow (shadow adapter + base)
        W_shadow = base_weights + self.get_delta_shadow()
        shadow_perf = eval_fn(W_shadow, self.validation_samples)

        # Ternary validation gate: TRUE, FALSE, UNKNOWN
        decision = self._ternary_gate(baseline_perf, shadow_perf)

        if decision == "TRUE":
            # Performance improved → commit shadow → primary
            self._ensure_device_weight_set(shadow=True)
            self.sync_shadow_weights_to_host()
            buffers = self._ensure_device_buffers()
            if buffers is None:
                raise RuntimeError("Cannot commit shadow weights without sovereign device buffers.")
            size_a = self.A.nbytes
            size_b = self.B.nbytes
            loader.memcpy_dtod(buffers.A_weights.ptr, buffers.A_shadow_weights.ptr, size_a)
            loader.memcpy_dtod(buffers.B_weights.ptr, buffers.B_shadow_weights.ptr, size_b)
            loader.memcpy_dtod(buffers.A_transposed.ptr, buffers.A_shadow_transposed.ptr, size_a)
            loader.memcpy_dtod(buffers.B_transposed.ptr, buffers.B_shadow_transposed.ptr, size_b)
            self.A.copy_from(self.A_shadow)
            self.B.copy_from(self.B_shadow)
            self._primary_host_dirty = False
            self._primary_device_dirty = False

            self.baseline_performance = shadow_perf
            self.accepted_count += 1

            improvement = shadow_perf - baseline_perf

            # Record success
            self.performance_history.append({
                'step': self.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'improvement': improvement,
                'accepted': True,
                'decision': 'TRUE'
            })

            print(f"[{self.specialist_name}] ✓ Update accepted: "
                  f"{baseline_perf:.4f} → {shadow_perf:.4f} (+{improvement:.4f})")

            self.update_count += 1
            return True, baseline_perf, shadow_perf

        elif decision == "FALSE":
            # Excessive degradation → reject
            self.rejected_count += 1
            degradation = baseline_perf - shadow_perf

            self.performance_history.append({
                'step': self.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'degradation': degradation,
                'accepted': False,
                'reason': 'excessive_degradation',
                'decision': 'FALSE'
            })

            print(f"[{self.specialist_name}] ✗ Update rejected: "
                  f"{baseline_perf:.4f} → {shadow_perf:.4f} (-{degradation:.4f}) "
                  f"- Excessive degradation")

            self.update_count += 1
            return False, baseline_perf, shadow_perf

        else:  # UNKNOWN
            # Insufficient evidence → accumulate data
            self.rejected_count += 1
            improvement = shadow_perf - baseline_perf

            self.performance_history.append({
                'step': self.update_count,
                'baseline': baseline_perf,
                'shadow': shadow_perf,
                'improvement': improvement,
                'accepted': False,
                'reason': 'insufficient_evidence',
                'decision': 'UNKNOWN'
            })

            print(f"[{self.specialist_name}] ? Update deferred: "
                  f"{baseline_perf:.4f} → {shadow_perf:.4f} (+{improvement:.4f}) "
                  f"- Insufficient evidence")

            self.update_count += 1
            return False, baseline_perf, shadow_perf

    def _ternary_gate(self, baseline_perf: float, shadow_perf: float) -> str:
        """
        Ternary validation gate for sovereign training.

        Decision logic:
        - TRUE: Shadow significantly better (improvement >= min_improvement)
        - FALSE: Shadow significantly worse (degradation > max_degradation)
        - UNKNOWN: Marginal difference (accumulate more evidence)

        Args:
            baseline_perf: Primary adapter performance
            shadow_perf: Shadow adapter performance

        Returns:
            "TRUE" | "FALSE" | "UNKNOWN"
        """
        improvement = shadow_perf - baseline_perf
        degradation = baseline_perf - shadow_perf

        # TRUE: Clear improvement
        if improvement >= self.config.min_improvement:
            return "TRUE"

        # FALSE: Excessive degradation
        elif degradation > self.config.max_degradation:
            return "FALSE"

        # UNKNOWN: Marginal difference (neither clearly better nor worse)
        else:
            return "UNKNOWN"

    def get_stats(self) -> Dict[str, Any]:
        """Get adapter statistics."""
        acceptance_rate = self.accepted_count / max(self.update_count, 1)

        return {
            'specialist_name': self.specialist_name,
            'shape': self.shape,
            'rank': self.rank,
            'params': self.get_num_params(),
            'memory_mb': self.get_memory_mb(),
            'update_count': self.update_count,
            'accepted': self.accepted_count,
            'rejected': self.rejected_count,
            'acceptance_rate': acceptance_rate,
            'baseline_performance': self.baseline_performance,
            'recent_history': self.performance_history[-10:]
        }

    def save_checkpoint(self, checkpoint_dir: Path):
        """Save adapter checkpoint with metadata."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save weights
        self.save(checkpoint_dir / f'{self.specialist_name}_adapter.npz')

        # Save metadata
        metadata = {
            'config': {
                'rank': self.config.rank,
                'alpha': self.config.alpha,
                'learning_rate': self.config.learning_rate
            },
            'stats': self.get_stats(),
            'performance_history': self.performance_history
        }

        serializable_metadata = _to_serializable(metadata)

        with open(checkpoint_dir / f'{self.specialist_name}_metadata.json', 'w') as f:
            json.dump(serializable_metadata, f, indent=2)

        print(f"[{self.specialist_name}] Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_dir: Path):
        """Load adapter checkpoint."""
        # Load weights
        self.load(checkpoint_dir / f'{self.specialist_name}_adapter.npz')

        # Load metadata
        with open(checkpoint_dir / f'{self.specialist_name}_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.performance_history = metadata.get('performance_history', [])
        stats = metadata.get('stats', {})
        self.baseline_performance = stats.get('baseline_performance', 0.0)
        self.fork_to_shadow()

        print(f"[{self.specialist_name}] Checkpoint loaded from {checkpoint_dir}")
        print(f"  Baseline performance: {self.baseline_performance:.4f}")
        print(f"  Acceptance rate: {stats.get('acceptance_rate', 0)*100:.1f}%")
