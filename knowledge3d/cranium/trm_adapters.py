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

import numpy as np

from knowledge3d.cranium.ptx_runtime.rpn_math_core import DeviceTensor, RPNMathCore
from knowledge3d.cranium.sovereign import loader


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
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
    require_gpu: bool = True          # Enforce GPU-only path


@dataclass
class AdapterDeviceBuffers:
    """GPU-side buffers for sovereign adapter updates."""

    dims: int
    rank: int
    gradient: DeviceTensor
    grad_a: DeviceTensor
    grad_b: DeviceTensor
    A: DeviceTensor
    B: DeviceTensor
    A_transposed: DeviceTensor
    B_transposed: DeviceTensor
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
        self.A = np.random.randn(shape[0], self.rank).astype(np.float32) * init_std
        self.B = np.random.randn(self.rank, shape[1]).astype(np.float32) * init_std

        # Zero-initialize B for stable training (LoRA best practice)
        self.B.fill(0.0)

    def get_delta(self) -> np.ndarray:
        """
        Reconstruct full adapter delta.

        Returns: ΔW = α × (A @ B)  [D×D]
        """
        return self.alpha * (self.A @ self.B)

    def apply_gradient(self, gradient: np.ndarray, lr: float = 0.001):
        if self._ensure_math_core():
            self.apply_gradient_rpn(gradient, lr)
            return

        if self.config.require_gpu:
            raise RuntimeError(
                f"[{self.specialist_name}] GPU math core required but unavailable; "
                "set require_gpu=False to enable CPU fallback."
            )

        self._apply_gradient_cpu(gradient, lr)

    def _apply_gradient_cpu(self, gradient: np.ndarray, lr: float) -> None:
        """
        Update adapter weights given gradient for full ΔW.

        Uses chain rule to compute gradients for A and B:
            ∂L/∂A = ∂L/∂ΔW @ B.T
            ∂L/∂B = A.T @ ∂L/∂ΔW

        Args:
            gradient: Gradient w.r.t. full ΔW [D×D]
            lr: Learning rate
        """
        if gradient.shape != self.shape:
            raise ValueError(f"Gradient shape {gradient.shape} != adapter shape {self.shape}")

        # Gradient clipping (prevent instability)
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1.0:
            gradient = gradient / grad_norm

        # Compute gradients for A and B using chain rule
        grad_A = gradient @ self.B.T  # [D×D] @ [D×r] = [D×r]
        grad_B = self.A.T @ gradient  # [r×D] @ [D×D] = [r×D]

        # Gradient descent
        self.A -= lr * grad_A
        self.B -= lr * grad_B
        return None

    def apply_gradient_rpn(self, gradient: np.ndarray, lr: float = 0.001) -> float:
        """
        Sovereign RPN-based gradient application.

        Args:
            gradient: Full ΔW gradient matrix
            lr: Learning rate

        Returns:
            Gradient norm after clipping
        """
        buffers = self._ensure_device_buffers()
        if buffers is None or self._math_core is None:
            if self.config.require_gpu:
                raise RuntimeError(
                    f"[{self.specialist_name}] GPU buffers unavailable for RPN training"
                )
            self._apply_gradient_cpu(gradient, lr)
            return float(np.linalg.norm(gradient))

        dims = buffers.dims
        rank = buffers.rank
        if gradient.shape != (dims, dims):
            raise ValueError(f"Gradient shape {gradient.shape} != adapter shape {(dims, dims)}")

        grad_host = np.ascontiguousarray(gradient, dtype=np.float32)
        RPNMathCore.copy_to_device(grad_host, buffers.gradient.ptr)
        RPNMathCore.copy_to_device(self.A, buffers.A.ptr)
        RPNMathCore.copy_to_device(self.B, buffers.B.ptr)

        b_t_host = np.ascontiguousarray(self.B.T, dtype=np.float32)
        RPNMathCore.copy_to_device(b_t_host, buffers.B_transposed.ptr)
        a_t_host = np.ascontiguousarray(self.A.T, dtype=np.float32)
        RPNMathCore.copy_to_device(a_t_host, buffers.A_transposed.ptr)

        grad_vec = self._vector_view(buffers.gradient)
        grad_norm = self._math_core.vector_norm(grad_vec)

        clip = self.config.gradient_clip
        if clip > 0.0 and grad_norm > clip:
            scale = clip / max(grad_norm, 1e-6)
            self._scale_vector(grad_vec, buffers.grad_scale, 'grad_scale_value', scale)
            grad_norm = clip

        # Compute grad_A = gradient @ B.T
        self._math_core.matmul(buffers.grad_a, buffers.gradient, buffers.B_transposed)

        # Compute grad_B = A.T @ gradient
        self._math_core.matmul(buffers.grad_b, buffers.A_transposed, buffers.gradient)

        # Update A
        a_vec = self._vector_view(buffers.grad_a)
        a_dest = self._vector_view(buffers.A)
        self._scale_vector(a_vec, buffers.a_scale, 'a_scale_value', -lr)
        self._math_core.vec_add3(a_dest, a_dest, a_vec, buffers.a_zero)

        # Update B
        b_vec = self._vector_view(buffers.grad_b)
        b_dest = self._vector_view(buffers.B)
        self._scale_vector(b_vec, buffers.b_scale, 'b_scale_value', -lr)
        self._math_core.vec_add3(b_dest, b_dest, b_vec, buffers.b_zero)

        # Sync back to host
        RPNMathCore.copy_to_host(buffers.A.ptr, self.A)
        RPNMathCore.copy_to_host(buffers.B.ptr, self.B)

        return float(grad_norm)

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return self.A.size + self.B.size

    def get_memory_mb(self) -> float:
        """Get memory footprint in MB (fp32)."""
        return self.get_num_params() * 4 / (1024**2)

    def save(self, path: Path):
        """Save adapter to disk."""
        np.savez_compressed(
            path,
            A=self.A,
            B=self.B,
            alpha=self.alpha,
            rank=self.rank,
            shape=self.shape
        )

    def load(self, path: Path):
        """Load adapter from disk."""
        data = np.load(path)
        self.A = data['A']
        self.B = data['B']
        self.alpha = float(data['alpha'])
        self.rank = int(data['rank'])
        self.shape = tuple(data['shape'])


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
        self.A_shadow = np.zeros_like(self.A)
        self.B_shadow = np.zeros_like(self.B)

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
        grad_a = DeviceTensor(alloc(a_len), dims, rank)
        grad_b = DeviceTensor(alloc(b_len), rank, dims)
        A_dev = DeviceTensor(alloc(a_len), dims, rank)
        B_dev = DeviceTensor(alloc(b_len), rank, dims)
        A_transposed = DeviceTensor(alloc(a_len), rank, dims)
        B_transposed = DeviceTensor(alloc(b_len), dims, rank)
        grad_scale = DeviceTensor(alloc(grad_len), grad_len, 1)
        a_scale = DeviceTensor(alloc(a_len), a_len, 1)
        b_scale = DeviceTensor(alloc(b_len), b_len, 1)
        a_zero = DeviceTensor(alloc(a_len), a_len, 1)
        b_zero = DeviceTensor(alloc(b_len), b_len, 1)

        buffers = AdapterDeviceBuffers(
            dims=dims,
            rank=rank,
            gradient=gradient,
            grad_a=grad_a,
            grad_b=grad_b,
            A=A_dev,
            B=B_dev,
            A_transposed=A_transposed,
            B_transposed=B_transposed,
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
        return buffers

    def fork_to_shadow(self):
        """Copy primary weights → shadow for testing."""
        np.copyto(self.A_shadow, self.A)
        np.copyto(self.B_shadow, self.B)

    def get_delta_shadow(self) -> np.ndarray:
        """Get shadow delta: ΔW_shadow = α × (A_shadow @ B_shadow)"""
        return self.alpha * (self.A_shadow @ self.B_shadow)

    def apply_gradient_to_shadow(self, gradient: np.ndarray,
                                 lr: Optional[float] = None):
        """
        Apply gradient to shadow weights.

        Primary weights unchanged - testing update safely.
        """
        lr = lr or self.config.learning_rate

        if self._ensure_math_core():
            primary_A = self.A
            primary_B = self.B
            try:
                self.A = self.A_shadow
                self.B = self.B_shadow
                self.apply_gradient_rpn(gradient, lr)
            finally:
                self.A = primary_A
                self.B = primary_B
            return

        # CPU fallback
        if self.config.require_gpu:
            raise RuntimeError(
                f"[{self.specialist_name}] GPU math core required for shadow update"
            )

        grad_norm = np.linalg.norm(gradient)
        if grad_norm > self.config.gradient_clip:
            gradient = gradient / grad_norm * self.config.gradient_clip

        grad_A = gradient @ self.B_shadow.T
        grad_B = self.A_shadow.T @ gradient

        self.A_shadow -= lr * grad_A
        self.B_shadow -= lr * grad_B

    def validate_and_commit(self, base_weights: np.ndarray,
                           eval_fn: Callable[[np.ndarray, List], float]) -> Tuple[bool, float, float]:
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
            np.copyto(self.A, self.A_shadow)
            np.copyto(self.B, self.B_shadow)

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

        print(f"[{self.specialist_name}] Checkpoint loaded from {checkpoint_dir}")
        print(f"  Baseline performance: {self.baseline_performance:.4f}")
        print(f"  Acceptance rate: {stats.get('acceptance_rate', 0)*100:.1f}%")
