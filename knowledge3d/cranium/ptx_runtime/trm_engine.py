"""Tiny Recursive Model (TRM) GPU engine using CuPy + NVRTC.

Based on "Less is More: Recursive Reasoning with Tiny Networks" (2025)
- 7M parameter 2-layer MLP
- Recursive refinement: z ← net(x,y,z), y ← net(y,z)
- Adaptive halting via drift measurement
- Codex: Integrated with sub-100µs latency guard

This uses CuPy (like guard.py) instead of cuda-python bindings for compatibility
with the k3d-cranium environment. Follows Codex's proven pattern.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import cupy as cp
import numpy as np

# Codex: Import latency guard for GPU-native SLA enforcement
try:
    from knowledge3d.cranium.bridges.guard import LatencyGuard
    LATENCY_GUARD_AVAILABLE = True
except ImportError:
    LATENCY_GUARD_AVAILABLE = False

from knowledge3d.cranium.utils.cupy_env import ensure_nvrtc_include_path

ensure_nvrtc_include_path()

logger = logging.getLogger(__name__)


# CUDA C++ source for TRM kernel (NVRTC-compiled via CuPy, not handwritten PTX)
CUDA_TRM_SOURCE = r"""
extern "C" {

// SwiGLU activation: x * sigmoid(gate)
__device__ inline float swiglu(float x, float gate) {
    float sig = 1.0f / (1.0f + expf(-gate));
    return x * sig;
}

// Recursive update: z_new = net(x + y + z_old)
// TRM core operation
__global__ void trm_recursive_update(
    const float* __restrict__ x,          // question (batch, 512)
    const float* __restrict__ y,          // answer (batch, 512)
    const float* __restrict__ z_old,      // latent_old (batch, 512)
    const float* __restrict__ w1,         // weights layer 1 (512, 1024)
    const float* __restrict__ w2,         // weights layer 2 (1024, 512)
    float* __restrict__ z_new,            // latent_new (batch, 512)
    float* __restrict__ delta_norm,       // ||z_new - z_old||_2 (batch,)
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || dim_idx >= 512) return;

    __shared__ float combined_input[512];
    __shared__ float hidden[1024];
    __shared__ float output_local[512];

    // Combine: x + y + z
    int idx = batch_idx * 512 + dim_idx;
    combined_input[dim_idx] = x[idx] + y[idx] + z_old[idx];
    __syncthreads();

    // MLP Layer 1: input -> hidden with SwiGLU
    // Each thread computes 2 hidden units (1024 / 512 = 2)
    for (int h = 0; h < 2; h++) {
        int hidden_idx = dim_idx * 2 + h;
        if (hidden_idx < 1024) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < 512; i++) {
                sum += combined_input[i] * w1[i * 1024 + hidden_idx];
            }
            // SwiGLU: even indices are x, odd are gates
            if (hidden_idx % 2 == 0) {
                hidden[hidden_idx] = sum;
            } else {
                float x_val = hidden[hidden_idx - 1];
                hidden[hidden_idx - 1] = swiglu(x_val, sum);
                hidden[hidden_idx] = 0.0f;
            }
        }
    }
    __syncthreads();

    // MLP Layer 2: hidden -> output
    float sum = 0.0f;
    #pragma unroll 8
    for (int h = 0; h < 1024; h++) {
        sum += hidden[h] * w2[h * 512 + dim_idx];
    }
    output_local[dim_idx] = sum;
    z_new[idx] = sum;
    __syncthreads();

    // Compute drift ||z_new - z_old||_2 (parallel reduction)
    __shared__ float diff_squared[512];
    float diff = output_local[dim_idx] - z_old[idx];
    diff_squared[dim_idx] = diff * diff;
    __syncthreads();

    // Reduction in shared memory (thread 0 does final sum)
    if (dim_idx == 0) {
        float total = 0.0f;
        for (int i = 0; i < 512; i++) {
            total += diff_squared[i];
        }
        delta_norm[batch_idx] = sqrtf(total);
    }
}

// Answer refinement: y_new = net(y_old + z)
__global__ void trm_answer_refine(
    const float* __restrict__ y_old,      // answer_old (batch, 512)
    const float* __restrict__ z,          // latent (batch, 512)
    const float* __restrict__ w1,         // weights layer 1
    const float* __restrict__ w2,         // weights layer 2
    float* __restrict__ y_new,            // answer_new (batch, 512)
    int batch_size
) {
    int batch_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (batch_idx >= batch_size || dim_idx >= 512) return;

    __shared__ float combined[512];
    __shared__ float hidden[1024];

    int idx = batch_idx * 512 + dim_idx;
    combined[dim_idx] = y_old[idx] + z[idx];
    __syncthreads();

    // MLP Layer 1
    for (int h = 0; h < 2; h++) {
        int hidden_idx = dim_idx * 2 + h;
        if (hidden_idx < 1024) {
            float sum = 0.0f;
            #pragma unroll 4
            for (int i = 0; i < 512; i++) {
                sum += combined[i] * w1[i * 1024 + hidden_idx];
            }
            if (hidden_idx % 2 == 0) {
                hidden[hidden_idx] = sum;
            } else {
                float x_val = hidden[hidden_idx - 1];
                hidden[hidden_idx - 1] = swiglu(x_val, sum);
                hidden[hidden_idx] = 0.0f;
            }
        }
    }
    __syncthreads();

    // MLP Layer 2
    float sum = 0.0f;
    #pragma unroll 8
    for (int h = 0; h < 1024; h++) {
        sum += hidden[h] * w2[h * 512 + dim_idx];
    }
    y_new[idx] = sum;
}

}  // extern "C"
"""


@dataclass
class TRMConfig:
    """TRM hyperparameters from paper."""
    input_dim: int = 512
    hidden_dim: int = 1024
    output_dim: int = 512
    n_recursions: int = 6      # Paper optimal
    T_iterations: int = 3       # Paper optimal
    epsilon: float = 1e-4       # Halting threshold
    ema_rate: float = 0.999     # Weight stability
    latency_threshold_us: float = 95.0  # Codex: Sub-100µs SLA


class TRMEngine:
    """GPU-resident Tiny Recursive Model using CuPy + NVRTC.

    Follows Codex's proven guard.py pattern:
    - CuPy RawKernel compiles CUDA C++ → PTX at runtime
    - No handwritten PTX syntax errors
    - Full GPU-native execution
    - Compatible with k3d-cranium environment

    Codex: Integrated with latency guard for <95µs SLA enforcement.
    """

    def __init__(self, config: Optional[TRMConfig] = None) -> None:
        self.config = config or TRMConfig()
        self._compile_kernels()
        self._allocate_weights()

        # Codex: Initialize GPU-native latency guard
        self.enforce_latency_sla = LATENCY_GUARD_AVAILABLE
        if self.enforce_latency_sla:
            self.latency_guard = LatencyGuard(threshold_us=self.config.latency_threshold_us)
            self.sla_breach_count = 0
        else:
            self.latency_guard = None
            warnings.warn("LatencyGuard not available; running without SLA enforcement")

    def _compile_kernels(self) -> None:
        """Compile TRM kernels via CuPy's RawKernel (NVRTC backend).

        Codex pattern: Same as guard.py but with inline CUDA instead of PTX file.
        """
        # Compile via CuPy RawKernel (uses NVRTC under the hood)
        self._recursive_update_kernel = cp.RawKernel(
            CUDA_TRM_SOURCE,
            "trm_recursive_update",
            options=("--use_fast_math",),  # Enable fast math for performance
        )

        self._answer_refine_kernel = cp.RawKernel(
            CUDA_TRM_SOURCE,
            "trm_answer_refine",
            options=("--use_fast_math",),
        )

        logger.info("TRM kernels compiled via CuPy RawKernel (NVRTC)")

    def _allocate_weights(self) -> None:
        """Allocate and initialize 2-layer MLP weights on GPU."""
        # W1: (512, 1024), W2: (1024, 512) = ~1.05M params (float32)
        w1_size = (self.config.input_dim, self.config.hidden_dim)
        w2_size = (self.config.hidden_dim, self.config.output_dim)

        # Xavier/Glorot initialization
        # Use NumPy on CPU then transfer to avoid cupy.random compilation issues
        w1_init = np.random.randn(*w1_size).astype(np.float32) * np.sqrt(2.0 / self.config.input_dim)
        w2_init = np.random.randn(*w2_size).astype(np.float32) * np.sqrt(2.0 / self.config.hidden_dim)

        self.w1 = cp.asarray(w1_init)
        self.w2 = cp.asarray(w2_init)

        total_params = self.w1.size + self.w2.size
        logger.info(f"Allocated TRM weights: {total_params / 1e6:.2f}M parameters on GPU")

    def recursive_refine(
        self,
        question: np.ndarray,  # (batch, 512)
        answer: Optional[np.ndarray] = None,  # (batch, 512)
        latent: Optional[np.ndarray] = None,  # (batch, 512)
        return_timing: bool = True
    ) -> Tuple[cp.ndarray, cp.ndarray, int, Optional[float]]:
        """Run TRM recursive refinement: z ← net(x,y,z), y ← net(y,z).

        Args:
            question: Input question embedding (batch, 512)
            answer: Initial answer (defaults to zeros)
            latent: Initial latent state (defaults to zeros)
            return_timing: Whether to return GPU-native timing

        Returns:
            (refined_answer, final_latent, num_steps, elapsed_us)

        Codex: Uses GPU-native latency guard for SLA enforcement.
        """
        batch_size = question.shape[0]

        # Transfer to GPU (or use directly if already CuPy array)
        if isinstance(question, np.ndarray):
            x = cp.asarray(question, dtype=cp.float32)
        else:
            x = question.astype(cp.float32)

        if answer is None:
            y = cp.zeros_like(x)
        elif isinstance(answer, np.ndarray):
            y = cp.asarray(answer, dtype=cp.float32)
        else:
            y = answer.astype(cp.float32)

        if latent is None:
            z = cp.zeros_like(x)
        elif isinstance(latent, np.ndarray):
            z = cp.asarray(latent, dtype=cp.float32)
        else:
            z = latent.astype(cp.float32)

        # Allocate buffers
        z_new = cp.empty_like(z)
        delta_norm = cp.zeros(batch_size, dtype=cp.float32)

        # Codex: Start GPU-native latency guard before any GPU work
        if self.latency_guard is not None:
            self.latency_guard.start()

        # TRM recursive loop
        num_steps = 0
        max_steps = self.config.n_recursions * self.config.T_iterations

        for step in range(max_steps):
            # z_new = net(x + y + z)
            grid = (batch_size,)
            block = (512,)
            self._recursive_update_kernel(
                grid, block,
                (x, y, z, self.w1, self.w2, z_new, delta_norm, batch_size)
            )

            # Check drift for halting
            # Transfer to CPU for comparison to avoid cupy compilation issues
            if np.all(delta_norm.get() < self.config.epsilon):
                num_steps = step + 1
                break

            # z = z_new
            z, z_new = z_new, z  # Swap references (avoid copy)

            # y_new = net(y + z)
            self._answer_refine_kernel(
                grid, block,
                (y, z, self.w1, self.w2, y, batch_size)  # In-place update of y
            )

            num_steps = step + 1

        cp.cuda.Stream.null.synchronize()

        # Codex: Get GPU-native timing (ground truth, not CUDA events)
        elapsed_us = None
        if self.latency_guard is not None:
            elapsed_ns, breached = self.latency_guard.stop()
            elapsed_us = elapsed_ns / 1000.0
            if breached:
                self.sla_breach_count += 1
                warnings.warn(
                    f"⚠️  Latency SLA breach #{self.sla_breach_count}: "
                    f"{elapsed_us:.2f}µs (target: <{self.config.latency_threshold_us}µs)"
                )

        if return_timing:
            return y, z, num_steps, elapsed_us
        else:
            return y, z, num_steps, None

    def close(self) -> None:
        """Release GPU resources."""
        # CuPy handles memory automatically via garbage collection
        if hasattr(self, "w1"):
            del self.w1
        if hasattr(self, "w2"):
            del self.w2
        if hasattr(self, "_recursive_update_kernel"):
            del self._recursive_update_kernel
        if hasattr(self, "_answer_refine_kernel"):
            del self._answer_refine_kernel

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["TRMEngine", "TRMConfig"]
