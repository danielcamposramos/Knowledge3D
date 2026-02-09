"""ARC-specific GPU/PTX operations with JIT kernels.

This module avoids static PTX binary coupling for ARC ranking/oracle helpers.
Kernels are compiled at runtime against the active CUDA stack, which keeps
device compatibility aligned with the current host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Keep CUDA headers discoverable for CuPy NVRTC JIT when CUDA_PATH is missing.
if "CUDA_PATH" not in os.environ and Path("/usr/include/cuda_fp16.h").exists():
    os.environ["CUDA_PATH"] = "/usr"

try:  # pragma: no cover - optional GPU dependency
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore
    _HAS_CUPY = False


@dataclass
class ArcPtxRankingResult:
    """Result envelope for GPU-ranked candidate pools."""

    scores: np.ndarray
    ranked_indices: list[int]
    top_index: int
    mode: str


class ARCPTXOps:
    """JIT-backed ARC kernels for sovereignty-aligned ranking operations."""

    _WEIGHTED_SCORE_KERNEL = r"""
    extern "C" __global__
    void weighted_score_kernel(
        const float* source_precision,
        const float* quality_prior,
        const float* train_similarity,
        const float* novelty,
        const float* grammar_confidence,
        const float* cross_modal,
        const float* compositional,
        const float* reuse,
        const float* family_bonus,
        float* out_score,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        out_score[idx] =
            0.26f * source_precision[idx] +
            0.20f * quality_prior[idx] +
            0.16f * train_similarity[idx] +
            0.08f * novelty[idx] +
            0.12f * grammar_confidence[idx] +
            0.08f * cross_modal[idx] +
            0.06f * compositional[idx] +
            0.04f * reuse[idx] +
            family_bonus[idx];
    }
    """

    _ARGMAX_KERNEL = r"""
    extern "C" __global__
    void argmax_kernel(const float* scores, const int n, int* out_idx) {
        if (threadIdx.x != 0 || blockIdx.x != 0) return;
        int best_idx = 0;
        float best_val = scores[0];
        for (int i = 1; i < n; ++i) {
            float v = scores[i];
            if (v > best_val) {
                best_val = v;
                best_idx = i;
            }
        }
        out_idx[0] = best_idx;
    }
    """

    def __init__(self) -> None:
        self._weighted_kernel = None
        self._argmax_kernel = None

    @property
    def available(self) -> bool:
        return bool(_HAS_CUPY and cp is not None)

    def _ensure_kernels(self) -> None:
        if not self.available:
            raise RuntimeError("cupy_unavailable")
        assert cp is not None
        if self._weighted_kernel is None:
            self._weighted_kernel = cp.RawKernel(self._WEIGHTED_SCORE_KERNEL, "weighted_score_kernel")
        if self._argmax_kernel is None:
            self._argmax_kernel = cp.RawKernel(self._ARGMAX_KERNEL, "argmax_kernel")

    def rank_candidates_ternary(
        self,
        *,
        source_precision: Iterable[float],
        quality_prior: Iterable[float],
        train_similarity: Iterable[float],
        novelty: Iterable[float],
        grammar_confidence: Iterable[float],
        cross_modal: Iterable[float],
        compositional: Iterable[float],
        reuse: Iterable[float],
        family_bonus: Iterable[float],
    ) -> ArcPtxRankingResult:
        """Compute weighted scores and top winner on GPU."""
        self._ensure_kernels()
        assert cp is not None

        source_precision_gpu = cp.asarray(np.asarray(list(source_precision), dtype=np.float32))
        n = int(source_precision_gpu.size)
        if n <= 0:
            return ArcPtxRankingResult(scores=np.asarray([], dtype=np.float32), ranked_indices=[], top_index=0, mode="gpu_empty")

        quality_prior_gpu = cp.asarray(np.asarray(list(quality_prior), dtype=np.float32))
        train_similarity_gpu = cp.asarray(np.asarray(list(train_similarity), dtype=np.float32))
        novelty_gpu = cp.asarray(np.asarray(list(novelty), dtype=np.float32))
        grammar_conf_gpu = cp.asarray(np.asarray(list(grammar_confidence), dtype=np.float32))
        cross_modal_gpu = cp.asarray(np.asarray(list(cross_modal), dtype=np.float32))
        compositional_gpu = cp.asarray(np.asarray(list(compositional), dtype=np.float32))
        reuse_gpu = cp.asarray(np.asarray(list(reuse), dtype=np.float32))
        family_bonus_gpu = cp.asarray(np.asarray(list(family_bonus), dtype=np.float32))

        out_score_gpu = cp.zeros(n, dtype=cp.float32)
        threads = 128
        blocks = (n + threads - 1) // threads

        self._weighted_kernel(
            (blocks,),
            (threads,),
            (
                source_precision_gpu,
                quality_prior_gpu,
                train_similarity_gpu,
                novelty_gpu,
                grammar_conf_gpu,
                cross_modal_gpu,
                compositional_gpu,
                reuse_gpu,
                family_bonus_gpu,
                out_score_gpu,
                np.int32(n),
            ),
        )

        top_idx_gpu = cp.zeros(1, dtype=cp.int32)
        self._argmax_kernel((1,), (1,), (out_score_gpu, np.int32(n), top_idx_gpu))
        cp.cuda.runtime.deviceSynchronize()

        score_cpu = cp.asnumpy(out_score_gpu).astype(np.float32, copy=False)
        ranked_indices = np.argsort(score_cpu)[::-1].astype(int).tolist()
        top_idx = int(cp.asnumpy(top_idx_gpu)[0])
        if top_idx in ranked_indices:
            ranked_indices = [top_idx] + [idx for idx in ranked_indices if idx != top_idx]

        return ArcPtxRankingResult(
            scores=score_cpu,
            ranked_indices=ranked_indices,
            top_index=top_idx,
            mode="gpu_jit_ternary_ranking",
        )


ARC_PTX_OPS = ARCPTXOps()

