"""ARC-specific GPU/PTX operations with JIT kernels.

This module avoids static PTX binary coupling for ARC reasoning helpers.
Kernels are compiled at runtime against the active CUDA stack, which keeps
device compatibility aligned with the current host.
"""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Keep CUDA headers discoverable for CuPy NVRTC JIT when CUDA_PATH is missing.
def _configure_cuda_header_paths() -> None:
    include_candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        include_candidates.append(Path(conda_prefix) / "targets" / "x86_64-linux" / "include")
        include_candidates.append(Path(conda_prefix) / "include")
    include_candidates.extend(
        [
            Path("/usr/include"),
            Path("/usr/local/cuda/include"),
            Path("/opt/cuda/include"),
        ]
    )

    chosen_include: Path | None = None
    for inc in include_candidates:
        if (inc / "cuda_fp16.h").exists():
            chosen_include = inc
            break
    if chosen_include is None:
        return

    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = str(chosen_include.parent)
    if "CUPY_INCLUDE_PATH" not in os.environ:
        os.environ["CUPY_INCLUDE_PATH"] = str(chosen_include)


_configure_cuda_header_paths()

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
    """JIT-backed ARC kernels for sovereignty-aligned ARC operations."""

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

    _DISCOVERY_SCORE_KERNEL = r"""
    extern "C" __global__
    void discovery_score_kernel(
        const float* confidence,
        const float* source_prior,
        const float* family_match,
        const float* novelty,
        float* out_score,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        out_score[idx] =
            0.45f * confidence[idx] +
            0.25f * source_prior[idx] +
            0.20f * family_match[idx] +
            0.10f * novelty[idx];
    }
    """

    _VALIDITY_SCORE_KERNEL = r"""
    extern "C" __global__
    void validity_score_kernel(
        const float* family_match,
        const float* shape_match,
        const float* palette_match,
        const float* object_match,
        const float w_family,
        const float w_shape,
        const float w_palette,
        const float w_object,
        float* out_score,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        out_score[idx] =
            w_family * family_match[idx] +
            w_shape * shape_match[idx] +
            w_palette * palette_match[idx] +
            w_object * object_match[idx];
    }
    """

    _FEATURE_EXTRACT_KERNEL = r"""
    extern "C" __global__
    void extract_pattern_features_kernel(
        const int* source_ids,
        const int* family_ids,
        const int expected_family,
        float* out_source_prior,
        float* out_family_match,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;

        const int src = source_ids[idx];
        float src_prior = 0.30f;  // unknown default
        if (src == 1) src_prior = 0.46f;      // contrastive_anti
        else if (src == 2) src_prior = 0.45f; // legacy_pipeline
        else if (src == 3) src_prior = 0.41f; // multi_galaxy_composition
        else if (src == 4) src_prior = 0.32f; // traditional
        else if (src == 5) src_prior = 0.19f; // autonomous_generation

        const int cand = family_ids[idx];
        float fam_match = 0.0f;
        // 0 unknown, 1 spatial, 2 spatial_or_recolor, 3 scale_or_translate,
        // 4 filter_or_count, 5 mixed
        if (expected_family == 0) {
            fam_match = 1.0f;
        } else if (expected_family == cand) {
            fam_match = 1.0f;
        } else if (expected_family == 1 || expected_family == 2) {
            fam_match = (cand == 1 || cand == 2) ? 1.0f : 0.0f;
        } else if (expected_family == 3) {
            fam_match = (cand == 3 || cand == 5) ? 1.0f : 0.0f;
        } else if (expected_family == 4) {
            fam_match = (cand == 4 || cand == 5) ? 1.0f : 0.0f;
        } else if (expected_family == 5) {
            fam_match = (cand == 5 || cand == 2 || cand == 3 || cand == 4) ? 1.0f : 0.0f;
        }

        out_source_prior[idx] = src_prior;
        out_family_match[idx] = fam_match;
    }
    """

    _FILTER_THRESHOLD_KERNEL = r"""
    extern "C" __global__
    void filter_by_threshold_kernel(
        const float* score,
        const float* family_match,
        const float threshold,
        const int strict_mode,
        unsigned char* keep_mask,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        unsigned char keep = (score[idx] >= threshold) ? 1 : 0;
        if (strict_mode && family_match[idx] < 0.5f) keep = 0;
        keep_mask[idx] = keep;
    }
    """

    _VALIDITY_FLAGS_KERNEL = r"""
    extern "C" __global__
    void check_grid_validity_kernel(
        const float* family_match,
        const float* shape_match,
        const float* palette_match,
        const float* object_match,
        const unsigned char* keep_mask,
        unsigned char* reject_family,
        unsigned char* reject_shape,
        unsigned char* reject_palette,
        unsigned char* reject_object,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        if (keep_mask[idx]) {
            reject_family[idx] = 0;
            reject_shape[idx] = 0;
            reject_palette[idx] = 0;
            reject_object[idx] = 0;
            return;
        }
        reject_family[idx] = (family_match[idx] < 0.5f) ? 1 : 0;
        reject_shape[idx] = (shape_match[idx] < 0.5f) ? 1 : 0;
        reject_palette[idx] = (palette_match[idx] < 0.5f) ? 1 : 0;
        reject_object[idx] = (object_match[idx] < 0.5f) ? 1 : 0;
    }
    """

    _COMPARE_GRIDS_KERNEL = r"""
    extern "C" __global__
    void compare_grids_kernel(
        const int* candidates,
        const int* expected,
        const int num_cells,
        float* out_scores,
        const int n
    ) {
        const int idx = blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) return;
        const int base = idx * num_cells;
        int matches = 0;
        for (int j = 0; j < num_cells; ++j) {
            if (candidates[base + j] == expected[j]) matches += 1;
        }
        out_scores[idx] = (num_cells > 0) ? ((float)matches / (float)num_cells) : 0.0f;
    }
    """

    def __init__(self) -> None:
        profile_flag = str(os.environ.get("K3D_PTX_PROFILE", "")).strip().lower()
        self._profile_enabled = profile_flag in {"1", "true", "yes", "on"}
        self._weighted_kernel = None
        self._argmax_kernel = None
        self._discovery_kernel = None
        self._validity_kernel = None
        self._feature_extract_kernel = None
        self._filter_threshold_kernel = None
        self._validity_flags_kernel = None
        self._compare_grids_kernel = None

    def _ptx_profile(self, label: str, start_evt: Any, end_evt: Any, **meta: Any) -> None:
        """Log kernel timing when K3D_PTX_PROFILE is enabled."""
        if not self._profile_enabled:
            return
        if not self.available:
            return
        assert cp is not None
        try:
            end_evt.synchronize()
            elapsed_ms = float(cp.cuda.get_elapsed_time(start_evt, end_evt))
        except Exception:
            return
        meta_str = " ".join(f"{k}={v}" for k, v in meta.items() if v is not None)
        if meta_str:
            print(f"[PTX PROFILE] {label}: {elapsed_ms:.3f}ms {meta_str}")
        else:
            print(f"[PTX PROFILE] {label}: {elapsed_ms:.3f}ms")

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
        if self._discovery_kernel is None:
            self._discovery_kernel = cp.RawKernel(self._DISCOVERY_SCORE_KERNEL, "discovery_score_kernel")
        if self._validity_kernel is None:
            self._validity_kernel = cp.RawKernel(self._VALIDITY_SCORE_KERNEL, "validity_score_kernel")
        if self._feature_extract_kernel is None:
            self._feature_extract_kernel = cp.RawKernel(self._FEATURE_EXTRACT_KERNEL, "extract_pattern_features_kernel")
        if self._filter_threshold_kernel is None:
            self._filter_threshold_kernel = cp.RawKernel(self._FILTER_THRESHOLD_KERNEL, "filter_by_threshold_kernel")
        if self._validity_flags_kernel is None:
            self._validity_flags_kernel = cp.RawKernel(self._VALIDITY_FLAGS_KERNEL, "check_grid_validity_kernel")
        if self._compare_grids_kernel is None:
            self._compare_grids_kernel = cp.RawKernel(self._COMPARE_GRIDS_KERNEL, "compare_grids_kernel")

    def _gpu_f32(self, values: Iterable[float]) -> Any:
        """Convert an iterable of numeric values into a float32 CuPy array."""
        assert cp is not None
        return cp.asarray(list(values), dtype=cp.float32)

    def _gpu_argsort_desc(self, values_gpu: Any) -> np.ndarray:
        """GPU argsort. CPU fallback is disallowed."""
        assert cp is not None
        try:
            return cp.asnumpy(cp.argsort(values_gpu)[::-1]).astype(np.int32, copy=False)
        except Exception:
            raise RuntimeError("ptx_argsort_failed")

    def _source_id(self, source: str) -> int:
        s = str(source or "unknown")
        if s == "contrastive_anti":
            return 1
        if s == "legacy_pipeline":
            return 2
        if s == "multi_galaxy_composition":
            return 3
        if s == "traditional":
            return 4
        if s == "autonomous_generation":
            return 5
        return 0

    def _family_id(self, family: str) -> int:
        f = str(family or "unknown")
        if f == "spatial":
            return 1
        if f == "spatial_or_recolor":
            return 2
        if f == "scale_or_translate":
            return 3
        if f == "filter_or_count":
            return 4
        if f == "mixed":
            return 5
        return 0

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
        evt_total_start = cp.cuda.Event()
        evt_total_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_total_start.record()

        source_precision_gpu = self._gpu_f32(source_precision)
        n = int(source_precision_gpu.size)
        if n <= 0:
            return ArcPtxRankingResult(scores=np.asarray([], dtype=np.float32), ranked_indices=[], top_index=0, mode="gpu_empty")

        quality_prior_gpu = self._gpu_f32(quality_prior)
        train_similarity_gpu = self._gpu_f32(train_similarity)
        novelty_gpu = self._gpu_f32(novelty)
        grammar_conf_gpu = self._gpu_f32(grammar_confidence)
        cross_modal_gpu = self._gpu_f32(cross_modal)
        compositional_gpu = self._gpu_f32(compositional)
        reuse_gpu = self._gpu_f32(reuse)
        family_bonus_gpu = self._gpu_f32(family_bonus)

        out_score_gpu = cp.zeros(n, dtype=cp.float32)
        threads = 128
        blocks = (n + threads - 1) // threads

        evt_weighted_start = cp.cuda.Event()
        evt_weighted_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_weighted_start.record()
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
        if self._profile_enabled:
            evt_weighted_end.record()
            self._ptx_profile("weighted_score_kernel", evt_weighted_start, evt_weighted_end, n=n)

        top_idx_gpu = cp.zeros(1, dtype=cp.int32)
        evt_argmax_start = cp.cuda.Event()
        evt_argmax_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_argmax_start.record()
        self._argmax_kernel((1,), (1,), (out_score_gpu, np.int32(n), top_idx_gpu))
        if self._profile_enabled:
            evt_argmax_end.record()
            self._ptx_profile("argmax_kernel", evt_argmax_start, evt_argmax_end, n=n)
        cp.cuda.runtime.deviceSynchronize()

        score_cpu = cp.asnumpy(out_score_gpu).astype(np.float32, copy=False)
        ranked_indices = self._gpu_argsort_desc(out_score_gpu).tolist()
        top_idx = int(cp.asnumpy(top_idx_gpu)[0])
        if top_idx in ranked_indices:
            ranked_indices = [top_idx] + [idx for idx in ranked_indices if idx != top_idx]

        if self._profile_enabled:
            evt_total_end.record()
            self._ptx_profile("rank_candidates_ternary.total", evt_total_start, evt_total_end, n=n)

        return ArcPtxRankingResult(
            scores=score_cpu,
            ranked_indices=ranked_indices,
            top_index=top_idx,
            mode="gpu_jit_ternary_ranking",
        )

    def discover_patterns_ptx(
        self,
        *,
        train_examples: list[dict[str, Any]],
        patterns: list[Any],
        top_k: int = 256,
    ) -> list[Any]:
        """PTX-sort discovered patterns by ARC-train consistency priors."""
        if not patterns:
            return []
        if not self.available:
            raise RuntimeError("arc_ptx_unavailable")
        self._ensure_kernels()
        assert cp is not None
        top_k = max(1, int(top_k))
        evt_total_start = cp.cuda.Event()
        evt_total_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_total_start.record()

        expected_family = self._infer_expected_family(train_examples)
        queries = [self._pattern_query(p) for p in patterns]
        sources = [self._pattern_source(p) for p in patterns]
        families = [self._infer_family_from_query(q) for q in queries]
        confidence = [self._pattern_confidence(p) for p in patterns]
        source_ids = [self._source_id(str(src)) for src in sources]
        family_ids = [self._family_id(str(fam)) for fam in families]
        expected_family_id = self._family_id(expected_family)
        freq = Counter(queries)
        novelty = [1.0 / float(max(1, freq.get(q, 1))) for q in queries]

        conf_gpu = self._gpu_f32(confidence)
        src_id_gpu = cp.asarray(source_ids, dtype=cp.int32)
        fam_id_gpu = cp.asarray(family_ids, dtype=cp.int32)
        src_gpu = cp.zeros(int(conf_gpu.size), dtype=cp.float32)
        fam_gpu = cp.zeros(int(conf_gpu.size), dtype=cp.float32)
        nov_gpu = self._gpu_f32(novelty)
        n = int(conf_gpu.size)
        threads = 256
        blocks = (n + threads - 1) // threads
        evt_feature_start = cp.cuda.Event()
        evt_feature_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_feature_start.record()
        self._feature_extract_kernel(
            (blocks,),
            (threads,),
            (
                src_id_gpu,
                fam_id_gpu,
                np.int32(expected_family_id),
                src_gpu,
                fam_gpu,
                np.int32(n),
            ),
        )
        if self._profile_enabled:
            evt_feature_end.record()
            self._ptx_profile("extract_pattern_features_kernel", evt_feature_start, evt_feature_end, n=n)
        out_score_gpu = cp.zeros(n, dtype=cp.float32)
        evt_discovery_start = cp.cuda.Event()
        evt_discovery_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_discovery_start.record()
        self._discovery_kernel(
            (blocks,),
            (threads,),
            (conf_gpu, src_gpu, fam_gpu, nov_gpu, out_score_gpu, np.int32(n)),
        )
        if self._profile_enabled:
            evt_discovery_end.record()
            self._ptx_profile("discovery_score_kernel", evt_discovery_start, evt_discovery_end, n=n)
        cp.cuda.runtime.deviceSynchronize()
        scores = cp.asnumpy(out_score_gpu).astype(np.float32, copy=False)
        ranked_indices = self._gpu_argsort_desc(out_score_gpu)[:top_k]
        ranked: list[Any] = []
        for idx in ranked_indices.tolist():
            pattern = patterns[int(idx)]
            score = float(scores[int(idx)])
            if isinstance(pattern, dict):
                pattern["ptx_discovery_score"] = score
            else:
                try:
                    setattr(pattern, "ptx_discovery_score", score)
                except Exception:
                    pass
            ranked.append(pattern)
        if self._profile_enabled:
            evt_total_end.record()
            self._ptx_profile("discover_patterns_ptx.total", evt_total_start, evt_total_end, n=n, top_k=top_k)
        return ranked

    def apply_validity_gates_relaxed_ptx(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        validity_profile: dict[str, Any],
        strictness: str = "medium",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Apply configurable strict/medium/relaxed validity gates on GPU."""
        if not ranked_candidates:
            return [], {
                "enabled": True,
                "mode": "ptx_validity",
                "strictness": strictness,
                "pre_count": 0,
                "post_count": 0,
                "filtered_count": 0,
                "fallback_to_ungated": False,
                "family_rejects": 0,
                "shape_rejects": 0,
                "palette_rejects": 0,
                "object_rejects": 0,
                "validity_reject_rate": 0.0,
            }
        if not self.available:
            raise RuntimeError("arc_ptx_unavailable")
        self._ensure_kernels()
        assert cp is not None
        evt_total_start = cp.cuda.Event()
        evt_total_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_total_start.record()

        strictness_key = str(strictness or "medium").strip().lower()
        if strictness_key not in {"strict", "medium", "relaxed"}:
            strictness_key = "medium"

        weights = {
            "strict": (0.45, 0.30, 0.15, 0.10, 0.92),
            "medium": (0.30, 0.30, 0.20, 0.20, 0.62),
            "relaxed": (0.20, 0.20, 0.25, 0.35, 0.45),
        }
        w_family, w_shape, w_palette, w_object, threshold = weights[strictness_key]

        # Fast-path: reuse already-computed constraint components from ranking.
        # This avoids re-running heavy per-candidate grid analysis on CPU.
        fam_v: list[float]
        shape_v: list[float]
        palette_v: list[float]
        obj_v: list[float]
        has_component_scores = True
        fam_v = []
        shape_v = []
        palette_v = []
        obj_v = []
        for item in ranked_candidates:
            comp = item.get("components", {}) if isinstance(item.get("components"), dict) else {}
            if not comp:
                has_component_scores = False
                break
            if "family_score" not in comp or "shape_score" not in comp or "palette_score" not in comp or "object_score" not in comp:
                has_component_scores = False
                break
            fam_v.append(float(comp.get("family_score", 0.0)))
            shape_v.append(float(comp.get("shape_score", 0.0)))
            palette_v.append(float(comp.get("palette_score", 0.0)))
            obj_v.append(float(comp.get("object_score", 0.0)))

        if not has_component_scores:
            fam_v = []
            shape_v = []
            palette_v = []
            obj_v = []
            test_grid = self._to_grid(validity_profile.get("test_input_grid"))
            test_shape = (
                (len(test_grid), len(test_grid[0]) if test_grid else 0)
                if test_grid
                else None
            )
            test_palette = self._palette_of(test_grid) if test_grid else set()
            expected_object_count = validity_profile.get("expected_object_count")
            expected_family = str(validity_profile.get("inferred_family", "unknown") or "unknown")
            # Only compute input object count if a downstream check can use it.
            needs_input_objects = bool(
                expected_object_count is not None
                or expected_family in {"filter_or_count", "mixed"}
                or strictness_key == "strict"
            )
            test_object_count = self._count_connected_objects(test_grid) if (test_grid and needs_input_objects) else None

            for item in ranked_candidates:
                grid = self._to_grid(item.get("candidate"))
                family_ok, shape_ok, palette_ok, object_ok = self._candidate_validity_bits_with_stats(
                    candidate_grid=grid,
                    profile=validity_profile,
                    test_shape=test_shape,
                    test_palette=test_palette,
                    test_object_count=test_object_count,
                )
                fam_v.append(1.0 if family_ok else 0.0)
                shape_v.append(1.0 if shape_ok else 0.0)
                palette_v.append(1.0 if palette_ok else 0.0)
                obj_v.append(1.0 if object_ok else 0.0)

        fam_gpu = cp.asarray(fam_v, dtype=cp.float32)
        shape_gpu = cp.asarray(shape_v, dtype=cp.float32)
        palette_gpu = cp.asarray(palette_v, dtype=cp.float32)
        object_gpu = cp.asarray(obj_v, dtype=cp.float32)
        n = int(fam_gpu.size)
        score_gpu = cp.zeros(n, dtype=cp.float32)
        threads = 256
        blocks = (n + threads - 1) // threads
        evt_validity_start = cp.cuda.Event()
        evt_validity_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_validity_start.record()
        self._validity_kernel(
            (blocks,),
            (threads,),
            (
                fam_gpu,
                shape_gpu,
                palette_gpu,
                object_gpu,
                np.float32(w_family),
                np.float32(w_shape),
                np.float32(w_palette),
                np.float32(w_object),
                score_gpu,
                np.int32(n),
            ),
        )
        if self._profile_enabled:
            evt_validity_end.record()
            self._ptx_profile("validity_score_kernel", evt_validity_start, evt_validity_end, n=n, strictness=strictness_key)
        cp.cuda.runtime.deviceSynchronize()
        keep_gpu = cp.zeros(n, dtype=cp.uint8)
        evt_filter_start = cp.cuda.Event()
        evt_filter_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_filter_start.record()
        self._filter_threshold_kernel(
            (blocks,),
            (threads,),
            (
                score_gpu,
                fam_gpu,
                np.float32(threshold),
                np.int32(1 if strictness_key == "strict" else 0),
                keep_gpu,
                np.int32(n),
            ),
        )
        if self._profile_enabled:
            evt_filter_end.record()
            self._ptx_profile("filter_by_threshold_kernel", evt_filter_start, evt_filter_end, n=n, threshold=float(threshold))

        rej_family_gpu = cp.zeros(n, dtype=cp.uint8)
        rej_shape_gpu = cp.zeros(n, dtype=cp.uint8)
        rej_palette_gpu = cp.zeros(n, dtype=cp.uint8)
        rej_object_gpu = cp.zeros(n, dtype=cp.uint8)
        evt_flags_start = cp.cuda.Event()
        evt_flags_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_flags_start.record()
        self._validity_flags_kernel(
            (blocks,),
            (threads,),
            (
                fam_gpu,
                shape_gpu,
                palette_gpu,
                object_gpu,
                keep_gpu,
                rej_family_gpu,
                rej_shape_gpu,
                rej_palette_gpu,
                rej_object_gpu,
                np.int32(n),
            ),
        )
        if self._profile_enabled:
            evt_flags_end.record()
            self._ptx_profile("check_grid_validity_kernel", evt_flags_start, evt_flags_end, n=n)
        cp.cuda.runtime.deviceSynchronize()

        keep_cpu = cp.asnumpy(keep_gpu).astype(np.uint8, copy=False)
        keep_idx = np.where(keep_cpu > np.uint8(0))[0].astype(np.int32, copy=False).tolist()
        filtered: list[dict[str, Any]] = [ranked_candidates[int(i)] for i in keep_idx]

        family_rejects = int(np.sum(cp.asnumpy(rej_family_gpu).astype(np.uint8, copy=False)))
        shape_rejects = int(np.sum(cp.asnumpy(rej_shape_gpu).astype(np.uint8, copy=False)))
        palette_rejects = int(np.sum(cp.asnumpy(rej_palette_gpu).astype(np.uint8, copy=False)))
        object_rejects = int(np.sum(cp.asnumpy(rej_object_gpu).astype(np.uint8, copy=False)))

        fallback = False
        if not filtered:
            filtered = []
            fallback = False
        pre_count = len(ranked_candidates)
        post_count = len(filtered)
        filtered_count = pre_count - post_count if not fallback else pre_count
        reject_rate = (filtered_count / pre_count) if pre_count else 0.0
        report = {
            "enabled": True,
            "mode": "ptx_validity",
            "strictness": strictness_key,
            "pre_count": pre_count,
            "post_count": post_count,
            "filtered_count": filtered_count,
            "fallback_to_ungated": fallback,
            "family_rejects": family_rejects,
            "shape_rejects": shape_rejects,
            "palette_rejects": palette_rejects,
            "object_rejects": object_rejects,
            "validity_reject_rate": reject_rate,
            "validity_threshold": float(threshold),
        }
        if self._profile_enabled:
            evt_total_end.record()
            self._ptx_profile("apply_validity_gates_relaxed_ptx.total", evt_total_start, evt_total_end, n=n, strictness=strictness_key)
        return filtered, report

    def check_oracle_fuzzy_ptx(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        expected_grid: list[list[int]],
        fuzzy_threshold: float,
        thresholds: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95),
    ) -> dict[str, Any]:
        """Compute fuzzy/exact oracle metrics on GPU from candidate grids."""
        if not ranked_candidates or not expected_grid:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_exact": False,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": False,
                "fuzzy_oracle_at_all": False,
                "fuzzy_best_score": 0.0,
                "fuzzy_best_rank": None,
                **{f"oracle_fuzzy_{self._threshold_key(t)}": False for t in thresholds},
            }
        if not self.available:
            raise RuntimeError("arc_ptx_unavailable")
        self._ensure_kernels()
        assert cp is not None
        evt_total_start = cp.cuda.Event()
        evt_total_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_total_start.record()

        exp_h = len(expected_grid)
        exp_w = len(expected_grid[0]) if exp_h else 0
        if exp_h == 0 or exp_w == 0:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_exact": False,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": False,
                "fuzzy_oracle_at_all": False,
                "fuzzy_best_score": 0.0,
                "fuzzy_best_rank": None,
                **{f"oracle_fuzzy_{self._threshold_key(t)}": False for t in thresholds},
            }

        expected_flat = [int(cell) for row in expected_grid for cell in row]
        candidate_rows: list[list[int]] = []
        for item in ranked_candidates:
            grid = self._to_grid(item.get("candidate"))
            if grid and len(grid) == exp_h and len(grid[0]) == exp_w:
                candidate_rows.append([int(cell) for row in grid for cell in row])
            else:
                resized = self._resize_nn(grid, exp_h, exp_w)
                candidate_rows.append([int(cell) for row in resized for cell in row])
        if not candidate_rows:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_exact": False,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": False,
                "fuzzy_oracle_at_all": False,
                "fuzzy_best_score": 0.0,
                "fuzzy_best_rank": None,
                **{f"oracle_fuzzy_{self._threshold_key(t)}": False for t in thresholds},
            }

        cand_gpu = cp.asarray(candidate_rows, dtype=cp.int32)
        exp_gpu = cp.asarray(expected_flat, dtype=cp.int32)
        n = int(cand_gpu.shape[0])
        num_cells = int(cand_gpu.shape[1])
        fuzzy_scores = cp.zeros(n, dtype=cp.float32)
        threads = 128
        blocks = (n + threads - 1) // threads
        evt_compare_start = cp.cuda.Event()
        evt_compare_end = cp.cuda.Event()
        if self._profile_enabled:
            evt_compare_start.record()
        self._compare_grids_kernel(
            (blocks,),
            (threads,),
            (
                cand_gpu.ravel(),
                exp_gpu,
                np.int32(num_cells),
                fuzzy_scores,
                np.int32(n),
            ),
        )
        if self._profile_enabled:
            evt_compare_end.record()
            self._ptx_profile("compare_grids_kernel", evt_compare_start, evt_compare_end, n=n, num_cells=num_cells)
        cp.cuda.runtime.deviceSynchronize()
        scores = cp.asnumpy(fuzzy_scores).astype(np.float32, copy=False)
        best_rank = int(np.argmax(scores))
        best_score = float(scores[best_rank])
        exact_idx = np.where(scores >= 0.999999)[0]
        exact_rank = int(exact_idx[0]) if exact_idx.size else None

        stratified_hits = {t: bool(np.any(scores >= float(t))) for t in thresholds}
        fuzzy_thr = float(max(0.5, min(0.99, fuzzy_threshold)))
        fuzzy_at_3 = bool(np.any(scores[:3] >= fuzzy_thr))
        fuzzy_at_10 = bool(np.any(scores[:10] >= fuzzy_thr))
        fuzzy_at_all = bool(np.any(scores >= fuzzy_thr))

        out = {
            "oracle_at_3": bool(exact_rank is not None and exact_rank < 3),
            "oracle_at_10": bool(exact_rank is not None and exact_rank < 10),
            "oracle_at_all": bool(exact_rank is not None),
            "correct_rank": exact_rank,
            "oracle_exact": bool(exact_rank is not None),
            "fuzzy_oracle_at_3": fuzzy_at_3,
            "fuzzy_oracle_at_10": fuzzy_at_10,
            "fuzzy_oracle_at_all": fuzzy_at_all,
            "fuzzy_best_score": best_score,
            "fuzzy_best_rank": best_rank,
            **{f"oracle_fuzzy_{self._threshold_key(t)}": bool(v) for t, v in stratified_hits.items()},
        }
        if self._profile_enabled:
            evt_total_end.record()
            self._ptx_profile("check_oracle_fuzzy_ptx.total", evt_total_start, evt_total_end, n=n, num_cells=num_cells)
        return out

    def _check_oracle_fuzzy_cpu(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        expected_grid: list[list[int]],
        fuzzy_threshold: float,
        thresholds: tuple[float, ...],
    ) -> dict[str, Any]:
        exp_h = len(expected_grid)
        exp_w = len(expected_grid[0]) if exp_h else 0
        if exp_h == 0 or exp_w == 0:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_exact": False,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": False,
                "fuzzy_oracle_at_all": False,
                "fuzzy_best_score": 0.0,
                "fuzzy_best_rank": None,
                **{f"oracle_fuzzy_{self._threshold_key(t)}": False for t in thresholds},
            }
        scores: list[float] = []
        for item in ranked_candidates:
            grid = self._to_grid(item.get("candidate"))
            resized = self._resize_nn(grid, exp_h, exp_w)
            matches = 0
            total = exp_h * exp_w
            for r in range(exp_h):
                for c in range(exp_w):
                    if int(resized[r][c]) == int(expected_grid[r][c]):
                        matches += 1
            scores.append((matches / total) if total else 0.0)
        if not scores:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_exact": False,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": False,
                "fuzzy_oracle_at_all": False,
                "fuzzy_best_score": 0.0,
                "fuzzy_best_rank": None,
                **{f"oracle_fuzzy_{self._threshold_key(t)}": False for t in thresholds},
            }
        best_rank = int(np.argmax(scores))
        best_score = float(scores[best_rank])
        exact_rank = None
        for i, v in enumerate(scores):
            if v >= 0.999999:
                exact_rank = i
                break
        fuzzy_thr = float(max(0.5, min(0.99, fuzzy_threshold)))
        stratified_hits = {t: any(v >= float(t) for v in scores) for t in thresholds}
        return {
            "oracle_at_3": bool(exact_rank is not None and exact_rank < 3),
            "oracle_at_10": bool(exact_rank is not None and exact_rank < 10),
            "oracle_at_all": bool(exact_rank is not None),
            "correct_rank": exact_rank,
            "oracle_exact": bool(exact_rank is not None),
            "fuzzy_oracle_at_3": any(v >= fuzzy_thr for v in scores[:3]),
            "fuzzy_oracle_at_10": any(v >= fuzzy_thr for v in scores[:10]),
            "fuzzy_oracle_at_all": any(v >= fuzzy_thr for v in scores),
            "fuzzy_best_score": best_score,
            "fuzzy_best_rank": best_rank,
            **{f"oracle_fuzzy_{self._threshold_key(t)}": bool(v) for t, v in stratified_hits.items()},
        }

    def _pattern_confidence(self, pattern: Any) -> float:
        if isinstance(pattern, dict):
            return self._clamp(float(pattern.get("confidence", 0.5)))
        return self._clamp(float(getattr(pattern, "confidence", 0.5)))

    def _pattern_source(self, pattern: Any) -> str:
        if isinstance(pattern, dict):
            return str(pattern.get("source", "unknown"))
        return str(getattr(pattern, "source", "unknown"))

    def _pattern_query(self, pattern: Any) -> str:
        if isinstance(pattern, dict):
            return str(pattern.get("query", "")).strip().lower()
        return str(getattr(pattern, "query", "")).strip().lower()

    def _pattern_family(self, pattern: Any) -> str:
        return self._infer_family_from_query(self._pattern_query(pattern))

    def _source_prior(self, source: str) -> float:
        priors = {
            "contrastive_anti": 0.46,
            "legacy_pipeline": 0.45,
            "multi_galaxy_composition": 0.41,
            "traditional": 0.32,
            "autonomous_generation": 0.19,
            "unknown": 0.30,
        }
        return self._clamp(float(priors.get(source, priors["unknown"])))

    def _infer_expected_family(self, train_examples: list[dict[str, Any]]) -> str:
        input_shapes: list[tuple[int, int]] = []
        output_shapes: list[tuple[int, int]] = []
        in_objects: list[int] = []
        out_objects: list[int] = []
        preserve_palette = True
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            in_grid = self._to_grid(pair.get("input"))
            out_grid = self._to_grid(pair.get("output"))
            if not in_grid or not out_grid:
                continue
            input_shapes.append((len(in_grid), len(in_grid[0])))
            output_shapes.append((len(out_grid), len(out_grid[0])))
            in_objects.append(self._count_connected_objects(in_grid))
            out_objects.append(self._count_connected_objects(out_grid))
            if self._palette_of(in_grid) != self._palette_of(out_grid):
                preserve_palette = False
        if not input_shapes or not output_shapes:
            return "unknown"
        shape_preserved = all(i == o for i, o in zip(input_shapes, output_shapes))
        object_preserved = all(i == o for i, o in zip(in_objects, out_objects))
        if shape_preserved and object_preserved:
            return "spatial_or_recolor" if not preserve_palette else "spatial"
        deltas = [(o[0] - i[0], o[1] - i[1]) for i, o in zip(input_shapes, output_shapes)]
        if deltas and len(set(deltas)) == 1:
            return "scale_or_translate"
        obj_deltas = [o - i for i, o in zip(in_objects, out_objects)]
        if obj_deltas and len(set(obj_deltas)) == 1:
            return "filter_or_count"
        return "mixed"

    def _infer_family_from_query(self, query: str) -> str:
        q = str(query or "").lower()
        if any(k in q for k in ("rotate", "mirror", "reflect", "flip")):
            return "spatial"
        if any(k in q for k in ("scale", "resize", "shrink", "translate")):
            return "scale_or_translate"
        if any(k in q for k in ("count", "filter", "object count", "connected component")):
            return "filter_or_count"
        if any(k in q for k in ("color transformation", "recolor")):
            return "spatial_or_recolor"
        return "mixed"

    def _families_compatible(self, expected_family: str, candidate_family: str) -> bool:
        if expected_family == "unknown":
            return True
        if expected_family == candidate_family:
            return True
        compatible = {
            "spatial": {"spatial", "spatial_or_recolor"},
            "spatial_or_recolor": {"spatial", "spatial_or_recolor"},
            "scale_or_translate": {"scale_or_translate", "mixed"},
            "filter_or_count": {"filter_or_count", "mixed"},
            "mixed": {"mixed", "spatial_or_recolor", "scale_or_translate", "filter_or_count"},
        }
        return candidate_family in compatible.get(expected_family, {expected_family})

    def _candidate_validity_bits(
        self,
        candidate_grid: list[list[int]],
        profile: dict[str, Any],
    ) -> tuple[bool, bool, bool, bool]:
        if not candidate_grid:
            return False, False, False, False
        expected_family = str(profile.get("inferred_family", "") or "unknown")
        test_grid = self._to_grid(profile.get("test_input_grid")) or candidate_grid
        family = self._infer_expected_family(
            [{"input": test_grid, "output": candidate_grid}]
        )
        family_ok = self._families_compatible(expected_family, family)

        expected_shape = profile.get("expected_shape")
        if expected_shape is None:
            shape_ok = True
        else:
            shape_ok = (len(candidate_grid), len(candidate_grid[0])) == tuple(expected_shape)

        output_palette = set(profile.get("output_palette", []))
        stable_palette_size = profile.get("stable_output_palette_size")
        candidate_palette = self._palette_of(candidate_grid)
        palette_ok = True
        if output_palette and profile.get("preserve_palette") is False:
            palette_ok = candidate_palette.issubset(output_palette)
        if palette_ok and stable_palette_size is not None:
            palette_ok = len(candidate_palette) == int(stable_palette_size)

        expected_object_count = profile.get("expected_object_count")
        object_ok = True
        if expected_object_count is not None:
            object_ok = self._count_connected_objects(candidate_grid) == int(expected_object_count)

        return family_ok, shape_ok, palette_ok, object_ok

    def _candidate_validity_bits_with_stats(
        self,
        *,
        candidate_grid: list[list[int]],
        profile: dict[str, Any],
        test_shape: tuple[int, int] | None,
        test_palette: set[int],
        test_object_count: int | None,
    ) -> tuple[bool, bool, bool, bool]:
        """Faster validity check path with precomputed input stats."""
        if not candidate_grid:
            return False, False, False, False

        expected_family = str(profile.get("inferred_family", "") or "unknown")
        candidate_shape = (len(candidate_grid), len(candidate_grid[0]) if candidate_grid else 0)
        candidate_palette = self._palette_of(candidate_grid)
        expected_shape = profile.get("expected_shape")
        output_palette = set(profile.get("output_palette", []))
        stable_palette_size = profile.get("stable_output_palette_size")
        expected_object_count = profile.get("expected_object_count")

        # Shape gate.
        shape_ok = True
        if expected_shape is not None:
            shape_ok = candidate_shape == tuple(expected_shape)

        # Palette gate.
        palette_ok = True
        if output_palette and profile.get("preserve_palette") is False:
            palette_ok = candidate_palette.issubset(output_palette)
        if palette_ok and stable_palette_size is not None:
            palette_ok = len(candidate_palette) == int(stable_palette_size)

        # Object gate.
        candidate_objects: int | None = None
        object_ok = True
        if expected_object_count is not None:
            candidate_objects = self._count_connected_objects(candidate_grid)
            object_ok = candidate_objects == int(expected_object_count)

        # Family gate (approximate, avoids repeated full infer).
        family_ok = True
        if expected_family and expected_family != "unknown":
            inferred = "mixed"
            if test_shape is not None:
                same_shape = candidate_shape == test_shape
                if same_shape:
                    if test_object_count is not None:
                        if candidate_objects is None:
                            candidate_objects = self._count_connected_objects(candidate_grid)
                        object_preserved = candidate_objects == test_object_count
                    else:
                        object_preserved = True
                    if object_preserved:
                        inferred = "spatial_or_recolor" if candidate_palette != test_palette else "spatial"
                    else:
                        inferred = "filter_or_count"
                else:
                    inferred = "scale_or_translate"
            family_ok = self._families_compatible(expected_family, inferred)

        return family_ok, shape_ok, palette_ok, object_ok

    def _resize_nn(self, grid: list[list[int]], target_h: int, target_w: int) -> list[list[int]]:
        if not grid or not grid[0]:
            return [[0 for _ in range(max(1, target_w))] for _ in range(max(1, target_h))]
        src_h = len(grid)
        src_w = len(grid[0])
        if src_h == target_h and src_w == target_w:
            return [row[:] for row in grid]
        out = [[0 for _ in range(max(1, target_w))] for _ in range(max(1, target_h))]
        for r in range(len(out)):
            rr = min(src_h - 1, int((r / max(1, len(out))) * src_h))
            for c in range(len(out[0])):
                cc = min(src_w - 1, int((c / max(1, len(out[0]))) * src_w))
                out[r][c] = int(grid[rr][cc])
        return out

    def _to_grid(self, value: Any) -> list[list[int]]:
        if value is None:
            return []
        if isinstance(value, list):
            if not value or not isinstance(value[0], list):
                return []
            out: list[list[int]] = []
            for row in value:
                if not isinstance(row, list):
                    return []
                out.append([int(cell) for cell in row])
            return out
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return self._to_grid(converted)
        return []

    def _palette_of(self, grid: list[list[int]]) -> set[int]:
        return {int(cell) for row in grid for cell in row}

    def _count_connected_objects(self, grid: list[list[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        h = len(grid)
        w = len(grid[0])
        visited = [[False] * w for _ in range(h)]
        components = 0
        for r in range(h):
            for c in range(w):
                if visited[r][c] or grid[r][c] == 0:
                    continue
                components += 1
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    rr, cc = stack.pop()
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if visited[nr][nc] or grid[nr][nc] == 0:
                            continue
                        visited[nr][nc] = True
                        stack.append((nr, nc))
        return components

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, float(value)))

    def _threshold_key(self, value: float) -> str:
        text = f"{float(value):.2f}"
        whole, frac = text.split(".")
        return f"{whole}_{frac}"


ARC_PTX_OPS = ARCPTXOps()
