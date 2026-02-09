"""Adapter bridge from Week 14 ARC benchmark to legacy SovereignAIPipeline."""

from __future__ import annotations

import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.ternary_quality_memory import TernaryQualityMemory

# Keep CUDA headers discoverable for CuPy JIT in environments that ship
# headers under /usr/include without exporting CUDA_PATH.
if "CUDA_PATH" not in os.environ and Path("/usr/include/cuda_fp16.h").exists():
    os.environ["CUDA_PATH"] = "/usr"

try:  # pragma: no cover - optional GPU dependency
    import cupy as cp  # type: ignore

    _HAS_CUPY = True
except Exception:  # pragma: no cover - CPU-only environments
    cp = None  # type: ignore
    _HAS_CUPY = False

try:  # pragma: no cover - optional PTX runtime dependency
    from knowledge3d.cranium.ptx import ARC_PTX_OPS

    _HAS_PTX_OPS = True
except Exception:  # pragma: no cover - keep benchmark runnable without PTX stack
    ARC_PTX_OPS = None  # type: ignore
    _HAS_PTX_OPS = False


@dataclass
class _GeneratedPattern:
    """Internal bookkeeping for autonomous ARC pattern generation."""

    pattern_id: str
    source_galaxy: str
    target_galaxy: str
    confidence: float
    query: str
    source: str
    pair_index: int | None = None


class ArcAgi2Adapter:
    """Translate Week 14 benchmark tasks into legacy ARC sovereign pipeline calls."""

    def __init__(
        self,
        *,
        use_enriched: bool = True,
        strict_legacy: bool = False,
        knowledgeverse: Knowledgeverse | None = None,
        enable_contrastive_learning: bool = False,
        enable_validity_gates: bool = False,
        enable_fuzzy_oracle: bool = False,
        fuzzy_oracle_threshold: float = 0.95,
        enable_ptx_ranking: bool = False,
    ):
        self.use_enriched = use_enriched
        self.strict_legacy = strict_legacy
        self.knowledgeverse = knowledgeverse
        self.enable_contrastive_learning = bool(enable_contrastive_learning)
        self.enable_validity_gates = bool(enable_validity_gates)
        self.enable_fuzzy_oracle = bool(enable_fuzzy_oracle)
        self.fuzzy_oracle_threshold = max(0.50, min(0.99, float(fuzzy_oracle_threshold)))
        self.enable_ptx_ranking = bool(enable_ptx_ranking)
        self._ptx_ranking_available = bool(self.enable_ptx_ranking and _HAS_CUPY and _HAS_PTX_OPS)
        ptx_unavailable_reasons: list[str] = []
        if self.enable_ptx_ranking and not _HAS_CUPY:
            ptx_unavailable_reasons.append("cupy_missing")
        if self.enable_ptx_ranking and not _HAS_PTX_OPS:
            ptx_unavailable_reasons.append("ptx_ops_unavailable")
        self._ptx_unavailable_reason = ",".join(ptx_unavailable_reasons) if ptx_unavailable_reasons else None
        if self.enable_ptx_ranking and self._ptx_ranking_available and ARC_PTX_OPS is not None:
            try:
                ARC_PTX_OPS.rank_candidates_ternary(
                    source_precision=[0.5],
                    quality_prior=[0.5],
                    train_similarity=[0.5],
                    novelty=[0.5],
                    grammar_confidence=[0.5],
                    cross_modal=[0.5],
                    compositional=[0.5],
                    reuse=[0.5],
                    family_bonus=[0.0],
                )
            except Exception as exc:
                self._ptx_ranking_available = False
                reason = f"ptx_selftest_failed:{exc}"
                self._ptx_unavailable_reason = (
                    f"{self._ptx_unavailable_reason},{reason}"
                    if self._ptx_unavailable_reason
                    else reason
                )
        self._last_ranking_debug: dict[str, Any] = {
            "ptx_used": False,
            "ptx_top_index": None,
            "ptx_mode": ("cpu_ptx_disabled" if not self.enable_ptx_ranking else "cpu_ptx_unavailable"),
            "ptx_error": self._ptx_unavailable_reason,
        }
        self.pipeline = None
        self._init_error: str | None = None
        self.quality_memory: TernaryQualityMemory | None = None
        if knowledgeverse is not None and hasattr(knowledgeverse, "storage_root"):
            state_path = Path(getattr(knowledgeverse, "storage_root")) / "checkpoints" / "arc_quality_memory.json"
            self.quality_memory = TernaryQualityMemory(state_path=state_path, emit_galaxy_entries=False)

        try:
            from knowledge3d.training.arc_agi import SovereignAIPipeline

            self.pipeline = SovereignAIPipeline(
                matryoshka_dim=512 if use_enriched else 128,
                hybrid_mode=use_enriched,
                knowledgeverse=self.knowledgeverse,
            )
        except Exception as exc:  # pragma: no cover - environment dependent.
            self._init_error = str(exc)
            if strict_legacy:
                raise

    def solve_task(
        self,
        task: dict[str, Any],
        *,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one ARC task.

        If the legacy pipeline is unavailable or fails and `strict_legacy=False`,
        an optional fallback solver can be used to keep the benchmark runnable.
        """
        if self.pipeline is None:
            return self._fallback_or_raise(task, "pipeline_unavailable", fallback_solver)

        task_id = str(task.get("id", "unknown"))
        test_block = task.get("test") or [{}]
        test_input = test_block[0].get("input")
        expected_output = test_block[0].get("output")
        train_examples = task.get("train") or []
        discovered_patterns = self.discover_patterns(train_examples)
        generated_patterns = [
            p
            for p in discovered_patterns
            if p.source in {"autonomous_generation", "contrastive_anti"}
        ]
        traditional_patterns = [p for p in discovered_patterns if p.source == "traditional"]
        cross_modal_patterns = [p for p in discovered_patterns if p.source == "multi_galaxy_composition"]
        contrastive_patterns = [p for p in discovered_patterns if p.source == "contrastive_anti"]

        try:
            result = self.pipeline.process_task(
                task_id=task_id,
                test_input=test_input,
                train_examples=train_examples,
                expected_output=expected_output,
                top_k=9 if self.use_enriched else 3,
                record_submission=False,
            )
            predicted = result.output_grid
            exact_match = self._grids_match(predicted, expected_output)
            generated_conf_mean = 0.0
            if generated_patterns:
                generated_conf_mean = sum(p.confidence for p in generated_patterns) / len(generated_patterns)
            validity_profile = self._build_validity_profile(
                train_examples=train_examples,
                test_input=test_input,
            )
            ranked_candidates = self._rank_candidates_for_task(
                test_input=test_input,
                legacy_prediction=predicted,
                discovered_patterns=discovered_patterns,
                validity_profile=validity_profile,
                return_debug=True,
            )
            if isinstance(ranked_candidates, tuple):
                ranked_candidates, ranking_debug = ranked_candidates
            else:
                ranking_debug = {}
            validity_report: dict[str, Any] = {
                "enabled": self.enable_validity_gates,
                "pre_count": len(ranked_candidates),
                "post_count": len(ranked_candidates),
                "filtered_count": 0,
                "fallback_to_ungated": False,
                "family_rejects": 0,
                "shape_rejects": 0,
                "palette_rejects": 0,
                "object_rejects": 0,
            }
            if self.enable_validity_gates and ranked_candidates:
                ranked_candidates, validity_report = self._apply_validity_gates(
                    ranked_candidates=ranked_candidates,
                    validity_profile=validity_profile,
                )
            ranked_prediction = predicted
            ranking_applied = bool(ranked_candidates)
            ranking_override_used = False
            ranking_top = None
            legacy_rank = None
            pre_top_source = str(ranking_debug.get("pre_top_source", "unknown"))
            if ranked_candidates:
                ranking_top = ranked_candidates[0]
                for item in ranked_candidates:
                    if item.get("pattern", {}).get("pattern_id") == "legacy_pipeline_output":
                        legacy_rank = item
                        break

                if self._should_apply_ranking_override(ranking_top, legacy_rank):
                    ranked_prediction = ranking_top.get("candidate", predicted)
                    ranking_override_used = True

            ranked_exact_match = self._grids_match(ranked_prediction, expected_output)
            oracle_metrics = self._compute_oracle_metrics(
                ranked_candidates,
                expected_output,
                fuzzy_threshold=(self.fuzzy_oracle_threshold if self.enable_fuzzy_oracle else None),
            )
            oracle_diagnostics = self.evaluate_task_with_oracle_diagnostics(
                predicted=ranked_prediction,
                expected=expected_output,
                validity_profile=validity_profile,
                validity_report=validity_report,
                oracle_metrics=oracle_metrics,
            )
            top_5 = ranked_candidates[:5]
            top_5_scores = [float(item.get("score", 0.0)) for item in top_5]
            top_5_sources = [str(item.get("pattern", {}).get("source", "unknown")) for item in top_5]
            score_range = (
                (max(top_5_scores) - min(top_5_scores)) if len(top_5_scores) >= 2 else 0.0
            )
            score_stddev = (
                statistics.pstdev(top_5_scores) if len(top_5_scores) >= 2 else 0.0
            )
            legacy_correct = bool(result.correct)
            # Preserve legacy benchmark semantics (fuzzy-aware correctness) so
            # ranking experiments cannot silently zero out baseline performance.
            final_correct = legacy_correct
            if ranking_override_used and expected_output is not None:
                # Override can only improve correctness when exact match is achieved.
                final_correct = legacy_correct or ranked_exact_match
            post_top_source = (
                str(ranking_top.get("pattern", {}).get("source", "unknown")) if ranking_top else "legacy_pipeline"
            )
            ranking_changed_top1 = bool(ranking_applied and pre_top_source != post_top_source)
            ptx_ranking_used = bool(ranking_debug.get("ptx_used", False))
            ptx_top_index = ranking_debug.get("ptx_top_index")
            ptx_mode = str(ranking_debug.get("ptx_mode", "cpu"))
            ptx_error = ranking_debug.get("ptx_error")
            self._update_quality_memory(
                ranked_candidates=ranked_candidates,
                ranking_top=ranking_top,
                final_correct=final_correct,
                oracle_metrics=oracle_metrics,
            )
            if self.knowledgeverse is not None:
                self.knowledgeverse.log_event(
                    event_type="arc_pattern_discovery",
                    event_data={
                        "task_id": task_id,
                        "total_patterns": len(discovered_patterns),
                        "generated_patterns": len(generated_patterns),
                        "traditional_patterns": len(traditional_patterns),
                        "cross_modal_patterns": len(cross_modal_patterns),
                        "contrastive_patterns": len(contrastive_patterns),
                        "confidence": (
                            sum(p.confidence for p in discovered_patterns) / len(discovered_patterns)
                            if discovered_patterns
                            else 0.0
                        ),
                        "specialist": "visual",
                    },
                )
                if ranking_applied and ranking_top is not None:
                    self.knowledgeverse.log_event(
                        event_type="arc_candidate_ranking",
                        event_data={
                            "task_id": task_id,
                            "selected_source": ranking_top.get("pattern", {}).get("source", "unknown"),
                            "selected_score": float(ranking_top.get("score", 0.0)),
                            "legacy_score": float(legacy_rank.get("score", 0.0)) if legacy_rank else 0.0,
                            "override_used": ranking_override_used,
                            "selected_pattern_id": ranking_top.get("pattern", {}).get("pattern_id", ""),
                            "specialist": "visual",
                            "quality_prior": (
                                float(ranking_top.get("quality_prior", 0.0))
                                if isinstance(ranking_top, dict)
                                else 0.0
                            ),
                            "ranking_components": ranking_top.get("components", {}),
                            "candidate_count": len(ranked_candidates),
                            "ranking_changed_top1": ranking_changed_top1,
                            "pre_top_source": pre_top_source,
                            "post_top_source": post_top_source,
                            "ptx_ranking_enabled": bool(self.enable_ptx_ranking),
                            "ptx_ranking_used": ptx_ranking_used,
                            "ptx_mode": ptx_mode,
                            "ptx_top_index": ptx_top_index,
                            "ptx_error": ptx_error,
                        },
                    )
                    self.knowledgeverse.log_event(
                        event_type="arc_ranking_scores",
                        event_data={
                            "task_id": task_id,
                            "top_5_scores": top_5_scores,
                            "top_5_sources": top_5_sources,
                            "score_range": float(score_range),
                            "score_stddev": float(score_stddev),
                            "oracle_at_3": oracle_metrics["oracle_at_3"],
                            "oracle_at_10": oracle_metrics["oracle_at_10"],
                            "oracle_at_all": oracle_metrics["oracle_at_all"],
                            "oracle_fuzzy_0_80": bool(oracle_metrics.get("oracle_fuzzy_0_80", False)),
                            "oracle_fuzzy_0_85": bool(oracle_metrics.get("oracle_fuzzy_0_85", False)),
                            "oracle_fuzzy_0_90": bool(oracle_metrics.get("oracle_fuzzy_0_90", False)),
                            "oracle_fuzzy_0_95": bool(oracle_metrics.get("oracle_fuzzy_0_95", False)),
                            "oracle_exact": bool(oracle_metrics.get("oracle_exact", False)),
                            "fuzzy_oracle_at_10": oracle_metrics.get("fuzzy_oracle_at_10", False),
                            "fuzzy_oracle_at_all": oracle_metrics.get("fuzzy_oracle_at_all", False),
                            "fuzzy_best_score": float(oracle_metrics.get("fuzzy_best_score", 0.0)),
                            "correct_rank": oracle_metrics["correct_rank"],
                            "specialist": "visual",
                            "ptx_ranking_enabled": bool(self.enable_ptx_ranking),
                            "ptx_ranking_used": ptx_ranking_used,
                            "ptx_mode": ptx_mode,
                            "ptx_top_index": ptx_top_index,
                            "ptx_error": ptx_error,
                        },
                    )
                    self.knowledgeverse.log_event(
                        event_type="arc_oracle_diagnostics",
                        event_data={
                            "task_id": task_id,
                            "validity_reject_rate": float(validity_report.get("validity_reject_rate", 0.0)),
                            "validity_filtered_count": int(validity_report.get("filtered_count", 0)),
                            "validity_fallback": bool(validity_report.get("fallback_to_ungated", False)),
                            "validity_family_rejects": int(validity_report.get("family_rejects", 0)),
                            "family_mismatch": bool(oracle_diagnostics.get("family_mismatch", False)),
                            "shape_mismatch": bool(oracle_diagnostics.get("shape_mismatch", False)),
                            "palette_mismatch": bool(oracle_diagnostics.get("palette_mismatch", False)),
                            "object_count_mismatch": bool(oracle_diagnostics.get("object_count_mismatch", False)),
                            "oracle_at_all": bool(oracle_metrics.get("oracle_at_all", False)),
                            "fuzzy_oracle_at_all": bool(oracle_metrics.get("fuzzy_oracle_at_all", False)),
                            "specialist": "visual",
                        },
                    )
            reasoning_trace = self._extract_reasoning_trace(result)
            if ranking_applied and ranking_top is not None:
                reasoning_trace.append(
                    "ranking::selected_source="
                    f"{ranking_top.get('pattern', {}).get('source', 'unknown')} "
                    f"score={float(ranking_top.get('score', 0.0)):.4f} "
                    f"override={ranking_override_used}"
                )
            return {
                "task_id": task_id,
                "correct": final_correct,
                "exact_match": ranked_exact_match,
                "legacy_correct": legacy_correct,
                "predicted": ranked_prediction,
                "legacy_predicted": predicted,
                "legacy_exact_match": exact_match,
                "expected": expected_output,
                "reasoning_trace": reasoning_trace,
                "patterns_used": self._count_patterns_used(result.best_program),
                "solver": "legacy_sovereign_pipeline",
                "score": float(result.score),
                "fuzzy_score": float(getattr(result, "fuzzy_score", 0.0)),
                "generated_patterns": [pattern.__dict__ for pattern in discovered_patterns],
                "generated_pattern_count": len(generated_patterns),
                "generated_pattern_confidence_mean": generated_conf_mean,
                "traditional_pattern_count": len(traditional_patterns),
                "cross_modal_pattern_count": len(cross_modal_patterns),
                "contrastive_pattern_count": len(contrastive_patterns),
                "ranking_applied": ranking_applied,
                "ranking_override_used": ranking_override_used,
                "ranking_top_score": float(ranking_top.get("score", 0.0)) if ranking_top else 0.0,
                "ranking_legacy_score": float(legacy_rank.get("score", 0.0)) if legacy_rank else 0.0,
                "ranking_top_components": ranking_top.get("components", {}) if ranking_top else {},
                "ranked_candidate_count": len(ranked_candidates),
                "pattern_source": post_top_source,
                "ranking_pre_top_source": pre_top_source,
                "ranking_post_top_source": post_top_source,
                "ranking_changed_top1": ranking_changed_top1,
                "ptx_ranking_enabled": bool(self.enable_ptx_ranking),
                "ptx_ranking_used": ptx_ranking_used,
                "ptx_ranking_mode": ptx_mode,
                "ptx_ranking_top_index": ptx_top_index,
                "ptx_ranking_error": ptx_error,
                "ranking_top_5_scores": top_5_scores,
                "ranking_top_5_sources": top_5_sources,
                "ranking_score_range": float(score_range),
                "ranking_score_stddev": float(score_stddev),
                "oracle_at_3": oracle_metrics["oracle_at_3"],
                "oracle_at_10": oracle_metrics["oracle_at_10"],
                "oracle_at_all": oracle_metrics["oracle_at_all"],
                "correct_rank": oracle_metrics["correct_rank"],
                "oracle_fuzzy_0_80": bool(oracle_metrics.get("oracle_fuzzy_0_80", False)),
                "oracle_fuzzy_0_85": bool(oracle_metrics.get("oracle_fuzzy_0_85", False)),
                "oracle_fuzzy_0_90": bool(oracle_metrics.get("oracle_fuzzy_0_90", False)),
                "oracle_fuzzy_0_95": bool(oracle_metrics.get("oracle_fuzzy_0_95", False)),
                "oracle_exact": bool(oracle_metrics.get("oracle_exact", False)),
                "fuzzy_oracle_at_3": bool(oracle_metrics.get("fuzzy_oracle_at_3", False)),
                "fuzzy_oracle_at_10": bool(oracle_metrics.get("fuzzy_oracle_at_10", False)),
                "fuzzy_oracle_at_all": bool(oracle_metrics.get("fuzzy_oracle_at_all", False)),
                "fuzzy_best_score": float(oracle_metrics.get("fuzzy_best_score", 0.0)),
                "fuzzy_best_rank": oracle_metrics.get("fuzzy_best_rank"),
                "validity_gates_enabled": self.enable_validity_gates,
                "validity_gate_report": validity_report,
                "validity_reject_rate": float(validity_report.get("validity_reject_rate", 0.0)),
                "validity_family_rejects": int(validity_report.get("family_rejects", 0)),
                "oracle_failure_modes": oracle_diagnostics,
            }
        except Exception as exc:
            return self._fallback_or_raise(task, str(exc), fallback_solver)

    def _fallback_or_raise(
        self,
        task: dict[str, Any],
        reason: str,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if self.strict_legacy or fallback_solver is None:
            raise RuntimeError(f"Legacy ARC pipeline unavailable: {reason}")
        fallback_result = fallback_solver(task, self.use_enriched)
        fallback_result["fallback_reason"] = reason
        fallback_result["solver"] = "trm_navigator_fallback"
        return fallback_result

    def _extract_reasoning_trace(self, result: Any) -> list[str]:
        lines = [
            f"program_type={result.program_type}",
            f"score={float(result.score):.4f}",
            f"fuzzy_score={float(getattr(result, 'fuzzy_score', 0.0)):.4f}",
        ]
        program = (result.best_program or "").strip()
        if program:
            snippet = program.splitlines()[:4]
            lines.extend(f"program::{line}" for line in snippet)
        signature = getattr(result, "signature", None)
        if signature:
            lines.append(f"signature={signature}")
        return lines

    def _count_patterns_used(self, program: str) -> int:
        patterns: set[str] = set()
        for raw_line in (program or "").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            token = line.split()[0]
            patterns.add(token)
        return len(patterns)

    def _grids_match(
        self,
        predicted: list[list[int]] | None,
        expected: list[list[int]] | None,
    ) -> bool:
        if not isinstance(predicted, list) or not isinstance(expected, list):
            return False
        if len(predicted) != len(expected):
            return False
        for pred_row, exp_row in zip(predicted, expected):
            if list(pred_row) != list(exp_row):
                return False
        return True

    def discover_patterns(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[_GeneratedPattern]:
        """
        Discover ARC patterns with multiple strategies:
        1. Traditional deterministic summaries.
        2. Autonomous generation via procedural galaxies.
        3. Cross-modal navigation composition.

        Generated entries are stored in Grammar so future runs can reuse
        discovered transforms.
        """
        if self.enable_contrastive_learning:
            return self.discover_patterns_contrastive(train_examples)
        generated: list[_GeneratedPattern] = []
        generated.extend(self._discover_patterns_traditional(train_examples))
        generated.extend(self._discover_patterns_with_autonomous_generation(train_examples))
        generated.extend(self._discover_patterns_cross_modal(train_examples))
        return generated

    def discover_patterns_contrastive(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[_GeneratedPattern]:
        """
        Contrastive discovery with forward/backward/fusion logic.

        Forward:
        - regular pattern discovery (traditional + autonomous + cross-modal)
        Backward:
        - anti-pattern generation from failure/opposite transform hypotheses
        Fusion:
        - deduplicate and keep highest-confidence pattern per key
        """
        forward: list[_GeneratedPattern] = []
        forward.extend(self._discover_patterns_traditional(train_examples))
        forward.extend(self._discover_patterns_with_autonomous_generation(train_examples))
        forward.extend(self._discover_patterns_cross_modal(train_examples))

        backward = self._discover_patterns_contrastive_backward(train_examples, forward)
        fused = self._fuse_pattern_sets(forward, backward)
        if self.knowledgeverse is not None:
            self.knowledgeverse.log_event(
                event_type="arc_contrastive_pattern_discovery",
                event_data={
                    "forward_count": len(forward),
                    "backward_count": len(backward),
                    "fusion_count": len(fused),
                    "specialist": "visual",
                    "confidence": (
                        sum(p.confidence for p in fused) / len(fused)
                        if fused
                        else 0.0
                    ),
                },
            )
        return fused

    def _discover_patterns_contrastive_backward(
        self,
        train_examples: list[dict[str, Any]],
        forward_patterns: list[_GeneratedPattern],
    ) -> list[_GeneratedPattern]:
        """
        Generate anti-patterns from opposite transform hypotheses.
        """
        if self.knowledgeverse is None:
            return []
        navigator = getattr(self.knowledgeverse, "trm_navigator", None)
        if navigator is None or not hasattr(navigator, "generate_from_procedural"):
            return []

        anti_patterns: list[_GeneratedPattern] = []
        seeds = forward_patterns[:4]
        if not seeds:
            # Build fallback seeds from examples if forward set is empty.
            for pair in train_examples[:2]:
                if not isinstance(pair, dict):
                    continue
                if not self._is_grid_like(pair.get("input")) or not self._is_grid_like(pair.get("output")):
                    continue
                seeds.append(
                    _GeneratedPattern(
                        pattern_id="contrastive_seed",
                        source_galaxy="Drawing",
                        target_galaxy="Grammar",
                        confidence=0.4,
                        query=self._describe_visual_transformation(pair.get("input"), pair.get("output")),
                        source="contrastive_seed",
                    )
                )

        for idx, seed in enumerate(seeds):
            anti_query = self._invert_transformation_query(seed.query)
            if not anti_query:
                continue
            for source_galaxy in ("3DObjects", "Reality", "Grammar"):
                result = navigator.generate_from_procedural(
                    query=f"anti-pattern visual transformation: {anti_query}",
                    source_galaxy=source_galaxy,
                    target_galaxy="Grammar",
                    store_result=True,
                )
                if "error" in result:
                    continue
                metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                anti_patterns.append(
                    _GeneratedPattern(
                        pattern_id=str(result.get("id", f"anti_{idx}")),
                        source_galaxy=str(metadata.get("source_galaxy", source_galaxy)),
                        target_galaxy="Grammar",
                        confidence=float(metadata.get("confidence", 0.65)),
                        query=anti_query,
                        source="contrastive_anti",
                        pair_index=idx,
                    )
                )
                break
        return anti_patterns

    def _fuse_pattern_sets(
        self,
        forward: list[_GeneratedPattern],
        backward: list[_GeneratedPattern],
    ) -> list[_GeneratedPattern]:
        """
        Deduplicate forward/backward patterns with confidence-aware fusion.
        """
        fused: dict[tuple[str, str], _GeneratedPattern] = {}
        for pattern in [*forward, *backward]:
            key = (pattern.pattern_id, pattern.query)
            existing = fused.get(key)
            if existing is None or pattern.confidence > existing.confidence:
                fused[key] = pattern
        return list(fused.values())

    def _invert_transformation_query(self, query: str) -> str:
        """
        Produce an opposite/contrastive transformation hypothesis.
        """
        q = str(query).lower()
        mappings = (
            ("rotate 90 degrees", "reflect across vertical axis"),
            ("reflect across vertical axis", "rotate 90 degrees"),
            ("reflect across horizontal axis", "rotate 90 degrees"),
            ("scale by factor", "shrink by factor"),
            ("shrink by factor", "scale by factor"),
            ("color transformation", "preserve color with spatial transform"),
            ("object count change", "preserve object count with relocation"),
        )
        for src, dst in mappings:
            if src in q:
                return q.replace(src, dst)
        # default exploration fallback
        return f"alternate transformation for: {q}"

    def _discover_patterns_traditional(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[_GeneratedPattern]:
        """Build simple deterministic transformation summaries."""
        discovered: list[_GeneratedPattern] = []
        for idx, pair in enumerate(train_examples[:3]):
            if not isinstance(pair, dict):
                continue
            input_grid = pair.get("input")
            output_grid = pair.get("output")
            if not self._is_grid_like(input_grid) or not self._is_grid_like(output_grid):
                continue
            query = f"traditional visual rule: {self._describe_visual_transformation(input_grid, output_grid)}"
            discovered.append(
                _GeneratedPattern(
                    pattern_id=f"traditional_pair_{idx}",
                    source_galaxy="Drawing",
                    target_galaxy="Grammar",
                    confidence=0.6,
                    query=query,
                    source="traditional",
                    pair_index=idx,
                )
            )
        return discovered

    def _discover_patterns_with_autonomous_generation(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[_GeneratedPattern]:
        """Generate procedural candidates from 3DObjects/Reality."""
        if self.knowledgeverse is None:
            return []
        navigator = getattr(self.knowledgeverse, "trm_navigator", None)
        if navigator is None or not hasattr(navigator, "generate_from_procedural"):
            return []

        generated: list[_GeneratedPattern] = []
        for train_idx, pair in enumerate(train_examples[:3]):
            if not isinstance(pair, dict):
                continue
            input_grid = pair.get("input")
            output_grid = pair.get("output")
            if not self._is_grid_like(input_grid) or not self._is_grid_like(output_grid):
                continue

            transform_query = self._describe_visual_transformation(input_grid, output_grid)
            query = f"visual transformation: {transform_query}"

            # First try spatial generation, then fallback to physical/procedural
            # systems and finally grammar-derived procedural memory.
            attempts = (
                ("3DObjects", "Grammar"),
                ("Reality", "Grammar"),
                ("Drawing", "Grammar"),
                ("Grammar", "Grammar"),
            )
            for source_galaxy, target_galaxy in attempts:
                result = navigator.generate_from_procedural(
                    query=query,
                    source_galaxy=source_galaxy,
                    target_galaxy=target_galaxy,
                    store_result=True,
                )
                if "error" in result:
                    continue
                metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
                generated.append(
                    _GeneratedPattern(
                        pattern_id=str(result.get("id", "")),
                        source_galaxy=str(metadata.get("source_galaxy", source_galaxy)),
                        target_galaxy=target_galaxy,
                        confidence=float(metadata.get("confidence", 0.7)),
                        query=query,
                        source="autonomous_generation",
                        pair_index=train_idx,
                    )
                )
                break
        return generated

    def _discover_patterns_cross_modal(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[_GeneratedPattern]:
        """Ask navigator for multi-galaxy composition hints."""
        if self.knowledgeverse is None:
            return []
        navigator = getattr(self.knowledgeverse, "trm_navigator", None)
        if navigator is None or not hasattr(navigator, "navigate_and_compose"):
            return []
        cross_modal_query = self._create_cross_modal_query(train_examples)
        if not cross_modal_query:
            return []
        try:
            composed = navigator.navigate_and_compose(
                query=cross_modal_query,
                specialist="visual",
                domain_hint="arc_spatial_reasoning",
                use_forward_backward=True,
            )
        except Exception:
            return []
        candidates = composed.get("candidates")
        if not isinstance(candidates, list):
            return []
        results: list[_GeneratedPattern] = []
        for idx, candidate in enumerate(candidates[:5]):
            if not isinstance(candidate, dict):
                continue
            entry = candidate.get("entry", {}) if isinstance(candidate.get("entry"), dict) else {}
            pattern_id = str(entry.get("id", f"cross_modal_candidate_{idx}"))
            conf = float(candidate.get("confidence", candidate.get("score", 0.6)))
            results.append(
                _GeneratedPattern(
                    pattern_id=pattern_id,
                    source_galaxy="Drawing+Math+Reality",
                    target_galaxy="Grammar",
                    confidence=conf,
                    query=cross_modal_query,
                    source="multi_galaxy_composition",
                    pair_index=idx,
                )
            )
        return results

    def _describe_visual_transformation(
        self,
        input_grid: Any,
        output_grid: Any,
    ) -> str:
        """Describe input->output spatial change for generation queries."""
        in_grid = self._to_grid(input_grid)
        out_grid = self._to_grid(output_grid)
        if not in_grid or not out_grid:
            return "unknown visual transformation"

        input_grid = in_grid
        output_grid = out_grid
        in_h = len(input_grid)
        in_w = len(input_grid[0]) if in_h else 0
        out_h = len(output_grid)
        out_w = len(output_grid[0]) if out_h else 0

        descriptions: list[str] = []

        if in_h > 0 and in_w > 0 and in_h == out_w and in_w == out_h:
            descriptions.append("rotate 90 degrees")

        if in_h > 0 and in_w > 0 and (out_h != in_h or out_w != in_w):
            scale_h = (out_h / in_h) if in_h else 1.0
            scale_w = (out_w / in_w) if in_w else 1.0
            if scale_h > 1.0 or scale_w > 1.0:
                descriptions.append(f"scale by factor {max(scale_h, scale_w):.2f}")
            elif scale_h < 1.0 or scale_w < 1.0:
                descriptions.append(f"shrink by factor {min(scale_h, scale_w):.2f}")

        if self._grid_flip_h(input_grid) == output_grid:
            descriptions.append("reflect across vertical axis")
        elif self._grid_flip_v(input_grid) == output_grid:
            descriptions.append("reflect across horizontal axis")

        unique_in = {value for row in input_grid for value in row}
        unique_out = {value for row in output_grid for value in row}
        if unique_in != unique_out:
            descriptions.append(f"color transformation ({len(unique_in)} to {len(unique_out)} colors)")

        in_objects = self._count_connected_objects(input_grid)
        out_objects = self._count_connected_objects(output_grid)
        if in_objects != out_objects:
            descriptions.append(f"object count change ({in_objects} to {out_objects})")

        if descriptions:
            return ", ".join(descriptions)
        if in_h == out_h and in_w == out_w:
            return "recolor or local shape transform"
        return f"transform {in_h}x{in_w} grid to {out_h}x{out_w} grid"

    def _create_cross_modal_query(self, train_examples: list[dict[str, Any]]) -> str | None:
        """Build a short query describing common pattern over train pairs."""
        features: list[str] = []
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            input_grid = pair.get("input")
            output_grid = pair.get("output")
            if not self._is_grid_like(input_grid) or not self._is_grid_like(output_grid):
                continue
            features.append(self._describe_visual_transformation(input_grid, output_grid))
        if not features:
            return None
        feature = max(set(features), key=features.count)
        return f"spatial reasoning pattern: {feature} (ARC task)"

    def _rank_candidates_for_task(
        self,
        *,
        test_input: Any,
        legacy_prediction: Any,
        discovered_patterns: list[_GeneratedPattern],
        validity_profile: dict[str, Any] | None = None,
        return_debug: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Build + rank candidates using discovered pattern quality signals.

        Includes legacy output as a baseline candidate plus pattern-driven
        generated variants derived from discovered transformations.
        """
        input_grid = self._to_grid(test_input)
        legacy_grid = self._to_grid(legacy_prediction)
        if not input_grid:
            return []

        candidate_map: dict[tuple[tuple[int, ...], ...], tuple[list[list[int]], dict[str, Any]]] = {}
        if legacy_grid:
            sig = self._grid_signature(legacy_grid)
            candidate_map[sig] = (
                legacy_grid,
                {
                    "pattern_id": "legacy_pipeline_output",
                    "source": "legacy_pipeline",
                    "confidence": 0.65,
                    "metadata": {"composition_depth": 1, "reuse_count": 10},
                    "query": "legacy sovereign pipeline",
                },
            )

        for pattern in discovered_patterns:
            generated_grid = self._generate_candidate_from_pattern(input_grid, pattern)
            if not generated_grid:
                continue
            sig = self._grid_signature(generated_grid)
            pattern_payload = {
                "pattern_id": pattern.pattern_id,
                "source": pattern.source,
                "confidence": pattern.confidence,
                "query": pattern.query,
                "metadata": {
                    "source_galaxy": pattern.source_galaxy,
                    "target_galaxy": pattern.target_galaxy,
                    "composition_depth": self._infer_composition_depth_from_query(pattern.query),
                    "reuse_count": self._estimate_pattern_reuse(pattern.pattern_id),
                },
            }
            existing = candidate_map.get(sig)
            if existing is None:
                candidate_map[sig] = (generated_grid, pattern_payload)
            else:
                # Keep whichever source has stronger ranking priors.
                _, existing_pattern = existing
                if self._pattern_priority(pattern_payload) > self._pattern_priority(existing_pattern):
                    candidate_map[sig] = (generated_grid, pattern_payload)

        candidates: list[list[list[int]]] = []
        patterns: list[dict[str, Any]] = []
        for grid, payload in candidate_map.values():
            candidates.append(grid)
            patterns.append(payload)
        pre_top_source = str(patterns[0].get("source", "unknown")) if patterns else "none"
        ranked = self._rank_candidates(
            candidates,
            patterns,
            test_input=input_grid,
            validity_profile=(validity_profile or {}),
        )
        ranker_debug = dict(getattr(self, "_last_ranking_debug", {}) or {})
        if return_debug:
            return ranked, {
                "pre_top_source": pre_top_source,
                "candidate_pool_size": len(candidates),
                "ptx_used": bool(ranker_debug.get("ptx_used", False)),
                "ptx_top_index": ranker_debug.get("ptx_top_index"),
                "ptx_mode": str(ranker_debug.get("ptx_mode", "cpu")),
                "ptx_error": ranker_debug.get("ptx_error"),
            }
        return ranked

    def _pattern_priority(self, pattern: dict[str, Any]) -> float:
        """Lightweight priority for dedup collisions."""
        source = self._get_source_score(pattern)
        confidence = self._get_grammar_confidence(pattern)
        cross_modal = self._get_cross_modal_score(pattern)
        return (0.50 * source) + (0.30 * confidence) + (0.20 * cross_modal)

    def _should_apply_ranking_override(
        self,
        top_ranked: dict[str, Any] | None,
        legacy_rank: dict[str, Any] | None,
    ) -> bool:
        """
        Conservative gating: only override legacy if ranking evidence is strong.
        """
        if not top_ranked:
            return False
        top_pattern = top_ranked.get("pattern", {})
        top_source = str(top_pattern.get("source", ""))
        if top_source not in {"autonomous_generation", "multi_galaxy_composition"}:
            return False
        top_score = float(top_ranked.get("score", 0.0))
        legacy_score = float(legacy_rank.get("score", 0.0)) if legacy_rank else 0.0
        components = top_ranked.get("components", {})
        grammar_conf = float(components.get("grammar_confidence", 0.0))
        cross_modal = float(components.get("cross_modal", 0.0))
        compositional = float(components.get("compositional", 0.0))
        # Require clear margin and multi-signal quality.
        if top_score < (legacy_score + 0.15):
            return False
        if grammar_conf < 0.75:
            return False
        if cross_modal < 0.70:
            return False
        if compositional < 0.70:
            return False
        return True

    def _rank_candidates(
        self,
        candidates: list[list[list[int]]],
        patterns: list[dict[str, Any]],
        *,
        test_input: list[list[int]] | None = None,
        validity_profile: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Rank candidate grids using ternary + train-consistency quality signals.

        Scoring is aligned with ternary contrastive guidance:
        - Source precision priors (legacy/contrastive/autonomous)
        - Online quality priors (ternary memory)
        - Train-pair consistency similarity
        - Novelty (avoid duplicate pools)
        - Cross-modal and compositional evidence
        """
        profile = validity_profile or {}
        source_precision = {
            "legacy_pipeline": 0.45,
            "contrastive_anti": 0.46,
            "autonomous_generation": 0.19,
            "traditional": 0.32,
            "multi_galaxy_composition": 0.41,
            "unknown": 0.30,
        }
        signatures: list[tuple[tuple[int, ...], ...]] = [self._grid_signature(grid) for grid in candidates]
        duplicate_counts = Counter(signatures)
        scored_candidates: list[dict[str, Any]] = []
        for idx, candidate in enumerate(candidates):
            pattern = patterns[idx] if idx < len(patterns) else {}
            grammar_conf = self._get_grammar_confidence(pattern)
            cross_modal = self._get_cross_modal_score(pattern)
            source = self._get_source_score(pattern)
            source_precision_score = self._get_source_precision(pattern, source_precision)
            compositional = self._get_compositional_score(pattern)
            reuse = self._get_reuse_score(pattern)
            quality_prior = self._get_quality_prior(pattern)
            train_similarity = self._compute_train_pair_similarity(
                candidate_grid=candidate,
                test_input=(test_input or []),
                profile=profile,
            )
            novelty = self._compute_novelty(
                signature=signatures[idx],
                duplicate_count=duplicate_counts.get(signatures[idx], 1),
            )
            family = self._classify_candidate_family(
                input_grid=(test_input or []),
                output_grid=candidate,
            )
            family_match = self._family_matches_profile(family=family, profile=profile)
            family_bonus = 0.06 if family_match else -0.20
            scored_candidates.append(
                {
                    "candidate": candidate,
                    "score": 0.0,
                    "quality_prior": quality_prior,
                    "components": {
                        "grammar_confidence": grammar_conf,
                        "cross_modal": cross_modal,
                        "source": source,
                        "source_precision": source_precision_score,
                        "compositional": compositional,
                        "reuse": reuse,
                        "quality_prior": quality_prior,
                        "train_similarity": train_similarity,
                        "novelty": novelty,
                        "family": family,
                        "family_match": family_match,
                        "family_bonus": family_bonus,
                    },
                    "pattern": pattern,
                }
            )

        ranked = self._score_and_sort_candidates(scored_candidates)
        return ranked

    def _score_and_sort_candidates(self, scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Compute final scores and sort candidates.

        CPU path is default. PTX path is used when explicitly enabled and GPU
        runtime is available; failures fail closed into CPU path with telemetry.
        """
        self._last_ranking_debug = {
            "ptx_used": False,
            "ptx_top_index": None,
            "ptx_mode": (
                "cpu_ptx_disabled"
                if not self.enable_ptx_ranking
                else (
                    "cpu_ptx_unavailable"
                    if not self._ptx_ranking_available
                    else "cpu_fallback"
                )
            ),
            "ptx_error": self._ptx_unavailable_reason,
        }
        if not scored_candidates:
            return []

        if self._ptx_ranking_available:
            try:
                return self._score_and_sort_candidates_ptx(scored_candidates)
            except Exception as exc:
                self._last_ranking_debug = {
                    "ptx_used": False,
                    "ptx_top_index": None,
                    "ptx_mode": "cpu_fallback_after_ptx_error",
                    "ptx_error": str(exc),
                }

        # CPU deterministic fallback
        for item in scored_candidates:
            components = item.get("components", {})
            item["score"] = (
                0.26 * float(components.get("source_precision", 0.0))
                + 0.20 * float(components.get("quality_prior", 0.0))
                + 0.16 * float(components.get("train_similarity", 0.0))
                + 0.08 * float(components.get("novelty", 0.0))
                + 0.12 * float(components.get("grammar_confidence", 0.0))
                + 0.08 * float(components.get("cross_modal", 0.0))
                + 0.06 * float(components.get("compositional", 0.0))
                + 0.04 * float(components.get("reuse", 0.0))
                + float(components.get("family_bonus", 0.0))
            )
        scored_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return scored_candidates

    def _score_and_sort_candidates_ptx(self, scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        GPU scoring path: weighted score fusion on GPU + PTX top-1 selection.
        """
        if ARC_PTX_OPS is None or not ARC_PTX_OPS.available:
            raise RuntimeError("arc_ptx_ops_unavailable")

        source_precision = [float(item.get("components", {}).get("source_precision", 0.0)) for item in scored_candidates]
        quality_prior = [float(item.get("components", {}).get("quality_prior", 0.0)) for item in scored_candidates]
        train_similarity = [float(item.get("components", {}).get("train_similarity", 0.0)) for item in scored_candidates]
        novelty = [float(item.get("components", {}).get("novelty", 0.0)) for item in scored_candidates]
        grammar_conf = [float(item.get("components", {}).get("grammar_confidence", 0.0)) for item in scored_candidates]
        cross_modal = [float(item.get("components", {}).get("cross_modal", 0.0)) for item in scored_candidates]
        compositional = [float(item.get("components", {}).get("compositional", 0.0)) for item in scored_candidates]
        reuse = [float(item.get("components", {}).get("reuse", 0.0)) for item in scored_candidates]
        family_bonus = [float(item.get("components", {}).get("family_bonus", 0.0)) for item in scored_candidates]

        ranking = ARC_PTX_OPS.rank_candidates_ternary(
            source_precision=source_precision,
            quality_prior=quality_prior,
            train_similarity=train_similarity,
            novelty=novelty,
            grammar_confidence=grammar_conf,
            cross_modal=cross_modal,
            compositional=compositional,
            reuse=reuse,
            family_bonus=family_bonus,
        )
        score_cpu = ranking.scores.astype(float).tolist()
        for idx, item in enumerate(scored_candidates):
            item["score"] = float(score_cpu[idx])

        ranked_indices = list(ranking.ranked_indices)
        ptx_top_index = int(ranking.top_index)

        self._last_ranking_debug = {
            "ptx_used": True,
            "ptx_top_index": ptx_top_index,
            "ptx_mode": str(ranking.mode),
            "ptx_error": None,
        }
        return [scored_candidates[idx] for idx in ranked_indices]

    def _get_source_precision(
        self,
        pattern: dict[str, Any],
        priors: dict[str, float],
    ) -> float:
        source = str(pattern.get("source", "unknown") or "unknown")
        return self._clamp(float(priors.get(source, priors.get("unknown", 0.30))))

    def _compute_train_pair_similarity(
        self,
        *,
        candidate_grid: list[list[int]],
        test_input: list[list[int]],
        profile: dict[str, Any],
    ) -> float:
        """
        Estimate candidate consistency with train-derived profile.
        """
        if not candidate_grid or not test_input:
            return 0.0

        scores: list[float] = []
        expected_shape = profile.get("expected_shape")
        candidate_shape = (len(candidate_grid), len(candidate_grid[0]) if candidate_grid else 0)
        if expected_shape is not None:
            scores.append(1.0 if candidate_shape == tuple(expected_shape) else 0.0)
        else:
            scores.append(0.5)

        expected_object_count = profile.get("expected_object_count")
        if expected_object_count is not None:
            cand_objects = self._count_connected_objects(candidate_grid)
            diff = abs(cand_objects - int(expected_object_count))
            scores.append(self._clamp(1.0 - (diff / max(1, int(expected_object_count)))))
        else:
            scores.append(0.5)

        output_palette = set(profile.get("output_palette", []))
        cand_palette = self._palette_of(candidate_grid)
        if output_palette:
            overlap = len(cand_palette & output_palette)
            union = len(cand_palette | output_palette)
            scores.append((overlap / union) if union else 0.0)
        else:
            scores.append(0.5)

        expected_family = str(profile.get("inferred_family", "") or "")
        if expected_family:
            family = self._classify_candidate_family(
                input_grid=test_input,
                output_grid=candidate_grid,
            )
            scores.append(1.0 if self._families_compatible(expected_family, family) else 0.0)
        else:
            scores.append(0.5)
        return self._clamp(sum(scores) / len(scores))

    def _compute_novelty(
        self,
        *,
        signature: tuple[tuple[int, ...], ...],
        duplicate_count: int,
    ) -> float:
        if duplicate_count <= 1:
            return 1.0
        # Penalize repeated outputs while still keeping alternatives alive.
        return self._clamp(1.0 / float(duplicate_count))

    def _build_validity_profile(
        self,
        *,
        train_examples: list[dict[str, Any]],
        test_input: Any,
    ) -> dict[str, Any]:
        """
        Build non-leaking validity constraints from train pairs.

        Constraints are inferred from train input->output relations and then
        applied to candidate outputs for test inputs.
        """
        input_shapes: list[tuple[int, int]] = []
        output_shapes: list[tuple[int, int]] = []
        input_object_counts: list[int] = []
        output_object_counts: list[int] = []
        output_palettes: set[int] = set()
        output_palette_sizes: list[int] = []
        preserve_shape = True
        preserve_palette = True
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            in_grid = self._to_grid(pair.get("input"))
            out_grid = self._to_grid(pair.get("output"))
            if not in_grid or not out_grid:
                continue
            in_shape = (len(in_grid), len(in_grid[0]))
            out_shape = (len(out_grid), len(out_grid[0]))
            input_shapes.append(in_shape)
            output_shapes.append(out_shape)
            in_palette = self._palette_of(in_grid)
            out_palette = self._palette_of(out_grid)
            if in_shape != out_shape:
                preserve_shape = False
            if in_palette != out_palette:
                preserve_palette = False
            output_palettes |= out_palette
            output_palette_sizes.append(len(out_palette))
            input_object_counts.append(self._count_connected_objects(in_grid))
            output_object_counts.append(self._count_connected_objects(out_grid))

        expected_shape = None
        test_grid = self._to_grid(test_input)
        if input_shapes and output_shapes and test_grid:
            same_shape_relation = all(i == o for i, o in zip(input_shapes, output_shapes))
            if same_shape_relation:
                expected_shape = (len(test_grid), len(test_grid[0]))
            else:
                swaps = [i[0] == o[1] and i[1] == o[0] for i, o in zip(input_shapes, output_shapes)]
                if swaps and all(swaps):
                    expected_shape = (len(test_grid[0]), len(test_grid))
                else:
                    deltas = [(o[0] - i[0], o[1] - i[1]) for i, o in zip(input_shapes, output_shapes)]
                    if deltas and len(set(deltas)) == 1:
                        dh, dw = deltas[0]
                        expected_shape = (max(1, len(test_grid) + dh), max(1, len(test_grid[0]) + dw))
        shape_deltas = [(o[0] - i[0], o[1] - i[1]) for i, o in zip(input_shapes, output_shapes)]
        consistent_shape_delta = shape_deltas[0] if shape_deltas and len(set(shape_deltas)) == 1 else None

        expected_object_delta = None
        expected_object_count = None
        if input_object_counts and output_object_counts:
            deltas = [out_c - in_c for in_c, out_c in zip(input_object_counts, output_object_counts)]
            if len(set(deltas)) == 1:
                expected_object_delta = deltas[0]
                if test_grid:
                    expected_object_count = max(
                    0,
                    self._count_connected_objects(test_grid) + int(expected_object_delta),
                )
        inferred_family = self._infer_transformation_family(
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            input_object_counts=input_object_counts,
            output_object_counts=output_object_counts,
            preserve_palette=preserve_palette,
        )
        stable_palette_size = (
            output_palette_sizes[0]
            if output_palette_sizes and len(set(output_palette_sizes)) == 1
            else None
        )

        return {
            "inferred_family": inferred_family,
            "shape_deltas": shape_deltas,
            "expected_shape_delta": consistent_shape_delta,
            "expected_shape": expected_shape,
            "test_input_grid": test_grid,
            "preserve_shape": preserve_shape,
            "preserve_palette": preserve_palette,
            "output_palette": sorted(output_palettes),
            "stable_output_palette_size": stable_palette_size,
            "expected_object_delta": expected_object_delta,
            "expected_object_count": expected_object_count,
        }

    def _infer_transformation_family(
        self,
        *,
        input_shapes: list[tuple[int, int]],
        output_shapes: list[tuple[int, int]],
        input_object_counts: list[int],
        output_object_counts: list[int],
        preserve_palette: bool,
    ) -> str:
        """
        Infer a coarse transform family from train pairs.
        """
        if not input_shapes or not output_shapes:
            return "unknown"
        shape_preserved = all(inp == out for inp, out in zip(input_shapes, output_shapes))
        object_preserved = all(inp == out for inp, out in zip(input_object_counts, output_object_counts))
        if shape_preserved and object_preserved:
            return "spatial_or_recolor" if not preserve_palette else "spatial"

        # All pairs share same shape delta: scale/translate family.
        shape_deltas = [(out[0] - inp[0], out[1] - inp[1]) for inp, out in zip(input_shapes, output_shapes)]
        if shape_deltas and len(set(shape_deltas)) == 1:
            return "scale_or_translate"

        # Object counts drift consistently while shape may stay.
        object_deltas = [out - inp for inp, out in zip(input_object_counts, output_object_counts)]
        if object_deltas and len(set(object_deltas)) == 1:
            return "filter_or_count"
        return "mixed"

    def _classify_candidate_family(
        self,
        *,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> str:
        """
        Classify candidate family relative to a test input.
        """
        if not input_grid or not output_grid:
            return "unknown"
        in_shape = (len(input_grid), len(input_grid[0]) if input_grid else 0)
        out_shape = (len(output_grid), len(output_grid[0]) if output_grid else 0)
        in_objects = self._count_connected_objects(input_grid)
        out_objects = self._count_connected_objects(output_grid)
        if in_shape == out_shape and in_objects == out_objects:
            in_palette = self._palette_of(input_grid)
            out_palette = self._palette_of(output_grid)
            return "spatial_or_recolor" if in_palette != out_palette else "spatial"
        if in_shape != out_shape:
            return "scale_or_translate"
        if in_objects != out_objects:
            return "filter_or_count"
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

    def _family_matches_profile(self, *, family: str, profile: dict[str, Any]) -> bool:
        expected_family = str(profile.get("inferred_family", "") or "unknown")
        return self._families_compatible(expected_family, family)

    def _apply_validity_gates(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        validity_profile: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Filter invalid candidates using train-derived constraints."""
        filtered: list[dict[str, Any]] = []
        family_rejects = 0
        shape_rejects = 0
        palette_rejects = 0
        object_rejects = 0
        for item in ranked_candidates:
            candidate_grid = self._to_grid(item.get("candidate"))
            if not candidate_grid:
                continue
            passes, reason = self._candidate_passes_validity(candidate_grid, validity_profile)
            if passes:
                filtered.append(item)
            elif reason == "family":
                family_rejects += 1
            elif reason == "shape":
                shape_rejects += 1
            elif reason == "palette":
                palette_rejects += 1
            elif reason == "object":
                object_rejects += 1

        fallback = False
        post = filtered
        if not filtered:
            fallback = True
            post = ranked_candidates
        pre_count = len(ranked_candidates)
        post_count = len(post)
        filtered_count = family_rejects + shape_rejects + palette_rejects + object_rejects
        reject_rate = (filtered_count / pre_count) if pre_count else 0.0
        report = {
            "enabled": True,
            "pre_count": pre_count,
            "post_count": post_count,
            "filtered_count": filtered_count,
            "fallback_to_ungated": fallback,
            "family_rejects": family_rejects,
            "shape_rejects": shape_rejects,
            "palette_rejects": palette_rejects,
            "object_rejects": object_rejects,
            "validity_reject_rate": reject_rate,
        }
        return post, report

    def _candidate_passes_validity(
        self,
        candidate_grid: list[list[int]],
        profile: dict[str, Any],
    ) -> tuple[bool, str]:
        expected_family = str(profile.get("inferred_family", "") or "")
        if expected_family:
            test_shape_grid = self._to_grid(profile.get("test_input_grid"))
            if not test_shape_grid:
                test_shape_grid = candidate_grid
            family = self._classify_candidate_family(
                input_grid=test_shape_grid,
                output_grid=candidate_grid,
            )
            if not self._families_compatible(expected_family, family):
                return False, "family"

        expected_shape = profile.get("expected_shape")
        if expected_shape is not None:
            cand_shape = (len(candidate_grid), len(candidate_grid[0]) if candidate_grid else 0)
            if cand_shape != tuple(expected_shape):
                return False, "shape"

        output_palette = set(profile.get("output_palette", []))
        if output_palette and profile.get("preserve_palette") is False:
            cand_palette = self._palette_of(candidate_grid)
            # Candidate should not invent colors unseen in train outputs.
            if not cand_palette.issubset(output_palette):
                return False, "palette"
        stable_palette_size = profile.get("stable_output_palette_size")
        if stable_palette_size is not None:
            cand_palette_size = len(self._palette_of(candidate_grid))
            if int(cand_palette_size) != int(stable_palette_size):
                return False, "palette"

        expected_object_count = profile.get("expected_object_count")
        if expected_object_count is not None:
            cand_objects = self._count_connected_objects(candidate_grid)
            if cand_objects != int(expected_object_count):
                return False, "object"
        return True, ""

    def _get_quality_prior(self, pattern: dict[str, Any]) -> float:
        """
        Retrieve compact quality prior from ternary quality memory.

        Returns normalized value in [0, 1] where 0.5 is neutral.
        """
        if self.quality_memory is None:
            return 0.5
        pattern_id = str(pattern.get("pattern_id", "")).strip()
        if not pattern_id:
            return 0.5
        prior = self.quality_memory.get_prior(pattern_id)
        if prior is None:
            return 0.5
        return self._clamp((prior.prior + 1.0) / 2.0)

    def _update_quality_memory(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        ranking_top: dict[str, Any] | None,
        final_correct: bool,
        oracle_metrics: dict[str, Any],
    ) -> None:
        if self.quality_memory is None:
            return
        transfer_signal = 1.0 if bool(oracle_metrics.get("oracle_at_all")) else -1.0
        for idx, item in enumerate(ranked_candidates[:5]):
            pattern = item.get("pattern", {})
            pattern_id = str(pattern.get("pattern_id", "")).strip()
            if not pattern_id:
                continue
            confidence = float(item.get("score", 0.5))
            if idx == 0 and ranking_top is not None:
                outcome = 1 if final_correct else -1
            else:
                # Keep non-selected candidates as uncertain feedback to avoid over-penalization.
                outcome = 0
            try:
                self.quality_memory.update(
                    pattern_id=pattern_id,
                    outcome=outcome,
                    confidence=confidence,
                    transfer_signal=transfer_signal,
                    knowledgeverse=self.knowledgeverse,
                    specialist="visual",
                    galaxy="Grammar",
                    source="arc_agi_2_adapter",
                )
            except Exception:
                continue

    def _compute_oracle_metrics(
        self,
        ranked_candidates: list[dict[str, Any]],
        expected_output: Any,
        *,
        fuzzy_threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Compute oracle@k diagnostics from ranked candidate list.
        """
        expected_grid = self._to_grid(expected_output)
        if not expected_grid or not ranked_candidates:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_fuzzy_0_80": False,
                "oracle_fuzzy_0_85": False,
                "oracle_fuzzy_0_90": False,
                "oracle_fuzzy_0_95": False,
                "oracle_exact": False,
                "fuzzy_oracle_at_3": False,
                "fuzzy_oracle_at_10": False,
                "fuzzy_oracle_at_all": False,
                "fuzzy_best_score": 0.0,
                "fuzzy_best_rank": None,
            }
        matches: list[int] = []
        fuzzy_scores: list[tuple[int, float]] = []
        for idx, item in enumerate(ranked_candidates):
            candidate_grid = self._to_grid(item.get("candidate"))
            if self._grids_match(candidate_grid, expected_grid):
                matches.append(idx)
            fuzzy_scores.append((idx, self._fuzzy_grid_similarity(candidate_grid, expected_grid)))
        stratified_thresholds = (0.80, 0.85, 0.90, 0.95)
        stratified_hits = {
            threshold: any(score >= threshold for _, score in fuzzy_scores)
            for threshold in stratified_thresholds
        }
        best_fuzzy_rank: int | None = None
        best_fuzzy_score = 0.0
        if fuzzy_scores:
            best_fuzzy_rank, best_fuzzy_score = max(fuzzy_scores, key=lambda it: it[1])
        threshold = 1.01 if fuzzy_threshold is None else float(fuzzy_threshold)
        fuzzy_at_3 = any(score >= threshold for idx, score in fuzzy_scores if idx < 3)
        fuzzy_at_10 = any(score >= threshold for idx, score in fuzzy_scores if idx < 10)
        fuzzy_at_all = any(score >= threshold for _, score in fuzzy_scores)
        if not matches:
            return {
                "oracle_at_3": False,
                "oracle_at_10": False,
                "oracle_at_all": False,
                "correct_rank": None,
                "oracle_fuzzy_0_80": bool(stratified_hits.get(0.80, False)),
                "oracle_fuzzy_0_85": bool(stratified_hits.get(0.85, False)),
                "oracle_fuzzy_0_90": bool(stratified_hits.get(0.90, False)),
                "oracle_fuzzy_0_95": bool(stratified_hits.get(0.95, False)),
                "oracle_exact": False,
                "fuzzy_oracle_at_3": fuzzy_at_3,
                "fuzzy_oracle_at_10": fuzzy_at_10,
                "fuzzy_oracle_at_all": fuzzy_at_all,
                "fuzzy_best_score": best_fuzzy_score,
                "fuzzy_best_rank": best_fuzzy_rank,
            }
        best_rank = min(matches)
        return {
            "oracle_at_3": best_rank < 3,
            "oracle_at_10": best_rank < 10,
            "oracle_at_all": True,
            "correct_rank": best_rank,
            "oracle_fuzzy_0_80": bool(stratified_hits.get(0.80, False)),
            "oracle_fuzzy_0_85": bool(stratified_hits.get(0.85, False)),
            "oracle_fuzzy_0_90": bool(stratified_hits.get(0.90, False)),
            "oracle_fuzzy_0_95": bool(stratified_hits.get(0.95, False)),
            "oracle_exact": True,
            "fuzzy_oracle_at_3": fuzzy_at_3,
            "fuzzy_oracle_at_10": fuzzy_at_10,
            "fuzzy_oracle_at_all": fuzzy_at_all,
            "fuzzy_best_score": best_fuzzy_score,
            "fuzzy_best_rank": best_fuzzy_rank,
        }

    def evaluate_task_with_oracle_diagnostics(
        self,
        *,
        predicted: Any,
        expected: Any,
        validity_profile: dict[str, Any],
        validity_report: dict[str, Any],
        oracle_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Diagnose why oracle misses occur.

        This is diagnostic-only and never mutates task outcome.
        """
        predicted_grid = self._to_grid(predicted)
        expected_grid = self._to_grid(expected)
        shape_mismatch = bool(predicted_grid and expected_grid and (len(predicted_grid) != len(expected_grid) or len(predicted_grid[0]) != len(expected_grid[0])))
        palette_mismatch = bool(predicted_grid and expected_grid and (self._palette_of(predicted_grid) != self._palette_of(expected_grid)))
        object_count_mismatch = bool(
            predicted_grid
            and expected_grid
            and (self._count_connected_objects(predicted_grid) != self._count_connected_objects(expected_grid))
        )
        family_mismatch = bool(int(validity_report.get("family_rejects", 0)) > 0)
        fuzzy_score = float(oracle_metrics.get("fuzzy_best_score", 0.0))
        root_cause = "unknown"
        if bool(oracle_metrics.get("oracle_at_all", False)):
            root_cause = "ranking_or_selection"
        elif family_mismatch:
            root_cause = "family_constraints"
        elif shape_mismatch:
            root_cause = "shape_constraints"
        elif palette_mismatch:
            root_cause = "palette_constraints"
        elif object_count_mismatch:
            root_cause = "object_count_constraints"
        elif fuzzy_score >= 0.80:
            root_cause = "near_miss_generation"
        else:
            root_cause = "generation_gap"
        return {
            "family_mismatch": family_mismatch,
            "shape_mismatch": shape_mismatch,
            "palette_mismatch": palette_mismatch,
            "object_count_mismatch": object_count_mismatch,
            "fuzzy_best_score": fuzzy_score,
            "root_cause": root_cause,
            "validity_profile": validity_profile,
            "validity_reject_rate": float(validity_report.get("validity_reject_rate", 0.0)),
        }

    def _get_grammar_confidence(self, pattern: dict[str, Any]) -> float:
        """Return normalized confidence from pattern metadata."""
        if "confidence" in pattern:
            try:
                return self._clamp(float(pattern["confidence"]))
            except Exception:
                return 0.5
        metadata = pattern.get("metadata", {}) if isinstance(pattern.get("metadata"), dict) else {}
        if "confidence" in metadata:
            try:
                return self._clamp(float(metadata["confidence"]))
            except Exception:
                return 0.5
        entry = pattern.get("pattern", {}) if isinstance(pattern.get("pattern"), dict) else {}
        entry_meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        if "confidence" in entry_meta:
            try:
                return self._clamp(float(entry_meta["confidence"]))
            except Exception:
                return 0.5
        return 0.5

    def _get_cross_modal_score(self, pattern: dict[str, Any]) -> float:
        """Score higher when evidence spans multiple modalities/galaxies."""
        source = str(pattern.get("source", ""))
        if "cross_modal" in source or "multi_galaxy" in source:
            return 1.0
        metadata = pattern.get("metadata", {}) if isinstance(pattern.get("metadata"), dict) else {}
        if metadata.get("cross_modal"):
            return 0.8
        source_galaxy = str(metadata.get("source_galaxy", ""))
        if "+" in source_galaxy:
            return 0.85
        entry = pattern.get("pattern", {}) if isinstance(pattern.get("pattern"), dict) else {}
        entry_meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        if entry_meta.get("symlink"):
            return 0.7
        return 0.4

    def _get_source_score(self, pattern: dict[str, Any]) -> float:
        """Prefer autonomous and cross-modal sources over traditional."""
        source = str(pattern.get("source", "traditional"))
        if source == "contrastive_anti":
            return 0.9
        if source == "autonomous_generation":
            return 1.0
        if "cross_modal" in source or "multi_galaxy" in source:
            return 0.8
        if source == "legacy_pipeline":
            return 0.65
        if source == "traditional":
            return 0.5
        return 0.3

    def _get_compositional_score(self, pattern: dict[str, Any]) -> float:
        """Prefer patterns with deeper composition."""
        metadata = pattern.get("metadata", {}) if isinstance(pattern.get("metadata"), dict) else {}
        depth = int(metadata.get("composition_depth", 1))
        entry = pattern.get("pattern", {}) if isinstance(pattern.get("pattern"), dict) else {}
        entry_meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        depth = max(depth, int(entry_meta.get("composition_depth", 1)))
        if depth >= 3:
            return 1.0
        if depth == 2:
            return 0.7
        return 0.4

    def _get_reuse_score(self, pattern: dict[str, Any]) -> float:
        """Score based on prior usage frequency."""
        metadata = pattern.get("metadata", {}) if isinstance(pattern.get("metadata"), dict) else {}
        reuse = float(metadata.get("reuse_count", 0))
        entry = pattern.get("pattern", {}) if isinstance(pattern.get("pattern"), dict) else {}
        entry_meta = entry.get("metadata", {}) if isinstance(entry.get("metadata"), dict) else {}
        reuse = max(reuse, float(entry_meta.get("query_count", 0)))
        return self._clamp(reuse / 10.0)

    def _generate_candidate_from_pattern(
        self,
        input_grid: list[list[int]],
        pattern: _GeneratedPattern,
    ) -> list[list[int]]:
        """
        Apply a lightweight transform inferred from pattern query text.

        This keeps adapter-side ranking deterministic while leveraging
        discovered procedural hints.
        """
        q = pattern.query.lower()
        if "reflect across vertical axis" in q:
            return self._grid_flip_h(input_grid)
        if "reflect across horizontal axis" in q:
            return self._grid_flip_v(input_grid)
        if "rotate 90 degrees" in q:
            return self._grid_rotate_90(input_grid)
        if "scale by factor" in q:
            factor = self._parse_scale_factor(q)
            return self._grid_scale(input_grid, factor)
        if "shrink by factor" in q:
            factor = self._parse_scale_factor(q)
            return self._grid_scale(input_grid, factor)
        if "alternate transformation" in q:
            return self._grid_rotate_90(self._grid_flip_h(input_grid))
        # For color/local transforms we keep input as a conservative candidate.
        return [row[:] for row in input_grid]

    def _grid_rotate_90(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        return [[grid[h - 1 - r][c] for r in range(h)] for c in range(w)]

    def _grid_scale(self, grid: list[list[int]], factor: float) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        if factor <= 0:
            return [row[:] for row in grid]
        if abs(factor - 1.0) < 1e-6:
            return [row[:] for row in grid]
        h = len(grid)
        w = len(grid[0])
        new_h = max(1, int(round(h * factor)))
        new_w = max(1, int(round(w * factor)))
        out = [[0 for _ in range(new_w)] for _ in range(new_h)]
        for r in range(new_h):
            src_r = min(h - 1, int(r / factor))
            for c in range(new_w):
                src_c = min(w - 1, int(c / factor))
                out[r][c] = int(grid[src_r][src_c])
        return out

    def _parse_scale_factor(self, query: str) -> float:
        tokens = query.replace(",", " ").split()
        for idx, token in enumerate(tokens):
            if token == "factor" and idx + 1 < len(tokens):
                try:
                    return float(tokens[idx + 1])
                except Exception:
                    continue
        return 1.0

    def _estimate_pattern_reuse(self, pattern_id: str) -> int:
        if not self.knowledgeverse:
            return 0
        try:
            grammar = self.knowledgeverse.galaxy_manager.get_galaxy("Grammar")
        except Exception:
            return 0
        entries = getattr(grammar, "entries", [])
        count = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", ""))
            if pattern_id and pattern_id == entry_id:
                count += 1
        return count

    def _infer_composition_depth_from_query(self, query: str) -> int:
        q = query.lower()
        depth = 1
        # Each comma-separated transformation hint increases composition depth.
        depth += q.count(",")
        if " and " in q:
            depth += q.count(" and ")
        return min(6, max(1, depth))

    def _grid_signature(self, grid: list[list[int]]) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(cell) for cell in row) for row in grid)

    def _palette_of(self, grid: list[list[int]]) -> set[int]:
        return {int(cell) for row in grid for cell in row}

    def _resize_grid_nn(
        self,
        grid: list[list[int]],
        target_h: int,
        target_w: int,
    ) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
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

    def _fuzzy_grid_similarity(
        self,
        predicted: list[list[int]],
        expected: list[list[int]],
    ) -> float:
        """
        Similarity in [0, 1] for near-miss diagnostics.
        """
        if not predicted or not expected:
            return 0.0
        exp_h = len(expected)
        exp_w = len(expected[0]) if exp_h else 0
        pred_h = len(predicted)
        pred_w = len(predicted[0]) if pred_h else 0
        if exp_h == 0 or exp_w == 0 or pred_h == 0 or pred_w == 0:
            return 0.0

        # Compare on expected canvas.
        pred_norm = self._resize_grid_nn(predicted, exp_h, exp_w)
        matches = 0
        total = exp_h * exp_w
        for r in range(exp_h):
            for c in range(exp_w):
                if int(pred_norm[r][c]) == int(expected[r][c]):
                    matches += 1
        cell_score = matches / total if total else 0.0

        pred_palette = self._palette_of(predicted)
        exp_palette = self._palette_of(expected)
        if pred_palette or exp_palette:
            palette_score = len(pred_palette & exp_palette) / len(pred_palette | exp_palette)
        else:
            palette_score = 1.0

        pred_obj = self._count_connected_objects(predicted)
        exp_obj = self._count_connected_objects(expected)
        denom = max(1, exp_obj)
        object_score = max(0.0, 1.0 - (abs(pred_obj - exp_obj) / denom))

        return self._clamp((0.70 * cell_score) + (0.20 * palette_score) + (0.10 * object_score))

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def _grid_flip_h(self, grid: list[list[int]]) -> list[list[int]]:
        return [list(reversed(row)) for row in grid]

    def _grid_flip_v(self, grid: list[list[int]]) -> list[list[int]]:
        return list(reversed(grid))

    def _is_grid_like(self, value: Any) -> bool:
        grid = self._to_grid(value)
        return bool(grid)

    def _to_grid(self, value: Any) -> list[list[int]]:
        """Normalize list/tuple/ndarray-like values into int grid lists."""
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
        # Support numpy arrays without importing numpy.
        if hasattr(value, "tolist"):
            converted = value.tolist()
            if isinstance(converted, list):
                return self._to_grid(converted)
        return []

    def _count_connected_objects(self, grid: list[list[int]]) -> int:
        """Count 4-connected components of non-zero cells."""
        if not grid or not grid[0]:
            return 0
        h = len(grid)
        w = len(grid[0])
        visited = [[False] * w for _ in range(h)]

        def neighbors(r: int, c: int) -> list[tuple[int, int]]:
            out: list[tuple[int, int]] = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.append((rr, cc))
            return out

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
                    for nr, nc in neighbors(rr, cc):
                        if visited[nr][nc] or grid[nr][nc] == 0:
                            continue
                        visited[nr][nc] = True
                        stack.append((nr, nc))
        return components
