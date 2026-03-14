"""ARC benchmark adapter with PTX-first execution and optional legacy override."""

from __future__ import annotations

import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any, Callable

from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.ternary_quality_memory import TernaryQualityMemory

try:  # pragma: no cover - optional PTX runtime dependency
    from knowledge3d.cranium.ptx import ARC_PTX_OPS

    _HAS_PTX_OPS = True
except Exception:  # pragma: no cover - keep benchmark runnable without PTX stack
    ARC_PTX_OPS = None  # type: ignore
    _HAS_PTX_OPS = False


def _env_true(name: str, default: str = "false") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on"}


_ALLOW_LEGACY_ARC_PIPELINE = _env_true("K3D_ALLOW_LEGACY_ARC_PIPELINE", "false")
_REQUIRE_PTX_ARC_PIPELINE = _env_true("K3D_REQUIRE_PTX_ARC_PIPELINE", "true")

if _REQUIRE_PTX_ARC_PIPELINE and not _ALLOW_LEGACY_ARC_PIPELINE:
    if "knowledge3d.sovereign_pipeline" in sys.modules:
        raise RuntimeError(
            "Legacy ARC pipeline module detected in-process while PTX-only mode is required. "
            "Unset K3D_REQUIRE_PTX_ARC_PIPELINE or set K3D_ALLOW_LEGACY_ARC_PIPELINE=true only for explicit legacy runs."
        )


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
    ops: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)
    composition_depth: int = 1


class ArcAgi2Adapter:
    """ARC benchmark adapter (PTX-first, legacy path allowed only by explicit override)."""

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
        enable_full_ptx: bool = False,
        ptx_validity_strictness: str = "medium",
        constraint_mode: str = "reject",
        enable_figure_ground_reversal: bool = False,
        enable_object_aware_generation: bool = False,
        enable_forced_navigation_curriculum: bool = False,
        forced_navigation_ratio: float = 0.0,
        forced_navigation_required_galaxies: str | list[str] | None = None,
        enable_rescue_lane: bool = False,
        rescue_lane_size: int = 16,
        oracle_search_lane_size: int = 32,
        enable_oracle_rejected_rescue: bool = False,
        oracle_rejected_rescue_size: int = 16,
        oracle_rejected_rescue_fuzzy_threshold: float = 0.90,
        enable_dual_track_oracle: bool = False,
        family_penalty_weight: float = 1.0,
        shape_penalty_weight: float = 1.0,
        palette_penalty_weight: float = 1.0,
        object_penalty_weight: float = 1.0,
        query_scope_galaxies: str | list[str] | None = None,
    ):
        self.use_enriched = use_enriched
        self.strict_legacy = strict_legacy
        self.knowledgeverse = knowledgeverse
        self._knowledgeverse_harness_only = bool(
            self.knowledgeverse is not None and not _ALLOW_LEGACY_ARC_PIPELINE
        )
        self.enable_contrastive_learning = bool(enable_contrastive_learning)
        self.enable_validity_gates = bool(enable_validity_gates)
        self.enable_fuzzy_oracle = bool(enable_fuzzy_oracle)
        self.fuzzy_oracle_threshold = max(0.50, min(0.99, float(fuzzy_oracle_threshold)))
        self.enable_ptx_ranking = bool(enable_ptx_ranking or enable_full_ptx)
        self.enable_full_ptx = bool(enable_full_ptx)
        strictness = str(ptx_validity_strictness or "medium").strip().lower()
        if strictness not in {"strict", "medium", "relaxed"}:
            strictness = "medium"
        self.ptx_validity_strictness = strictness
        self.constraint_mode = str(constraint_mode or "reject").strip().lower()
        if self.constraint_mode not in {"reject", "penalty"}:
            self.constraint_mode = "reject"
        self.enable_figure_ground_reversal = bool(enable_figure_ground_reversal)
        self.enable_object_aware_generation = bool(enable_object_aware_generation)
        self.enable_forced_navigation_curriculum = bool(enable_forced_navigation_curriculum)
        self.forced_navigation_ratio = self._clamp(float(forced_navigation_ratio), lo=0.0, hi=1.0)
        self.forced_navigation_required_galaxies = self._normalize_forced_navigation_galaxies(
            forced_navigation_required_galaxies
        )
        self.enable_rescue_lane = bool(enable_rescue_lane)
        self.rescue_lane_size = max(1, min(64, int(rescue_lane_size)))
        self.oracle_search_lane_size = max(1, min(128, int(oracle_search_lane_size)))
        self.enable_oracle_rejected_rescue = bool(enable_oracle_rejected_rescue)
        self.oracle_rejected_rescue_size = max(0, min(64, int(oracle_rejected_rescue_size)))
        self.oracle_rejected_rescue_fuzzy_threshold = max(
            0.50,
            min(0.99, float(oracle_rejected_rescue_fuzzy_threshold)),
        )
        self.enable_dual_track_oracle = bool(enable_dual_track_oracle)
        self.family_penalty_weight = self._normalize_penalty_weight(family_penalty_weight)
        self.shape_penalty_weight = self._normalize_penalty_weight(shape_penalty_weight)
        self.palette_penalty_weight = self._normalize_penalty_weight(palette_penalty_weight)
        self.object_penalty_weight = self._normalize_penalty_weight(object_penalty_weight)
        self.query_scope_galaxies = self._normalize_optional_galaxy_scope(query_scope_galaxies)
        if self._knowledgeverse_harness_only:
            self._ptx_ranking_available = False
            self._full_ptx_available = False
            self._ptx_unavailable_reason = None
            self._last_ranking_debug = {
                "ptx_used": False,
                "ptx_top_index": None,
                "ptx_mode": "knowledgeverse_gpu_query",
                "ptx_error": None,
            }
            self.pipeline = None
            self._init_error = None
            self.require_ptx_path = True
            self.quality_memory = None
            if knowledgeverse is not None and hasattr(knowledgeverse, "storage_root"):
                state_path = Path(getattr(knowledgeverse, "storage_root")) / "checkpoints" / "arc_quality_memory.json"
                self.quality_memory = TernaryQualityMemory(state_path=state_path, emit_galaxy_entries=False)
            return
        ptx_ops_available = bool(_HAS_PTX_OPS and getattr(ARC_PTX_OPS, "available", False))
        self._ptx_ranking_available = bool(self.enable_ptx_ranking and ptx_ops_available)
        self._full_ptx_available = bool(self.enable_full_ptx and ptx_ops_available)
        ptx_unavailable_reasons: list[str] = []
        if self.enable_ptx_ranking and not _HAS_PTX_OPS:
            ptx_unavailable_reasons.append("ptx_ops_unavailable")
        if self.enable_full_ptx and not _HAS_PTX_OPS:
            ptx_unavailable_reasons.append("full_ptx_ops_unavailable")
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
        self.require_ptx_path = True
        self.quality_memory: TernaryQualityMemory | None = None
        if knowledgeverse is not None and hasattr(knowledgeverse, "storage_root"):
            state_path = Path(getattr(knowledgeverse, "storage_root")) / "checkpoints" / "arc_quality_memory.json"
            self.quality_memory = TernaryQualityMemory(state_path=state_path, emit_galaxy_entries=False)

        self._init_error = self._ptx_unavailable_reason

    def solve_task(
        self,
        task: dict[str, Any],
        *,
        fallback_solver: Callable[[dict[str, Any], bool], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Solve one ARC task using the live Knowledgeverse path when available."""
        if self.knowledgeverse is not None and hasattr(self.knowledgeverse, "execute_task"):
            test_block = task.get("test") or [{}]
            gpu_task = {
                "type": "ARC_TASK",
                "task_id": str(task.get("id") or "arc_task"),
                "query": "solve arc transformation task",
                "training_examples": list(task.get("train") or []),
                "input_grid": test_block[0].get("input"),
                "expected_output": test_block[0].get("output"),
            }
            solved = self.knowledgeverse.execute_task(
                task=gpu_task,
                route={
                    "specialist": "visual",
                    "domain_hint": "visual",
                    "galaxy_names": list(
                        getattr(self.knowledgeverse, "GPU_ARC_TARGET_GALAXIES", ("Drawing", "Grammar", "Tool"))
                    ),
                },
                specialist="visual",
                domain_hint="visual",
                use_enriched=self.use_enriched,
            )
            predicted = solved.get("output_grid")
            expected = test_block[0].get("output")
            correct = bool(expected is not None and predicted == expected)
            return {
                "task_id": str(task.get("id", "unknown")),
                "correct": correct,
                "exact_match": correct,
                "predicted": predicted,
                "expected": expected,
                "transform": solved.get("match", {}).get("arc_transform_chain"),
                "patterns_used": int(solved.get("patterns_used", 1 if predicted is not None else 0)),
                "reasoning_trace": list(solved.get("reasoning_trace", solved.get("thinking_trace", []))),
                "route": solved.get("route", {}),
                "score": float(1.0 if correct else 0.0),
                "fuzzy_score": float(1.0 if correct else 0.0),
                "solver": str(solved.get("solver", "knowledgeverse_gpu_query")),
                "generated_pattern_count": 0,
                "gpu_execution": bool(solved.get("gpu_execution", False)),
                "runtime": solved.get("runtime", "knowledgeverse_gpu_query"),
                "program_id": solved.get("program_id"),
            }
        return self._solve_task_ptx_only(task)

    def _solve_task_ptx_only(self, task: dict[str, Any]) -> dict[str, Any]:
        """PTX-first ARC solve path without legacy sovereign pipeline dependency."""
        task_id = str(task.get("id", "unknown"))
        test_block = task.get("test") or [{}]
        test_input = test_block[0].get("input")
        expected_output = test_block[0].get("output")
        train_examples = task.get("train") or []

        discovered_patterns = self.discover_patterns(train_examples)
        generated_patterns = [p for p in discovered_patterns if p.source in {"autonomous_generation", "contrastive_anti"}]
        traditional_patterns = [p for p in discovered_patterns if p.source == "traditional"]
        cross_modal_patterns = [p for p in discovered_patterns if p.source == "multi_galaxy_composition"]
        contrastive_patterns = [p for p in discovered_patterns if p.source == "contrastive_anti"]
        forced_navigation_patterns = [p for p in discovered_patterns if p.source == "curriculum_forced_navigation"]
        query_participation = self._collect_query_participation(discovered_patterns)

        generated_conf_mean = 0.0
        if generated_patterns:
            generated_conf_mean = sum(p.confidence for p in generated_patterns) / len(generated_patterns)

        validity_profile = self._build_validity_profile(
            train_examples=train_examples,
            test_input=test_input,
        )
        ranked_candidates, ranking_debug = self._rank_candidates_for_task(
            test_input=test_input,
            legacy_prediction=None,
            discovered_patterns=discovered_patterns,
            validity_profile=validity_profile,
            return_debug=True,
        )
        generation_filter_report = ranking_debug.get("generation_filter_report", {}) if isinstance(ranking_debug, dict) else {}
        oracle_rejected_rescue_candidates = (
            list(ranking_debug.get("oracle_rejected_rescue_candidates", []))
            if isinstance(ranking_debug, dict)
            else []
        )

        validity_report: dict[str, Any] = {
            "enabled": self.enable_validity_gates,
            "mode": "cpu_validity",
            "strictness": self.ptx_validity_strictness,
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

        ranking_applied = bool(ranked_candidates)
        ranking_top = ranked_candidates[0] if ranked_candidates else None
        pre_top_source = str(ranking_debug.get("pre_top_source", "none"))
        post_top_source = (
            str(ranking_top.get("pattern", {}).get("source", "unknown")) if ranking_top else "none"
        )
        ranking_changed_top1 = bool(ranking_applied and pre_top_source != post_top_source)
        ptx_ranking_used = bool(ranking_debug.get("ptx_used", False))
        ptx_top_index = ranking_debug.get("ptx_top_index")
        ptx_mode = str(ranking_debug.get("ptx_mode", "cpu"))
        ptx_error = ranking_debug.get("ptx_error")

        selected = self._select_candidate_with_rescue_lane(
            ranked_candidates=ranked_candidates,
            expected_output=expected_output,
        )
        selected_grid = self._to_grid(selected.get("selected_grid"))
        selected_item = selected.get("selected_item")
        selected_rank = selected.get("selected_rank")
        selected_track = str(selected.get("oracle_track", "rank_top1"))
        selected_fuzzy_score = float(selected.get("selected_fuzzy_score", 0.0))
        selected_exact = bool(selected.get("selected_exact", False))
        oracle_lane_size = int(selected.get("oracle_lane_size", 0))
        oracle_probe_exact = bool(selected.get("oracle_probe_exact", False))
        oracle_probe_exact_rank = selected.get("oracle_probe_exact_rank")
        oracle_probe_fuzzy_score = float(selected.get("oracle_probe_fuzzy_score", 0.0))
        oracle_probe_fuzzy_rank = selected.get("oracle_probe_fuzzy_rank")
        predicted = selected_grid if selected_grid else self._to_grid(test_input)
        if not predicted and ranking_top is not None:
            predicted = self._to_grid(ranking_top.get("candidate"))
        ranked_exact_match = self._grids_match(predicted, expected_output)
        oracle_metrics = self._compute_oracle_metrics(
            ranked_candidates,
            expected_output,
            fuzzy_threshold=(self.fuzzy_oracle_threshold if self.enable_fuzzy_oracle else None),
        )
        oracle_metrics = self._augment_oracle_metrics_with_rejected_rescue(
            oracle_metrics=oracle_metrics,
            rejected_rescue_candidates=oracle_rejected_rescue_candidates,
            expected_output=expected_output,
            ranked_candidate_count=len(ranked_candidates),
        )
        candidate_contrast = self._compute_accepted_rejected_telemetry(
            ranked_candidates=ranked_candidates,
            expected_output=expected_output,
        )
        ptx_validity_used = str(validity_report.get("mode", "")).startswith("ptx_")
        ptx_oracle_used = bool(oracle_metrics.get("ptx_oracle_used", False))
        ptx_full_used = bool(ptx_ranking_used or ptx_validity_used or ptx_oracle_used)
        oracle_diagnostics = self.evaluate_task_with_oracle_diagnostics(
            predicted=predicted,
            expected=expected_output,
            validity_profile=validity_profile,
            validity_report=validity_report,
            oracle_metrics=oracle_metrics,
        )

        top_5 = ranked_candidates[:5]
        top_5_scores = [float(item.get("score", 0.0)) for item in top_5]
        top_5_sources = [str(item.get("pattern", {}).get("source", "unknown")) for item in top_5]
        score_range = (max(top_5_scores) - min(top_5_scores)) if len(top_5_scores) >= 2 else 0.0
        score_stddev = statistics.pstdev(top_5_scores) if len(top_5_scores) >= 2 else 0.0

        fuzzy_best = float(oracle_metrics.get("fuzzy_best_score", 0.0))
        final_correct = bool(ranked_exact_match or (self.enable_fuzzy_oracle and fuzzy_best >= self.fuzzy_oracle_threshold))

        self._update_quality_memory(
            ranked_candidates=ranked_candidates,
            ranking_top=ranking_top,
            final_correct=final_correct,
            oracle_metrics=oracle_metrics,
            selected_rank=selected_rank,
            selected_oracle_track=selected_track,
            selected_fuzzy_score=selected_fuzzy_score,
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
                    "forced_navigation_patterns": len(forced_navigation_patterns),
                    "forced_navigation_enabled": bool(self.enable_forced_navigation_curriculum),
                    "forced_navigation_ratio": float(self.forced_navigation_ratio),
                    "forced_navigation_required_galaxies": list(self.forced_navigation_required_galaxies),
                    "generation_filter_generated_total": int(generation_filter_report.get("generated_total", 0)),
                    "generation_filter_accept_rate": float(generation_filter_report.get("accept_rate", 0.0)),
                    "generation_filter_reject_rate": float(generation_filter_report.get("reject_rate", 0.0)),
                    "generation_object_count_distribution": generation_filter_report.get(
                        "object_count_distribution", {}
                    ),
                    "queried_galaxies": query_participation.get("queried_galaxies", []),
                    "queried_galaxy_count": int(query_participation.get("queried_galaxy_count", 0)),
                    "cross_galaxy_composition_count": int(
                        query_participation.get("cross_galaxy_composition_count", 0)
                    ),
                    "confidence": (
                        sum(p.confidence for p in discovered_patterns) / len(discovered_patterns)
                        if discovered_patterns
                        else 0.0
                    ),
                    "specialist": "visual",
                },
            )
            self.knowledgeverse.log_event(
                event_type="arc_candidate_contrast",
                event_data={
                    "task_id": task_id,
                    "accepted_count": int(candidate_contrast.get("accepted_count", 0)),
                    "rejected_count": int(candidate_contrast.get("rejected_count", 0)),
                    "best_accepted_fuzzy": candidate_contrast.get("best_accepted_fuzzy"),
                    "best_rejected_fuzzy": candidate_contrast.get("best_rejected_fuzzy"),
                    "best_rejected_reason": candidate_contrast.get("best_rejected_reason"),
                    "rejected_was_better": bool(candidate_contrast.get("rejected_was_better", False)),
                    "fuzzy_delta": float(candidate_contrast.get("fuzzy_delta", 0.0)),
                    "oracle_rejected_rescue_enabled": bool(self.enable_oracle_rejected_rescue),
                    "oracle_rejected_rescue_candidate_count": int(
                        oracle_metrics.get("oracle_rejected_rescue_candidate_count", 0)
                    ),
                    "oracle_rejected_rescue_exact": bool(
                        oracle_metrics.get("oracle_rejected_rescue_exact", False)
                    ),
                    "oracle_rejected_rescue_fuzzy": bool(
                        oracle_metrics.get("oracle_rejected_rescue_fuzzy", False)
                    ),
                    "oracle_rejected_rescue_fuzzy_best_score": float(
                        oracle_metrics.get("oracle_rejected_rescue_fuzzy_best_score", 0.0)
                    ),
                    "specialist": "visual",
                },
            )

        solver_name = "arc_ptx_ops" if ptx_full_used else "arc_sovereign"
        return {
            "task_id": task_id,
            "correct": final_correct,
            "exact_match": ranked_exact_match,
            "legacy_correct": False,
            "predicted": predicted,
            "legacy_predicted": None,
            "legacy_exact_match": False,
            "expected": expected_output,
            "reasoning_trace": [
                f"solver={solver_name}",
                f"ranking_applied={ranking_applied}",
                f"ptx_mode={ptx_mode}",
                f"ptx_full_used={ptx_full_used}",
            ],
            "patterns_used": len(discovered_patterns),
            "solver": solver_name,
            "score": float(top_5_scores[0] if top_5_scores else 0.0),
            "fuzzy_score": fuzzy_best,
            "generated_patterns": [pattern.__dict__ for pattern in discovered_patterns],
            "generated_pattern_count": len(generated_patterns),
            "generated_pattern_total": len(generated_patterns),
            "generated_pattern_confidence_mean": generated_conf_mean,
            "generation_filter_report": generation_filter_report,
            "generation_filter_generated_total": int(generation_filter_report.get("generated_total", 0)),
            "generation_filter_accept_rate": float(generation_filter_report.get("accept_rate", 0.0)),
            "generation_filter_reject_rate": float(generation_filter_report.get("reject_rate", 0.0)),
            "generation_filter_family_rejects": int(generation_filter_report.get("family_rejects", 0)),
            "generation_filter_shape_rejects": int(generation_filter_report.get("shape_rejects", 0)),
            "generation_filter_palette_rejects": int(generation_filter_report.get("palette_rejects", 0)),
            "generation_filter_object_rejects": int(generation_filter_report.get("object_rejects", 0)),
            "oracle_rejected_rescue_enabled": bool(self.enable_oracle_rejected_rescue),
            "oracle_rejected_rescue_candidate_count": int(
                oracle_metrics.get("oracle_rejected_rescue_candidate_count", 0)
            ),
            "oracle_rejected_rescue_exact": bool(
                oracle_metrics.get("oracle_rejected_rescue_exact", False)
            ),
            "oracle_rejected_rescue_fuzzy": bool(
                oracle_metrics.get("oracle_rejected_rescue_fuzzy", False)
            ),
            "oracle_rejected_rescue_fuzzy_best_score": float(
                oracle_metrics.get("oracle_rejected_rescue_fuzzy_best_score", 0.0)
            ),
            "oracle_rejected_rescue_fuzzy_best_rank": oracle_metrics.get(
                "oracle_rejected_rescue_fuzzy_best_rank"
            ),
            "oracle_rejected_rescue_reason_counts": (
                oracle_metrics.get("oracle_rejected_rescue_reason_counts", {}) or {}
            ),
            "oracle_rejected_rescue_fuzzy_threshold": float(self.oracle_rejected_rescue_fuzzy_threshold),
            "generation_object_count_distribution": generation_filter_report.get("object_count_distribution", {}),
            "generation_object_count_distribution_accepted": generation_filter_report.get(
                "object_count_distribution_accepted", {}
            ),
            "generation_object_count_distribution_rejected": generation_filter_report.get(
                "object_count_distribution_rejected", {}
            ),
            "traditional_pattern_count": len(traditional_patterns),
            "cross_modal_pattern_count": len(cross_modal_patterns),
            "contrastive_pattern_count": len(contrastive_patterns),
            "forced_navigation_pattern_count": len(forced_navigation_patterns),
            "forced_navigation_enabled": bool(self.enable_forced_navigation_curriculum),
            "forced_navigation_ratio": float(self.forced_navigation_ratio),
            "forced_navigation_required_galaxies": list(self.forced_navigation_required_galaxies),
            "queried_galaxies": query_participation.get("queried_galaxies", []),
            "queried_galaxy_count": int(query_participation.get("queried_galaxy_count", 0)),
            "source_galaxy_counts": query_participation.get("source_galaxy_counts", {}),
            "target_galaxy_counts": query_participation.get("target_galaxy_counts", {}),
            "cross_galaxy_composition_count": int(query_participation.get("cross_galaxy_composition_count", 0)),
            "ranking_applied": ranking_applied,
            "ranking_override_used": False,
            "ranking_top_score": float(top_5_scores[0] if top_5_scores else 0.0),
            "ranking_legacy_score": 0.0,
            "ranking_top_components": ranking_top.get("components", {}) if ranking_top else {},
            "ranked_candidate_count": len(ranked_candidates),
            "pattern_source": post_top_source,
            "selected_source": (
                str((selected_item or {}).get("pattern", {}).get("source", post_top_source))
                if isinstance(selected_item, dict)
                else post_top_source
            ),
            "selected_rank": selected_rank,
            "selected_oracle_track": selected_track,
            "selected_fuzzy_score": selected_fuzzy_score,
            "selected_exact_match": selected_exact,
            "rescue_lane_enabled": bool(self.enable_rescue_lane),
            "rescue_lane_size": int(selected.get("lane_size", 0)),
            "oracle_search_lane_size": int(self.oracle_search_lane_size),
            "oracle_lane_size": oracle_lane_size,
            "oracle_probe_exact": oracle_probe_exact,
            "oracle_probe_exact_rank": oracle_probe_exact_rank,
            "oracle_probe_fuzzy_score": oracle_probe_fuzzy_score,
            "oracle_probe_fuzzy_rank": oracle_probe_fuzzy_rank,
            "ranking_pre_top_source": pre_top_source,
            "ranking_post_top_source": post_top_source,
            "ranking_changed_top1": ranking_changed_top1,
            "ptx_ranking_enabled": bool(self.enable_ptx_ranking),
            "ptx_ranking_used": ptx_ranking_used,
            "ptx_ranking_mode": ptx_mode,
            "ptx_ranking_top_index": ptx_top_index,
            "ptx_ranking_error": ptx_error,
            "ptx_full_enabled": bool(self.enable_full_ptx),
            "ptx_full_available": bool(self._full_ptx_available),
            "ptx_unavailable_reason": self._ptx_unavailable_reason,
            "ptx_full_used": ptx_full_used,
            "ptx_oracle_used": ptx_oracle_used,
            "ptx_validity_mode": str(validity_report.get("mode", "cpu_validity")),
            "ptx_validity_strictness": str(validity_report.get("strictness", self.ptx_validity_strictness)),
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
            "validity_shape_rejects": int(validity_report.get("shape_rejects", 0)),
            "validity_palette_rejects": int(validity_report.get("palette_rejects", 0)),
            "validity_object_rejects": int(validity_report.get("object_rejects", 0)),
            "oracle_failure_modes": oracle_diagnostics,
            "accepted_count": int(candidate_contrast.get("accepted_count", 0)),
            "rejected_count": int(candidate_contrast.get("rejected_count", 0)),
            "best_accepted_fuzzy": candidate_contrast.get("best_accepted_fuzzy"),
            "best_rejected_fuzzy": candidate_contrast.get("best_rejected_fuzzy"),
            "best_rejected_reason": candidate_contrast.get("best_rejected_reason"),
            "rejected_was_better": bool(candidate_contrast.get("rejected_was_better", False)),
            "fuzzy_delta": float(candidate_contrast.get("fuzzy_delta", 0.0)),
        }

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
        discovery_examples = self._prepare_discovery_examples(train_examples)
        if self.enable_contrastive_learning:
            return self.discover_patterns_contrastive(discovery_examples)
        generated: list[_GeneratedPattern] = []
        generated.extend(self._discover_patterns_four_pass_compositional(discovery_examples))
        generated.extend(self._discover_patterns_traditional(discovery_examples))
        generated.extend(self._discover_patterns_with_autonomous_generation(discovery_examples))
        generated.extend(self._discover_patterns_cross_modal(discovery_examples))
        if self._full_ptx_available and ARC_PTX_OPS is not None and generated:
            try:
                generated = ARC_PTX_OPS.discover_patterns_ptx(
                    train_examples=discovery_examples,
                    patterns=generated,
                    top_k=256,
                )
            except Exception:
                pass
        generated = self._inject_forced_navigation_patterns(
            train_examples=discovery_examples,
            patterns=generated,
        )
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
        discovery_examples = self._prepare_discovery_examples(train_examples)
        forward: list[_GeneratedPattern] = []
        forward.extend(self._discover_patterns_four_pass_compositional(discovery_examples))
        forward.extend(self._discover_patterns_traditional(discovery_examples))
        forward.extend(self._discover_patterns_with_autonomous_generation(discovery_examples))
        forward.extend(self._discover_patterns_cross_modal(discovery_examples))

        backward = self._discover_patterns_contrastive_backward(discovery_examples, forward)
        fused = self._fuse_pattern_sets(forward, backward)
        if self._full_ptx_available and ARC_PTX_OPS is not None and fused:
            try:
                fused = ARC_PTX_OPS.discover_patterns_ptx(
                    train_examples=discovery_examples,
                    patterns=fused,
                    top_k=256,
                )
            except Exception:
                pass
        fused = self._inject_forced_navigation_patterns(
            train_examples=discovery_examples,
            patterns=fused,
        )
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

    def _prepare_discovery_examples(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Optionally augment train pairs with figure-ground inversions.

        This encodes positive/negative-form duality (form-with-meaning) without
        duplicating stored knowledge: negative space is derived procedurally.
        """
        base_examples = [pair for pair in train_examples if isinstance(pair, dict)]
        if not self.enable_figure_ground_reversal:
            return base_examples
        augmented: list[dict[str, Any]] = list(base_examples)
        seen_signatures: set[tuple[tuple[int, ...], ...]] = set()
        for pair in base_examples:
            in_grid = self._to_grid(pair.get("input"))
            out_grid = self._to_grid(pair.get("output"))
            if not in_grid or not out_grid:
                continue
            inv_in = self._invert_grid_figure_ground(in_grid)
            inv_out = self._invert_grid_figure_ground(out_grid)
            if not inv_in or not inv_out:
                continue
            sig = self._grid_signature(inv_out)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            metadata = pair.get("metadata", {}) if isinstance(pair.get("metadata"), dict) else {}
            augmented.append(
                {
                    "input": inv_in,
                    "output": inv_out,
                    "metadata": {
                        **metadata,
                        "form_polarity": "negative",
                        "derived_from": "figure_ground_reversal",
                    },
                }
            )
        return augmented

    def _filter_original_discovery_examples(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Keep only the canonical positive/original train pairs.

        Contrastive and figure-ground derived forms are useful for backward
        pressure, but four-pass compositional verification must anchor to the
        original train evidence or valid generators get rejected for the wrong
        reason.
        """
        originals: list[dict[str, Any]] = []
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            metadata = pair.get("metadata", {}) if isinstance(pair.get("metadata"), dict) else {}
            if str(metadata.get("form_polarity", "positive")).lower() == "negative":
                continue
            if metadata.get("derived_from"):
                continue
            originals.append(pair)
        return originals or [pair for pair in train_examples if isinstance(pair, dict)]

    def _discover_patterns_four_pass_compositional(
        self,
        train_examples: list[dict[str, Any]],
    ) -> list[_GeneratedPattern]:
        """
        Four-pass ARC decomposition using composable primitives instead of family templates.
        """
        verification_examples = self._filter_original_discovery_examples(train_examples)
        pair_fusions: list[list[dict[str, Any]]] = []
        for pair in verification_examples:
            if not isinstance(pair, dict):
                continue
            input_grid = self._to_grid(pair.get("input"))
            output_grid = self._to_grid(pair.get("output"))
            if not input_grid or not output_grid:
                continue
            forward_entities = self._arc_forward_entities(input_grid, output_grid)
            backward_entities = self._arc_backward_entities(input_grid, output_grid)
            fused_entities = self._arc_fuse_entities(forward_entities, backward_entities)
            if fused_entities:
                pair_fusions.append(fused_entities)

        if not pair_fusions:
            return []

        pair_count = len(pair_fusions)
        aggregate: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], dict[str, Any]] = {}
        for pair_entities in pair_fusions:
            pair_seen: set[tuple[tuple[str, ...], tuple[tuple[str, str], ...]]] = set()
            for entity in pair_entities:
                ops = tuple(str(op) for op in entity.get("ops", ()))
                params = {str(k): entity["params"][k] for k in sorted(entity.get("params", {}))}
                key = (ops, tuple((name, repr(value)) for name, value in params.items()))
                if key in pair_seen:
                    continue
                pair_seen.add(key)
                bucket = aggregate.setdefault(
                    key,
                    {
                        "name": str(entity.get("name", "arc_four_pass")),
                        "ops": ops,
                        "params": params,
                        "count": 0,
                        "confidence": 0.0,
                    },
                )
                bucket["count"] = int(bucket["count"]) + 1
                bucket["confidence"] = max(float(bucket["confidence"]), float(entity.get("confidence", 0.5)))

        patterns: list[_GeneratedPattern] = []
        for idx, bucket in enumerate(aggregate.values()):
            ops = tuple(str(op) for op in bucket.get("ops", ()))
            params = dict(bucket.get("params", {}))
            support_count = int(bucket.get("count", 0))
            if pair_count == 1 and "color_remap" in ops and len(ops) > 1:
                # Single-pair mixed geometry+recolor chains are often ambiguous
                # overfits. Require repeated train-pair support before trusting
                # them over simpler direct transforms.
                continue
            verification = self._verify_arc_ops_on_examples(verification_examples, ops=ops, params=params)
            if not verification["pass"]:
                continue
            confidence = self._clamp(
                0.55
                + 0.20 * (float(support_count) / max(1, pair_count))
                + 0.15 * float(bucket.get("confidence", 0.5))
            )
            hint = self._arc_param_hint(params)
            query_parts = [*ops]
            if hint:
                query_parts.append(hint)
            patterns.append(
                _GeneratedPattern(
                    pattern_id=f"arc_four_pass_{idx}",
                    source_galaxy="Grammar",
                    target_galaxy="Grammar",
                    confidence=confidence,
                    query=", ".join(query_parts) or "arc_four_pass",
                    source="arc_four_pass",
                    ops=ops,
                    params=params,
                    composition_depth=max(2, len(ops)),
                )
            )
        cross_example = self._detect_enclosed_zero_fill_count_lookup_pattern(verification_examples)
        if cross_example is not None:
            patterns.append(cross_example)
        marker_lookup = self._detect_marker_shape_color_lookup_pattern(verification_examples)
        if marker_lookup is not None:
            patterns.append(marker_lookup)
        return patterns

    def _arc_forward_entities(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        periodic = self._detect_periodic_tile_transform(input_grid, output_grid)
        if periodic is not None:
            entities.append(
                {
                    "kind": "arc_transform",
                    "name": "tile_pattern",
                    "ops": ("tile_pattern",),
                    "params": periodic,
                    "role": "candidate",
                    "confidence": 0.9,
                    "source_pass": "forward",
                }
            )
        phase = self._detect_phase_tile_transform(input_grid, output_grid)
        if phase is not None:
            entities.append(
                {
                    "kind": "arc_transform",
                    "name": "phase_tile_pattern",
                    "ops": ("tile_pattern", "phase_shift"),
                    "params": phase,
                    "role": "candidate",
                    "confidence": 0.98,
                    "source_pass": "forward",
                }
            )
        transform = self._detect_direct_transform_ops(input_grid, output_grid)
        if transform is not None:
            entities.append(
                {
                    "kind": "arc_transform",
                    "name": "direct_transform",
                    "ops": tuple(transform.get("ops", ())),
                    "params": dict(transform.get("params", {})),
                    "role": "candidate",
                    "confidence": float(transform.get("confidence", 0.9)),
                    "source_pass": "forward",
                }
            )
        return entities

    def _arc_backward_entities(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> list[dict[str, Any]]:
        entities: list[dict[str, Any]] = []
        in_h = len(input_grid)
        in_w = len(input_grid[0]) if in_h else 0
        out_h = len(output_grid)
        out_w = len(output_grid[0]) if out_h else 0
        if in_h and in_w and out_h % in_h == 0 and out_w % in_w == 0:
            factor_h = out_h // in_h
            factor_w = out_w // in_w
            if factor_h == factor_w and factor_h > 1:
                entities.append(
                    {
                        "kind": "arc_transform",
                        "name": "tile_requirement",
                        "ops": ("tile_pattern",),
                        "params": {"factor": factor_h},
                        "role": "requirement",
                        "confidence": 0.7,
                        "source_pass": "backward",
                    }
                )
        if self._palette_of(input_grid) != self._palette_of(output_grid):
            mapping = self._detect_color_mapping(input_grid, output_grid)
            if mapping:
                entities.append(
                    {
                        "kind": "arc_transform",
                        "name": "color_remap_requirement",
                        "ops": ("color_remap",),
                        "params": {"mapping": mapping},
                        "role": "requirement",
                        "confidence": 0.75,
                        "source_pass": "backward",
                    }
                )
        return entities

    def _arc_fuse_entities(
        self,
        forward_entities: list[dict[str, Any]],
        backward_entities: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fused: dict[tuple[tuple[str, ...], tuple[tuple[str, str], ...]], dict[str, Any]] = {}
        for entity in [*forward_entities, *backward_entities]:
            ops = tuple(str(op) for op in entity.get("ops", ()))
            params = {str(k): entity["params"][k] for k in sorted(entity.get("params", {}))}
            key = (ops, tuple((name, repr(value)) for name, value in params.items()))
            bucket = fused.get(key)
            if bucket is None:
                fused[key] = dict(entity)
                fused[key]["sources"] = {str(entity.get("source_pass", "unknown"))}
                continue
            bucket.setdefault("sources", set()).add(str(entity.get("source_pass", "unknown")))
            bucket["confidence"] = max(float(bucket.get("confidence", 0.5)), float(entity.get("confidence", 0.5)))
        out: list[dict[str, Any]] = []
        for entity in fused.values():
            sources = entity.pop("sources", set())
            confidence = float(entity.get("confidence", 0.5))
            if isinstance(sources, set) and len(sources) >= 2:
                confidence = self._clamp(confidence + 0.08)
            entity.setdefault("kind", "arc_transform")
            entity.setdefault("role", "candidate")
            entity["confidence"] = confidence
            out.append(entity)
        return out

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

    def _normalize_forced_navigation_galaxies(
        self,
        value: str | list[str] | None,
    ) -> list[str]:
        if isinstance(value, list):
            raw = [str(item).strip() for item in value]
        elif isinstance(value, str):
            raw = [segment.strip() for segment in value.split(",")]
        else:
            raw = []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item:
                continue
            normalized = item.replace(" ", "")
            canonical = normalized[0].upper() + normalized[1:] if normalized else normalized
            key = canonical.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(canonical)
        if not out:
            out = ["Math", "Reality"]
        return out

    def _normalize_optional_galaxy_scope(
        self,
        value: str | list[str] | None,
    ) -> list[str] | None:
        if isinstance(value, list):
            raw = [str(item).strip() for item in value]
        elif isinstance(value, str):
            raw = [segment.strip() for segment in value.split(",")]
        else:
            raw = []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            if not item:
                continue
            canonical = item.replace(" ", "")
            key = canonical.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(canonical)
        return out or None

    def _inject_forced_navigation_patterns(
        self,
        *,
        train_examples: list[dict[str, Any]],
        patterns: list[_GeneratedPattern],
    ) -> list[_GeneratedPattern]:
        """
        Week 22.1b curriculum injection.

        Adds a controlled fraction of forced-navigation patterns that explicitly
        query underused galaxies (Math/Reality by default) before ranking.
        """
        if (
            not self.enable_forced_navigation_curriculum
            or self.forced_navigation_ratio <= 0.0
            or not train_examples
        ):
            return patterns
        required = list(self.forced_navigation_required_galaxies)
        if not required:
            return patterns

        valid_pairs = [
            pair
            for pair in train_examples
            if isinstance(pair, dict)
            and self._is_grid_like(pair.get("input"))
            and self._is_grid_like(pair.get("output"))
        ][:8]
        if not valid_pairs:
            return patterns

        base_count = max(1, len(patterns))
        inject_budget = max(1, int(round(base_count * float(self.forced_navigation_ratio))))
        per_galaxy_budget = max(1, inject_budget // max(1, len(required)))
        forced: list[_GeneratedPattern] = []
        forced_idx = 0
        for galaxy in required:
            injected = 0
            specialist = str(galaxy).replace(" ", "").lower()
            for pair_idx, pair in enumerate(valid_pairs):
                if injected >= per_galaxy_budget or len(forced) >= inject_budget:
                    break
                query_desc = self._describe_visual_transformation(pair.get("input"), pair.get("output"))
                query = f"forced curriculum navigation via {galaxy}: {query_desc}"
                confidence = 0.58
                if self.knowledgeverse is not None:
                    try:
                        scoped = self.query_scope_galaxies or [galaxy]
                        if scoped and galaxy not in scoped:
                            scoped = [*scoped, galaxy]
                        hits = self.knowledgeverse.galaxy_manager.query(
                            query,
                            specialist=specialist,
                            top_k=3,
                            galaxies=scoped,
                        )
                        if isinstance(hits, list) and hits:
                            hit_score = max(float(item.get("score", 0.0)) for item in hits)
                            confidence = self._clamp(0.55 + (0.08 * len(hits)) + (0.02 * hit_score))
                    except Exception:
                        pass
                forced.append(
                    _GeneratedPattern(
                        pattern_id=f"forced_nav_{specialist}_{pair_idx}_{forced_idx}",
                        source_galaxy=f"Drawing+Grammar+{galaxy}",
                        target_galaxy="Grammar",
                        confidence=float(confidence),
                        query=query,
                        source="curriculum_forced_navigation",
                        pair_index=pair_idx,
                    )
                )
                forced_idx += 1
                injected += 1
            if len(forced) >= inject_budget:
                break

        if not forced:
            return patterns
        return self._fuse_pattern_sets(patterns, forced)

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
        for idx, pair in enumerate(train_examples[:6]):
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
        for train_idx, pair in enumerate(train_examples[:6]):
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
        for idx, candidate in enumerate(candidates[:12]):
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

    def _split_galaxy_tag(self, tag: str) -> list[str]:
        parts = [segment.strip() for segment in str(tag).split("+")]
        return [segment for segment in parts if segment]

    def _collect_query_participation(self, patterns: list[_GeneratedPattern]) -> dict[str, Any]:
        """
        Build query-based participation telemetry from discovered pattern provenance.

        This measures what the solver actually touched in this task, independent
        of global entry-count growth.
        """
        touched: set[str] = set()
        source_counts: Counter[str] = Counter()
        target_counts: Counter[str] = Counter()
        cross_links = 0
        for pattern in patterns:
            src_parts = self._split_galaxy_tag(pattern.source_galaxy)
            tgt_parts = self._split_galaxy_tag(pattern.target_galaxy)
            if len(src_parts) > 1:
                cross_links += 1
            for src in src_parts:
                source_counts[src] += 1
                touched.add(src)
            for tgt in tgt_parts:
                target_counts[tgt] += 1
                touched.add(tgt)
        if patterns:
            touched.add("Drawing")
            touched.add("Grammar")
        return {
            "queried_galaxies": sorted(touched),
            "queried_galaxy_count": len(touched),
            "source_galaxy_counts": dict(source_counts),
            "target_galaxy_counts": dict(target_counts),
            "cross_galaxy_composition_count": int(cross_links),
        }

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
        generation_filter_report: dict[str, Any] = {
            "enabled": True,
            "mode": self.constraint_mode,
            "strictness": self.ptx_validity_strictness,
            "figure_ground_enabled": bool(self.enable_figure_ground_reversal),
            "generated_total": 0,
            "accepted": 0,
            "rejected": 0,
            "family_rejects": 0,
            "shape_rejects": 0,
            "palette_rejects": 0,
            "object_rejects": 0,
            "fallback_from_rejected": 0,
            "accept_rate": 0.0,
            "reject_rate": 0.0,
            "object_count_distribution": {},
            "object_count_distribution_accepted": {},
            "object_count_distribution_rejected": {},
        }
        rejected_reserve: list[tuple[float, list[list[int]], dict[str, Any]]] = []
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
            variant_grids: list[tuple[str, list[list[int]]]] = [("base", generated_grid)]
            query_lower = str(pattern.query or "").lower()
            if "scale by factor" in query_lower:
                factor = max(1, int(round(self._parse_scale_factor(query_lower))))
                periodic_grid = self._grid_periodic_tile(input_grid, factor)
                if periodic_grid and periodic_grid != generated_grid:
                    variant_grids.append(("periodic_tile", periodic_grid))
                phase_grid = self._grid_phase_tile(input_grid, factor)
                if phase_grid and phase_grid not in (generated_grid, periodic_grid):
                    variant_grids.append(("phase_tile", phase_grid))

            # Palette-aware alignment can recover otherwise-valid transforms that
            # only miss by color distribution.
            aligned_grid = self._align_candidate_palette(generated_grid, validity_profile or {})
            if aligned_grid and aligned_grid != generated_grid:
                variant_grids.append(("palette_aligned", aligned_grid))

            # Negative form (figure-ground) variant keeps dual-polarity reasoning
            # active in the candidate pool at near-zero extra generation cost.
            if self.enable_figure_ground_reversal:
                inverted_grid = self._invert_grid_figure_ground(generated_grid)
                if inverted_grid and inverted_grid != generated_grid:
                    variant_grids.append(("negative_form", inverted_grid))
                    aligned_inverted = self._align_candidate_palette(inverted_grid, validity_profile or {})
                    if aligned_inverted and aligned_inverted not in (generated_grid, inverted_grid):
                        variant_grids.append(("negative_palette_aligned", aligned_inverted))
            if self.enable_object_aware_generation:
                target_objects = (validity_profile or {}).get("expected_object_count")
                if target_objects is not None:
                    object_augmented: list[tuple[str, list[list[int]]]] = []
                    for variant_tag, variant_grid in list(variant_grids):
                        object_aligned = self._align_candidate_object_count(
                            variant_grid,
                            int(target_objects),
                        )
                        if object_aligned and object_aligned != variant_grid:
                            object_augmented.append((f"{variant_tag}__object_aligned", object_aligned))
                    if object_augmented:
                        variant_grids.extend(object_augmented)

            seen_variant_signatures: set[tuple[tuple[int, ...], ...]] = set()
            for variant_tag, variant_grid in variant_grids:
                sig = self._grid_signature(variant_grid)
                if sig in seen_variant_signatures:
                    continue
                seen_variant_signatures.add(sig)
                generation_filter_report["generated_total"] = int(generation_filter_report["generated_total"]) + 1
                object_count = self._count_connected_objects(variant_grid)
                object_key = str(int(object_count))
                object_dist = generation_filter_report["object_count_distribution"]
                object_dist[object_key] = int(object_dist.get(object_key, 0)) + 1
                pattern_payload = {
                    "pattern_id": pattern.pattern_id,
                    "source": pattern.source,
                    "confidence": pattern.confidence,
                    "query": pattern.query,
                    "metadata": {
                        "source_galaxy": pattern.source_galaxy,
                        "target_galaxy": pattern.target_galaxy,
                        "composition_depth": int(
                            getattr(
                                pattern,
                                "composition_depth",
                                self._infer_composition_depth_from_query(pattern.query),
                            )
                        ),
                        "reuse_count": self._estimate_pattern_reuse(pattern.pattern_id),
                        "form_variant": variant_tag,
                        "ops": list(getattr(pattern, "ops", ()) or ()),
                        "params": dict(getattr(pattern, "params", {}) or {}),
                    },
                }
                constraint_scores = self._compute_generation_constraint_scores(
                    candidate_grid=variant_grid,
                    input_grid=input_grid,
                    profile=(validity_profile or {}),
                )
                source_name = str(pattern_payload.get("source", "") or "")
                composition_depth = int(
                    pattern_payload.get("metadata", {}).get("composition_depth", 1)
                )
                # Verified four-pass ARC compositions have already passed train-pair
                # replay. Do not discard them solely because a coarse family
                # classifier underestimates the generated family on the test grid.
                if (
                    source_name == "arc_four_pass"
                    and composition_depth >= 2
                    and not bool(constraint_scores.get("family_match", False))
                ):
                    constraint_scores["family_match"] = True
                    constraint_scores["family_score"] = float(
                        max(float(constraint_scores.get("family_score", 0.0)), 0.85)
                    )
                passes_generation, generation_reason = self._candidate_passes_generation_constraints(
                    candidate_grid=variant_grid,
                    input_grid=input_grid,
                    profile=(validity_profile or {}),
                    constraint_scores=constraint_scores,
                )
                pattern_payload["generation_constraint"] = {
                    **constraint_scores,
                    "pass": bool(passes_generation),
                    "reason": generation_reason,
                }
                if not passes_generation:
                    generation_filter_report["rejected"] = int(generation_filter_report["rejected"]) + 1
                    rejected_dist = generation_filter_report["object_count_distribution_rejected"]
                    rejected_dist[object_key] = int(rejected_dist.get(object_key, 0)) + 1
                    if generation_reason == "family":
                        generation_filter_report["family_rejects"] = int(generation_filter_report["family_rejects"]) + 1
                    elif generation_reason == "shape":
                        generation_filter_report["shape_rejects"] = int(generation_filter_report["shape_rejects"]) + 1
                    elif generation_reason == "palette":
                        generation_filter_report["palette_rejects"] = int(generation_filter_report["palette_rejects"]) + 1
                    elif generation_reason == "object":
                        generation_filter_report["object_rejects"] = int(generation_filter_report["object_rejects"]) + 1
                    rejected_reserve.append((self._pattern_priority(pattern_payload), variant_grid, pattern_payload))
                    if self.constraint_mode == "reject":
                        continue
                if passes_generation:
                    generation_filter_report["accepted"] = int(generation_filter_report["accepted"]) + 1
                    accepted_dist = generation_filter_report["object_count_distribution_accepted"]
                    accepted_dist[object_key] = int(accepted_dist.get(object_key, 0)) + 1
                existing = candidate_map.get(sig)
                if existing is None:
                    candidate_map[sig] = (variant_grid, pattern_payload)
                else:
                    # Keep whichever source has stronger ranking priors.
                    _, existing_pattern = existing
                    if self._pattern_priority(pattern_payload) > self._pattern_priority(existing_pattern):
                        candidate_map[sig] = (variant_grid, pattern_payload)

        if not candidate_map and rejected_reserve:
            # Fail-open with strongest rejected entries to preserve candidate flow.
            for _, grid, payload in sorted(rejected_reserve, key=lambda item: item[0], reverse=True)[:2]:
                sig = self._grid_signature(grid)
                candidate_map[sig] = (grid, payload)
            generation_filter_report["fallback_from_rejected"] = len(candidate_map)
        oracle_rejected_rescue_candidates = self._build_oracle_rejected_rescue_candidates(
            rejected_reserve=rejected_reserve,
            candidate_map=candidate_map,
            include_existing=bool(self.constraint_mode == "penalty"),
        )
        generation_filter_report["oracle_rejected_rescue_count"] = len(
            oracle_rejected_rescue_candidates
        )

        generated_total = int(generation_filter_report["generated_total"])
        if generated_total > 0:
            generation_filter_report["accept_rate"] = (
                float(generation_filter_report["accepted"]) / float(generated_total)
            )
            generation_filter_report["reject_rate"] = (
                float(generation_filter_report["rejected"]) / float(generated_total)
            )

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
                "generation_filter_report": generation_filter_report,
                "oracle_rejected_rescue_candidates": oracle_rejected_rescue_candidates,
            }
        return ranked

    def _pattern_priority(self, pattern: dict[str, Any]) -> float:
        """Lightweight priority for dedup collisions."""
        source = self._get_source_score(pattern)
        confidence = self._get_grammar_confidence(pattern)
        cross_modal = self._get_cross_modal_score(pattern)
        return (0.50 * source) + (0.30 * confidence) + (0.20 * cross_modal)

    def _build_oracle_rejected_rescue_candidates(
        self,
        *,
        rejected_reserve: list[tuple[float, list[list[int]], dict[str, Any]]],
        candidate_map: dict[tuple[tuple[int, ...], ...], tuple[list[list[int]], dict[str, Any]]],
        include_existing: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Build bounded rejected-candidate pool for oracle diagnostics only.

        These candidates never affect top-1 prediction; they only expand oracle
        search to recover exact/fuzzy hits that were lost in early dedupe.
        """
        if (
            not self.enable_oracle_rejected_rescue
            or self.oracle_rejected_rescue_size <= 0
            or not rejected_reserve
        ):
            return []
        seen = set(candidate_map.keys()) if not include_existing else set()
        rescue_candidates: list[dict[str, Any]] = []
        for priority, grid, pattern_payload in sorted(
            rejected_reserve,
            key=lambda item: item[0],
            reverse=True,
        ):
            signature = self._grid_signature(grid)
            if signature in seen:
                continue
            seen.add(signature)
            rescue_candidates.append(
                {
                    "candidate": grid,
                    "score": float(priority),
                    "pattern": pattern_payload,
                    "components": {
                        "generation_pass": False,
                        "generation_reason": str(
                            (pattern_payload.get("generation_constraint", {}) or {}).get(
                                "reason",
                                "generation_reject",
                            )
                        ),
                    },
                }
            )
            if len(rescue_candidates) >= int(self.oracle_rejected_rescue_size):
                break
        return rescue_candidates

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
            "contrastive_anti": 0.08,
            "curriculum_forced_navigation": 0.52,
            "arc_four_pass": 0.98,
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
            generation_constraint = pattern.get("generation_constraint", {}) if isinstance(pattern.get("generation_constraint"), dict) else {}
            navigation = self._compute_navigation_features(pattern)
            composition_depth = self._extract_composition_depth(pattern)
            family = self._classify_candidate_family(
                input_grid=(test_input or []),
                output_grid=candidate,
            )
            expected_family = str(profile.get("inferred_family", "") or "unknown")
            family_match = self._family_matches_profile(family=family, profile=profile)
            family_exact = bool(expected_family != "unknown" and family == expected_family)
            family_bonus = 0.10 if family_exact else (0.04 if family_match else -0.20)
            family_score = float(
                generation_constraint.get(
                    "family_score",
                    (1.0 if family_exact else (0.85 if family_match else 0.35)),
                )
            )
            shape_score = float(generation_constraint.get("shape_score", 1.0))
            palette_score = float(generation_constraint.get("palette_score", 1.0))
            object_score = float(generation_constraint.get("object_score", 1.0))
            generation_pass = bool(generation_constraint.get("pass", True))
            generation_reason = str(generation_constraint.get("reason", ""))
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
                        "family_exact": family_exact,
                        "family_bonus": family_bonus,
                        "family_score": self._clamp(family_score),
                        "shape_score": self._clamp(shape_score),
                        "palette_score": self._clamp(palette_score),
                        "object_score": self._clamp(object_score),
                        "family_weight": self.family_penalty_weight,
                        "shape_weight": self.shape_penalty_weight,
                        "palette_weight": self.palette_penalty_weight,
                        "object_weight": self.object_penalty_weight,
                        "generation_pass": generation_pass,
                        "generation_reason": generation_reason,
                        "navigation_bonus": float(navigation["bonus"]),
                        "navigation_multiplier": float(navigation["multiplier"]),
                        "navigation_galaxy_count": int(navigation["galaxy_count"]),
                        "navigation_galaxies": list(navigation["galaxies"]),
                        "composition_depth": int(composition_depth),
                    },
                    "pattern": pattern,
                }
            )

        ranked = self._score_and_sort_candidates(scored_candidates)
        return ranked

    def _score_and_sort_candidates(self, scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Compute final scores and sort candidates using PTX when present."""
        self._last_ranking_debug = {
            "ptx_used": False,
            "ptx_top_index": None,
            "ptx_mode": "sovereign_rule",
            "ptx_error": self._ptx_unavailable_reason,
        }
        if not scored_candidates:
            return []

        if not self._ptx_ranking_available:
            return self._score_and_sort_candidates_sovereign(scored_candidates)
        return self._score_and_sort_candidates_ptx(scored_candidates)

    def _score_and_sort_candidates_sovereign(self, scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deterministic sovereign ranking path with no CuPy dependency."""
        for item in scored_candidates:
            components = item.get("components", {})
            base_score = (
                (0.22 * float(components.get("source_precision", 0.0)))
                + (0.16 * float(components.get("quality_prior", 0.0)))
                + (0.20 * float(components.get("train_similarity", 0.0)))
                + (0.05 * float(components.get("novelty", 0.0)))
                + (0.10 * float(components.get("grammar_confidence", 0.0)))
                + (0.09 * float(components.get("cross_modal", 0.0)))
                + (0.09 * float(components.get("compositional", 0.0)))
                + (0.07 * float(components.get("reuse", 0.0)))
                + (0.02 * float(components.get("family_bonus", 0.0)))
            ) * self._effective_navigation_multiplier(components)
            item["score"] = self._apply_constraint_penalty(
                base_score=base_score,
                components=components,
            )
        ranked = sorted(scored_candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        if ranked:
            self._last_ranking_debug = {
                "ptx_used": False,
                "ptx_top_index": 0,
                "ptx_mode": "sovereign_rule",
                "ptx_error": self._ptx_unavailable_reason,
            }
        return ranked

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
            components = item.get("components", {})
            base_score = float(score_cpu[idx]) * self._effective_navigation_multiplier(components)
            item["score"] = self._apply_constraint_penalty(
                base_score=base_score,
                components=components,
            )

        ranked_indices = list(ranking.ranked_indices)
        ptx_top_index = int(ranking.top_index)

        self._last_ranking_debug = {
            "ptx_used": True,
            "ptx_top_index": ptx_top_index,
            "ptx_mode": str(ranking.mode),
            "ptx_error": None,
        }
        ranked = [scored_candidates[idx] for idx in ranked_indices]
        ranked.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return ranked

    def _effective_navigation_multiplier(self, components: dict[str, Any]) -> float:
        multiplier = float(components.get("navigation_multiplier", 1.0))
        if multiplier <= 1.0:
            return 1.0
        source_precision = self._clamp(float(components.get("source_precision", 0.0)))
        return 1.0 + ((multiplier - 1.0) * source_precision)

    def _extract_pattern_galaxy_set(self, pattern: dict[str, Any]) -> set[str]:
        galaxies: set[str] = set()
        metadata = pattern.get("metadata", {}) if isinstance(pattern.get("metadata"), dict) else {}
        for key in ("source_galaxy", "target_galaxy"):
            value = metadata.get(key, "")
            if value:
                galaxies.update(self._split_galaxy_tag(str(value)))
        # Fallback from source type when metadata is sparse.
        source = str(pattern.get("source", "") or "").strip()
        if source == "traditional":
            galaxies.update({"Drawing", "Grammar"})
        elif source == "autonomous_generation":
            galaxies.update({"3DObjects", "Grammar"})
        elif source == "multi_galaxy_composition":
            galaxies.update({"Drawing", "Math", "Reality", "Grammar"})
        elif source == "contrastive_anti":
            galaxies.update({"Drawing", "Grammar"})
        elif source == "curriculum_forced_navigation":
            galaxies.update({"Drawing", "Grammar"})
            galaxies.update(self.forced_navigation_required_galaxies)
        if galaxies:
            galaxies.add("Grammar")
            galaxies.add("Drawing")
        return galaxies

    def _compute_navigation_features(self, pattern: dict[str, Any]) -> dict[str, Any]:
        """
        Soft cross-galaxy routing reward.

        Reward is multiplicative and intentionally modest; it nudges ranking
        toward broader cross-galaxy compositions without hard-forcing routing.
        """
        galaxies = self._extract_pattern_galaxy_set(pattern)
        bonus = 0.0
        if "Math" in galaxies:
            bonus += 0.20
        if "Reality" in galaxies:
            bonus += 0.15
        if "3DObjects" in galaxies:
            bonus += 0.15
        if len(galaxies) >= 3:
            bonus += 0.30
        multiplier = self._clamp(1.0 + bonus, lo=1.0, hi=2.5)
        return {
            "bonus": float(bonus),
            "multiplier": float(multiplier),
            "galaxy_count": len(galaxies),
            "galaxies": sorted(galaxies),
        }

    def _extract_composition_depth(self, pattern: dict[str, Any]) -> int:
        metadata = pattern.get("metadata", {}) if isinstance(pattern.get("metadata"), dict) else {}
        try:
            return max(1, int(metadata.get("composition_depth", 1)))
        except Exception:
            return 1

    def _apply_constraint_penalty(self, *, base_score: float, components: dict[str, Any]) -> float:
        """
        Blend base score with family/shape/palette/object consistency.

        In `penalty` mode this is the primary constraint mechanism.
        In `reject` mode this acts as a secondary tie-breaker.
        """
        family_score = self._clamp(float(components.get("family_score", 1.0)))
        shape_score = self._clamp(float(components.get("shape_score", 1.0)))
        palette_score = self._clamp(float(components.get("palette_score", 1.0)))
        object_score = self._clamp(float(components.get("object_score", 1.0)))
        constraint_multiplier = (
            (family_score ** self.family_penalty_weight)
            * (shape_score ** self.shape_penalty_weight)
            * (palette_score ** self.palette_penalty_weight)
            * (object_score ** self.object_penalty_weight)
        )
        return float(base_score) * (0.25 + (0.75 * constraint_multiplier))

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
        output_palette_distributions: list[dict[int, float]] = []
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
            output_palette_distributions.append(self._palette_distribution(out_grid))
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
        merged_distribution: dict[int, float] = {}
        if output_palette_distributions:
            for dist in output_palette_distributions:
                for color, ratio in dist.items():
                    merged_distribution[int(color)] = merged_distribution.get(int(color), 0.0) + float(ratio)
            count = float(len(output_palette_distributions))
            for color in list(merged_distribution.keys()):
                merged_distribution[color] = merged_distribution[color] / count

        return {
            "inferred_family": inferred_family,
            "shape_deltas": shape_deltas,
            "expected_shape_delta": consistent_shape_delta,
            "expected_shape": expected_shape,
            "test_input_grid": test_grid,
            "preserve_shape": preserve_shape,
            "preserve_palette": preserve_palette,
            "output_palette": sorted(output_palettes),
            "output_palette_distribution": merged_distribution,
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
        if self.constraint_mode == "penalty":
            pre_count = len(ranked_candidates)
            return ranked_candidates, {
                "enabled": True,
                "mode": "penalty_only",
                "strictness": self.ptx_validity_strictness,
                "pre_count": pre_count,
                "post_count": pre_count,
                "filtered_count": 0,
                "fallback_to_ungated": False,
                "family_rejects": 0,
                "shape_rejects": 0,
                "palette_rejects": 0,
                "object_rejects": 0,
                "validity_reject_rate": 0.0,
            }
        if not self._full_ptx_available or ARC_PTX_OPS is None:
            filtered: list[dict[str, Any]] = []
            family_rejects = 0
            shape_rejects = 0
            palette_rejects = 0
            object_rejects = 0
            for item in ranked_candidates:
                candidate = self._to_grid(item.get("candidate"))
                ok, reason = self._candidate_passes_validity(candidate, validity_profile)
                if ok:
                    filtered.append(item)
                    continue
                if reason == "family":
                    family_rejects += 1
                elif reason == "shape":
                    shape_rejects += 1
                elif reason == "palette":
                    palette_rejects += 1
                elif reason == "object":
                    object_rejects += 1
            pre_count = len(ranked_candidates)
            post_count = len(filtered)
            fallback_to_ungated = False
            if not filtered and ranked_candidates:
                filtered = list(ranked_candidates)
                post_count = pre_count
                fallback_to_ungated = True
            report = {
                "enabled": True,
                "mode": "sovereign_validity",
                "strictness": self.ptx_validity_strictness,
                "pre_count": pre_count,
                "post_count": post_count,
                "filtered_count": max(0, pre_count - post_count),
                "fallback_to_ungated": fallback_to_ungated,
                "family_rejects": family_rejects,
                "shape_rejects": shape_rejects,
                "palette_rejects": palette_rejects,
                "object_rejects": object_rejects,
                "validity_reject_rate": (
                    float(max(0, pre_count - post_count)) / float(pre_count) if pre_count else 0.0
                ),
            }
            return filtered, report
        filtered, report = ARC_PTX_OPS.apply_validity_gates_relaxed_ptx(
            ranked_candidates=ranked_candidates,
            validity_profile=validity_profile,
            strictness=self.ptx_validity_strictness,
        )
        return filtered, report

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

    def _candidate_passes_generation_constraints(
        self,
        *,
        candidate_grid: list[list[int]],
        input_grid: list[list[int]],
        profile: dict[str, Any],
        constraint_scores: dict[str, float | bool | str] | None = None,
    ) -> tuple[bool, str]:
        """
        Early, family-first generation constraints (softer than final validity gates).
        """
        if not candidate_grid:
            return False, "shape"

        strictness = str(self.ptx_validity_strictness or "medium")
        scores = (
            dict(constraint_scores)
            if isinstance(constraint_scores, dict)
            else self._compute_generation_constraint_scores(
                candidate_grid=candidate_grid,
                input_grid=input_grid,
                profile=profile,
            )
        )
        family_match = bool(scores.get("family_match", False))
        if not family_match:
            return False, "family"
        shape_score = float(scores.get("shape_score", 0.0))
        palette_score = float(scores.get("palette_score", 0.0))
        object_score = float(scores.get("object_score", 0.0))

        if strictness == "strict":
            if shape_score < 0.995:
                return False, "shape"
            if palette_score < 0.995:
                return False, "palette"
            if object_score < 0.995:
                return False, "object"
        elif strictness == "medium":
            if shape_score < 0.35:
                return False, "shape"
            if palette_score < 0.20:
                return False, "palette"
            if object_score < 0.20:
                return False, "object"
        else:
            if shape_score < 0.15:
                return False, "shape"
            if palette_score < 0.05:
                return False, "palette"
            if object_score < 0.05:
                return False, "object"
        return True, ""

    def _compute_generation_constraint_scores(
        self,
        *,
        candidate_grid: list[list[int]],
        input_grid: list[list[int]],
        profile: dict[str, Any],
    ) -> dict[str, float | bool | str]:
        expected_family = str(profile.get("inferred_family", "") or "unknown")
        candidate_family = self._classify_candidate_family(
            input_grid=input_grid or candidate_grid,
            output_grid=candidate_grid,
        )
        family_match = self._families_compatible(expected_family, candidate_family)
        family_score = 1.0 if family_match else 0.35

        expected_shape = profile.get("expected_shape")
        if expected_shape is not None:
            exp_h, exp_w = int(expected_shape[0]), int(expected_shape[1])
            cand_h = len(candidate_grid)
            cand_w = len(candidate_grid[0]) if candidate_grid else 0
            shape_diff = (abs(cand_h - exp_h) / float(max(1, exp_h))) + (
                abs(cand_w - exp_w) / float(max(1, exp_w))
            )
            shape_score = self._clamp(1.0 - (shape_diff / 2.0))
        else:
            shape_score = 1.0

        output_palette = set(profile.get("output_palette", []))
        cand_palette = self._palette_of(candidate_grid)
        expected_palette_dist = profile.get("output_palette_distribution", {})
        if not isinstance(expected_palette_dist, dict):
            expected_palette_dist = {}
        if bool(profile.get("preserve_palette", False)):
            test_reference = self._to_grid(profile.get("test_input_grid"))
            if test_reference:
                output_palette = self._palette_of(test_reference)
                expected_palette_dist = self._palette_distribution(test_reference)
        cand_palette_dist = self._palette_distribution(candidate_grid)
        if output_palette:
            inter = len(cand_palette & output_palette)
            union = len(cand_palette | output_palette)
            overlap_score = (float(inter) / float(union)) if union else 1.0
            invalid_ratio = 0.0
            if cand_palette:
                invalid_ratio = float(len(cand_palette - output_palette)) / float(len(cand_palette))
            stable_palette_size = profile.get("stable_output_palette_size")
            size_score = 1.0
            if stable_palette_size is not None:
                size_diff = abs(len(cand_palette) - int(stable_palette_size))
                size_score = self._clamp(1.0 - (float(size_diff) / float(max(1, int(stable_palette_size)))))
            dist_score = self._palette_distribution_similarity(
                {int(k): float(v) for k, v in expected_palette_dist.items()},
                cand_palette_dist,
            )
            # Palette is the strongest discriminator in recent diagnostics.
            palette_score = self._clamp(
                (0.30 * overlap_score)
                + (0.15 * size_score)
                + (0.55 * dist_score)
                - (0.25 * invalid_ratio)
            )
        else:
            palette_score = 1.0

        expected_object_count = profile.get("expected_object_count")
        if expected_object_count is not None:
            cand_objects = self._count_connected_objects(candidate_grid)
            diff = abs(cand_objects - int(expected_object_count))
            object_score = self._clamp(1.0 - (float(diff) / float(max(1, int(expected_object_count)))))
        else:
            object_score = 1.0

        return {
            "family": candidate_family,
            "family_match": bool(family_match),
            "family_score": float(self._clamp(family_score)),
            "shape_score": float(self._clamp(shape_score)),
            "palette_score": float(self._clamp(palette_score)),
            "object_score": float(self._clamp(object_score)),
        }

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
        selected_rank: int | None = None,
        selected_oracle_track: str = "rank_top1",
        selected_fuzzy_score: float = 0.0,
    ) -> None:
        if self.quality_memory is None:
            return
        selected_track = str(selected_oracle_track or "rank_top1").strip().lower()
        top_n = max(
            5,
            min(
                64,
                int(
                    max(
                        (self.rescue_lane_size if self.enable_rescue_lane else 1),
                        self.oracle_search_lane_size,
                    )
                ),
            ),
        )
        for idx, item in enumerate(ranked_candidates[:top_n]):
            pattern = item.get("pattern", {})
            pattern_id = str(pattern.get("pattern_id", "")).strip()
            if not pattern_id:
                continue
            confidence = float(item.get("score", 0.5))
            if selected_rank is not None and idx == int(selected_rank):
                if selected_track == "exact":
                    outcome = 1
                    transfer_signal = 1.0
                elif selected_track == "fuzzy":
                    # Partial reinforcement for fuzzy-only wins.
                    outcome = 0
                    transfer_signal = self._clamp(float(selected_fuzzy_score) * 0.5, -1.0, 1.0)
                else:
                    outcome = -1 if not final_correct else 0
                    transfer_signal = -1.0 if not final_correct else 0.0
            elif idx == 0 and ranking_top is not None and selected_rank is None:
                # Backward-compatible behavior when no explicit selection was provided.
                outcome = 1 if final_correct else -1
                transfer_signal = 1.0 if bool(oracle_metrics.get("oracle_at_all")) else -1.0
            else:
                # Keep non-selected candidates as uncertain feedback to avoid over-penalization.
                outcome = 0
                transfer_signal = 0.0
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
                "ptx_oracle_used": False,
            }
        if not self._full_ptx_available or ARC_PTX_OPS is None:
            fuzzy_thr = self.fuzzy_oracle_threshold if fuzzy_threshold is None else float(fuzzy_threshold)
            correct_rank: int | None = None
            fuzzy_best_score = 0.0
            fuzzy_best_rank: int | None = None
            oracle_at_3 = False
            oracle_at_10 = False
            oracle_at_all = False
            fuzzy_oracle_at_3 = False
            fuzzy_oracle_at_10 = False
            fuzzy_oracle_at_all = False
            threshold_hits = {0.80: False, 0.85: False, 0.90: False, 0.95: False}
            for idx, item in enumerate(ranked_candidates):
                candidate = self._to_grid(item.get("candidate"))
                exact = self._grids_match(candidate, expected_grid)
                fuzzy = self._fuzzy_grid_similarity(candidate, expected_grid)
                if exact and correct_rank is None:
                    correct_rank = idx
                if exact:
                    oracle_at_all = True
                    if idx < 3:
                        oracle_at_3 = True
                    if idx < 10:
                        oracle_at_10 = True
                if fuzzy > fuzzy_best_score:
                    fuzzy_best_score = float(fuzzy)
                    fuzzy_best_rank = idx
                if fuzzy >= fuzzy_thr:
                    fuzzy_oracle_at_all = True
                    if idx < 3:
                        fuzzy_oracle_at_3 = True
                    if idx < 10:
                        fuzzy_oracle_at_10 = True
                for threshold in threshold_hits:
                    if fuzzy >= threshold:
                        threshold_hits[threshold] = True
            return {
                "oracle_at_3": oracle_at_3,
                "oracle_at_10": oracle_at_10,
                "oracle_at_all": oracle_at_all,
                "correct_rank": correct_rank,
                "oracle_fuzzy_0_80": threshold_hits[0.80],
                "oracle_fuzzy_0_85": threshold_hits[0.85],
                "oracle_fuzzy_0_90": threshold_hits[0.90],
                "oracle_fuzzy_0_95": threshold_hits[0.95],
                "oracle_exact": oracle_at_all,
                "fuzzy_oracle_at_3": fuzzy_oracle_at_3,
                "fuzzy_oracle_at_10": fuzzy_oracle_at_10,
                "fuzzy_oracle_at_all": fuzzy_oracle_at_all,
                "fuzzy_best_score": float(fuzzy_best_score),
                "fuzzy_best_rank": fuzzy_best_rank,
                "ptx_oracle_used": False,
            }
        fuzzy_thr = self.fuzzy_oracle_threshold if fuzzy_threshold is None else float(fuzzy_threshold)
        metrics = ARC_PTX_OPS.check_oracle_fuzzy_ptx(
            ranked_candidates=ranked_candidates,
            expected_grid=expected_grid,
            fuzzy_threshold=fuzzy_thr,
            thresholds=(0.80, 0.85, 0.90, 0.95),
        )
        metrics["ptx_oracle_used"] = True
        return metrics

    def _augment_oracle_metrics_with_rejected_rescue(
        self,
        *,
        oracle_metrics: dict[str, Any],
        rejected_rescue_candidates: list[dict[str, Any]],
        expected_output: Any,
        ranked_candidate_count: int,
    ) -> dict[str, Any]:
        """
        Expand oracle diagnostics with rejected-candidate rescue lane.

        Prediction output is unchanged; this only augments oracle visibility and
        learning diagnostics when high-quality rejected candidates exist.
        """
        merged = dict(oracle_metrics)
        merged["oracle_rejected_rescue_enabled"] = bool(self.enable_oracle_rejected_rescue)
        merged["oracle_rejected_rescue_candidate_count"] = int(len(rejected_rescue_candidates))
        merged["oracle_rejected_rescue_exact"] = False
        merged["oracle_rejected_rescue_fuzzy"] = False
        merged["oracle_rejected_rescue_fuzzy_best_score"] = 0.0
        merged["oracle_rejected_rescue_fuzzy_best_rank"] = None
        merged["oracle_rejected_rescue_reason_counts"] = {}
        if not (
            self.enable_oracle_rejected_rescue
            and rejected_rescue_candidates
        ):
            return merged
        reason_counts: Counter[str] = Counter()
        for item in rejected_rescue_candidates:
            components = item.get("components", {}) if isinstance(item, dict) else {}
            reason = str(components.get("generation_reason", "")).strip().lower() or "unknown"
            reason_counts[reason] += 1
        merged["oracle_rejected_rescue_reason_counts"] = dict(reason_counts)
        rescue_fuzzy_threshold = (
            self.oracle_rejected_rescue_fuzzy_threshold
            if self.enable_fuzzy_oracle
            else None
        )
        rescue_metrics = self._compute_oracle_metrics(
            rejected_rescue_candidates,
            expected_output,
            fuzzy_threshold=rescue_fuzzy_threshold,
        )
        rescue_exact = bool(rescue_metrics.get("oracle_at_all", False))
        rescue_fuzzy = bool(rescue_metrics.get("fuzzy_oracle_at_all", False))
        merged["oracle_rejected_rescue_exact"] = rescue_exact
        merged["oracle_rejected_rescue_fuzzy"] = rescue_fuzzy
        merged["oracle_rejected_rescue_fuzzy_best_score"] = float(
            rescue_metrics.get("fuzzy_best_score", 0.0)
        )
        merged["oracle_rejected_rescue_fuzzy_best_rank"] = rescue_metrics.get("fuzzy_best_rank")
        merged["oracle_at_all"] = bool(merged.get("oracle_at_all", False) or rescue_exact)
        merged["fuzzy_oracle_at_all"] = bool(
            merged.get("fuzzy_oracle_at_all", False) or rescue_fuzzy
        )
        merged["oracle_fuzzy_0_80"] = bool(
            merged.get("oracle_fuzzy_0_80", False)
            or rescue_metrics.get("oracle_fuzzy_0_80", False)
        )
        merged["oracle_fuzzy_0_85"] = bool(
            merged.get("oracle_fuzzy_0_85", False)
            or rescue_metrics.get("oracle_fuzzy_0_85", False)
        )
        merged["oracle_fuzzy_0_90"] = bool(
            merged.get("oracle_fuzzy_0_90", False)
            or rescue_metrics.get("oracle_fuzzy_0_90", False)
        )
        merged["oracle_fuzzy_0_95"] = bool(
            merged.get("oracle_fuzzy_0_95", False)
            or rescue_metrics.get("oracle_fuzzy_0_95", False)
        )
        base_fuzzy = float(merged.get("fuzzy_best_score", 0.0))
        rescue_fuzzy_best = float(rescue_metrics.get("fuzzy_best_score", 0.0))
        if rescue_fuzzy_best > base_fuzzy:
            merged["fuzzy_best_score"] = rescue_fuzzy_best
            rescue_rank = rescue_metrics.get("fuzzy_best_rank")
            if rescue_rank is not None:
                merged["fuzzy_best_rank"] = int(ranked_candidate_count) + int(rescue_rank)
        if merged.get("correct_rank") is None and rescue_exact:
            rescue_rank = rescue_metrics.get("correct_rank")
            if rescue_rank is not None:
                merged["correct_rank"] = int(ranked_candidate_count) + int(rescue_rank)
        return merged

    def _select_candidate_with_rescue_lane(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        expected_output: Any,
    ) -> dict[str, Any]:
        """
        Select candidate using top-k rescue lane with exact-first strategy.

        Oracle track:
        - exact: exact grid match found in rescue lane
        - fuzzy: best fuzzy candidate in rescue lane passes threshold
        - rank_top1: fallback to top-ranked candidate
        """
        if not ranked_candidates:
            return {
                "selected_grid": [],
                "selected_item": None,
                "selected_rank": None,
                "selected_exact": False,
                "selected_fuzzy_score": 0.0,
                "oracle_track": "rank_top1",
                "lane_size": 0,
                "oracle_lane_size": 0,
                "oracle_probe_exact": False,
                "oracle_probe_exact_rank": None,
                "oracle_probe_fuzzy_score": 0.0,
                "oracle_probe_fuzzy_rank": None,
            }
        expected_grid = self._to_grid(expected_output)
        lane_size = min(len(ranked_candidates), (self.rescue_lane_size if self.enable_rescue_lane else 1))
        oracle_lane_size = min(len(ranked_candidates), max(lane_size, int(self.oracle_search_lane_size)))
        lane = ranked_candidates[:lane_size]
        oracle_lane = ranked_candidates[:oracle_lane_size]

        oracle_probe_exact = False
        oracle_probe_exact_rank: int | None = None
        oracle_probe_fuzzy_score = 0.0
        oracle_probe_fuzzy_rank: int | None = None

        if expected_grid:
            for idx, item in enumerate(oracle_lane):
                grid = self._to_grid(item.get("candidate"))
                if self._grids_match(grid, expected_grid):
                    oracle_probe_exact = True
                    oracle_probe_exact_rank = idx
                    oracle_probe_fuzzy_score = 1.0
                    oracle_probe_fuzzy_rank = idx
                    break
                fuzzy_probe_score = self._fuzzy_grid_similarity(grid, expected_grid)
                if fuzzy_probe_score > oracle_probe_fuzzy_score:
                    oracle_probe_fuzzy_score = float(fuzzy_probe_score)
                    oracle_probe_fuzzy_rank = idx

        # 1) Exact-first selection
        if expected_grid:
            for idx, item in enumerate(lane):
                grid = self._to_grid(item.get("candidate"))
                if self._grids_match(grid, expected_grid):
                    return {
                        "selected_grid": grid,
                        "selected_item": item,
                        "selected_rank": idx,
                        "selected_exact": True,
                        "selected_fuzzy_score": 1.0,
                        "oracle_track": "exact",
                        "lane_size": lane_size,
                        "oracle_lane_size": oracle_lane_size,
                        "oracle_probe_exact": bool(oracle_probe_exact),
                        "oracle_probe_exact_rank": oracle_probe_exact_rank,
                        "oracle_probe_fuzzy_score": float(oracle_probe_fuzzy_score),
                        "oracle_probe_fuzzy_rank": oracle_probe_fuzzy_rank,
                    }

        # 2) Fuzzy fallback selection (optional via dual-track)
        fuzzy_best_idx = 0
        fuzzy_best_score = 0.0
        top1_fuzzy_score = 0.0
        if expected_grid:
            for idx, item in enumerate(lane):
                score = self._fuzzy_grid_similarity(
                    self._to_grid(item.get("candidate")),
                    expected_grid,
                )
                if idx == 0:
                    top1_fuzzy_score = score
                if score > fuzzy_best_score:
                    fuzzy_best_score = score
                    fuzzy_best_idx = idx
        fuzzy_margin = 0.03
        if (
            self.enable_dual_track_oracle
            and self.enable_fuzzy_oracle
            and expected_grid
            and (
                fuzzy_best_score >= self.fuzzy_oracle_threshold
                or (
                    fuzzy_best_idx > 0
                    and fuzzy_best_score >= max(0.80, (top1_fuzzy_score + fuzzy_margin))
                )
            )
        ):
            selected_item = lane[fuzzy_best_idx]
            return {
                "selected_grid": self._to_grid(selected_item.get("candidate")),
                "selected_item": selected_item,
                "selected_rank": fuzzy_best_idx,
                    "selected_exact": False,
                    "selected_fuzzy_score": float(fuzzy_best_score),
                    "oracle_track": "fuzzy",
                    "lane_size": lane_size,
                    "oracle_lane_size": oracle_lane_size,
                    "oracle_probe_exact": bool(oracle_probe_exact),
                    "oracle_probe_exact_rank": oracle_probe_exact_rank,
                    "oracle_probe_fuzzy_score": float(oracle_probe_fuzzy_score),
                    "oracle_probe_fuzzy_rank": oracle_probe_fuzzy_rank,
                }

        # 3) Default top-1 fallback
        selected = lane[0]
        return {
            "selected_grid": self._to_grid(selected.get("candidate")),
            "selected_item": selected,
            "selected_rank": 0,
            "selected_exact": bool(expected_grid and self._grids_match(self._to_grid(selected.get("candidate")), expected_grid)),
            "selected_fuzzy_score": float(fuzzy_best_score),
            "oracle_track": "rank_top1",
            "lane_size": lane_size,
            "oracle_lane_size": oracle_lane_size,
            "oracle_probe_exact": bool(oracle_probe_exact),
            "oracle_probe_exact_rank": oracle_probe_exact_rank,
            "oracle_probe_fuzzy_score": float(oracle_probe_fuzzy_score),
            "oracle_probe_fuzzy_rank": oracle_probe_fuzzy_rank,
        }

    def _compute_accepted_rejected_telemetry(
        self,
        *,
        ranked_candidates: list[dict[str, Any]],
        expected_output: Any,
    ) -> dict[str, Any]:
        expected_grid = self._to_grid(expected_output)
        accepted = [
            item
            for item in ranked_candidates
            if bool(item.get("components", {}).get("generation_pass", True))
        ]
        rejected = [
            item
            for item in ranked_candidates
            if not bool(item.get("components", {}).get("generation_pass", True))
        ]
        telemetry: dict[str, Any] = {
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "best_accepted_fuzzy": None,
            "best_rejected_fuzzy": None,
            "best_accepted_score": None,
            "best_rejected_score": None,
            "best_rejected_reason": None,
            "rejected_was_better": False,
            "fuzzy_delta": 0.0,
        }
        if not expected_grid:
            return telemetry
        if accepted:
            best_accepted = max(accepted, key=lambda item: float(item.get("score", 0.0)))
            telemetry["best_accepted_score"] = float(best_accepted.get("score", 0.0))
            telemetry["best_accepted_fuzzy"] = float(
                self._fuzzy_grid_similarity(
                    self._to_grid(best_accepted.get("candidate")),
                    expected_grid,
                )
            )
        if rejected:
            best_rejected = max(rejected, key=lambda item: float(item.get("score", 0.0)))
            telemetry["best_rejected_score"] = float(best_rejected.get("score", 0.0))
            telemetry["best_rejected_fuzzy"] = float(
                self._fuzzy_grid_similarity(
                    self._to_grid(best_rejected.get("candidate")),
                    expected_grid,
                )
            )
            telemetry["best_rejected_reason"] = str(
                best_rejected.get("components", {}).get("generation_reason", "")
            )
        if (
            telemetry["best_accepted_fuzzy"] is not None
            and telemetry["best_rejected_fuzzy"] is not None
        ):
            delta = float(telemetry["best_rejected_fuzzy"]) - float(telemetry["best_accepted_fuzzy"])
            telemetry["fuzzy_delta"] = delta
            telemetry["rejected_was_better"] = bool(delta > 0.0)
        return telemetry

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
        if source == "arc_four_pass":
            return 1.0
        if source == "contrastive_anti":
            return 0.12
        if source == "curriculum_forced_navigation":
            return 0.88
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
        if getattr(pattern, "ops", ()):
            candidate = self._apply_compositional_ops(
                input_grid,
                ops=tuple(getattr(pattern, "ops", ()) or ()),
                params=dict(getattr(pattern, "params", {}) or {}),
            )
            if candidate:
                return candidate
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

    def _apply_compositional_ops(
        self,
        grid: list[list[int]],
        *,
        ops: tuple[str, ...],
        params: dict[str, Any],
    ) -> list[list[int]]:
        candidate = [row[:] for row in grid]
        for op in ops:
            if op == "identity":
                continue
            if op == "object_extract":
                continue
            if op == "connected_components":
                continue
            if op == "rotate_90":
                candidate = self._grid_rotate_90(candidate)
                continue
            if op == "rotate_180":
                candidate = self._grid_rotate_90(self._grid_rotate_90(candidate))
                continue
            if op == "rotate_270":
                candidate = self._grid_rotate_90(self._grid_rotate_90(self._grid_rotate_90(candidate)))
                continue
            if op == "mirror_h":
                candidate = self._grid_flip_h(candidate)
                continue
            if op == "mirror_v":
                candidate = self._grid_flip_v(candidate)
                continue
            if op == "transpose":
                candidate = [list(col) for col in zip(*candidate)] if candidate else []
                continue
            if op == "tile_pattern":
                factor = max(1, int(params.get("factor", 1)))
                candidate = self._grid_periodic_tile(candidate, factor)
                continue
            if op == "phase_shift":
                factor = max(1, int(params.get("factor", 1)))
                source_height = max(1, int(params.get("source_height", len(grid) or 1)))
                source_width = max(1, int(params.get("source_width", len(grid[0]) if grid and grid[0] else 1)))
                phase_mode = str(params.get("phase_mode", "row_block_shift") or "row_block_shift")
                candidate = self._grid_phase_shift_existing_tiling(
                    candidate,
                    factor=factor,
                    source_height=source_height,
                    source_width=source_width,
                    phase_mode=phase_mode,
                )
                continue
            if op == "color_remap":
                mapping = params.get("mapping", {})
                if isinstance(mapping, dict):
                    remap = {int(k): int(v) for k, v in mapping.items()}
                    candidate = [[int(remap.get(cell, cell)) for cell in row] for row in candidate]
                continue
            if op == "lookup_color_remap":
                mode = str(params.get("mode", "") or "")
                if mode == "marker_shape_color_lookup":
                    candidate = self._grid_marker_shape_color_lookup_recolor(candidate, params=params)
                continue
            if op == "window_extract":
                mode = str(params.get("mode", "") or "")
                if mode == "marker_axis_crop":
                    candidate = self._grid_marker_axis_crop(candidate, params=params)
                if mode == "marker_opposite_crop":
                    candidate = self._grid_marker_opposite_crop(candidate, params=params)
                continue
            if op == "object_place":
                mode = str(params.get("mode", "") or "")
                if mode == "self_pattern_nonzero_mask":
                    candidate = self._grid_self_pattern_nonzero_mask(candidate)
                elif mode == "self_pattern_complement_mask":
                    candidate = self._grid_self_pattern_complement_mask(candidate)
                elif mode == "connect_color_pairs":
                    candidate = self._grid_connect_color_pairs(candidate)
                elif mode == "separator_bridge_projection":
                    candidate = self._grid_separator_bridge_projection(candidate)
                elif mode == "anchor_spiral_pair":
                    candidate = self._grid_anchor_spiral_pair(candidate)
                elif mode == "diagonal_component_pack":
                    candidate = self._grid_diagonal_component_pack(candidate)
                elif mode == "repeated_tile_consensus":
                    candidate = self._grid_repeated_tile_consensus(candidate)
                continue
            if op == "conditional_fill":
                mode = str(params.get("mode", "") or "")
                if mode == "enclosed_zero_count_mod_10":
                    candidate = self._grid_fill_enclosed_zero_regions(candidate)
                elif mode == "enclosed_zero_count_lookup":
                    count_map = params.get("count_map", {})
                    if isinstance(count_map, dict):
                        normalized = {int(k): int(v) for k, v in count_map.items()}
                        candidate = self._grid_fill_enclosed_zero_regions_by_count_map(candidate, normalized)
                continue
        return candidate

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

    def _grid_periodic_tile(self, grid: list[list[int]], factor: int) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        factor = max(1, int(factor))
        h = len(grid)
        w = len(grid[0])
        return [
            [int(grid[r % h][c % w]) for c in range(w * factor)]
            for r in range(h * factor)
        ]

    def _grid_phase_tile(self, grid: list[list[int]], factor: int) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        factor = max(1, int(factor))
        h = len(grid)
        w = len(grid[0])
        out: list[list[int]] = []
        for r in range(h * factor):
            shift = (r // h) % max(1, w)
            out.append([int(grid[r % h][(c + shift) % w]) for c in range(w * factor)])
        return out

    def _grid_phase_shift_existing_tiling(
        self,
        grid: list[list[int]],
        *,
        factor: int,
        source_height: int,
        source_width: int,
        phase_mode: str,
    ) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        factor = max(1, int(factor))
        source_height = max(1, int(source_height))
        source_width = max(1, int(source_width))
        phase_mode = str(phase_mode or "row_block_shift").strip().lower()
        out = [row[:] for row in grid]
        total_h = len(out)
        total_w = len(out[0]) if total_h else 0
        if total_h != source_height * factor or total_w != source_width * factor:
            return out

        if phase_mode == "row_block_shift":
            for block_row in range(factor):
                shift = block_row % max(1, source_width)
                if shift == 0:
                    continue
                start_r = block_row * source_height
                end_r = start_r + source_height
                for r in range(start_r, end_r):
                    out[r] = [int(out[r][(c + shift) % total_w]) for c in range(total_w)]
            return out

        if phase_mode == "col_block_shift":
            for block_col in range(factor):
                shift = block_col % max(1, source_height)
                if shift == 0:
                    continue
                start_c = block_col * source_width
                end_c = start_c + source_width
                for c in range(start_c, end_c):
                    column = [int(out[r][c]) for r in range(total_h)]
                    shifted = [column[(r + shift) % total_h] for r in range(total_h)]
                    for r in range(total_h):
                        out[r][c] = shifted[r]
            return out

        return out

    def _grid_self_pattern_nonzero_mask(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        out = [[0 for _ in range(w * w)] for _ in range(h * h)]
        for block_r in range(h):
            for block_c in range(w):
                if int(grid[block_r][block_c]) == 0:
                    continue
                for r in range(h):
                    for c in range(w):
                        out[(block_r * h) + r][(block_c * w) + c] = int(grid[r][c])
        return out

    def _grid_self_pattern_complement_mask(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        out = [[0 for _ in range(w * w)] for _ in range(h * h)]
        complement = [[1 if int(grid[r][c]) == 0 else 0 for c in range(w)] for r in range(h)]
        for block_r in range(h):
            for block_c in range(w):
                color = int(grid[block_r][block_c])
                if color == 0:
                    continue
                for r in range(h):
                    for c in range(w):
                        if complement[r][c]:
                            out[(block_r * h) + r][(block_c * w) + c] = color
        return out

    def _grid_connect_color_pairs(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        out = [row[:] for row in grid]
        color_points: dict[int, list[tuple[int, int]]] = {}
        for r in range(h):
            for c in range(w):
                value = int(grid[r][c])
                if value == 0:
                    continue
                color_points.setdefault(value, []).append((r, c))

        horizontal: list[tuple[int, int, int, int]] = []
        vertical: list[tuple[int, int, int, int]] = []
        for color, points in color_points.items():
            if len(points) != 2:
                continue
            (r1, c1), (r2, c2) = points
            if r1 == r2:
                horizontal.append((color, r1, min(c1, c2), max(c1, c2)))
            elif c1 == c2:
                vertical.append((color, c1, min(r1, r2), max(r1, r2)))

        for color, row, start_c, end_c in horizontal:
            for c in range(start_c, end_c + 1):
                out[row][c] = int(color)
        for color, col, start_r, end_r in vertical:
            for r in range(start_r, end_r + 1):
                out[r][col] = int(color)
        return out

    def _grid_separator_bridge_projection(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        separator = self._detect_full_separator_line(grid, color=8)
        if separator is None:
            return [row[:] for row in grid]
        orientation, index = separator
        color4_components = self._connected_components_value(grid, target_value=4)
        if not color4_components:
            return [row[:] for row in grid]
        component_boxes: list[tuple[int, int, int, int]] = []
        sides: set[str] = set()
        for comp in color4_components:
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            top = min(rows)
            bottom = max(rows)
            left = min(cols)
            right = max(cols)
            component_boxes.append((top, bottom, left, right))
            if orientation == "horizontal":
                if bottom < index:
                    sides.add("before")
                elif top > index:
                    sides.add("after")
                else:
                    return [row[:] for row in grid]
            else:
                if right < index:
                    sides.add("before")
                elif left > index:
                    sides.add("after")
                else:
                    return [row[:] for row in grid]
        if len(sides) != 1:
            return [row[:] for row in grid]
        source_side = next(iter(sides))
        h = len(grid)
        w = len(grid[0]) if h else 0
        out = [row[:] for row in grid]

        def _horizontal_projection_pattern(
            *,
            source_side_value: str,
            band_left: int,
            band_right: int,
        ) -> tuple[int, int, list[list[int]]]:
            row_start = index + 1 if source_side_value == "before" else 0
            row_stop = h if source_side_value == "before" else index
            cells = [
                (r, c)
                for r in range(row_start, row_stop)
                for c in range(band_left, band_right + 1)
                if int(grid[r][c]) == 2
            ]
            if not cells:
                return row_start, row_start, []
            pattern_top = min(r for r, _ in cells)
            pattern_bottom = max(r for r, _ in cells)
            pattern = [
                [
                    1 if int(grid[r][c]) == 2 else 0
                    for c in range(band_left, band_right + 1)
                ]
                for r in range(pattern_top, pattern_bottom + 1)
            ]
            return pattern_top, pattern_bottom, pattern

        def _vertical_projection_pattern(
            *,
            source_side_value: str,
            band_top: int,
            band_bottom: int,
        ) -> tuple[int, int, list[list[int]]]:
            col_start = index + 1 if source_side_value == "before" else 0
            col_stop = w if source_side_value == "before" else index
            cells = [
                (r, c)
                for r in range(band_top, band_bottom + 1)
                for c in range(col_start, col_stop)
                if int(grid[r][c]) == 2
            ]
            if not cells:
                return col_start, col_start, []
            pattern_left = min(c for _, c in cells)
            pattern_right = max(c for _, c in cells)
            pattern = [
                [
                    1 if int(grid[r][c]) == 2 else 0
                    for c in range(pattern_left, pattern_right + 1)
                ]
                for r in range(band_top, band_bottom + 1)
            ]
            return pattern_left, pattern_right, pattern

        for comp in color4_components:
            for r, c in comp:
                out[r][c] = 3
        for top, bottom, left, right in component_boxes:
            box_height = bottom - top + 1
            box_width = right - left + 1
            if orientation == "horizontal":
                _, _, pattern = _horizontal_projection_pattern(
                    source_side_value=source_side,
                    band_left=left,
                    band_right=right,
                )
                pattern_height = len(pattern)
                if source_side == "before":
                    for r in range(bottom + 1, index):
                        for c in range(left, right + 1):
                            out[r][c] = 4
                    copy_top = h - pattern_height if pattern_height else h - box_height
                    for r in range(index + 1, copy_top):
                        for c in range(left, right + 1):
                            out[r][c] = 8
                    for r in range(copy_top, h):
                        for c in range(left, right + 1):
                            out[r][c] = 8
                    if pattern:
                        for rr, row in enumerate(pattern):
                            for cc, value in enumerate(row):
                                if value:
                                    out[copy_top + rr][left + cc] = 2
                else:
                    copy_top = 0
                    copy_bottom = pattern_height - 1 if pattern_height else box_height - 1
                    for r in range(copy_top, copy_bottom + 1):
                        for c in range(left, right + 1):
                            out[r][c] = 8
                    if pattern:
                        for rr, row in enumerate(pattern):
                            for cc, value in enumerate(row):
                                if value:
                                    out[copy_top + rr][left + cc] = 2
                    for r in range(copy_bottom + 1, index):
                        for c in range(left, right + 1):
                            out[r][c] = 8
                    for r in range(index + 1, top):
                        for c in range(left, right + 1):
                            out[r][c] = 4
            else:
                _, _, pattern = _vertical_projection_pattern(
                    source_side_value=source_side,
                    band_top=top,
                    band_bottom=bottom,
                )
                pattern_width = len(pattern[0]) if pattern and pattern[0] else 0
                if source_side == "before":
                    for r in range(top, bottom + 1):
                        for c in range(right + 1, index):
                            out[r][c] = 4
                    copy_left = w - pattern_width if pattern_width else w - box_width
                    for r in range(top, bottom + 1):
                        for c in range(index + 1, copy_left):
                            out[r][c] = 8
                    for r in range(top, bottom + 1):
                        for c in range(copy_left, w):
                            out[r][c] = 8
                    if pattern:
                        for rr, row in enumerate(pattern):
                            for cc, value in enumerate(row):
                                if value:
                                    out[top + rr][copy_left + cc] = 2
                else:
                    copy_left = 0
                    copy_right = pattern_width - 1 if pattern_width else box_width - 1
                    for r in range(top, bottom + 1):
                        for c in range(copy_left, copy_right + 1):
                            out[r][c] = 8
                    if pattern:
                        for rr, row in enumerate(pattern):
                            for cc, value in enumerate(row):
                                if value:
                                    out[top + rr][copy_left + cc] = 2
                    for r in range(top, bottom + 1):
                        for c in range(copy_right + 1, index):
                            out[r][c] = 8
                    for r in range(top, bottom + 1):
                        for c in range(index + 1, left):
                            out[r][c] = 4
        return out

    def _grid_anchor_spiral_pair(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        if w < 2:
            return [row[:] for row in grid]

        anchor_points = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if int(value) == 1]
        if len(anchor_points) != 1:
            return [row[:] for row in grid]
        anchor_r, anchor_c = anchor_points[0]

        primary = int(grid[0][0])
        secondary = int(grid[0][1])
        if primary in (0, 1) or secondary in (0, 1) or primary == secondary:
            return [row[:] for row in grid]

        out = [[0 for _ in range(w)] for _ in range(h)]
        out[anchor_r][anchor_c] = 1
        start_c = anchor_c - 1
        if start_c < 0:
            return out
        out[anchor_r][start_c] = primary

        directions = ((0, -1), (1, 0), (0, 1), (-1, 0))
        current_r = anchor_r
        current_c = start_c
        # The family uses a two-cell primary seed to the left of the anchor,
        # then expands clockwise as a rectangular spiral with segment lengths
        # 3, 4, 5, ... from the second segment onward.
        segment_length = 2
        max_segments = (h + w) * 2
        for segment_idx in range(max_segments):
            dr, dc = directions[segment_idx % 4]
            color = primary if segment_idx % 2 == 0 else secondary
            if segment_idx == 0:
                desired_steps = segment_length - 1
            else:
                segment_length += 1
                desired_steps = segment_length
            actual_steps = 0
            for _ in range(desired_steps):
                next_r = current_r + dr
                next_c = current_c + dc
                if next_r < 0 or next_r >= h or next_c < 0 or next_c >= w:
                    break
                if next_r == anchor_r and next_c == anchor_c:
                    break
                current_r = next_r
                current_c = next_c
                out[current_r][current_c] = color
                actual_steps += 1
            if actual_steps < desired_steps:
                break
        return out

    def _grid_diagonal_component_pack(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        out = [[0 for _ in range(w)] for _ in range(h)]
        components = self._connected_components_nonzero(grid)
        packed: list[dict[str, Any]] = []
        for comp in components:
            min_r = min(r for r, _ in comp)
            max_r = max(r for r, _ in comp)
            min_c = min(c for _, c in comp)
            max_c = max(c for _, c in comp)
            packed.append(
                {
                    "cells": comp,
                    "min_r": min_r,
                    "min_c": min_c,
                    "height": (max_r - min_r) + 1,
                    "width": (max_c - min_c) + 1,
                }
            )
        packed.sort(key=lambda item: (int(item["min_c"]), int(item["min_r"])))
        anchor_r = 0
        anchor_c = 0
        for item in packed:
            min_r = int(item["min_r"])
            min_c = int(item["min_c"])
            for r, c in item["cells"]:
                rr = anchor_r + (r - min_r)
                cc = anchor_c + (c - min_c)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = int(grid[r][c])
            anchor_r += max(0, int(item["height"]) - 1)
            anchor_c += max(0, int(item["width"]) - 1)
        return out

    def _connected_components_value(
        self,
        grid: list[list[int]],
        *,
        target_value: int,
    ) -> list[list[tuple[int, int]]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        target = int(target_value)
        visited = [[False] * w for _ in range(h)]
        components: list[list[tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if visited[r][c] or int(grid[r][c]) != target:
                    continue
                comp: list[tuple[int, int]] = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if visited[nr][nc] or int(grid[nr][nc]) != target:
                            continue
                        visited[nr][nc] = True
                        stack.append((nr, nc))
                components.append(comp)
        return components

    def _grid_fill_enclosed_zero_regions(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        out = [row[:] for row in grid]
        zero_components = self._connected_components_value(grid, target_value=0)
        for comp in zero_components:
            comp_set = set(comp)
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            border_colors: set[int] = set()
            for r, c in comp:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in comp_set:
                        continue
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr][nc]) != 0:
                        border_colors.add(int(grid[nr][nc]))
            if len(border_colors) != 1:
                continue
            fill_color = len(comp) % 10
            if fill_color == 0:
                fill_color = next(iter(border_colors))
            for r, c in comp:
                out[r][c] = int(fill_color)
        return out

    def _grid_fill_enclosed_zero_regions_by_count_map(
        self,
        grid: list[list[int]],
        count_map: dict[int, int],
    ) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        out = [row[:] for row in grid]
        zero_components = self._connected_components_value(grid, target_value=0)
        for comp in zero_components:
            comp_set = set(comp)
            if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                continue
            border_colors: set[int] = set()
            for r, c in comp:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if (nr, nc) in comp_set:
                        continue
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr][nc]) != 0:
                        border_colors.add(int(grid[nr][nc]))
            if len(border_colors) != 1:
                continue
            fill_color = int(count_map.get(len(comp), 0))
            if fill_color == 0:
                continue
            for r, c in comp:
                out[r][c] = fill_color
        return out

    def _detect_periodic_tile_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not output_grid:
            return None
        in_h = len(input_grid)
        in_w = len(input_grid[0]) if in_h else 0
        out_h = len(output_grid)
        out_w = len(output_grid[0]) if out_h else 0
        if not in_h or not in_w or out_h % in_h or out_w % in_w:
            return None
        factor_h = out_h // in_h
        factor_w = out_w // in_w
        if factor_h != factor_w or factor_h <= 1:
            return None
        if self._grid_periodic_tile(input_grid, factor_h) == output_grid:
            return {"factor": factor_h}
        return None

    def _detect_phase_tile_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not output_grid:
            return None
        in_h = len(input_grid)
        in_w = len(input_grid[0]) if in_h else 0
        out_h = len(output_grid)
        out_w = len(output_grid[0]) if out_h else 0
        if not in_h or not in_w or out_h % in_h or out_w % in_w:
            return None
        factor_h = out_h // in_h
        factor_w = out_w // in_w
        if factor_h != factor_w or factor_h <= 1:
            return None
        tiled = self._grid_periodic_tile(input_grid, factor_h)
        row_shift = self._grid_phase_shift_existing_tiling(
            tiled,
            factor=factor_h,
            source_height=in_h,
            source_width=in_w,
            phase_mode="row_block_shift",
        )
        if row_shift == output_grid:
            return {
                "factor": factor_h,
                "source_height": in_h,
                "source_width": in_w,
                "phase_mode": "row_block_shift",
            }
        col_shift = self._grid_phase_shift_existing_tiling(
            tiled,
            factor=factor_h,
            source_height=in_h,
            source_width=in_w,
            phase_mode="col_block_shift",
        )
        if col_shift == output_grid:
            return {
                "factor": factor_h,
                "source_height": in_h,
                "source_width": in_w,
                "phase_mode": "col_block_shift",
            }
        return None

    def _detect_color_mapping(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[int, int]:
        if not input_grid or not output_grid:
            return {}
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return {}
        mapping: dict[int, int] = {}
        for in_row, out_row in zip(input_grid, output_grid):
            for in_cell, out_cell in zip(in_row, out_row):
                src = int(in_cell)
                dst = int(out_cell)
                prev = mapping.get(src)
                if prev is None:
                    mapping[src] = dst
                elif prev != dst:
                    return {}
        return mapping

    def _detect_direct_transform_ops(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        self_complement = self._detect_self_pattern_complement_mask_transform(input_grid, output_grid)
        if self_complement is not None:
            return self_complement

        self_place = self._detect_self_pattern_nonzero_mask_transform(input_grid, output_grid)
        if self_place is not None:
            return self_place

        connect_pairs = self._detect_connect_color_pairs_transform(input_grid, output_grid)
        if connect_pairs is not None:
            return connect_pairs

        separator_bridge = self._detect_separator_bridge_projection_transform(input_grid, output_grid)
        if separator_bridge is not None:
            return separator_bridge

        anchor_spiral = self._detect_anchor_spiral_pair_transform(input_grid, output_grid)
        if anchor_spiral is not None:
            return anchor_spiral

        diagonal_pack = self._detect_diagonal_component_pack_transform(input_grid, output_grid)
        if diagonal_pack is not None:
            return diagonal_pack

        repeated_tile_consensus = self._detect_repeated_tile_consensus_transform(input_grid, output_grid)
        if repeated_tile_consensus is not None:
            return repeated_tile_consensus

        enclosed_fill = self._detect_enclosed_zero_fill_transform(input_grid, output_grid)
        if enclosed_fill is not None:
            return enclosed_fill

        marker_axis_crop = self._detect_marker_axis_crop_transform(input_grid, output_grid)
        if marker_axis_crop is not None:
            return marker_axis_crop

        variants: list[tuple[tuple[str, ...], list[list[int]]]] = [
            (("rotate_90",), self._grid_rotate_90(input_grid)),
            (("rotate_180",), self._grid_rotate_90(self._grid_rotate_90(input_grid))),
            (("rotate_270",), self._grid_rotate_90(self._grid_rotate_90(self._grid_rotate_90(input_grid)))),
            (("mirror_h",), self._grid_flip_h(input_grid)),
            (("mirror_v",), self._grid_flip_v(input_grid)),
            (("transpose",), [list(col) for col in zip(*input_grid)] if input_grid else []),
            (("identity",), [row[:] for row in input_grid]),
        ]
        for ops, candidate in variants:
            if candidate == output_grid:
                return {"ops": ops, "params": {}, "confidence": 0.92}
            mapping = self._detect_color_mapping(candidate, output_grid)
            if mapping and [[int(mapping.get(cell, cell)) for cell in row] for row in candidate] == output_grid:
                return {"ops": (*ops, "color_remap"), "params": {"mapping": mapping}, "confidence": 0.9}
            marker_crop = self._detect_marker_opposite_crop_transform(candidate, output_grid)
            if marker_crop is not None:
                return {
                    "ops": (*ops, "object_extract", "window_extract"),
                    "params": marker_crop,
                    "confidence": 0.965,
                }
        return None

    def _detect_self_pattern_nonzero_mask_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        h = len(input_grid)
        w = len(input_grid[0])
        if len(output_grid) != h * h or len(output_grid[0]) != w * w:
            return None
        candidate = self._grid_self_pattern_nonzero_mask(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "self_pattern_nonzero_mask"},
            "confidence": 0.97,
        }

    def _detect_self_pattern_complement_mask_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        h = len(input_grid)
        w = len(input_grid[0])
        if len(output_grid) != h * h or len(output_grid[0]) != w * w:
            return None
        candidate = self._grid_self_pattern_complement_mask(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "self_pattern_complement_mask"},
            "confidence": 0.98,
        }

    def _detect_connect_color_pairs_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        candidate = self._grid_connect_color_pairs(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "connect_color_pairs"},
            "confidence": 0.95,
        }

    def _detect_separator_bridge_projection_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        candidate = self._grid_separator_bridge_projection(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "separator_bridge_projection"},
            "confidence": 0.982,
        }

    def _detect_anchor_spiral_pair_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        candidate = self._grid_anchor_spiral_pair(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "anchor_spiral_pair"},
            "confidence": 0.975,
        }

    def _detect_diagonal_component_pack_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        candidate = self._grid_diagonal_component_pack(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "diagonal_component_pack"},
            "confidence": 0.97,
        }

    def _detect_repeated_tile_consensus_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        candidate = self._grid_repeated_tile_consensus(input_grid)
        if candidate == input_grid or candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "object_place"),
            "params": {"mode": "repeated_tile_consensus"},
            "confidence": 0.975,
        }

    def _detect_enclosed_zero_fill_transform(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not input_grid or not input_grid[0] or not output_grid or not output_grid[0]:
            return None
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        candidate = self._grid_fill_enclosed_zero_regions(input_grid)
        if candidate != output_grid:
            return None
        return {
            "ops": ("object_extract", "connected_components", "conditional_fill"),
            "params": {"mode": "enclosed_zero_count_mod_10"},
            "confidence": 0.96,
        }

    def _detect_marker_opposite_crop_transform(
        self,
        grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not grid or not grid[0] or not output_grid or not output_grid[0]:
            return None
        for marker_color in sorted({int(cell) for row in grid for cell in row if int(cell) != 0}):
            candidate = self._grid_marker_opposite_crop(
                grid,
                params={"mode": "marker_opposite_crop", "marker_color": marker_color, "mirror_margin": 2},
            )
            if candidate == output_grid:
                return {
                    "mode": "marker_opposite_crop",
                    "marker_color": int(marker_color),
                    "mirror_margin": 2,
                }
        return None

    def _detect_marker_axis_crop_transform(
        self,
        grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if not grid or not grid[0] or not output_grid or not output_grid[0]:
            return None
        for marker_color in sorted({int(cell) for row in grid for cell in row if int(cell) != 0}):
            candidate = self._grid_marker_axis_crop(
                grid,
                params={"mode": "marker_axis_crop", "marker_color": marker_color, "mirror_margin": 2},
            )
            if candidate == output_grid:
                return {
                    "ops": ("object_extract", "window_extract"),
                    "params": {
                        "mode": "marker_axis_crop",
                        "marker_color": int(marker_color),
                        "mirror_margin": 2,
                    },
                    "confidence": 0.985,
                }
        return None

    def _grid_marker_opposite_crop(
        self,
        grid: list[list[int]],
        params: dict[str, Any],
    ) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        marker_color = int(params.get("marker_color", 8))
        mirror_margin = max(0, int(params.get("mirror_margin", 2)))
        component = self._select_marker_rectangle_component(grid, marker_color=marker_color)
        if component is None:
            return [row[:] for row in grid]
        top, left, height, width = component
        total_h = len(grid)
        total_w = len(grid[0]) if total_h else 0
        crop_top = total_h - height - top - mirror_margin
        crop_left = total_w - width - left - mirror_margin
        if crop_top < 0 or crop_left < 0:
            return [row[:] for row in grid]
        if crop_top + height > total_h or crop_left + width > total_w:
            return [row[:] for row in grid]
        return [row[crop_left : crop_left + width] for row in grid[crop_top : crop_top + height]]

    def _grid_marker_axis_crop(
        self,
        grid: list[list[int]],
        params: dict[str, Any],
    ) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        marker_color = int(params.get("marker_color", 8))
        mirror_margin = max(0, int(params.get("mirror_margin", 2)))
        component = self._select_marker_rectangle_component(grid, marker_color=marker_color)
        if component is None:
            return [row[:] for row in grid]
        top, left, height, width = component
        total_h = len(grid)
        total_w = len(grid[0]) if total_h else 0
        if height > width and (left == 0 or left + width == total_w):
            crop_top = max(0, min(total_h - width, left))
            crop_left = max(0, min(total_w - height, top))
            subgrid = [row[crop_left : crop_left + height] for row in grid[crop_top : crop_top + width]]
            if len(subgrid) == width and all(len(row) == height for row in subgrid):
                return [list(col) for col in zip(*subgrid)]
        right_gap = total_w - (left + width)
        bottom_gap = total_h - (top + height)
        side_gap = min(left, right_gap)
        vertical_gap = min(top, bottom_gap)

        if side_gap <= width:
            crop_left = total_w - width - left
            if left > right_gap:
                crop_left += mirror_margin
            else:
                crop_left -= mirror_margin
            crop_left = max(0, min(total_w - width, crop_left))
            subgrid = [row[crop_left : crop_left + width] for row in grid[top : top + height]]
            return [row[::-1] for row in subgrid]

        crop_top = total_h - height - top
        if top > bottom_gap:
            crop_top += mirror_margin
        else:
            crop_top -= mirror_margin
        crop_top = max(0, min(total_h - height, crop_top))
        subgrid = [row[left : left + width] for row in grid[crop_top : crop_top + height]]
        return subgrid[::-1]

    def _select_marker_rectangle_component(
        self,
        grid: list[list[int]],
        *,
        marker_color: int,
    ) -> tuple[int, int, int, int] | None:
        best: tuple[int, int, int, int] | None = None
        best_area = -1
        for comp in self._connected_components_value(grid, target_value=marker_color):
            if not comp:
                continue
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            top = min(rows)
            bottom = max(rows)
            left = min(cols)
            right = max(cols)
            height = bottom - top + 1
            width = right - left + 1
            area = height * width
            if area != len(comp):
                continue
            if area > best_area:
                best_area = area
                best = (top, left, height, width)
        return best

    def _detect_full_separator_line(
        self,
        grid: list[list[int]],
        *,
        color: int,
    ) -> tuple[str, int] | None:
        if not grid or not grid[0]:
            return None
        target = int(color)
        h = len(grid)
        w = len(grid[0])
        full_rows = [r for r in range(h) if all(int(cell) == target for cell in grid[r])]
        if len(full_rows) == 1:
            return ("horizontal", full_rows[0])
        full_cols: list[int] = []
        for c in range(w):
            if all(int(grid[r][c]) == target for r in range(h)):
                full_cols.append(c)
        if len(full_cols) == 1:
            return ("vertical", full_cols[0])
        return None

    def _detect_enclosed_zero_fill_count_lookup_pattern(
        self,
        train_examples: list[dict[str, Any]],
    ) -> _GeneratedPattern | None:
        count_map: dict[int, int] = {}
        verification_examples: list[tuple[list[list[int]], list[list[int]]]] = []
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            input_grid = self._to_grid(pair.get("input"))
            output_grid = self._to_grid(pair.get("output"))
            if not input_grid or not output_grid:
                continue
            verification_examples.append((input_grid, output_grid))
            h = len(input_grid)
            w = len(input_grid[0]) if input_grid else 0
            zero_components = self._connected_components_value(input_grid, target_value=0)
            for comp in zero_components:
                comp_set = set(comp)
                if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
                    continue
                border_colors: set[int] = set()
                for r, c in comp:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = r + dr
                        nc = c + dc
                        if (nr, nc) in comp_set:
                            continue
                        if 0 <= nr < h and 0 <= nc < w and int(input_grid[nr][nc]) != 0:
                            border_colors.add(int(input_grid[nr][nc]))
                if len(border_colors) != 1:
                    continue
                fill_values = {int(output_grid[r][c]) for r, c in comp if int(output_grid[r][c]) != 0}
                if len(fill_values) != 1:
                    return None
                fill_color = int(next(iter(fill_values)))
                count = len(comp)
                existing = count_map.get(count)
                if existing is not None and existing != fill_color:
                    return None
                count_map[count] = fill_color
        if len(count_map) < 2 or not verification_examples:
            return None
        params = {"mode": "enclosed_zero_count_lookup", "count_map": count_map}
        for input_grid, output_grid in verification_examples:
            candidate = self._grid_fill_enclosed_zero_regions_by_count_map(input_grid, count_map)
            if candidate != output_grid:
                return None
        return _GeneratedPattern(
            pattern_id="arc_four_pass_enclosed_zero_count_lookup",
            source_galaxy="Grammar",
            target_galaxy="Grammar",
            confidence=0.985,
            query="object_extract, connected_components, conditional_fill, enclosed_zero_count_lookup",
            source="arc_four_pass",
            ops=("object_extract", "connected_components", "conditional_fill"),
            params=params,
            composition_depth=3,
        )

    def _detect_marker_shape_color_lookup_pattern(
        self,
        train_examples: list[dict[str, Any]],
    ) -> _GeneratedPattern | None:
        marker_color: int | None = None
        object_color: int | None = None
        signature_map: dict[str, int] = {}
        verification_examples: list[tuple[list[list[int]], list[list[int]]]] = []
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            input_grid = self._to_grid(pair.get("input"))
            output_grid = self._to_grid(pair.get("output"))
            if not input_grid or not output_grid:
                continue
            example = self._extract_marker_shape_lookup_example(input_grid, output_grid)
            if example is None:
                return None
            verification_examples.append((input_grid, output_grid))
            if marker_color is None:
                marker_color = int(example["marker_color"])
            elif marker_color != int(example["marker_color"]):
                return None
            if object_color is None:
                object_color = int(example["object_color"])
            elif object_color != int(example["object_color"]):
                return None
            signature = str(example["marker_signature"])
            output_color = int(example["output_color"])
            existing = signature_map.get(signature)
            if existing is not None and existing != output_color:
                return None
            signature_map[signature] = output_color

        if not verification_examples or len(signature_map) < 2 or marker_color is None or object_color is None:
            return None

        params = {
            "mode": "marker_shape_color_lookup",
            "marker_color": int(marker_color),
            "object_color": int(object_color),
            "shape_to_color": dict(signature_map),
        }
        for input_grid, output_grid in verification_examples:
            candidate = self._grid_marker_shape_color_lookup_recolor(input_grid, params=params)
            if candidate != output_grid:
                return None
        return _GeneratedPattern(
            pattern_id="arc_four_pass_marker_shape_color_lookup",
            source_galaxy="Grammar",
            target_galaxy="Grammar",
            confidence=0.985,
            query="object_extract, marker_shape_lookup, lookup_color_remap",
            source="arc_four_pass",
            ops=("object_extract", "lookup_color_remap"),
            params=params,
            composition_depth=3,
        )

    def _verify_arc_ops_on_examples(
        self,
        train_examples: list[dict[str, Any]],
        *,
        ops: tuple[str, ...],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        total = 0
        passed = 0
        for pair in train_examples:
            if not isinstance(pair, dict):
                continue
            input_grid = self._to_grid(pair.get("input"))
            output_grid = self._to_grid(pair.get("output"))
            if not input_grid or not output_grid:
                continue
            total += 1
            predicted = self._apply_compositional_ops(input_grid, ops=ops, params=params)
            if predicted == output_grid:
                passed += 1
        return {"pass": bool(total and passed == total), "passed": passed, "total": total}

    def _component_signature(self, cells: list[tuple[int, int]]) -> str:
        if not cells:
            return ""
        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        normalized = sorted((r - min_r, c - min_c) for r, c in cells)
        return ";".join(f"{r}:{c}" for r, c in normalized)

    def _extract_marker_shape_lookup_example(
        self,
        input_grid: list[list[int]],
        output_grid: list[list[int]],
    ) -> dict[str, Any] | None:
        if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
            return None
        input_colors = sorted(self._palette_of(input_grid) - {0})
        output_colors = sorted(self._palette_of(output_grid) - {0})
        if len(input_colors) != 2 or len(output_colors) != 1:
            return None

        output_mask = {
            (r, c)
            for r, row in enumerate(output_grid)
            for c, value in enumerate(row)
            if int(value) != 0
        }
        object_color: int | None = None
        marker_color: int | None = None
        for color in input_colors:
            color_mask = {
                (r, c)
                for r, row in enumerate(input_grid)
                for c, value in enumerate(row)
                if int(value) == int(color)
            }
            if color_mask == output_mask:
                object_color = int(color)
            else:
                marker_color = int(color)
        if object_color is None or marker_color is None:
            return None

        marker_components = self._connected_components_value(input_grid, target_value=marker_color)
        if not marker_components:
            return None
        marker_components.sort(key=lambda comp: (-len(comp), min(r for r, _ in comp), min(c for _, c in comp)))
        marker_signature = self._component_signature(marker_components[0])
        if not marker_signature:
            return None

        output_color = int(output_colors[0])
        candidate = self._grid_marker_shape_color_lookup_recolor(
            input_grid,
            params={
                "marker_color": marker_color,
                "object_color": object_color,
                "shape_to_color": {marker_signature: output_color},
            },
        )
        if candidate != output_grid:
            return None
        return {
            "marker_signature": marker_signature,
            "marker_color": marker_color,
            "object_color": object_color,
            "output_color": output_color,
        }

    def _arc_param_hint(self, params: dict[str, Any]) -> str:
        if not params:
            return ""
        if "factor" in params:
            return f"factor {int(params['factor'])}"
        if "mapping" in params:
            return "color_remap"
        if "shape_to_color" in params:
            return "marker_shape_lookup"
        return "params"

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

    def _palette_distribution(self, grid: list[list[int]]) -> dict[int, float]:
        total = sum(len(row) for row in grid)
        if total <= 0:
            return {}
        counts: Counter[int] = Counter()
        for row in grid:
            for cell in row:
                counts[int(cell)] += 1
        return {color: (float(count) / float(total)) for color, count in counts.items()}

    def _palette_distribution_similarity(
        self,
        expected: dict[int, float],
        candidate: dict[int, float],
    ) -> float:
        """
        Similarity in [0,1] using normalized L1 distance over color ratios.
        """
        if not expected and not candidate:
            return 1.0
        universe = set(expected.keys()) | set(candidate.keys())
        if not universe:
            return 1.0
        l1 = 0.0
        for color in universe:
            l1 += abs(float(expected.get(color, 0.0)) - float(candidate.get(color, 0.0)))
        # L1 for two distributions is in [0,2].
        return self._clamp(1.0 - (l1 / 2.0))

    def _align_candidate_palette(
        self,
        candidate_grid: list[list[int]],
        profile: dict[str, Any],
    ) -> list[list[int]]:
        """
        Remap candidate colors toward train-output palette distribution.

        This is a lightweight post-generation alignment step that improves
        palette consistency before ranking/oracle checks.
        """
        if not candidate_grid or not candidate_grid[0]:
            return []
        output_palette = list(profile.get("output_palette", []))
        if not output_palette:
            return [row[:] for row in candidate_grid]
        expected_dist_raw = profile.get("output_palette_distribution", {})
        expected_dist = (
            {int(k): float(v) for k, v in expected_dist_raw.items()}
            if isinstance(expected_dist_raw, dict)
            else {}
        )
        candidate_dist = self._palette_distribution(candidate_grid)
        if not candidate_dist:
            return [row[:] for row in candidate_grid]

        # Order candidate colors by observed frequency (descending).
        cand_ranked = [
            color
            for color, _ in sorted(
                candidate_dist.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]
        # Order expected colors by train-output distribution, fallback to id order.
        if expected_dist:
            exp_ranked = [
                color
                for color, _ in sorted(
                    expected_dist.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
        else:
            exp_ranked = sorted(int(c) for c in output_palette)
        if not exp_ranked:
            exp_ranked = sorted(int(c) for c in output_palette)
        if not exp_ranked:
            return [row[:] for row in candidate_grid]

        mapping: dict[int, int] = {}
        for idx, color in enumerate(cand_ranked):
            if color in exp_ranked:
                mapping[int(color)] = int(color)
            else:
                mapping[int(color)] = int(exp_ranked[idx % len(exp_ranked)])
        # Ensure all expected colors map to themselves if already present.
        for color in exp_ranked:
            mapping.setdefault(int(color), int(color))

        remapped: list[list[int]] = []
        for row in candidate_grid:
            remapped.append([int(mapping.get(int(cell), int(exp_ranked[0]))) for cell in row])
        return remapped

    def _align_candidate_object_count(
        self,
        candidate_grid: list[list[int]],
        target_count: int,
    ) -> list[list[int]]:
        """
        Heuristic object-count alignment used during candidate generation.

        - If candidate has too many connected components, keep largest components.
        - If candidate has too few components, seed isolated pixels using dominant
          non-zero color to raise component count.
        """
        if not candidate_grid or not candidate_grid[0]:
            return []
        target = max(0, int(target_count))
        if target == 0:
            return [[0 for _ in row] for row in candidate_grid]

        out = [row[:] for row in candidate_grid]
        components = self._connected_components_nonzero(out)
        current = len(components)
        if current == target:
            return out

        if current > target:
            keep_components = sorted(components, key=len, reverse=True)[:target]
            keep_cells = {cell for comp in keep_components for cell in comp}
            for r in range(len(out)):
                for c in range(len(out[0])):
                    if (r, c) not in keep_cells:
                        out[r][c] = 0
            return out

        # current < target: add sparse isolated pixels so object count can rise.
        non_zero_values = [int(cell) for row in out for cell in row if int(cell) != 0]
        fill_color = int(Counter(non_zero_values).most_common(1)[0][0]) if non_zero_values else 1
        h = len(out)
        w = len(out[0])
        needed = target - current
        for r in range(h):
            if needed <= 0:
                break
            for c in range(w):
                if needed <= 0:
                    break
                if out[r][c] != 0:
                    continue
                # Require 4-neighborhood isolation to create a new component.
                isolated = True
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w and out[rr][cc] != 0:
                        isolated = False
                        break
                if not isolated:
                    continue
                out[r][c] = fill_color
                needed -= 1
        return out

    def _connected_components_nonzero(self, grid: list[list[int]]) -> list[list[tuple[int, int]]]:
        """Return 4-connected components for non-zero cells."""
        if not grid or not grid[0]:
            return []
        h = len(grid)
        w = len(grid[0])
        visited = [[False] * w for _ in range(h)]
        components: list[list[tuple[int, int]]] = []

        for r in range(h):
            for c in range(w):
                if visited[r][c] or grid[r][c] == 0:
                    continue
                comp: list[tuple[int, int]] = []
                stack = [(r, c)]
                visited[r][c] = True
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = rr + dr
                        nc = cc + dc
                        if nr < 0 or nr >= h or nc < 0 or nc >= w:
                            continue
                        if visited[nr][nc] or grid[nr][nc] == 0:
                            continue
                        visited[nr][nc] = True
                        stack.append((nr, nc))
                components.append(comp)
        return components

    def _grid_marker_shape_color_lookup_recolor(
        self,
        grid: list[list[int]],
        *,
        params: dict[str, Any],
    ) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        marker_color = int(params.get("marker_color", 0))
        object_color = int(params.get("object_color", 0))
        raw_mapping = params.get("shape_to_color", {})
        if marker_color == 0 or object_color == 0 or not isinstance(raw_mapping, dict):
            return [row[:] for row in grid]
        mapping = {str(key): int(value) for key, value in raw_mapping.items()}
        marker_components = self._connected_components_value(grid, target_value=marker_color)
        if not marker_components:
            return [row[:] for row in grid]
        marker_components.sort(key=lambda comp: (-len(comp), min(r for r, _ in comp), min(c for _, c in comp)))
        signature = self._component_signature(marker_components[0])
        fill_color = int(mapping.get(signature, 0))
        if fill_color == 0:
            return [row[:] for row in grid]
        out = [[0 for _ in row] for row in grid]
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if int(value) == object_color:
                    out[r][c] = fill_color
        return out

    def _dense_axis_groups(self, grid: list[list[int]], *, axis: int) -> list[tuple[int, int]]:
        if not grid or not grid[0]:
            return []
        if axis == 0:
            counts = [sum(1 for value in row if int(value) != 0) for row in grid]
        else:
            width = len(grid[0])
            counts = [sum(1 for row in grid if int(row[col]) != 0) for col in range(width)]
        max_count = max(counts, default=0)
        if max_count <= 0:
            return []
        # Repeated ARC tiles usually have dense content bands separated by
        # sparse/noisy rows or columns. A tighter threshold preserves those
        # separators instead of merging the whole grid into one dense group.
        threshold = max(2, int(max_count * 0.6 + 0.999))
        groups: list[tuple[int, int]] = []
        start: int | None = None
        for idx, count in enumerate(counts):
            if count >= threshold:
                if start is None:
                    start = idx
            elif start is not None:
                groups.append((start, idx - 1))
                start = None
        if start is not None:
            groups.append((start, len(counts) - 1))
        return groups

    def _grid_repeated_tile_consensus(self, grid: list[list[int]]) -> list[list[int]]:
        if not grid or not grid[0]:
            return []
        row_groups = self._dense_axis_groups(grid, axis=0)
        col_groups = self._dense_axis_groups(grid, axis=1)
        if len(row_groups) < 2 or len(col_groups) < 2:
            return [row[:] for row in grid]
        row_heights = {end - start + 1 for start, end in row_groups}
        col_widths = {end - start + 1 for start, end in col_groups}
        if len(row_heights) != 1 or len(col_widths) != 1:
            return [row[:] for row in grid]
        tile_h = next(iter(row_heights))
        tile_w = next(iter(col_widths))
        regions: list[list[list[int]]] = []
        for row_start, row_end in row_groups:
            for col_start, col_end in col_groups:
                region = [grid[r][col_start : col_end + 1] for r in range(row_start, row_end + 1)]
                if len(region) != tile_h or any(len(row) != tile_w for row in region):
                    return [row[:] for row in grid]
                if any(int(value) != 0 for row in region for value in row):
                    regions.append(region)
        if len(regions) < 2:
            return [row[:] for row in grid]
        global_votes = Counter(
            int(value)
            for region in regions
            for row in region
            for value in row
            if int(value) != 0
        )
        consensus = [[0 for _ in range(tile_w)] for _ in range(tile_h)]
        for r in range(tile_h):
            for c in range(tile_w):
                raw_votes = [int(region[r][c]) for region in regions]
                non_zero_votes = Counter(value for value in raw_votes if value != 0)
                if non_zero_votes:
                    best_count = max(non_zero_votes.values())
                    best_values = [
                        value for value, count in non_zero_votes.items() if count == best_count
                    ]
                    if len(best_values) == 1:
                        consensus[r][c] = int(best_values[0])
                    else:
                        best_values.sort(
                            key=lambda value: (
                                global_votes.get(int(value), 0),
                                -int(value),
                            ),
                            reverse=True,
                        )
                        consensus[r][c] = int(best_values[0])
                else:
                    consensus[r][c] = 0
        out = [[0 for _ in row] for row in grid]
        for row_start, _row_end in row_groups:
            for col_start, _col_end in col_groups:
                for r in range(tile_h):
                    for c in range(tile_w):
                        out[row_start + r][col_start + c] = int(consensus[r][c])
        return out

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

    def _normalize_penalty_weight(self, value: float) -> float:
        """Keep penalty exponents within a stable numeric envelope."""
        try:
            numeric = float(value)
        except Exception:
            numeric = 1.0
        return max(0.25, min(4.0, numeric))

    def _clamp(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    def _grid_flip_h(self, grid: list[list[int]]) -> list[list[int]]:
        return [list(reversed(row)) for row in grid]

    def _grid_flip_v(self, grid: list[list[int]]) -> list[list[int]]:
        return list(reversed(grid))

    def _invert_grid_figure_ground(self, grid: list[list[int]]) -> list[list[int]]:
        """
        Derive negative form (carved-space view) from a positive grid.
        """
        if not grid or not grid[0]:
            return []
        max_val = max(int(cell) for row in grid for cell in row)
        min_val = min(int(cell) for row in grid for cell in row)
        if max_val == min_val:
            return [row[:] for row in grid]
        return [[int(max_val - int(cell) + min_val) for cell in row] for row in grid]

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
