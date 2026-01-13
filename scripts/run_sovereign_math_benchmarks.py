#!/usr/bin/env python3
"""
Sovereign Math Benchmark Runner

Uses ONLY sovereign components:
- ModularRPNEngine (PTX-based GPU execution)
- SovereignComposer (Galaxy-based RPN composition)
- Grammar Galaxy word rules

NO CuPy, NO numpy in the hot path.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Allow running as a script without requiring `PYTHONPATH=.`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure sovereign GPU flags are set (mirrors scripts/k3d_env.sh defaults)
os.environ.setdefault("K3D_PTX_STRICT", "1")
os.environ.setdefault("K3D_FORCE_PTX_FUSE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Load all galaxies and registry up front
from knowledge3d.training.math_benchmarks.galaxy_loader import UNIFIED_GALAXY
from knowledge3d.training.math_benchmarks.symbol_registry import (
    SYMBOL_REGISTRY,
    populate_registry_from_galaxy,
)
# Sovereign imports only
from knowledge3d.training.math_benchmarks.math_knowledge_loader import MathKnowledgeLoader
from knowledge3d.training.math_benchmarks.sovereign_composer import SovereignComposer
from knowledge3d.training.math_benchmarks.word_problem_solver import WordProblemSolver
from knowledge3d.training.math_benchmarks.benchmark_evaluator import MathBenchmarkEvaluator
from knowledge3d.training.math_benchmarks.math_output_adapter import MathOutputAdapter
from knowledge3d.training.math_benchmarks.math_templates import get_all_templates
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.training.math_benchmarks.rpn_validator import is_valid_rpn
from knowledge3d.cranium.math_galaxy_population import (
    populate_role_patterns,
    populate_theorem_patterns,
)
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarRule
from knowledge3d.training.math_benchmarks.calculus_grammar_rules import get_calculus_rules
from knowledge3d.training.math_benchmarks.theorem_router import TheoremRouter
from knowledge3d.training.math_benchmarks.latex_normalizer import normalize_latex_to_natural
from knowledge3d.training.math_benchmarks.recursive_solver import RecursiveSolver


def _build_theorem_rules(
    theorem_patterns: List[Dict[str, Any]],
) -> Tuple[List[GrammarRule], Dict[str, Dict[str, Any]]]:
    rules: List[GrammarRule] = []
    meta: Dict[str, Dict[str, Any]] = {}
    for pattern in theorem_patterns:
        precond = pattern.get("precondition", {}) or {}
        cues = []
        for cue in precond.get("context_cues", []) or []:
            token = str(cue or "").strip().lower()
            if len(token) >= 2:
                cues.append(token)
        for tag in pattern.get("semantic_tags", []) or []:
            token = str(tag or "").strip().lower()
            if len(token) >= 2:
                cues.append(token)
        deduped: List[str] = []
        seen: set[str] = set()
        for token in cues:
            if token in seen:
                continue
            seen.add(token)
            deduped.append(token)
        deduped = deduped[:8]
        if not deduped:
            continue
        regex = "|".join(re.escape(token) for token in deduped)
        pattern_text = rf"(?i)\b({regex})\b" if regex else str(pattern.get("pattern_id") or "")
        rpn_program = " ".join(str(tok) for tok in pattern.get("transformation", {}).get("rpn_program", []))
        rule_id = f"theorem:{pattern.get('pattern_id')}"
        domain = f"theorem_{pattern.get('domain') or 'math'}"
        semantics = {
            "pattern_type": "word_sequence",
            "word_pattern": [{"word_in": list(deduped)}],
            "match_mode": "subsequence",
            "max_skip": 6,
        }
        rule = GrammarRule(
            rule_id=rule_id,
            language="math",
            pattern=pattern_text,
            rpn_program=rpn_program,
            domain=domain,
            semantics=semantics,
            description="theorem_pattern",
        )
        rules.append(rule)
        meta[rule_id] = {
            "pattern_id": pattern.get("pattern_id"),
            "grammar_rule": pattern.get("grammar_rule"),
            "semantic_tags": list(pattern.get("semantic_tags") or []),
        }
    return rules, meta


class SovereignBenchmarkRunner:
    """Run math benchmarks using sovereign components only."""

    def __init__(
        self,
        *,
        use_trm_navigator: bool = False,
        disable_retrieval: bool = False,
        shadow_readonly: bool = False,
        load_all_galaxies: bool = False,
        enable_book_galaxies: bool = False,
        book_galaxy_root: str | None = None,
        book_max_books: int = 8,
        book_top_k: int = 5,
        thinking_budget: int = 0,
        verbose: bool = False,
        router_weights: Optional[str] = None,
    ):
        print(UNIFIED_GALAXY.report())
        populate_registry_from_galaxy()
        stats = SYMBOL_REGISTRY.compression_stats()
        print(
            f"Symbol Registry: {stats['unique_symbols']} symbols, "
            f"{stats['cross_domain_symbols']} cross-domain, "
            f"{stats['compression_ratio']:.1f}x compression"
        )

        self.composer = SovereignComposer()
        self.word_solver = WordProblemSolver()
        self.template_rules = get_all_templates()
        self.knowledge_loader = MathKnowledgeLoader()
        knowledge_stats = self.knowledge_loader.load_all()
        kb_symbols_added = self.knowledge_loader.populate_math_galaxy()
        # Load full knowledge base; LOD/FOS filtering happens semantically at match time
        knowledge_rules_raw = self.knowledge_loader.to_grammar_rules()
        self.knowledge_rules = [r for r in knowledge_rules_raw if is_valid_rpn(getattr(r, "rpn_program", ""))]
        invalid_count = len(knowledge_rules_raw) - len(self.knowledge_rules)
        # Build domain buckets with precompiled regex to avoid hot-path CPU thrash
        import re as _re
        self.knowledge_rules_by_domain: Dict[str, List[tuple]] = {}
        for rule in self.knowledge_rules:
            domain = getattr(rule, "domain", "math_kb")
            comp = _re.compile(rule.pattern)
            self.knowledge_rules_by_domain.setdefault(domain, []).append((comp, rule))
        # Global fallback pools sorted by pattern length (avoid rebuilding rules)
        self.knowledge_rules_sorted = sorted(self.knowledge_rules, key=lambda r: len(r.pattern), reverse=True)
        self.knowledge_rules_fallback = self.knowledge_rules_sorted[:256]
        self.knowledge_rules_fallback_compiled = [(_re.compile(rule.pattern), rule) for rule in self.knowledge_rules_fallback]
        # Extended compiled pool (wider net) capped to keep perf bounded
        self.knowledge_rules_len_compiled = [(_re.compile(rule.pattern), rule) for rule in self.knowledge_rules_sorted[:512]]
        print(
            f"Loaded Knowledge: {knowledge_stats['formulas']} formulas, "
            f"{knowledge_stats['rules']} rules, {knowledge_stats['rpn_patterns']} RPN patterns, "
            f"{len(self.knowledge_rules)} valid grammar rules (filtered from {len(knowledge_rules_raw)}, dropped {invalid_count} invalid), "
            f"{kb_symbols_added} symbols added to Math Galaxy"
        )
        self.evaluator = MathBenchmarkEvaluator()
        self.adapter = MathOutputAdapter()
        self.engine = ModularRPNEngine()
        self._use_trm_navigator = bool(use_trm_navigator)
        self._disable_retrieval = bool(disable_retrieval)
        self._shadow_readonly = bool(shadow_readonly)
        self._load_all_galaxies = bool(load_all_galaxies)
        self._enable_book_galaxies = bool(enable_book_galaxies)
        self._book_galaxy_root = book_galaxy_root
        self._book_max_books = int(max(0, book_max_books))
        self._book_top_k = int(max(0, book_top_k))
        self._thinking_budget = int(max(0, thinking_budget))
        self._verbose = bool(verbose)
        self._router_weights = str(router_weights) if router_weights else None
        self._role_patterns: List[Dict[str, Any]] = []
        self._theorem_patterns: List[Dict[str, Any]] = []
        self._theorem_rules: List[GrammarRule] = []
        self._theorem_rule_meta: Dict[str, Dict[str, Any]] = {}
        self._theorem_grammar_rules: Dict[str, GrammarRule] = {}
        self._theorem_router: Optional[TheoremRouter] = None
        self._trm_navigator = None
        self._recursive_solver = RecursiveSolver(verbose=self._verbose)
        self._log_galaxy = None
        self._ttc_best_source_counts: Dict[str, Dict[str, int]] = {}
        self._ttc_usage_stats: Dict[str, Dict[str, int]] = {}
        self._shadow_copy = None
        self._galaxy_reader = None
        if self._load_all_galaxies:
            artifact_dirs = ["/K3D/Knowledge3D.local/galaxies/books_v5_clean2"]
            role_patterns = populate_role_patterns(
                artifact_dirs=artifact_dirs,
                min_examples=3,
            )
            print(f"Loaded {len(role_patterns)} role patterns from books_v5_clean2")
            theorem_patterns = populate_theorem_patterns(
                artifact_dirs=artifact_dirs,
                min_examples=2,
            )
            print(f"Loaded {len(theorem_patterns)} theorem patterns from books_v5_clean2")
            print(f"Theorem patterns: {[p['pattern_id'] for p in theorem_patterns]}")
            self._role_patterns = role_patterns
            self._theorem_patterns = theorem_patterns
            self._theorem_rules, self._theorem_rule_meta = _build_theorem_rules(theorem_patterns)
            if self._theorem_rules:
                print(f"Built {len(self._theorem_rules)} theorem grammar rules")
            calculus_rules = get_calculus_rules()
            self._theorem_grammar_rules = {r.rule_id: r for r in calculus_rules}
            if self._theorem_grammar_rules:
                print(f"Loaded {len(self._theorem_grammar_rules)} calculus grammar rules")
            router_strategy = "learned" if self._router_weights else "heuristic"
            self._theorem_router = TheoremRouter(
                self._theorem_grammar_rules.keys(),
                strategy=router_strategy,
                learned_weights_path=self._router_weights,
            )
            if self._router_weights:
                print(f"Loaded theorem router weights from {self._router_weights}")

    def set_log_galaxy(self, log_galaxy) -> None:
        self._log_galaxy = log_galaxy
        if self._use_trm_navigator:
            from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
            from knowledge3d.training.math_benchmarks.trm_math_navigator import TRMMathNavigator
            from knowledge3d.training.math_benchmarks.trm_galaxy_reader import TRMGalaxyReader
            from knowledge3d.training.arc_agi.math_grammar_rules import GALAXY_AWARE_RULES

            drawing_galaxy = UNIFIED_GALAXY.galaxies.get("drawing")
            grammar_galaxy = UNIFIED_GALAXY.galaxies.get("grammar")
            if drawing_galaxy is not None and grammar_galaxy is not None:
                self._shadow_copy = DualShadowCopy(drawing_galaxy, grammar_galaxy, staged=True)

                local_dir = os.getenv("K3D_LOCAL_DIR", "/K3D/Knowledge3D.local")
                checkpoint_path = Path(local_dir) / "checkpoints" / "math_benchmarks" / "shadow_copy.json"
                if checkpoint_path.exists():
                    self._shadow_copy.load(checkpoint_path)

            galaxy_reader = None
            try:
                galaxy_reader = TRMGalaxyReader(
                    word_galaxy=UNIFIED_GALAXY.galaxies.get("word_galaxy"),
                    grammar_galaxy=UNIFIED_GALAXY.galaxies.get("grammar"),
                    math_galaxy=UNIFIED_GALAXY.galaxies.get("math_symbols"),
                    generic_equations_galaxy=UNIFIED_GALAXY.galaxies.get("generic_equations"),
                    rule_bank=list(GALAXY_AWARE_RULES) + list(self._theorem_rules),
                    shadow_copy=self._shadow_copy,
                    use_retrieval=not self._disable_retrieval,
                    reality_galaxy=UNIFIED_GALAXY.galaxies.get("reality"),
                    drawing_galaxy=UNIFIED_GALAXY.galaxies.get("drawing"),
                    enable_cross_domain=self._load_all_galaxies,
                    enable_book_galaxies=self._enable_book_galaxies,
                    book_galaxy_root=self._book_galaxy_root,
                    book_max_books=self._book_max_books,
                    book_top_k=self._book_top_k,
                    thinking_budget=self._thinking_budget,
                    verbose=self._verbose,
                    theorem_router=self._theorem_router,
                    theorem_rule_meta=self._theorem_rule_meta,
                    theorem_grammar_rules=self._theorem_grammar_rules,
                )
            except Exception:
                galaxy_reader = None
            self._galaxy_reader = galaxy_reader

            self._trm_navigator = TRMMathNavigator(
                rule_bank=(list(self.template_rules) + list(UNIFIED_GALAXY.get_grammar_rules())),
                math_galaxy=UNIFIED_GALAXY.galaxies.get("math_symbols"),
                rpn_engine=self.engine,
                shadow_copy=self._shadow_copy,
                galaxy_reader=galaxy_reader,
                # In benchmark mode we record ONLY when evaluator confirms correctness.
                record_on_confidence=False,
            )
        self.base = Path("/K3D/K3D_llama_cpp/datasets")
        self.solve_stats = {
            "trm": 0,
            "template": 0,
            "composer": 0,
            "word": 0,
            "grammar": 0,
            "knowledge": 0,
            "fail": 0,
            "grammar_attempts": 0,
            "knowledge_attempts": 0,
        }
        self._failures: List[Dict[str, Any]] = []
        self._failure_details: List[Dict[str, Any]] = []
        self._failure_rule_counts: Dict[str, int] = {}
        self._retrieval_events: List[Dict[str, Any]] = []
        # Phase 5 (analysis): bounded stdout logs for grep-driven triage.
        if self._verbose:
            self._analysis_log_limits = {"success": 200, "no_rule_match": 200, "exploration": 200}
        else:
            self._analysis_log_limits = {"success": 50, "no_rule_match": 50, "exploration": 50}

    def _extract_expected_num(self, expected: Any) -> Optional[float]:
        if isinstance(expected, dict) and "expected_num" in expected:
            try:
                return float(expected.get("expected_num")) if expected.get("expected_num") is not None else None
            except Exception:
                return None
        try:
            truth_str = str(expected)
        except Exception:
            return None
        try:
            m = re.search(r"####\s*([-+]?\d[\d,]*\.?\d*)", truth_str)
            if m:
                return float(str(m.group(1)).replace(",", ""))
        except Exception:
            pass
        try:
            nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", truth_str)
            if nums:
                return float(str(nums[-1]).replace(",", ""))
        except Exception:
            return None
        return None

    def _extract_got_num(self, got: Any) -> Optional[float]:
        if isinstance(got, dict) and "got_num" in got:
            try:
                return float(got.get("got_num")) if got.get("got_num") is not None else None
            except Exception:
                return None
        try:
            return float(got) if got is not None else None
        except Exception:
            return None

    def _get_failure_debug(self, failure: Dict[str, Any]) -> Dict[str, Any]:
        trace = failure.get("trace", {}) if isinstance(failure.get("trace", {}), dict) else {}
        meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
        comp = meta.get("read_composition", {}) if isinstance(meta.get("read_composition", {}), dict) else {}
        test_time = meta.get("test_time", {}) if isinstance(meta.get("test_time", {}), dict) else {}

        rpn = str(meta.get("rpn_program") or trace.get("rpn_program") or "")
        template_used = meta.get("template_used")
        if isinstance(comp.get("template_used"), str) and comp.get("template_used"):
            template_used = comp.get("template_used")

        # BookGalaxy debug (when enabled in TRMGalaxyReader TTC path).
        book_summary: Dict[str, Any] = {}
        try:
            read_trace = meta.get("read_trace", {}) if isinstance(meta.get("read_trace", {}), dict) else {}
            bg = read_trace.get("book_galaxy", {}) if isinstance(read_trace.get("book_galaxy", {}), dict) else {}
            hits = bg.get("hits", []) if isinstance(bg.get("hits", []), list) else []

            page_hits = [h for h in hits if isinstance(h, dict) and h.get("book_id") and h.get("page")]
            template_hits: List[Dict[str, Any]] = []
            artifact_hits: List[Dict[str, Any]] = []
            artifact_selection: List[Dict[str, Any]] = []
            template_selection: List[Dict[str, Any]] = []
            for h in hits:
                if not isinstance(h, dict):
                    continue
                if isinstance(h.get("templates"), list):
                    template_hits.extend([t for t in h.get("templates", []) if isinstance(t, dict)])
                if isinstance(h.get("artifacts"), list):
                    artifact_hits.extend([a for a in h.get("artifacts", []) if isinstance(a, dict)])
                if isinstance(h.get("artifact_selection"), list):
                    artifact_selection.extend([a for a in h.get("artifact_selection", []) if isinstance(a, dict)])
                if isinstance(h.get("template_selection"), list):
                    template_selection.extend([t for t in h.get("template_selection", []) if isinstance(t, dict)])

            top_books = []
            try:
                # rank by page hit score desc (best-effort)
                ranked = sorted(page_hits, key=lambda x: float(x.get("score") or 0.0), reverse=True)
                for ph in ranked[:3]:
                    top_books.append(
                        {
                            "book_id": ph.get("book_id"),
                            "page": ph.get("page"),
                            "score": ph.get("score"),
                            "domain": ph.get("domain"),
                            "title": ph.get("title"),
                        }
                    )
            except Exception:
                top_books = []

            top_artifacts = []
            try:
                ranked_a = sorted(artifact_hits, key=lambda x: float(x.get("score") or 0.0), reverse=True)
                for ah in ranked_a[:3]:
                    top_artifacts.append(
                        {
                            "book_id": ah.get("book_id"),
                            "page": ah.get("page"),
                            "score": ah.get("score"),
                            "type": ah.get("artifact_type"),
                            "name": str(ah.get("name") or "")[:120],
                            "n_conditions": len(ah.get("conditions") or []) if isinstance(ah.get("conditions"), list) else 0,
                        }
                    )
            except Exception:
                top_artifacts = []

            book_summary = {
                "page_hits": len(page_hits),
                "template_hits": len(template_hits),
                "artifact_hits": len(artifact_hits),
                "artifact_selection": artifact_selection[:6],
                "template_selection": template_selection[:6],
                "top_books": top_books,
                "top_artifacts": top_artifacts,
            }
        except Exception:
            book_summary = {}

        return {
            "rpn": rpn,
            "template_used": template_used,
            "patterns_used": comp.get("patterns_used"),
            "patterns_matched": comp.get("patterns_matched"),
            "best_source": test_time.get("best_source"),
            "book_seed_count": test_time.get("book_seed_count"),
            "book_seed_sample": test_time.get("book_seed_sample"),
            "book_sourced_seed_count": test_time.get("book_sourced_seed_count"),
            "book_sourced_seed_sample": test_time.get("book_sourced_seed_sample"),
            "book_sourced_candidates_evaluated": test_time.get("book_sourced_candidates_evaluated"),
            "book_sourced_eval": test_time.get("book_sourced_eval"),
            "rejected_by_reason": test_time.get("rejected_by_reason"),
            "candidates_evaluated": test_time.get("candidates_evaluated"),
            "plausible_candidates_seen": test_time.get("plausible_candidates_seen"),
            "book_summary": book_summary,
        }

    def _classify_wrong_computation_type(
        self,
        *,
        text: str,
        expected: Optional[float],
        got: Optional[float],
        rpn: str,
    ) -> str:
        low = (text or "").lower()
        expr = str(rpn or "")
        ops = {tok for tok in expr.split() if tok in {"+", "-", "*", "/"}}
        op_count = sum(1 for tok in expr.split() if tok in {"+", "-", "*", "/"})
        multi_step = sum(1 for cue in ("then", "after", "next", "finally", "and then") if cue in low)

        # Percent normalization / bounds.
        if "%" in low or "percent" in low:
            if any(w in low for w in ("percentage of", "as a percentage", "expressed as a percentage", "what percent")):
                if got is not None and (got < -1e-9 or got > 100.0 + 1e-6):
                    return "percent_out_of_bounds"
            if " 100 /" not in f" {expr} " and "/ 100" not in f" {expr} ":
                if "*" in ops and got is not None and expected is not None:
                    if abs(got) > abs(expected) * 5 and abs(got) > 1000:
                        return "percent_missing_divide_100"
            if got is not None and expected is not None:
                if abs(got - expected) > 1e-6 and abs(got) > 10 * max(1.0, abs(expected)):
                    return "percent_magnitude"

        # Multi-step incomplete chain (often missing operations).
        if multi_step > 0 and op_count < multi_step + 1:
            return "multi_step_incomplete"

        # Relative/comparative chains (very common GSM8K failure mode).
        if any(w in low for w in ("more than", "less than", "twice", "double", "triple", "half", "times as many", "as many")):
            return "relative_chain"

        # Pairs: "X pairs" implies ×2, which is commonly missed.
        if any(w in low for w in (" pair ", " pairs ")):
            if " 2 *" not in f" {expr} " and "2 *" not in expr:
                return "pairs_missing_x2"

        # Extraction-only: no operators at all.
        if op_count == 0:
            return "no_operation"

        # Intent/operator mismatch.
        if any(w in low for w in ("left", "remaining", "rest")) and "+" in ops and "-" not in ops:
            return "remaining_used_add"
        if any(w in low for w in ("total", "altogether", "in all", "combined")) and "-" in ops and "+" not in ops:
            return "total_used_subtract"
        if any(w in low for w in ("each", "per", "every")) and "+" in ops and "*" not in ops and "/" not in ops:
            return "rate_used_add_only"

        # Cost/unit problems: division often signals the wrong aggregation strategy.
        if "$" in text and any(w in low for w in ("cost", "costs", "price", "ticket")):
            if "/" in ops and "+" not in ops and "-" not in ops:
                return "cost_divide_instead_aggregate"
            if any(w in low for w in ("how much more", "difference", "more expensive")):
                if "+" in ops and "*" in ops and "-" not in ops:
                    return "cost_difference_missing_subtract"
            if any(w in low for w in ("adult", "child", "children")):
                # Often requires splitting counts and summing two products.
                return "cost_multi_group"

        # "Price per ..." problems typically require division; pure products/sums are suspicious.
        if ("price" in low or "cost" in low) and ("per" in low or "each" in low):
            if "/" not in ops:
                return "missing_division"

        # Missing-term signature: many numbers in text but few literals used in rpn.
        try:
            n_prompt_nums = len(re.findall(r"[-+]?\d*\.?\d+", low))
        except Exception:
            n_prompt_nums = 0
        n_rpn_nums = 0
        for tok in expr.split():
            if tok in {"+", "-", "*", "/"}:
                continue
            try:
                float(tok)
            except Exception:
                continue
            n_rpn_nums += 1
        if n_prompt_nums >= 4 and n_rpn_nums <= 2:
            return "missing_terms"

        # Generic magnitude mismatch.
        if got is not None and expected is not None and abs(expected) > 1e-9:
            ratio = abs(got / expected)
            if ratio >= 100 or ratio <= 0.01:
                return "wrong_magnitude"

        return "other"

    def _print_wrong_computation_report(
        self, *, failures: List[Dict[str, Any]], top_k: int = 3, examples_per: int = 3, max_examples_total: int = 20
    ) -> None:
        if not failures:
            return
        wrongs: List[Dict[str, Any]] = []
        for f in failures:
            text = str(f.get("text", "") or "")
            low = text.lower()
            solver = str(f.get("solver", "") or "")
            if solver in ("fail",):
                continue
            if any(w in low for w in ["step", "then", "after", "first", "second", "third", "finally"]):
                continue
            if len(low.split()) > 50:
                continue
            wrongs.append(f)
        if not wrongs:
            return

        families: Dict[str, List[Dict[str, Any]]] = {}
        for f in wrongs:
            text = str(f.get("text", "") or "")
            expected_num = f.get("expected_num") if f.get("expected_num") is not None else self._extract_expected_num(f.get("expected"))
            got_num = f.get("got_num") if f.get("got_num") is not None else self._extract_got_num(f.get("got"))
            dbg = self._get_failure_debug(f)
            bug = self._classify_wrong_computation_type(
                text=text,
                expected=expected_num,
                got=got_num,
                rpn=str(dbg.get("rpn") or ""),
            )
            families.setdefault(bug, []).append({**f, "_expected_num": expected_num, "_got_num": got_num, "_dbg": dbg})

        ordered = sorted(families.items(), key=lambda kv: len(kv[1]), reverse=True)[: max(1, int(top_k))]
        print("\n=== WRONG_COMPUTATION ANALYSIS (verbose) ===")
        print(f"  total_wrong_computation: {len(wrongs)}")
        printed = 0
        for idx, (bug, items) in enumerate(ordered, start=1):
            print(f"\n  [{idx}] Category: {bug} (failures: {len(items)})")
            for ex in items[: max(1, int(examples_per))]:
                if printed >= int(max_examples_total):
                    break
                dbg = ex.get("_dbg", {}) if isinstance(ex.get("_dbg", {}), dict) else {}
                print(f"    - text: {str(ex.get('text',''))[:160].replace(chr(10), ' ')}")
                print(f"      template: {dbg.get('template_used')}")
                print(f"      rpn: {str(dbg.get('rpn') or '')[:160]}")
                print(f"      expected: {ex.get('_expected_num')} got: {ex.get('_got_num')}")
                if dbg.get("template_used") in {"test_time_compute", "test_time_compute_fallback"}:
                    print(
                        "      source:"
                        f" best_source={dbg.get('best_source')}"
                        f" book_seed_count={dbg.get('book_seed_count')}"
                    )
                    if dbg.get("book_sourced_seed_count"):
                        print(
                            "      book_sourced:"
                            f" seeds={dbg.get('book_sourced_seed_count')}"
                            f" evaluated={dbg.get('book_sourced_candidates_evaluated')}"
                        )
                        ev = dbg.get("book_sourced_eval")
                        if isinstance(ev, list) and ev:
                            for e in ev[:3]:
                                if not isinstance(e, dict):
                                    continue
                                print(
                                    "        - book_eval:"
                                    f" plausible={e.get('plausible')}"
                                    f" reason={e.get('reason')}"
                                    f" result={e.get('result')}"
                                    f" conf={e.get('confidence')}"
                                    f" rpn={str(e.get('rpn') or '')[:60]}"
                                )
                bs = dbg.get("book_summary", {}) if isinstance(dbg.get("book_summary", {}), dict) else {}
                if bs.get("page_hits") or bs.get("template_hits") or bs.get("artifact_hits"):
                    print(
                        "      book_hits:"
                        f" pages={bs.get('page_hits', 0)}"
                        f" templates={bs.get('template_hits', 0)}"
                        f" artifacts={bs.get('artifact_hits', 0)}"
                    )
                    sel = bs.get("artifact_selection", [])
                    if isinstance(sel, list) and sel:
                        for s in sel[:3]:
                            if not isinstance(s, dict):
                                continue
                            print(
                                "        - selected_artifact:"
                                f" {s.get('artifact_type')} score_internal={s.get('score_internal')}"
                                f" book={s.get('book_id')} page={s.get('page')}"
                                f" conds={len(s.get('conditions') or [])}"
                                f" name={str(s.get('name') or '')[:80]}"
                            )
                            er = s.get("emitted_rpn", [])
                            if isinstance(er, list) and er:
                                print(f"          emitted_rpn: {str(er[0])[:140]}")
                    tops = bs.get("top_artifacts", [])
                    if isinstance(tops, list) and tops:
                        for t in tops[:3]:
                            if not isinstance(t, dict):
                                continue
                            print(
                                "        - artifact:"
                                f" {t.get('type')} score={t.get('score')}"
                                f" book={t.get('book_id')} page={t.get('page')}"
                                f" conds={t.get('n_conditions')}"
                                f" name={t.get('name')}"
                            )
                rej = dbg.get("rejected_by_reason")
                if isinstance(rej, dict) and rej:
                    top_rej = sorted(rej.items(), key=lambda kv: kv[1], reverse=True)[:3]
                    print(f"      rejected_by_reason(top3): {top_rej}")
                if dbg.get("candidates_evaluated") is not None:
                    print(f"      ttc_candidates: {dbg.get('candidates_evaluated')} plausible_seen: {dbg.get('plausible_candidates_seen')}")
                printed += 1
            if printed >= int(max_examples_total):
                break

    def _print_non_wrong_failure_samples(
        self,
        *,
        failures: List[Dict[str, Any]],
        max_per_category: int = 3,
    ) -> None:
        if not failures:
            return

        buckets: Dict[str, List[Dict[str, Any]]] = {
            "no_rule_match": [],
            "multi_step_needed": [],
            "word_problem": [],
        }
        for f in failures:
            solver = str(f.get("solver", "") or "")
            text = str(f.get("text", "") or "")
            low = text.lower()
            if solver in ("fail",):
                buckets["no_rule_match"].append(f)
                continue
            if any(w in low for w in ["step", "then", "after", "first", "second", "third", "finally"]):
                buckets["multi_step_needed"].append(f)
                continue
            if len(low.split()) > 50:
                buckets["word_problem"].append(f)
                continue

        if not any(buckets.values()):
            return

        print("\n=== FAILURE SAMPLES (verbose) ===")
        for cat in ("no_rule_match", "multi_step_needed", "word_problem"):
            items = buckets.get(cat, [])
            if not items:
                continue
            print(f"  {cat}: {len(items)}")
            for ex in items[: max(1, int(max_per_category))]:
                dbg = self._get_failure_debug(ex)
                expected_num = ex.get("expected_num") if ex.get("expected_num") is not None else self._extract_expected_num(ex.get("expected"))
                got_num = ex.get("got_num") if ex.get("got_num") is not None else self._extract_got_num(ex.get("got"))
                print(f"    - text: {str(ex.get('text',''))[:160].replace(chr(10), ' ')}")
                print(f"      expected: {expected_num} got: {got_num}")
                if dbg.get("template_used") or dbg.get("rpn"):
                    print(f"      template: {dbg.get('template_used')} rpn: {str(dbg.get('rpn') or '')[:120]}")

    def load_dataset(self, name: str) -> List[Dict[str, Any]]:
        """Load benchmark dataset. Skips gracefully when files are missing."""
        problems: List[Dict[str, Any]] = []

        if name == "gsm8k":
            path = self.base / "GSM8K/grade_school_math/data/train.jsonl"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        p = json.loads(line)
                        p["source"] = "gsm8k"
                        problems.append(p)
            return problems

        if name == "math":
            path = self.base / "math/data/train.jsonl"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        p = json.loads(line)
                        p["source"] = "math"
                        text = p.get("problem", p.get("question", ""))
                        if isinstance(text, str) and text:
                            p["normalized_problem"] = normalize_latex_to_natural(text)
                        problems.append(p)
            return problems

        if name == "omni_math":
            path = self.base / "Omni-MATH/Omni-Math.jsonl"
            if path.exists():
                with open(path) as f:
                    for line in f:
                        p = json.loads(line)
                        p["source"] = "omni_math"
                        problems.append(p)
            return problems

        if name == "amc_aime":
            amc_path = self.base / "AMC-AIME/data"
            for jsonl in amc_path.glob("*.jsonl") if amc_path.exists() else []:
                with open(jsonl) as f:
                    for line in f:
                        p = json.loads(line)
                        p["source"] = "amc_aime"
                        problems.append(p)
            return problems

        if name == "mmlu":
            mmlu_path = self.base / "MMLU/data/test"
            for csv_file in mmlu_path.glob("*.csv") if mmlu_path.exists() else []:
                import csv

                with open(csv_file) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 5:
                            problems.append(
                                {
                                    "question": row[0],
                                    "choices": row[1:5],
                                    "answer": row[5] if len(row) > 5 else "A",
                                    "source": "mmlu",
                                }
                            )
            return problems

        return problems

    def solve_problem_with_meta(self, problem: Dict[str, Any]) -> Tuple[Any, str, Dict[str, Any]]:
        """Solve a problem and return (result, solver_name, trace)."""
        text = problem.get("normalized_problem", problem.get("normalized_question"))
        if not text:
            text = problem.get("problem", problem.get("question", ""))
        source = problem.get("source", "")
        solution = problem.get("answer", problem.get("solution", ""))

        # MMLU: multiple-choice → return A/B/C/D
        if source == "mmlu":
            result = self._solve_mmlu(problem)
            return result, "mmlu", {"rule_used": None, "rpn_program": "", "source": source}

        # Try -1: TRM navigator (rule routing + execution)
        if self._trm_navigator is not None:
            try:
                result, meta = self._trm_navigator.solve(text)
                trace = {"rule_used": meta.get("rule_used"), "rpn_program": meta.get("rpn_program", ""), "meta": meta}
                # Extra guardrails for known over-broad arithmetic triggers inside longer word problems.
                rule_used = str(meta.get("rule_used") or "")
                if rule_used in {"gsm_plus", "gsm_a_times_b", "gsm_divided_by"}:
                    nums = re.findall(r"\d+\.?\d*", text)
                    alpha_words = re.findall(r"[A-Za-z]+", text)
                    if len(nums) > 2 and len(alpha_words) > 8:
                        result = None
                if result is not None and self._validate_answer(result, text, source):
                    return result, "trm", trace
            except Exception:
                pass

        # Try -0.5: Recursive Solver (Compositional Decomposition)
        # This handles complex calculus forms that single regexes miss ((3x-4)/(2x+3)).
        if self._recursive_solver:
            rec_result = self._recursive_solver.solve(text)
            if rec_result is not None:
                if self._validate_answer(rec_result, text, source):
                    return rec_result, "recursive", {"rule_used": "compositional_solver"}

        # Try 0: Curated parametric templates (fast, high-quality)
        for rule in self.template_rules:
            result, template_trace = self._apply_template_with_trace(rule, text)
            if result is not None and template_trace is not None:
                if self._validate_answer(result, text, source):
                    return result, "template", template_trace

        # Try 1: Composite matcher (contextual extraction + multi-match composition)
        composite_result = self._try_composite_match(text, source)
        if composite_result is not None:
            value, trace = composite_result
            if self._validate_answer(value, text, source):
                return value, "word", trace

        # Try 1b: Core grammar galaxy rules (word/competition/calculus/etc.)
        grammar_attempt = self._try_grammar_rules(text, source)
        if grammar_attempt is not None:
            value, trace = grammar_attempt
            if self._validate_answer(value, text, source):
                return value, "grammar", trace

        # Try 2: Knowledge-derived grammar rules (scored, top-N, domain-filtered)
        knowledge_attempt = self._try_knowledge_rules(text, source)
        if knowledge_attempt is not None:
            value, trace = knowledge_attempt
            if self._validate_answer(value, text, source):
                return value, "knowledge", trace

        # Try 1.5: Galaxy composer (safe-gated for GSM8K)
        if self._looks_like_expression(text, source):
            rpn_str = self.composer.compose(text)
            if rpn_str and rpn_str.strip():
                try:
                    result = self.engine.evaluate(rpn_str)
                    if result is not None and self._validate_answer(result, text, source):
                        return result, "composer", {"rule_used": "composer", "rpn_program": rpn_str}
                except Exception:
                    pass

        # Try 2: Word problem solver for natural language (last resort; can be over-broad)
        word_result = self.word_solver.solve(text)
        if isinstance(word_result, dict):
            rpn_program = word_result.get("rpn_program", "")
            if rpn_program:
                try:
                    result = self.engine.evaluate(rpn_program)
                    if result is not None and self._validate_answer(result, text, source):
                        return (
                            result,
                            "word",
                            {"rule_used": "word_solver", "rpn_program": rpn_program, "matched_rules": word_result.get("matched_rules", [])},
                        )
                except Exception:
                    pass

        return None, "fail", {"rule_used": None, "rpn_program": ""}

    def _try_grammar_rules(self, text: str, source: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        grammar_matches = []
        for rule in UNIFIED_GALAXY.get_grammar_rules():
            try:
                import re as _re

                m = _re.search(rule.pattern, text, _re.IGNORECASE | _re.DOTALL)
                if m:
                    score = self._score_rule_match(rule, m, source)
                    grammar_matches.append((score, rule, m))
            except Exception:
                continue
        if not grammar_matches:
            return None
        self.solve_stats["grammar_attempts"] += len(grammar_matches)
        for _, rule, m in sorted(grammar_matches, key=lambda x: x[0], reverse=True)[:50]:
            try:
                rpn_program = rule.rpn_program
                if callable(rpn_program):
                    rpn_program = rpn_program(m)
                else:
                    for idx, group in enumerate(m.groups()):
                        rpn_program = rpn_program.replace(f"{{g{idx}}}", group)
                        rpn_program = rpn_program.replace(f"{{{idx}}}", group)
                if not is_valid_rpn(rpn_program):
                    continue
                if not rpn_program:
                    continue
                result = self.engine.evaluate(rpn_program)
                if result is None:
                    continue
                return (
                    float(result),
                    {
                        "rule_used": getattr(rule, "rule_id", None),
                        "pattern": getattr(rule, "pattern", ""),
                        "groups": list(m.groups()),
                        "rpn_program": rpn_program,
                    },
                )
            except Exception:
                continue
        return None

    def _try_knowledge_rules(self, text: str, source: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        # Knowledge-derived rules are very broad; on GSM8K-style word problems they
        # frequently match irrelevant number spans and cause wrong_computation.
        if source == "gsm8k" and not self._looks_like_expression(text, source):
            return None

        def _looks_like_math_fragment(fragment: str) -> bool:
            if source == "gsm8k":
                if "\\" in fragment:
                    return True
                return any(op in fragment for op in ("+", "-", "*", "/", "^", "="))
            # Non-GSM datasets can have wordy fragments (e.g., "derivative of ...").
            return len(fragment.strip()) >= 18

        scored_matches = []
        for comp, rule in self._iter_knowledge_rules(source):
            try:
                m = comp.search(text)
                if m:
                    score = self._score_rule_match(rule, m, source)
                    scored_matches.append((score, rule, m))
            except Exception:
                continue
        if not scored_matches:
            for comp, rule in self.knowledge_rules_fallback_compiled:
                try:
                    m = comp.search(text)
                    if m:
                        score = self._score_rule_match(rule, m, source)
                        scored_matches.append((score, rule, m))
                except Exception:
                    continue
        if not scored_matches:
            for comp, rule in self.knowledge_rules_len_compiled:
                try:
                    m = comp.search(text)
                    if m:
                        score = self._score_rule_match(rule, m, source)
                        scored_matches.append((score, rule, m))
                except Exception:
                    continue
        if not scored_matches:
            return None
        self.solve_stats["knowledge_attempts"] += len(scored_matches)
        top_matches = sorted(scored_matches, key=lambda x: x[0], reverse=True)[:80]
        best_score = top_matches[0][0] if top_matches else 0.0
        # Require the candidate to be near the top score (reduces random low-score hits).
        rel_floor = 0.85 if source == "gsm8k" else 0.75
        min_score = max(1.0, best_score * rel_floor)
        for score, rule, m in top_matches:
            if score < min_score:
                continue
            try:
                span = m.group(0) or ""
                if len(span) < 8:
                    continue
                if not _looks_like_math_fragment(span):
                    continue
                rpn_program = rule.rpn_program
                if callable(rpn_program):
                    rpn_program = rpn_program(m)
                else:
                    for idx, group in enumerate(m.groups()):
                        rpn_program = rpn_program.replace(f"{{g{idx}}}", group)
                        rpn_program = rpn_program.replace(f"{{{idx}}}", group)
                if not is_valid_rpn(rpn_program):
                    continue
                result = self.engine.evaluate(rpn_program)
                if result is None:
                    continue
                return (
                    float(result),
                    {
                        "rule_used": getattr(rule, "rule_id", None),
                        "pattern": getattr(rule, "pattern", ""),
                        "groups": list(m.groups()),
                        "rpn_program": rpn_program,
                    },
                )
            except Exception:
                continue
        return None

    def solve_problem(self, problem: Dict[str, Any]) -> Any:
        """Compatibility wrapper returning only the predicted value."""
        result, _, _ = self.solve_problem_with_meta(problem)
        return result

    def _apply_template(self, rule, text: str):
        """Apply a parametric template rule to text."""
        import re as _re
        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        try:
            m = _re.search(rule.pattern, text, _re.IGNORECASE | _re.DOTALL)

            # If no direct match, try after normalizing number words
            if not m:
                normalized = normalize_number_words(text)
                m = _re.search(rule.pattern, normalized, _re.IGNORECASE | _re.DOTALL)
            if not m:
                return None
            if not self._validate_captures(rule, m, text):
                return None
            rpn_program = rule.rpn_program
            for idx, group in enumerate(m.groups()):
                clean = str(group).replace(",", "").strip()
                rpn_program = rpn_program.replace(f"{{{idx}}}", clean)
            if not is_valid_rpn(rpn_program):
                return None
            result = self.engine.evaluate(rpn_program)
            if result is None:
                return None
            if not self._sanity_check_answer(result, text):
                return None
            return result
        except Exception:
            return None

    def _apply_template_with_trace(self, rule, text: str) -> Tuple[Any, Dict[str, Any] | None]:
        """Apply a template rule and return (result, trace) for diagnostics."""
        import re as _re
        from knowledge3d.training.math_benchmarks.number_words import normalize_number_words

        try:
            m = _re.search(rule.pattern, text, _re.IGNORECASE | _re.DOTALL)
            if not m:
                normalized = normalize_number_words(text)
                m = _re.search(rule.pattern, normalized, _re.IGNORECASE | _re.DOTALL)
            if not m:
                return None, None
            if not self._validate_captures(rule, m, text):
                return None, None
            rpn_program = rule.rpn_program
            for idx, group in enumerate(m.groups()):
                clean = str(group).replace(",", "").strip()
                rpn_program = rpn_program.replace(f"{{{idx}}}", clean)
            if not is_valid_rpn(rpn_program):
                return None, None
            result = self.engine.evaluate(rpn_program)
            if result is None:
                return None, None
            if not self._sanity_check_answer(result, text):
                return None, None
            return (
                result,
                {
                    "rule_used": getattr(rule, "rule_id", None),
                    "pattern": getattr(rule, "pattern", ""),
                    "groups": list(m.groups()),
                    "rpn_program": rpn_program,
                },
            )
        except Exception:
            return None, None

    def _validate_captures(self, rule, match, text: str) -> bool:
        """Lightweight semantic validation of captured numbers."""
        if len(match.groups()) < 2:
            return True
        rule_id = getattr(rule, "rule_id", "")
        captures = match.groups()
        text_lower = text.lower()
        # Reject over-broad arithmetic matches when the prompt contains extra numbers.
        if rule_id in {"gsm_plus", "gsm_a_times_b", "gsm_divided_by"}:
            try:
                nums = re.findall(r"\d+\.?\d*", text)
                alpha_words = re.findall(r"[A-Za-z]+", text)
                if len(nums) > 2 and len(alpha_words) > 8:
                    return False
            except Exception:
                return False
        # For subtraction-like rules, ensure first capture appears before subtraction cue.
        if any(key in rule_id for key in ["sub", "minus", "discount", "spent", "loss", "used", "lost"]):
            sub_keywords = ["minus", "subtract", "less", "spent", "ate", "gave", "lost", "uses", "used", "took", "gave away"]
            for kw in sub_keywords:
                if kw in text_lower:
                    kw_pos = text_lower.find(kw)
                    cap0_pos = text_lower.find(str(captures[0]).lower())
                    if cap0_pos > kw_pos >= 0:
                        return False
        return True

    def _sanity_check_answer(self, result: float, text: str) -> bool:
        """Reject obviously unreasonable answers to reduce bad template hits."""
        import re as _re

        try:
            nums = [float(n.replace(",", "")) for n in _re.findall(r"\d+\.?\d*", text)]
        except Exception:
            nums = []
        if not nums:
            return True
        max_num = max(nums)
        min_pos = min((n for n in nums if n > 0), default=0.0)
        if result < 0 and "negative" not in text.lower():
            return False
        if result > max_num * 1000:
            return False
        if min_pos > 0 and result < min_pos / 1000:
            return False
        return True

    def _score_rule_match(self, rule, match, source: str) -> float:
        """Lightweight scoring for knowledge rules to reduce noise."""
        specificity = len(rule.pattern)
        captures = len(match.groups())
        domain = getattr(rule, "domain", "")
        domain_boost = 0.0
        if source == "gsm8k" and "arithmetic" in domain:
            domain_boost += 1.0
        if source in {"math", "omni_math", "amc_aime"} and any(k in domain for k in ["calculus", "algebra", "sequences"]):
            domain_boost += 1.0
        if source == "mmlu" and "finance" in domain:
            domain_boost += 0.5
        return specificity * 0.01 + captures * 0.5 + domain_boost

    def _iter_knowledge_rules(self, source: str):
        """Yield compiled knowledge rules filtered by dataset domain."""
        domain_map = {
            "gsm8k": {"math_arithmetic", "math_finance", "math_geometry", "math_sequences", "math_kb"},
            "math": {"math_calculus", "math_algebra", "math_geometry", "math_sequences", "math_arithmetic", "math_kb"},
            "omni_math": {"math_calculus", "math_algebra", "math_geometry", "math_sequences", "math_arithmetic", "math_kb"},
            "amc_aime": {"math_calculus", "math_algebra", "math_geometry", "math_sequences", "math_number_theory", "math_kb"},
            "mmlu": {"math_finance", "math_statistics", "math_arithmetic", "math_kb"},
        }
        allowed = domain_map.get(source, None)
        if allowed:
            # Always include arithmetic/finance/geometry as shared basics
            allowed = set(allowed) | {"math_arithmetic", "math_finance", "math_geometry"}
            for domain in allowed:
                for comp_rule in self.knowledge_rules_by_domain.get(domain, []):
                    yield comp_rule
        else:
            # Fallback: limited prioritized set to avoid CPU exhaustion
            import re as _re
            for rule in self.knowledge_rules_fallback:
                yield (_re.compile(rule.pattern), rule)

    def _solve_mmlu(self, problem: Dict[str, Any]) -> str:
        """
        Solve MMLU multiple choice by evaluating each choice.

        Strategy:
        1. Try to compute numeric answer from the question
        2. Match computed answer against choices
        3. Fallback to default 'A'
        """
        question = problem.get("question", "")
        choices = problem.get("choices", [])
        if not choices:
            return "A"

        computed_answer = None

        # Word solver attempt
        word_result = self.word_solver.solve(question)
        if isinstance(word_result, dict):
            rpn = word_result.get("rpn_program", "")
            if rpn:
                try:
                    computed_answer = self.engine.evaluate(rpn)
                except Exception:
                    computed_answer = None

        # Composer attempt (LaTeX/infix)
        if computed_answer is None and "\\" in question:
            rpn_str = self.composer.compose(question)
            if rpn_str and rpn_str.strip():
                try:
                    computed_answer = self.engine.evaluate(rpn_str)
                except Exception:
                    computed_answer = None

        if computed_answer is not None:
            for i, choice in enumerate(choices):
                try:
                    choice_val = float(str(choice).strip())
                    if abs(choice_val - float(computed_answer)) < 1e-6:
                        return chr(65 + i)
                except (ValueError, TypeError):
                    pass

                nums = re.findall(r"[-+]?\d*\.?\d+", str(choice))
                if nums:
                    try:
                        choice_val = float(nums[-1])
                        if abs(choice_val - float(computed_answer)) < 1e-6:
                            return chr(65 + i)
                    except (ValueError, TypeError):
                        pass

        return "A"

    def run_benchmark(
        self,
        dataset_name: str,
        limit: int | None = None,
        *,
        start_index: int = 0,
        shuffle: bool = False,
        shuffle_seed: int = 0,
    ) -> Dict[str, Any]:
        """Run benchmark on a dataset."""
        problems = self.load_dataset(dataset_name)
        failures_start = len(self._failures)
        if shuffle and problems:
            rng = random.Random(int(shuffle_seed))
            rng.shuffle(problems)

        start = int(start_index or 0)
        if start < 0:
            start = 0
        if start:
            problems = problems[start:]

        if limit:
            problems = problems[:limit]

        if not problems:
            print(f"\nRunning {dataset_name}: 0 problems (dataset missing or empty)")
            return {"correct": 0, "total": 0, "accuracy": 0.0, "error": "missing_dataset"}

        label = dataset_name
        if start or shuffle:
            label = f"{dataset_name} (start={start} shuffle={bool(shuffle)} seed={int(shuffle_seed)})"
        print(f"\nRunning {label}: {len(problems)} problems")

        by_solver: Dict[str, int] = {k: 0 for k in ("trm", "template", "composer", "word", "grammar", "knowledge", "fail")}
        rule_hist: Dict[str, int] = {}
        correct = 0
        total = 0

        for i, problem in enumerate(problems):
            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1}/{len(problems)} ({100 * correct / max(1, total):.1f}% so far)")

            text = str(problem.get("problem", problem.get("question", "")))
            predicted, solver, trace = self.solve_problem_with_meta(problem)
            by_solver[solver] = by_solver.get(solver, 0) + 1
            if solver in self.solve_stats:
                self.solve_stats[solver] += 1
            rule_key = None
            rule_used = trace.get("rule_used")
            meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
            if solver == "trm":
                template_used = meta.get("template_used")
                if template_used == "theorem_router":
                    attempts = meta.get("theorem_attempts", [])
                    if isinstance(attempts, list) and attempts:
                        rule_key = attempts[0].get("rule_id") or rule_used
                if rule_key is None:
                    rule_key = rule_used or template_used
            else:
                rule_key = rule_used or trace.get("template_used")
            if isinstance(rule_key, str) and rule_key:
                rule_hist[rule_key] = int(rule_hist.get(rule_key, 0)) + 1
            ground_truth = problem.get("answer", problem.get("solution", ""))

            result = self.evaluator.evaluate(
                problem_id=str(i),
                predicted=predicted,
                ground_truth=ground_truth,
                source=dataset_name,
            )

            # Test-time compute attribution (Phase 7): track whether TTC solutions
            # came from book-seeded programs or generic candidate families.
            try:
                if solver == "trm":
                    meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
                    test_time = meta.get("test_time", {}) if isinstance(meta.get("test_time", {}), dict) else {}
                    best_source = test_time.get("best_source")
                    if isinstance(best_source, str) and best_source:
                        bucket = self._ttc_best_source_counts.setdefault(dataset_name, {})
                        bucket[best_source] = int(bucket.get(best_source, 0)) + 1
                        stats = self._ttc_usage_stats.setdefault(dataset_name, {})
                        stats["ttc_calls"] = int(stats.get("ttc_calls", 0)) + 1
                        try:
                            bsc = int(test_time.get("book_seed_count") or 0)
                        except Exception:
                            bsc = 0
                        try:
                            bssc = int(test_time.get("book_sourced_seed_count") or 0)
                        except Exception:
                            bssc = 0
                        if bsc > 0:
                            stats["with_book_seed"] = int(stats.get("with_book_seed", 0)) + 1
                        if bssc > 0:
                            stats["with_book_sourced_seed"] = int(stats.get("with_book_sourced_seed", 0)) + 1
            except Exception:
                pass

            # Retrieval diagnostics: record when TRM used a retrieved template.
            try:
                if solver == "trm" and trace.get("rule_used") == "galaxy_read":
                    meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
                    comp = meta.get("read_composition", {}) if isinstance(meta.get("read_composition", {}), dict) else {}
                    if comp.get("template_selected_by") == "retrieval":
                        # Extract numeric expected answer for GSM8K-style solutions.
                        expected_num = self._extract_expected_num(ground_truth)
                        self._retrieval_events.append(
                            {
                                "problem_text": str(text)[:220],
                                "expected": str(ground_truth)[:80],
                                "expected_num": expected_num,
                                "got": str(predicted)[:80],
                                "got_num": float(predicted) if predicted is not None else None,
                                "correct": bool(result.get("correct")),
                                "retrieval_score": comp.get("retrieval_score"),
                                "retrieved_template": comp.get("retrieved_template"),
                                "heuristic_template": comp.get("heuristic_template"),
                                "template_used": comp.get("template_used"),
                                "patterns_used": comp.get("patterns_used"),
                                "patterns_matched": comp.get("patterns_matched"),
                                "rpn": str(trace.get("rpn_program") or "")[:160],
                            }
                        )
            except Exception:
                pass

            total += 1
            if result["correct"]:
                correct += 1
                if self._analysis_log_limits.get("success", 0) > 0:
                    self._analysis_log_limits["success"] = int(self._analysis_log_limits.get("success", 0)) - 1
                    rule_used = trace.get("rule_used")
                    meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
                    comp = meta.get("read_composition", {}) if isinstance(meta.get("read_composition", {}), dict) else {}
                    template_used = comp.get("template_used")
                    patterns_used = comp.get("patterns_used")
                    # Extract numeric expected for GSM8K-style.
                    expected_num = self._extract_expected_num(ground_truth)
                    got_num = None
                    try:
                        got_num = float(predicted) if predicted is not None else None
                    except Exception:
                        got_num = None
                    print(
                        "[SUCCESS]"
                        f" ds={dataset_name}"
                        f" solver={solver}"
                        f" rule={rule_used}"
                        f" template={template_used}"
                        f" expected={expected_num}"
                        f" got={got_num}"
                        f" patterns={patterns_used}"
                        f" text={text[:120].replace(chr(10), ' ')}"
                    )
                self._record_correct_solve(problem, predicted, solver, trace, dataset_name)
                if solver == "recursive" and self._log_galaxy is not None and self._recursive_solver is not None:
                    trace_info = self._recursive_solver.get_last_trace()
                    self._log_galaxy.add_trace(
                        problem_text=text,
                        step_sequence=trace_info.get("step_sequence", []),
                        result=float(predicted) if predicted is not None else None,
                        success=True,
                        trace_lines=trace_info.get("trace_lines", []),
                        metadata={
                            "dataset": dataset_name,
                            "expression": trace_info.get("expression"),
                            "point": trace_info.get("point"),
                        },
                    )
            else:
                expected_num = None
                got_num = None
                try:
                    expected_num = self._extract_expected_num(ground_truth)
                except Exception:
                    expected_num = None
                try:
                    got_num = self._extract_got_num(predicted)
                except Exception:
                    got_num = None
                self._failures.append(
                    {
                        "text": str(text)[:300],
                        "expected": str(ground_truth)[:120],
                        "expected_num": expected_num,
                        "got": str(predicted)[:120],
                        "got_num": got_num,
                        "source": dataset_name,
                        "solver": solver,
                        "trm_tried": self._trm_navigator is not None,
                        "trace": trace,
                    }
                )
                self._log_failure_detail(problem, predicted, ground_truth, solver, trace)
                if solver == "fail" and self._analysis_log_limits.get("no_rule_match", 0) > 0:
                    self._analysis_log_limits["no_rule_match"] = int(self._analysis_log_limits.get("no_rule_match", 0)) - 1
                    nums = []
                    try:
                        nums = re.findall(r"[-+]?\d*\.?\d+", text)[:6]
                    except Exception:
                        nums = []
                    keywords = []
                    low = text.lower()
                    for k in ("per", "each", "every", "total", "altogether", "sum", "left", "remaining", "more", "less", "times", "twice", "divided", "split", "share"):
                        if k in low:
                            keywords.append(k)
                    print(
                        "[NO_RULE_MATCH]"
                        f" ds={dataset_name}"
                        f" nums={nums}"
                        f" kw={keywords[:8]}"
                        f" text={text[:180].replace(chr(10), ' ')}"
                    )

            # Phase 5B: print exploration trace for Galaxy reads (bounded).
            try:
                if (
                    solver == "trm"
                    and trace.get("rule_used") == "galaxy_read"
                    and self._analysis_log_limits.get("exploration", 0) > 0
                ):
                    self._analysis_log_limits["exploration"] = int(self._analysis_log_limits.get("exploration", 0)) - 1
                    meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
                    expl = meta.get("exploration", {}) if isinstance(meta.get("exploration", {}), dict) else {}
                    ts = expl.get("tsinghua", {}) if isinstance(expl.get("tsinghua", {}), dict) else {}
                    hubs = ts.get("hub_concepts", [])
                    checks = ts.get("rule_checks", None)
                    selected = ts.get("selected_rule_ids", [])
                    print(
                        "[EXPLORATION]"
                        f" ds={dataset_name}"
                        f" correct={bool(result.get('correct'))}"
                        f" concepts={expl.get('concepts')}"
                        f" hubs={hubs}"
                        f" checks={checks}"
                        f" selected={selected[:8]}"
                    )
            except Exception:
                pass

            # Phase 5B: annotate exploration entry with benchmark ground-truth.
            try:
                if solver == "trm" and trace.get("rule_used") == "galaxy_read" and self._shadow_copy is not None:
                    expected_num = self._extract_expected_num(ground_truth)
                    got_num = None
                    try:
                        got_num = float(predicted) if predicted is not None else None
                    except Exception:
                        got_num = None
                    self._shadow_copy.annotate_exploration_eval(
                        problem_text=text,
                        expected_num=expected_num,
                        got_num=got_num,
                        correct=bool(result.get("correct")),
                    )
            except Exception:
                pass

        accuracy = correct / total if total > 0 else 0.0
        print(f"Dataset: {dataset_name}")
        print(f"  Total: {total}")
        print(f"  Correct: {correct} ({100 * accuracy:.2f}%)")
        print(f"  Accuracy: {correct}/{total} = {100 * accuracy:.2f}%")
        print("  By solver:")
        for key in ("trm", "template", "composer", "word", "grammar", "knowledge", "fail"):
            if key in by_solver:
                print(f"    - {key}: {by_solver[key]}")
        if self._galaxy_reader is not None:
            try:
                stats = self._galaxy_reader.get_stats()
                print(f"  TRM reader retrieval hits: {stats.get('retrieval_hits', 0)}")
                if "retrieval_used" in stats:
                    print(f"  TRM reader retrieval used: {stats.get('retrieval_used', 0)}")
                print(f"  TRM reader template counts: {stats.get('template_counts', {})}")
            except Exception:
                pass
        if dataset_name in self._ttc_best_source_counts:
            counts = dict(self._ttc_best_source_counts.get(dataset_name, {}))
            if counts:
                print(f"  TTC best_source: {counts}")
        if dataset_name in self._ttc_usage_stats:
            stats = dict(self._ttc_usage_stats.get(dataset_name, {}))
            if stats:
                print(f"  TTC usage: {stats}")
        if rule_hist:
            total_rules = sum(rule_hist.values())
            entropy = 0.0
            for count in rule_hist.values():
                if count <= 0:
                    continue
                p = float(count) / float(total_rules)
                entropy -= p * math.log2(p)
            max_entropy = math.log2(len(rule_hist)) if len(rule_hist) > 1 else 0.0
            top_rules = sorted(rule_hist.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print(f"  Rule selection entropy: {entropy:.3f} (max {max_entropy:.3f})")
            print(f"  Top rules: {top_rules}")
        if self._retrieval_events:
            total_r = len(self._retrieval_events)
            ok_r = sum(1 for e in self._retrieval_events if e.get("correct"))
            diff_r = sum(1 for e in self._retrieval_events if e.get("retrieved_template") and e.get("heuristic_template") and e.get("retrieved_template") != e.get("heuristic_template"))
            diff_ok = sum(
                1
                for e in self._retrieval_events
                if e.get("correct")
                and e.get("retrieved_template")
                and e.get("heuristic_template")
                and e.get("retrieved_template") != e.get("heuristic_template")
            )
            print(f"  Retrieval used: {total_r} (correct: {ok_r}, wrong: {total_r - ok_r})")
            print(f"  Retrieval differed from heuristic: {diff_r} (correct: {diff_ok}, wrong: {diff_r - diff_ok})")
            wrong = [e for e in self._retrieval_events if not e.get("correct")]
            for e in wrong[:10]:
                print(
                    "  [retrieval wrong]"
                    f" score={e.get('retrieval_score')}"
                    f" retrieved={e.get('retrieved_template')}"
                    f" heuristic={e.get('heuristic_template')}"
                    f" patterns_used={e.get('patterns_used')}"
                    f" expected={e.get('expected_num')} got={e.get('got_num')}"
                )

        if self._verbose:
            try:
                self._print_wrong_computation_report(
                    failures=self._failures[failures_start:],
                    top_k=3,
                    examples_per=3,
                )
                self._print_non_wrong_failure_samples(
                    failures=self._failures[failures_start:],
                    max_per_category=3,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[verbose analysis] failed: {exc}")

        return {"correct": correct, "total": total, "accuracy": accuracy}

    def _record_correct_solve(
        self,
        problem: Dict[str, Any],
        predicted: Any,
        solver: str,
        trace: Dict[str, Any],
        dataset_name: str,
    ) -> None:
        """
        Record to shadow copy only when benchmark evaluator confirmed correctness.

        This prevents the system from "learning" wrong programs due to confidence
        heuristics that don’t use ground-truth.
        """
        if self._shadow_copy is None:
            return
        rpn_program = str(trace.get("rpn_program", "") or "").strip()
        if not rpn_program:
            return

        # Backfill compositional learning even when another solver got it right:
        # derive a Galaxy-reading template+pattern signature for the same correct RPN.
        self._record_composition_backfill(problem, rpn_program, dataset_name)

        if solver != "trm":
            return

        meta = trace.get("meta", {}) if isinstance(trace.get("meta", {}), dict) else {}
        rule_used = str(trace.get("rule_used") or meta.get("rule_used") or "")
        confidence = meta.get("confidence", 1.0)
        try:
            score = float(confidence)
        except Exception:
            score = 1.0

        task_signature = {
            "problem_text": str(problem.get("problem", problem.get("question", "")))[:200],
            "rule_id": rule_used or "trm",
            "result": str(predicted)[:64],
            "problem_type": dataset_name,
        }

        program_type = "math"
        semantic_context: Dict[str, Any] = {}
        if rule_used == "galaxy_read":
            program_type = "reading"
            semantic_context = {
                "word_patterns": meta.get("read_trace", {}).get("patterns", []) if isinstance(meta.get("read_trace", {}), dict) else [],
                "composition_strategy": meta.get("read_understanding", {}) if isinstance(meta.get("read_understanding", {}), dict) else {},
                "composition_meta": meta.get("read_composition", {}) if isinstance(meta.get("read_composition", {}), dict) else {},
            }

        try:
            self._shadow_copy.record(
                task_signature=task_signature,
                program=rpn_program,
                program_type=program_type,
                score=score,
                task_id=f"{program_type}_{hash(task_signature['problem_text']) % 10000}",
                semantic_context=semantic_context,
            )
        except Exception:
            return

        # Additionally record composition strategy so TRM can learn template selection.
        if rule_used == "galaxy_read":
            comp = meta.get("read_composition", {}) if isinstance(meta.get("read_composition", {}), dict) else {}
            template_used = comp.get("template_used")
            patterns_matched = comp.get("patterns_matched")
            if isinstance(template_used, str) and template_used and isinstance(patterns_matched, list) and patterns_matched:
                try:
                    self._shadow_copy.record(
                        task_signature=task_signature,
                        program=rpn_program,
                        program_type="composition",
                        score=score,
                        task_id=f"composition_{hash(task_signature['problem_text']) % 10000}",
                        semantic_context={
                            "template_used": template_used,
                            "patterns_matched": patterns_matched,
                            "composition_steps": comp.get("composition_steps", []),
                            "structure": comp.get("structure", {}) if isinstance(comp.get("structure", {}), dict) else {},
                        },
                    )
                except Exception:
                    pass

        # Calibrate rule confidence for routing.
        if rule_used:
            try:
                self._shadow_copy.update_pattern_confidence(rule_used, score)
            except Exception:
                pass
        if program_type == "reading":
            for p in semantic_context.get("word_patterns", []):
                rid = p.get("rule_id") if isinstance(p, dict) else None
                if rid:
                    try:
                        self._shadow_copy.update_pattern_confidence(str(rid), score)
                    except Exception:
                        continue

    def _record_composition_backfill(self, problem: Dict[str, Any], rpn_program: str, dataset_name: str) -> None:
        """
        Record composition strategy for a correct RPN even if TRM didn't produce it.

        This increases learning signal when accuracy is still low.
        """
        if self._shadow_copy is None or self._galaxy_reader is None:
            return
        text = str(problem.get("problem", problem.get("question", "")))
        try:
            understanding, read_trace = self._galaxy_reader.read_problem(text)
        except Exception:
            return
        if not getattr(understanding, "is_complete", lambda: False)():
            return
        try:
            composed = self._galaxy_reader.compose_rpn(understanding, trace=read_trace, problem_text=text)
        except Exception:
            return
        if not composed:
            return

        def _norm(s: str) -> str:
            return " ".join(str(s).split())

        if _norm(composed) != _norm(rpn_program):
            return

        comp = getattr(self._galaxy_reader, "get_last_composition_meta", lambda: {})()
        template_used = comp.get("template_used") if isinstance(comp, dict) else None
        patterns_matched = comp.get("patterns_matched") if isinstance(comp, dict) else None
        if not isinstance(template_used, str) or not template_used:
            return
        if not isinstance(patterns_matched, list) or not patterns_matched:
            return

        task_signature = {
            "problem_text": text[:200],
            "rule_id": "composition_backfill",
            "result": str(problem.get("answer", problem.get("solution", "")))[:64],
            "problem_type": dataset_name,
        }
        try:
            self._shadow_copy.record(
                task_signature=task_signature,
                program=rpn_program,
                program_type="composition",
                score=1.0,
                task_id=f"composition_backfill_{hash(task_signature['problem_text']) % 10000}",
                semantic_context={
                    "template_used": template_used,
                    "patterns_matched": patterns_matched,
                    "composition_steps": comp.get("composition_steps", []) if isinstance(comp, dict) else [],
                    "structure": comp.get("structure", {}) if isinstance(comp, dict) and isinstance(comp.get("structure", {}), dict) else {},
                },
            )
        except Exception:
            return

    def run_all(self, limit_per_dataset: int | None = None) -> Dict[str, Any]:
        """Run all benchmarks."""
        datasets = ["gsm8k", "math", "omni_math", "amc_aime", "mmlu"]
        results: Dict[str, Any] = {}

        print("=" * 60)
        print("SOVEREIGN MATH BENCHMARK")
        print("No CuPy. No numpy in hot path. Pure PTX + Galaxy.")
        print("=" * 60)

        total_correct = 0
        total_problems = 0

        for ds in datasets:
            try:
                r = self.run_benchmark(ds, limit=limit_per_dataset)
                results[ds] = r
                total_correct += r.get("correct", 0)
                total_problems += r.get("total", 0)
            except Exception as e:  # noqa: BLE001
                print(f"  {ds}: ERROR - {e}")
                results[ds] = {"correct": 0, "total": 0, "accuracy": 0.0, "error": str(e)}

        overall = total_correct / total_problems if total_problems > 0 else 0.0

        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Overall: {total_correct}/{total_problems} = {100 * overall:.2f}%")
        for ds, r in results.items():
            print(f"  {ds:12s}: {r.get('accuracy', 0) * 100:.2f}%")
        print(f"Solve path stats: {self.solve_stats}")
        print(f"Shadow Copy: {len(self._shadow_copy.library) if self._shadow_copy is not None else 0} discoveries recorded")

        failure_report = self._analyze_failures()
        results["_failures"] = failure_report
        if failure_report.get("total_failures"):
            print("\nFailure analysis:")
            print(f"  Total failures: {failure_report['total_failures']}")
            print(f"  Categories: {failure_report['categories']}")
            print("  Sample failures:")
            for item in failure_report.get("sample_failures", []):
                print(f"    - [{item.get('source')}/{item.get('solver')}] {item.get('text')}")
        self._print_top_failing_rules()

        self._finalize_shadow_copy()
        return results

    def _log_failure_detail(
        self,
        problem: Dict[str, Any],
        result: Any,
        expected: Any,
        solver: str,
        trace: Dict[str, Any],
    ) -> None:
        rule_used = trace.get("rule_used")
        rpn_program = trace.get("rpn_program", "") or ""
        entry = {
            "problem_text": str(problem.get("problem", problem.get("question", "")))[:200],
            "expected": str(expected)[:50],
            "got": str(result)[:50],
            "solver": solver,
            "rule_used": rule_used,
            "rpn_program": str(rpn_program)[:120],
            "source": problem.get("source", "unknown"),
        }
        self._failure_details.append(entry)
        if rule_used:
            key = f"{solver}:{rule_used}"
            self._failure_rule_counts[key] = self._failure_rule_counts.get(key, 0) + 1

    def _print_top_failing_rules(self, top_k: int = 10) -> None:
        if not self._failure_rule_counts:
            return
        pairs = sorted(self._failure_rule_counts.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
        print("\nTop failing rules:")
        for idx, (key, count) in enumerate(pairs, start=1):
            print(f"  {idx}. {key}: {count} failures")

    def _validate_answer(self, result: Any, problem_text: str, source: str) -> bool:
        if result is None:
            return False
        try:
            val = float(result)
        except Exception:
            return False
        if math.isnan(val) or math.isinf(val):
            return False

        text = (problem_text or "").lower()
        if source in ("gsm8k", "amc_aime", "omni_math"):
            if val < 0 and "negative" not in text:
                return False
            if val > 1_000_000:
                return False
            if any(q in text for q in ("how many", "number of", "how much", "total", "altogether")):
                if abs(val - round(val)) > 1e-6:
                    return False
        # AMC/AIME answers are typically non-negative integers (often mod 1000 in evaluator)
        if source == "amc_aime":
            if abs(val - round(val)) > 1e-6:
                return False
        return True

    def _looks_like_expression(self, text: str, source: str) -> bool:
        """
        Heuristic gate for invoking the Galaxy composer.

        Prevents composer from "hallucinating" arithmetic on long GSM8K word problems
        where parsing a random substring tends to produce wrong_computation.
        """
        if "\\" in text:
            return True
        if any(op in text for op in ("+", "-", "*", "/", "^", "=", "(", ")")):
            # If it is a long wordy prompt, composer tends to pick the wrong numbers.
            word_count = len(text.split())
            alpha_words = re.findall(r"[A-Za-z]+", text)
            if source == "gsm8k" and word_count > 18:
                return False
            if len(alpha_words) > 8 and word_count > 12:
                return False
            return True
        # GSM8K: only allow composer on short prompts.
        return False

    def _try_composite_match(self, problem_text: str, source: str) -> Optional[Tuple[float, Dict[str, Any]]]:
        if source != "gsm8k":
            return None
        text = problem_text
        # Base: "... has N ..." then apply gives/gets operations.
        base = re.search(
            r"(?:has|had|owns|starts with|began with|there (?:are|were))\s+(\d+)",
            text,
            re.IGNORECASE,
        )
        if not base:
            return None

        ops: List[str] = [base.group(1)]
        number_re = re.compile(r"\d+(?:\.\d+)?")

        def _segment_from(idx: int) -> str:
            # Take a short span up to sentence boundary (or a safe max).
            tail = text[idx : idx + 140]
            for sep in (".", "?", "!", "\n"):
                pos = tail.find(sep)
                if pos >= 0:
                    return tail[:pos]
            return tail

        # Subtraction verbs: subtract every number mentioned in the verb phrase
        for m in re.finditer(r"\b(gives?|gave|spent|lost|uses?|used|ate)\b", text, re.IGNORECASE):
            if m.start() < base.end():
                continue
            seg = _segment_from(m.end())
            amts = number_re.findall(seg)
            if len(amts) > 1 and re.search(r"\b(each|per)\b", seg, re.IGNORECASE):
                continue
            for amt in amts:
                ops.append(amt)
                ops.append("-")

        # Addition verbs
        for m in re.finditer(r"\b(receives?|gets?|gains?|earned|found)\b", text, re.IGNORECASE):
            if m.start() < base.end():
                continue
            seg = _segment_from(m.end())
            amts = number_re.findall(seg)
            if len(amts) > 1 and re.search(r"\b(each|per)\b", seg, re.IGNORECASE):
                continue
            for amt in amts:
                ops.append(amt)
                ops.append("+")

        # Buying: only count when it's explicitly "more/additional"
        for m in re.finditer(r"\b(buys?|bought)\b", text, re.IGNORECASE):
            if m.start() < base.end():
                continue
            seg = _segment_from(m.end())
            if not re.search(r"\b(more|additional)\b", seg, re.IGNORECASE):
                continue
            for amt in number_re.findall(seg):
                ops.append(amt)
                ops.append("+")

        if len(ops) <= 1:
            return None

        rpn = " ".join(ops)
        try:
            result = self.engine.evaluate(rpn)
        except Exception:
            return None
        return float(result), {
            "rule_used": "composite_matcher",
            "rpn_program": rpn,
            "base": base.group(1),
            "ops": ops[1:],
        }

    def _analyze_failures(self) -> Dict[str, Any]:
        """Categorize failures by type (heuristic, for baseline triage)."""
        categories: Dict[str, int] = {
            "no_rule_match": 0,
            "wrong_computation": 0,
            "multi_step_needed": 0,
            "word_problem": 0,
            "algebra_needed": 0,
            "unknown": 0,
        }

        for f in self._failures:
            text = str(f.get("text", "")).lower()
            solver = str(f.get("solver", ""))
            if solver in ("fail",):
                categories["no_rule_match"] += 1
                continue
            if "solve" in text or "x =" in text or "equation" in text or "x^2" in text or "x²" in text:
                categories["algebra_needed"] += 1
                continue
            if any(w in text for w in ["step", "then", "after", "first", "second", "third", "finally"]):
                categories["multi_step_needed"] += 1
                continue
            if len(text.split()) > 50:
                categories["word_problem"] += 1
                continue
            if solver in ("grammar", "knowledge", "template", "composer", "word", "trm"):
                categories["wrong_computation"] += 1
                continue
            categories["unknown"] += 1

        return {
            "total_failures": len(self._failures),
            "categories": categories,
            "sample_failures": self._failures[:10],
        }

    def _finalize_shadow_copy(self) -> None:
        if getattr(self, "_shadow_readonly", False):
            print("\n[SHADOW COPY] Read-only mode: skipping save/commit")
            return
        if self._shadow_copy is None:
            return
        try:
            local_dir = os.getenv("K3D_LOCAL_DIR", "/K3D/Knowledge3D.local")
            checkpoint_dir = Path(local_dir) / "checkpoints" / "math_benchmarks"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._shadow_copy.save(checkpoint_dir / "shadow_copy.json")
            self._shadow_copy.commit_pending()
            print(f"\n[SHADOW COPY] Recorded {len(self._shadow_copy.library)} discoveries")
        except Exception as exc:  # noqa: BLE001
            print(f"[SHADOW COPY] finalize failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="(Legacy) Limit problems per dataset")
    parser.add_argument("--dataset", type=str, default=None, help="(Legacy) Run single dataset")
    parser.add_argument("--datasets", nargs="*", default=None, help="Datasets to run (e.g. gsm8k math)")
    parser.add_argument("--max-problems", type=int, default=None, help="Max problems per dataset (new)")
    parser.add_argument(
        "--use-trm-navigator",
        action="store_true",
        help="Enable TRM-style rule routing (experimental, still sovereign).",
    )
    parser.add_argument(
        "--disable-retrieval",
        action="store_true",
        help="Disable shadow-copy template retrieval (heuristic-only template selection).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Skip the first N problems per dataset (applied after optional --shuffle).",
    )
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset order before slicing/limit.")
    parser.add_argument("--shuffle-seed", type=int, default=0, help="Seed used by --shuffle.")
    parser.add_argument(
        "--shadow-readonly",
        action="store_true",
        help="Load shadow copy but do not save/commit updates (useful for evaluation runs).",
    )
    parser.add_argument(
        "--load-all-galaxies",
        action="store_true",
        help="Enable cross-domain exploration hooks (Reality/Drawing galaxies) in TRM reader.",
    )
    parser.add_argument(
        "--enable-book-galaxies",
        action="store_true",
        help="Enable Book Galaxy lookups (requires prior ingestion under K3D_LOCAL_DIR/galaxies/books).",
    )
    parser.add_argument(
        "--book-galaxy-root",
        type=str,
        default=None,
        help="Override Book Galaxy root dir (defaults to K3D_LOCAL_DIR/galaxies/books).",
    )
    parser.add_argument(
        "--book-max-books",
        type=int,
        default=8,
        help="Max number of book galaxies to scan (default: 8).",
    )
    parser.add_argument(
        "--book-top-k",
        type=int,
        default=5,
        help="Top-K book page hits to use per problem (default: 5).",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Test-time compute budget for Galaxy reader (0 disables).",
    )
    parser.add_argument(
        "--router-weights",
        type=str,
        default=None,
        help="Path to learned theorem router weights (JSON mapping).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra diagnostics and a wrong-computation summary for the current run.",
    )
    parser.add_argument(
        "--calc-microbench",
        type=str,
        default=None,
        help="Path to a JSONL file for calculus micro-benchmarking (e.g. data/calculus_microbench.jsonl).",
    )
    parser.add_argument(
        "--log-galaxy-out",
        type=str,
        default=None,
        help="Write Log Galaxy traces to this JSONL path (microbench preferred).",
    )
    args = parser.parse_args()

    runner = SovereignBenchmarkRunner(
        use_trm_navigator=bool(args.use_trm_navigator),
        disable_retrieval=bool(args.disable_retrieval),
        shadow_readonly=bool(args.shadow_readonly),
        load_all_galaxies=bool(args.load_all_galaxies),
        enable_book_galaxies=bool(args.enable_book_galaxies),
        book_galaxy_root=args.book_galaxy_root,
        book_max_books=int(args.book_max_books),
        book_top_k=int(args.book_top_k),
        thinking_budget=int(args.thinking_budget or 0),
        verbose=bool(args.verbose),
        router_weights=args.router_weights,
    )
    if args.log_galaxy_out:
        from knowledge3d.training.math_benchmarks.log_galaxy import LogGalaxy

        runner.set_log_galaxy(LogGalaxy())

    max_problems = args.max_problems if args.max_problems is not None else args.limit

    if args.calc_microbench:
        # Microbench Mode: Run specific file, force recursive solver usage
        print(f"\n[Microbench] Loading problems from {args.calc_microbench}")
        path = Path(args.calc_microbench)
        if not path.exists():
            print(f"Error: File {path} not found.")
            return

        problems = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    p["source"] = "microbench"
                    problems.append(p)
        
        # Override load_dataset to return just these problems
        runner.load_dataset = lambda name: problems if name == "microbench" else []
        
        # Run
        runner.run_benchmark("microbench")
        if args.log_galaxy_out and runner._log_galaxy is not None:
            runner._log_galaxy.to_jsonl(args.log_galaxy_out)
            print(f"[Microbench] Wrote Log Galaxy to {args.log_galaxy_out}")
        return

    if args.dataset:
        runner.run_benchmark(
            args.dataset,
            limit=max_problems,
            start_index=int(args.start_index or 0),
            shuffle=bool(args.shuffle),
            shuffle_seed=int(args.shuffle_seed or 0),
        )
        if args.log_galaxy_out and runner._log_galaxy is not None:
            runner._log_galaxy.to_jsonl(args.log_galaxy_out)
            print(f"[Benchmark] Wrote Log Galaxy to {args.log_galaxy_out}")
        runner._finalize_shadow_copy()
        return

    if args.datasets:
        results: Dict[str, Any] = {}
        for ds in args.datasets:
            results[ds] = runner.run_benchmark(
                ds,
                limit=max_problems,
                start_index=int(args.start_index or 0),
                shuffle=bool(args.shuffle),
                shuffle_seed=int(args.shuffle_seed or 0),
            )
        if args.log_galaxy_out and runner._log_galaxy is not None:
            runner._log_galaxy.to_jsonl(args.log_galaxy_out)
            print(f"[Benchmark] Wrote Log Galaxy to {args.log_galaxy_out}")
        print(f"\nShadow Copy: {len(runner._shadow_copy.library) if runner._shadow_copy is not None else 0} discoveries recorded")
        failure_report = runner._analyze_failures()
        if failure_report.get("total_failures"):
            print("\nFailure analysis:")
            print(f"  Total failures: {failure_report['total_failures']}")
            print(f"  Categories: {failure_report['categories']}")
        runner._print_top_failing_rules()
        runner._finalize_shadow_copy()
        return

    else:
        runner.run_all(limit_per_dataset=max_problems)


if __name__ == "__main__":
    main()
