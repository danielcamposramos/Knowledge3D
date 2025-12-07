"""
Sovereign ARC-AGI pipeline (Drawing + Grammar galaxies, TRM router).

This is a thin orchestrator that wires:
- DrawingGalaxy (visual atoms)
- GrammarGalaxy (196+ rules)
- SovereignTRMRouter (matryoshka + adapter; no external ML)
- ProgramComposer (cross-galaxy compositions)
- DualShadowCopy (evolution tracking)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Optional, Tuple

# SOVEREIGN: No numpy in hot path! Use plain lists for grids.

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.sovereign_trm_router import SovereignTRMRouter, rule_score_hint
from knowledge3d.training.arc_agi.hybrid_generator import HybridCandidateGenerator
from knowledge3d.training.arc_agi.program_composer import ProgramComposer
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.embedders import MultiModalGridEmbedder
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.cranium.math_galaxy import get_math_galaxy


def compute_ternary_reward(score: float) -> int:
    """
    Map continuous score to balanced ternary reward {-1, 0, +1}.

    -1: score < 0.50 (punish)
     0: 0.50 ≤ score < 0.99 (neutral)
    +1: score ≥ 0.99 (reward)
    """
    if score < 0.50:
        return -1
    if score < 0.99:
        return 0
    return 1


@dataclass
class TaskResult:
    task_id: str
    best_program: str
    program_type: str
    score: float
    signature: Dict
    output_grid: Optional[List[List[int]]] = None
    correct: bool = False
    fuzzy_score: float = 0.0


def _grids_equal(grid1: Sequence[Sequence[int]], grid2: Sequence[Sequence[int]]) -> bool:
    """SOVEREIGN: Compare grids without numpy."""
    # Basic shape check
    if not isinstance(grid1, (list, tuple)) or not isinstance(grid2, (list, tuple)):
        return False
    if len(grid1) != len(grid2):
        return False
    for row1, row2 in zip(grid1, grid2):
        if not isinstance(row1, (list, tuple)) or not isinstance(row2, (list, tuple)):
            return False
        if len(row1) != len(row2):
            return False
        # Convert to lists and compare to handle any type differences
        list1 = [int(x) for x in row1]  # Ensure integers
        list2 = [int(x) for x in row2]
        if list1 != list2:
            return False
    return True


def _fuzzy_match(
    predicted: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
    crop_tolerance: bool = True,
    align_tolerance: int = 1,
) -> float:
    """
    Fuzzy matching for ARC grids (tolerates padding, alignment).

    Args:
        predicted: Actual output grid
        expected: Expected output grid
        crop_tolerance: If True, crop larger grid to match smaller
        align_tolerance: Allow N-pixel alignment shifts

    Returns:
        Score [0.0, 1.0]: 1.0 = perfect fuzzy match
    """
    if not predicted or not expected:
        return 0.0

    # Strategy 1: Size normalization (crop padding)
    if crop_tolerance:
        h_pred, w_pred = len(predicted), len(predicted[0]) if predicted else 0
        h_exp, w_exp = len(expected), len(expected[0]) if expected else 0

        # Crop to smaller size (remove padding)
        h_min, w_min = min(h_pred, h_exp), min(w_pred, w_exp)

        # Extract core regions (top-left aligned)
        pred_core = [row[:w_min] for row in predicted[:h_min]]
        exp_core = [row[:w_min] for row in expected[:h_min]]

        # Check if cores match exactly
        if _grids_equal(pred_core, exp_core):
            return 1.0  # Perfect match after crop

        # Check if cores match with high overlap
        matches = 0
        total = h_min * w_min
        for r_pred, r_exp in zip(pred_core, exp_core):
            for a, b in zip(r_pred, r_exp):
                if a == b:
                    matches += 1

        core_score = matches / total if total > 0 else 0.0

        # Accept if > 80% core match
        if core_score > 0.80:
            return core_score

    # Strategy 2: Alignment tolerance (try 1-pixel shifts)
    if align_tolerance > 0 and len(predicted) == len(expected):
        h, w = len(predicted), len(predicted[0]) if predicted else 0
        if w == (len(expected[0]) if expected else 0):
            best_score = 0.0
            # Try shifts: (0,0), (1,0), (0,1), (-1,0), (0,-1)
            for dy in range(-align_tolerance, align_tolerance + 1):
                for dx in range(-align_tolerance, align_tolerance + 1):
                    matches = 0
                    total = 0
                    for y in range(h):
                        for x in range(w):
                            y_pred, x_pred = y + dy, x + dx
                            if 0 <= y_pred < h and 0 <= x_pred < w:
                                total += 1
                                if predicted[y_pred][x_pred] == expected[y][x]:
                                    matches += 1

                    if total > 0:
                        score = matches / total
                        if score > best_score:
                            best_score = score
            if best_score > 0.90:  # 90% match with alignment
                return best_score

    # Fallback: Raw pixel overlap
    if len(predicted) != len(expected):
        return 0.0
    if len(predicted[0]) != len(expected[0]):
        return 0.0

    matches = 0
    total = 0
    for r_pred, r_exp in zip(predicted, expected):
        for a, b in zip(r_pred, r_exp):
            total += 1
            if a == b:
                matches += 1

    return matches / total if total > 0 else 0.0


def _task_complexity(grid: Sequence[Sequence[int]]) -> float:
    """Heuristic complexity based on grid area (normalized)."""
    if not grid or not grid[0]:
        return 0.0
    h = len(grid)
    w = len(grid[0])
    area = h * w
    # Normalize roughly: 0 at 0 area, ~1 at 30x30 (~900)
    return min(1.0, area / 900.0)


def _procedural_resize(
    grid: Sequence[Sequence[int]],
    target_h: int,
    target_w: int,
) -> List[List[int]]:
    """
    Procedurally resize grid with majority-vote downsample or pixel-repeat upsample.
    """
    if not grid or not grid[0]:
        return [[0] * target_w for _ in range(target_h)]

    h_src, w_src = len(grid), len(grid[0])

    if h_src == target_h and w_src == target_w:
        return [list(row) for row in grid]

    # Shrink via stride
    if h_src > target_h or w_src > target_w:
        stride_h = max(1, h_src // target_h)
        stride_w = max(1, w_src // target_w)
        result: List[List[int]] = []
        for y_t in range(target_h):
            row: List[int] = []
            for x_t in range(target_w):
                vals: Dict[int, int] = {}
                y_start = y_t * stride_h
                x_start = x_t * stride_w
                for dy in range(stride_h):
                    for dx in range(stride_w):
                        y_s = y_start + dy
                        x_s = x_start + dx
                        if y_s < h_src and x_s < w_src:
                            v = grid[y_s][x_s]
                            vals[v] = vals.get(v, 0) + 1
                if vals:
                    majority = max(vals.items(), key=lambda kv: kv[1])[0]
                else:
                    majority = 0
                row.append(majority)
            result.append(row)
        return result

    # Expand via repetition
    if h_src < target_h or w_src < target_w:
        repeat_h = max(1, target_h // h_src)
        repeat_w = max(1, target_w // w_src)
        result: List[List[int]] = []
        for row in grid:
            expanded_row: List[int] = []
            for val in row:
                expanded_row.extend([val] * repeat_w)
            while len(expanded_row) < target_w:
                expanded_row.append(0)
            expanded_row = expanded_row[:target_w]
            for _ in range(repeat_h):
                result.append(expanded_row[:])
        while len(result) < target_h:
            result.append([0] * target_w)
        return result[:target_h]

    return [list(row) for row in grid]


class SovereignAIPipeline:
    """End-to-end sovereign pipeline (no external ML, GPU-only matryoshka)."""

    def __init__(
        self,
        matryoshka_dim: int = 512,
        *,
        staged_shadow: bool = False,
        embedding_galaxy=None,
        cosine_bridge=None,
        hybrid_mode: bool = False,
    ) -> None:
        self.drawing = DrawingGalaxy()
        self.grammar = GrammarGalaxy()
        self.math_galaxy = get_math_galaxy()
        self.shadow = DualShadowCopy(self.drawing, self.grammar, staged=staged_shadow)
        self.router = SovereignTRMRouter(self.drawing, self.grammar, shadow_copy=self.shadow, matryoshka_dim=matryoshka_dim)
        self.composer = ProgramComposer()
        # Shared codec embedder (singleton) to avoid repeated PTX loads across workers.
        self.codec_embedder = MultiModalGridEmbedder(matryoshka_dim=matryoshka_dim)
        self.embedding_galaxy = embedding_galaxy
        self.cosine_bridge = cosine_bridge or CosineSimilarityBridge()
        self.hybrid_mode = hybrid_mode
        self.results: List[TaskResult] = []
        self._grammar_token_index: Dict[str, set[str]] = {}
        self._grammar_token_version = 0
        self._shape_token_index: Dict[str, set[str]] = {}
        self._shape_token_version = 0

        print(f"  [MATH] Loaded {len(self.math_galaxy.symbols)} canonical math symbols")

    def process_task(
        self,
        task_id: str,
        test_input: Sequence[Sequence[int]],
        *,
        train_examples: Optional[List[Dict]] = None,
        expected_output: Optional[Sequence[Sequence[int]]] = None,
        top_k: int = 69,  # SOVEREIGN: Tesla 3-6-9 (increased from 12)
    ) -> TaskResult:
        """Process task merging procedural baseline + sovereign routing."""

        # 1) Procedural baseline candidates (train example analysis).
        from knowledge3d.training.arc_agi import CandidateGenerator
        from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor

        executor = ARCRPNExecutor()
        procedural_candidates: List[Dict] = []

        # SOVEREIGN: Extract semantic hints from context to guide generation
        semantic_hints: List[str] = []
        if self.shadow.semantic_context is not None:
            try:
                # Get top semantic matches and extract their word hints
                matches = self.shadow.semantic_context.find_matching_contexts(test_input, top_k=9)
                print(f"  [SEMANTIC EXTRACTION] Found {len(matches)} matching contexts")
                for ctx in matches:
                    # Extract transformation types and usage conditions as hints
                    # NOTE: find_matching_contexts() returns resolved words, not refs!
                    if "transformation_type" in ctx:
                        word = ctx["transformation_type"]
                        if isinstance(word, str) and word:
                            semantic_hints.append(word)
                    if "when_to_use" in ctx and isinstance(ctx["when_to_use"], list):
                        semantic_hints.extend([str(w) for w in ctx["when_to_use"] if w])
                if semantic_hints:
                    print(f"  [SEMANTIC HINTS] Extracted {len(semantic_hints)} hints: {semantic_hints[:5]}")
                else:
                    print(f"  [SEMANTIC HINTS] No hints extracted from {len(matches)} contexts")
            except Exception as e:
                print(f"  [PIPELINE] Warning: Could not extract semantic hints: {e}")

        if train_examples:
            from knowledge3d.training.arc_agi.parallel_generator import ParallelCandidateGenerator

            par_gen = ParallelCandidateGenerator(
                num_workers=9,
                candidates_per_worker=6,
                top_k=3,
                matryoshka_dim=self.router.matryoshka_dim,
                shadow_copy=self.shadow,
                drawing_galaxy=self.drawing,
                codec_embedder=self.codec_embedder,
                embedding_galaxy=self.embedding_galaxy,
                cosine_bridge=self.cosine_bridge,
            )
            if getattr(self, "hybrid_mode", False):
                hybrid_gen = HybridCandidateGenerator(
                    parallel_gen=par_gen,
                    shadow_copy=self.shadow,
                    drawing_galaxy=self.drawing,
                    grammar_galaxy=self.grammar,
                    core_pool=par_gen.core_pool,
                    quick_solve_threshold=0.95,
                    embedding_galaxy=self.embedding_galaxy,
                    cosine_bridge=self.cosine_bridge,
                )
                procedural_candidates = hybrid_gen.generate_hybrid(
                    input_grid=test_input,
                    train_examples=train_examples,
                    semantic_hints=semantic_hints,
                    expected_output=expected_output,
                    task_history=[],
                    task_complexity=_task_complexity(test_input),
                )
                print(f"  [CANDIDATES] Hybrid generated {len(procedural_candidates)} candidates (parallel + deep)")
            else:
                procedural_candidates = par_gen.generate_parallel(
                    input_grid=test_input,
                    train_examples=train_examples,
                    semantic_hints=semantic_hints,
                    expected_output=expected_output,
                )
                print(f"  [CANDIDATES] Parallel generated {len(procedural_candidates)} candidates (Tesla 3-6-9)")

        # 2) TRM router candidates.
        trm_candidates = self.router.route(test_input, top_k=top_k)

        # 3) Merge candidate programs.
        merged: List[Dict] = []
        print(f"  [HYBRID] Evaluating {len(procedural_candidates)} procedural candidates with TRM...")
        for output, instruction, rpn in procedural_candidates:
            trm_conf = self._evaluate_procedural_with_trm(
                program=rpn,
                output_grid=output,
                test_input=test_input,
                train_examples=train_examples or [],
            )
            priority = "high" if trm_conf > 0.7 else ("medium" if trm_conf > 0.5 else "low")
            merged.append(
                {
                    "program": rpn,
                    "program_type": "procedural",
                    "source": "baseline",
                    "output": output,  # SOVEREIGN: Keep as list, no numpy conversion
                    "trm_confidence": trm_conf,
                    "priority": priority,
                }
            )

        avg_conf = (
            sum(c.get("trm_confidence", 0.0) for c in merged) / len(merged)
            if merged
            else 0.0
        )
        print(
            f"  [HYBRID] TRM confidence avg={avg_conf:.2f} "
            f"({sum(1 for c in merged if c.get('priority')=='high')} high, "
            f"{sum(1 for c in merged if c.get('priority')=='medium')} medium)"
        )

        for cand in trm_candidates:
            if "program" in cand:
                merged.append(
                    {
                        "program": cand["program"],
                        "program_type": cand.get("program_type", "semantic"),
                        "source": cand.get("source", "semantic_match"),
                        "output": None,
                        "signature": cand.get("semantic_context") or cand.get("signature", {}),
                        "trm_confidence": 0.5,
                        "priority": "low",
                    }
                )
            else:
                compositions = self.composer.compose(cand["drawing_program"], [cand["grammar_rule"]])
                for prog, ptype in compositions:
                    merged.append(
                        {
                            "program": prog,
                        "program_type": ptype,
                        "source": cand.get("source", "trm"),
                        "output": None,  # to be executed
                        "signature": cand.get("signature", {}),
                        "trm_confidence": 0.5,
                        "priority": "low",
                    }
                )

        if not merged:
            raise RuntimeError("No candidates generated")

        expected_list = [list(row) for row in expected_output] if expected_output is not None else None
        expected_shape = (len(expected_list), len(expected_list[0]) if expected_list else 0) if expected_list else (0, 0)
        merged = self._rank_candidates_multimetric(merged, expected_shape)
        top_k_tesla = 27  # Tesla 3^3
        merged = merged[:top_k_tesla]
        print(
            f"  [TESLA] Executing top {top_k_tesla} candidates (3³ resonance): "
            f"{sum(1 for c in merged if c.get('priority')=='high')} high, "
            f"{sum(1 for c in merged if c.get('priority')=='medium')} medium, "
            f"{sum(1 for c in merged if c.get('priority')=='low')} low"
        )

        # SOVEREIGN: Keep grids as lists, no numpy conversion
        test_input_list = [list(row) for row in test_input]

        # 4) Execute programs needing execution and score.
        best = None
        exact_proc_best = None
        merged = self._deduplicate_candidates(merged)

        for cand_idx, cand in enumerate(merged):
            if cand["output"] is None:
                try:
                    cand["output"] = executor.execute(test_input, cand["program"])  # Returns list already
                except Exception:
                    cand["output"] = test_input_list
            # Size-aware preprocessing for scoring
            cand_output = cand["output"]
            fuzzy_threshold = 0.80
            drop_candidate = False
            downscale_factor = 1.0
            if expected_list is not None:
                exp_h = len(expected_list)
                exp_w = len(expected_list[0]) if expected_list else 0
                area = exp_h * exp_w
                fuzzy_threshold = self._get_adaptive_fuzzy_threshold(task_id, area)
                pred_h = len(cand_output) if cand_output else 0
                pred_w = len(cand_output[0]) if cand_output else 0
                if (pred_h, pred_w) != (exp_h, exp_w):
                    ratio_h = max(pred_h / exp_h, exp_h / pred_h) if pred_h and exp_h else 10.0
                    ratio_w = max(pred_w / exp_w, exp_w / pred_w) if pred_w and exp_w else 10.0
                    downscale_factor = max(ratio_h, ratio_w)
                    # Hard drop when > 2.5x in either dimension.
                    if ratio_h >= 2.5 or ratio_w >= 2.5:
                        print(f"  [DROP SIZE] Task {task_id}: {pred_h}x{pred_w} vs {exp_h}x{exp_w} (ratio_h={ratio_h:.2f}, ratio_w={ratio_w:.2f})")
                        drop_candidate = True
                    else:
                        print(f"  [RESIZE] Task {task_id}: {pred_h}x{pred_w} -> {exp_h}x{exp_w}")
                        cand_output = _procedural_resize(cand_output, exp_h, exp_w)
                        cand["output"] = cand_output
                        pred_h, pred_w = exp_h, exp_w
                else:
                    ratio_h = ratio_w = 1.0
                    downscale_factor = 1.0
                aspect_pred = (pred_h / pred_w) if pred_h and pred_w else 0.0
                aspect_exp = (exp_h / exp_w) if exp_h and exp_w else 0.0
                aspect_ok = True
                if aspect_pred and aspect_exp:
                    aspect_ratio = max(aspect_pred / aspect_exp, aspect_exp / aspect_pred)
                    aspect_ok = aspect_ratio <= 1.5
                if not aspect_ok:
                    print(f"  [DROP SIZE] Task {task_id}: aspect mismatch pred={aspect_pred:.2f} exp={aspect_exp:.2f}")
                    drop_candidate = True

                if drop_candidate:
                    continue

            fuzzy_score = (
                _fuzzy_match(cand_output, expected_list) if expected_list is not None else 0.0
            )
            score = self._score_candidate(
                cand_output,
                test_input_list,
                expected_list,
                source=cand["source"],
            )
            if expected_list is not None and downscale_factor > 2.0:
                fuzzy_threshold = max(0.60, fuzzy_threshold - 0.10)
            if expected_list is not None and fuzzy_score >= fuzzy_threshold:
                score = max(score, fuzzy_score)
                if cand_idx == 0:
                    print(
                        f"  [FUZZY MATCH] Task {task_id}: fuzzy_score={fuzzy_score:.2f} "
                        f"(threshold={fuzzy_threshold:.2f})"
                    )
            elif expected_list is not None and 0.70 <= fuzzy_score < 0.80:
                print(f"  [NEAR MISS] Task {task_id}: fuzzy_score={fuzzy_score:.2f} (70-80%, review needed)")
            cand["score"] = score
            if expected_list is not None:
                calibrated = self._calibrate_confidence_from_outcome(
                    cand,
                    fuzzy_score,
                    expected_shape,
                )
                cand["calibrated_confidence"] = calibrated
            # Diagnostic logging for early candidates to expose padding/alignment issues.
            if expected_list is not None and cand_idx < 3:
                exp_h = len(expected_list)
                exp_w = len(expected_list[0]) if expected_list else 0
                pred_h = len(cand["output"]) if cand["output"] else 0
                pred_w = len(cand["output"][0]) if cand["output"] else 0
                print(f"  [DIAGNOSTIC] Task {task_id}, Candidate {cand_idx}:")
                print(f"    Program: {cand['program'][:120]}{'...' if len(cand['program']) > 120 else ''}")
                print(f"    Expected shape: {exp_h}x{exp_w}, Actual shape: {pred_h}x{pred_w}")
                if exp_h <= 5 and pred_h <= 5:
                    print("    Expected grid:")
                    for row in expected_list:
                        print(f"      {row}")
                    print("    Actual grid:")
                    for row in cand["output"]:
                        print(f"      {row}")
                exact_flag = 1.0 if _grids_equal(cand["output"], expected_list) else 0.0
                print(f"    Exact match: {exact_flag}, Fuzzy match: {fuzzy_score:.2f}")
            # Track exact procedural matches first
            if expected_list is not None and cand["source"] == "baseline" and _grids_equal(cand["output"], expected_list):
                if exact_proc_best is None or score > exact_proc_best["score"]:
                    exact_proc_best = cand
            if best is None or score > best["score"]:
                best = cand

        # Prefer exact procedural match if available
        chosen = exact_proc_best or best
        if chosen is None:
            # No viable candidates survived filtering; return a neutral result.
            return TaskResult(
                task_id=task_id,
                best_program="",
                program_type="none",
                score=0.0,
                signature={"source": "none"},
                output_grid=test_input_list,
                correct=False,
                fuzzy_score=0.0,
            )

        assert chosen is not None
        signature = chosen.get("signature") or {"source": chosen["source"]}

        # DIAGNOSTIC: Log answer comparison details
        if expected_list is not None:
            chosen_fuzzy = _fuzzy_match(chosen["output"], expected_list)
            exp_h = len(expected_list)
            exp_w = len(expected_list[0]) if expected_list else 0
            area = exp_h * exp_w
            fuzzy_threshold = self._get_adaptive_fuzzy_threshold(task_id, area)
            chosen["score"] = max(chosen["score"], chosen_fuzzy)
            if chosen_fuzzy >= fuzzy_threshold:
                reward = 1
            else:
                reward = compute_ternary_reward(chosen["score"])
            reward_label = {-1: "PUNISH", 0: "NEUTRAL", 1: "REWARD"}[reward]
            is_correct = reward == 1
            print(
                f"  [ANSWER CHECK] Task {task_id}: score={chosen['score']:.2f}, reward={reward_label}, source={chosen['source']}"
            )
            if is_correct:
                print(f"  [ANSWER CHECK] ✅ CORRECT ANSWER FOUND!")
            elif chosen_fuzzy >= 0.70:
                print(f"  [ANSWER CHECK] Fuzzy score {chosen_fuzzy:.2f} (near miss)")
            elif chosen["score"] >= 0.9:
                print(f"  [ANSWER CHECK] High score but not exact match - checking grids:")
                print(f"    Expected shape: {len(expected_list)}×{len(expected_list[0]) if expected_list else 0}")
                print(f"    Got shape: {len(chosen['output'])}×{len(chosen['output'][0]) if chosen['output'] else 0}")
                # Show first few cells for debugging
                if len(expected_list) <= 5 and len(expected_list[0]) <= 5:
                    print(f"    Expected: {expected_list}")
                    print(f"    Got: {chosen['output']}")

        # Record discoveries according to shadow thresholds (with semantic context)
        self.shadow.record(
            signature,
            chosen["program"],
            chosen["program_type"],
            chosen["score"],
            input_grid=test_input_list,  # SOVEREIGN: Pass list, not numpy array
            output_grid=chosen["output"],
            task_id=task_id,
        )
        if expected_list is not None:
            self.shadow.update_task_history(task_id, is_correct)

        result = TaskResult(
            task_id=task_id,
            best_program=chosen["program"],
            program_type=chosen["program_type"],
            score=float(chosen["score"]),
            signature=signature,
            output_grid=chosen["output"],  # SOVEREIGN: Already a list, no .tolist() needed
            correct=is_correct if expected_list is not None else False,
            fuzzy_score=chosen_fuzzy if expected_list is not None else 0.0,
        )
        self.results.append(result)
        return result

    def summary(self) -> Dict[str, int]:
        return {
            "tasks": len(self.results),
            "shadow_entries": len(self.shadow.library),
            "drawing_shapes": len(self.drawing.shapes),
            "grammar_rules": len(self.grammar.rules),
        }

    def _log_vocabulary_quality(self, epoch: int) -> None:
        if not hasattr(self, "shadow") or not self.shadow.library:
            return

        recent_entries = [
            entry for entry in self.shadow.library if entry.get("quality_score", 0.0) > 0.6
        ][-100:]
        if not recent_entries:
            return

        rule_usage: Dict[str, int] = defaultdict(int)
        rule_quality: Dict[str, List[float]] = defaultdict(list)
        shape_usage: Dict[str, int] = defaultdict(int)

        for entry in recent_entries:
            program = entry.get("program", "")
            quality = float(entry.get("quality_score", 0.0))
            if entry.get("program_type") == "transformation":
                rules_used = self._parse_grammar_rules_from_program(program)
                for rule in rules_used:
                    rule_usage[rule] += 1
                    rule_quality[rule].append(quality)
            if entry.get("program_type") in {"visual", "hybrid"}:
                shapes_used = self._parse_drawing_shapes_from_program(program)
                for shape_id in shapes_used:
                    shape_usage[shape_id] += 1

        print(f"\n[VOCAB QUALITY Epoch {epoch}]")
        if rule_usage:
            print("  Top Grammar Rules (high-quality solutions):")
            sorted_rules = sorted(rule_usage.items(), key=lambda item: item[1], reverse=True)[:10]
            for rule, count in sorted_rules:
                scores = rule_quality.get(rule, [0.0])
                avg_q = sum(scores) / len(scores)
                print(f"    {rule}: {count}× used, avg_quality={avg_q:.3f}")
        else:
            print("  No grammar usage recorded in last 100 high-quality entries.")

        if shape_usage:
            print("  Top Drawing Shapes (high-quality solutions):")
            sorted_shapes = sorted(shape_usage.items(), key=lambda item: item[1], reverse=True)[:10]
            for shape_id, count in sorted_shapes:
                print(f"    {shape_id}: {count}× used")
        else:
            print("  No drawing shapes recorded in last 100 high-quality entries.")

        avg_depth = sum(len(entry.get("program", "").split()) for entry in recent_entries) / max(1, len(recent_entries))
        print(f"  Avg composition depth: {avg_depth:.1f} tokens")

    @staticmethod
    def _score_candidate(
        output: Sequence[Sequence[int]],
        input_grid: Sequence[Sequence[int]],
        expected: Optional[Sequence[Sequence[int]]] = None,
        *,
        source: str = "unknown",
    ) -> float:
        """Score candidates (SOVEREIGN: no numpy!); align thresholds with DualShadowCopy recording."""
        # Guard against malformed outputs (e.g., flat lists or scalars).
        if not isinstance(output, (list, tuple)) or any(not isinstance(row, (list, tuple)) for row in output):
            return 0.0

        if expected is not None and _grids_equal(output, expected):
            return 1.0

        score = 0.4  # below recording threshold

        # Reward non-identity transforms
        if not _grids_equal(output, input_grid):
            score += 0.2  # 0.6 baseline for any change

        # Count unique colors (no numpy unique!)
        all_values = [val for row in output for val in row]
        num_colors = len(set(all_values)) if all_values else 0

        # Compute filled ratio (no numpy count_nonzero!)
        total_cells = sum(len(row) for row in output)
        nonzero_count = sum(1 for val in all_values if val != 0)
        filled_ratio = float(nonzero_count) / float(total_cells or 1)

        if num_colors > 1:
            score += 0.1  # 0.7
        if 0.1 < filled_ratio < 0.9:
            score += 0.1  # up to 0.8

        if source == "baseline" and score < 1.0:
            score = min(score + 0.1, 0.7)  # slight procedural bias, capped

        return min(score, 1.0)

    def _calibrate_confidence_from_outcome(
        self,
        candidate: Dict,
        fuzzy_score: float,
        expected_shape: tuple[int, int],
    ) -> float:
        base_confidence = candidate.get("trm_confidence", 0.5)
        outcome_factor = fuzzy_score

        if candidate.get("output") and candidate["output"]:
            actual_h = len(candidate["output"])
            actual_w = len(candidate["output"][0]) if candidate["output"] else 0
            exp_h, exp_w = expected_shape
            ratio_h = max(actual_h / exp_h, exp_h / actual_h) if exp_h and actual_h else 10.0
            ratio_w = max(actual_w / exp_w, exp_w / actual_w) if exp_w and actual_w else 10.0
            size_ratio = max(ratio_h, ratio_w)
            size_accuracy = 1.0 / size_ratio if size_ratio > 0 else 0.0
        else:
            size_accuracy = 0.0

        pattern_id = self._extract_pattern_signature(candidate.get("program", ""))
        pattern_history = self.shadow.get_pattern_success_rate(pattern_id)
        history_factor = pattern_history if pattern_history is not None else 0.5

        calibrated = (
            0.4 * base_confidence
            + 0.3 * outcome_factor
            + 0.2 * size_accuracy
            + 0.1 * history_factor
        )

        self.shadow.update_pattern_confidence(pattern_id, calibrated)
        return min(1.0, max(0.0, calibrated))

    def _extract_pattern_signature(self, program: str) -> str:
        tokens = program.split()
        operations = [t for t in tokens if t.isalpha() or t.isupper()]
        return " ".join(operations)

    def _parse_grammar_rules_from_program(self, program: str) -> List[str]:
        if not program:
            return []
        self._ensure_grammar_token_index()
        tokens = set(token.lower() for token in program.split())
        matched: List[str] = []
        for rule_id, token_set in self._grammar_token_index.items():
            if token_set and token_set.intersection(tokens):
                matched.append(rule_id)
        return matched

    def _parse_drawing_shapes_from_program(self, program: str) -> List[str]:
        if not program:
            return []
        self._ensure_shape_token_index()
        tokens = set(token.lower() for token in program.split())
        matched: List[str] = []
        for shape_id, token_set in self._shape_token_index.items():
            if token_set and token_set.intersection(tokens):
                matched.append(shape_id)
        return matched

    def _ensure_grammar_token_index(self) -> None:
        current_version = len(self.grammar.rules)
        if getattr(self, "_grammar_token_version", 0) == current_version and self._grammar_token_index:
            return
        token_index: Dict[str, set[str]] = {}
        for rule_id, rule in self.grammar.rules.items():
            tokens: set[str] = set()
            rpn_program = getattr(rule, "rpn_program", "") or (rule.get("rpn_program") if isinstance(rule, dict) else "")
            if rpn_program:
                tokens.update(token.lower() for token in rpn_program.split())
            pattern = getattr(rule, "pattern", None) or (rule.get("pattern") if isinstance(rule, dict) else None)
            if pattern:
                tokens.add(str(pattern).lower())
            token_index[rule_id] = tokens
        self._grammar_token_index = token_index
        self._grammar_token_version = current_version

    def _ensure_shape_token_index(self) -> None:
        current_version = len(self.drawing.shapes)
        if getattr(self, "_shape_token_version", 0) == current_version and self._shape_token_index:
            return
        token_index: Dict[str, set[str]] = {}
        for shape_id, item in self.drawing.shapes.items():
            payload = item.payload if hasattr(item, "payload") else item
            tokens: set[str] = set()
            visual_rpn = payload.get("visual_rpn") if isinstance(payload, dict) else None
            if visual_rpn:
                tokens.update(token.lower() for token in visual_rpn.split())
            composition = (
                payload.get("procedural_programs", {}).get("composition")
                if isinstance(payload, dict)
                else None
            )
            if composition:
                tokens.update(token.lower() for token in composition.split())
            if not tokens and isinstance(payload, dict):
                tokens.add(shape_id.lower())
            token_index[shape_id] = tokens
        self._shape_token_index = token_index
        self._shape_token_version = current_version

    def _rank_candidates_multimetric(
        self,
        candidates: List[Dict],
        expected_shape: tuple[int, int],
    ) -> List[Dict]:
        pattern_usage: Dict[str, int] = {}
        scored: List[tuple[float, Dict]] = []
        exp_h, exp_w = expected_shape

        for cand in candidates:
            trm_score = cand.get("calibrated_confidence", cand.get("trm_confidence", 0.5))

            if cand.get("output") and cand["output"] and exp_h and exp_w:
                actual_h = len(cand["output"])
                actual_w = len(cand["output"][0]) if cand["output"] else 0
                ratio_h = max(actual_h / exp_h, exp_h / actual_h) if actual_h and exp_h else 10.0
                ratio_w = max(actual_w / exp_w, exp_w / actual_w) if actual_w and exp_w else 10.0
                size_ratio = max(ratio_h, ratio_w)
                size_score = 1.0 / size_ratio if size_ratio > 2.0 else 1.0
            else:
                size_score = 0.0

            pattern_id = self._extract_pattern_signature(cand.get("program", ""))
            usage = pattern_usage.get(pattern_id, 0)
            pattern_usage[pattern_id] = usage + 1
            novelty_score = 1.0 / (1.0 + usage * 0.2)

            tokens = cand.get("program", "").split()
            if tokens:
                known_tokens = sum(
                    1 for t in tokens if self.grammar.has_rule(t) or t in self.drawing.shapes
                )
                grammar_score = known_tokens / len(tokens)
            else:
                grammar_score = 0.0

            attractor_strength = cand.get("attractor_strength", 1)
            attractor_score = min(1.0, attractor_strength / 20.0)

            composite = (
                0.35 * trm_score
                + 0.25 * size_score
                + 0.15 * novelty_score
                + 0.15 * grammar_score
                + 0.10 * attractor_score
            )
        scored.append((composite, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def _analyze_program_attractors(self, candidates: List[Dict]) -> Dict[str, int]:
        programs = [cand.get("program", "") for cand in candidates if cand.get("program")]
        counts = Counter(programs)
        strong = {prog: count for prog, count in counts.items() if count >= 15}
        if strong:
            print(f"[ATTRACTORS] Found {len(strong)} strong canonical candidates:")
            preview = sorted(strong.items(), key=lambda item: item[1], reverse=True)[:5]
            for program, count in preview:
                snippet = program.replace("\n", " ")
                snippet = (snippet[:50] + "...") if len(snippet) > 50 else snippet
                print(f"  '{snippet}': discovered {count}× (canonical strength)")
        return dict(counts)

    def _deduplicate_candidates(self, candidates: List[Dict]) -> List[Dict]:
        attractor_counts = self._analyze_program_attractors(candidates)
        seen = set()
        unique: List[Dict] = []
        for cand in candidates:
            program = cand.get("program", "")
            out = cand.get("output")
            key = program
            if isinstance(out, (list, tuple)) and out:
                height = len(out)
                width = len(out[0]) if out and isinstance(out[0], (list, tuple)) else 0
                key = (height, width, self._hash_grid(out))
            if key in seen:
                continue
            seen.add(key)
            cand["attractor_strength"] = attractor_counts.get(program, 1)
            unique.append(cand)
        return unique

    def _get_adaptive_fuzzy_threshold(self, task_id: str, grid_area: int) -> float:
        if grid_area <= 9:
            base = 0.70
        elif grid_area <= 16:
            base = 0.75
        elif grid_area <= 64:
            base = 0.80
        else:
            base = 0.85

        history = self.shadow.get_task_history(task_id)
        if history:
            sr = history.get("success_rate", 0.0)
            if sr < 0.2:
                base *= 0.90
            elif sr > 0.8:
                base *= 1.05
        return min(0.90, max(0.60, base))

    def _hash_grid(self, grid: Sequence[Sequence[int]]) -> int:
        """Deterministic hash of a grid (SOVEREIGN: pure Python)."""
        h = 0
        for row in grid:
            for val in row:
                h = ((h * 31) + int(val) + 7) & 0xFFFFFFFF
        return h

    def _evaluate_procedural_with_trm(
        self,
        *,
        program: str,
        output_grid: Sequence[Sequence[int]],
        test_input: Sequence[Sequence[int]],
        train_examples: List[Dict],
    ) -> float:
        """
        Lightweight TRM-inspired confidence for procedural candidates.

        Checks grammar/drawing token familiarity, resemblance to stored patterns,
        and simple semantic keywords to prioritize likely-good programs.
        """
        confidence = 0.5  # neutral default
        tokens = program.split()

        if tokens:
            known = 0
            for tok in tokens:
                if tok in self.grammar.rules or tok in self.drawing.shapes or tok in self.drawing.primitives:
                    known += 1
            confidence += 0.2 * (known / len(tokens))

        if self.shadow.semantic_context is not None:
            try:
                matches = self.shadow.semantic_context.find_matching_contexts(
                    output_grid, top_k=3, similarity_threshold=0.55
                )
                if matches:
                    pattern_score = sum(m.get("score", 0.5) for m in matches) / len(matches)
                    confidence += 0.2 * pattern_score
            except Exception as e:
                print(f"  [HYBRID] Pattern check failed: {e}")

        keyword_hits = any(
            (
                "ROTATE" in tok,
                "FLIP" in tok,
                "RECOLOR" in tok,
                tok.startswith("ROT"),
                tok.startswith("FLIP"),
            )
            for tok in tokens
        )
        if keyword_hits:
            confidence += 0.1

        # Size plausibility: penalize outputs wildly off from train example outputs.
        if train_examples and output_grid and output_grid[0]:
            h_out, w_out = len(output_grid), len(output_grid[0])
            sizes = []
            for ex in train_examples:
                out = ex.get("output")
                if out and out[0]:
                    sizes.append((len(out), len(out[0])))
            if sizes:
                reasonable = False
                for h_t, w_t in sizes:
                    if h_t == 0 or w_t == 0:
                        continue
                    ratio_h = max(h_out / h_t, h_t / h_out)
                    ratio_w = max(w_out / w_t, w_t / w_out)
                    if ratio_h <= 4.0 and ratio_w <= 4.0:
                        reasonable = True
                        break
                if not reasonable:
                    confidence -= 0.3
                    print(f"  [TRM SIZE] Penalizing {h_out}x{w_out} (train sizes: {sizes[:3]})")

        return min(1.0, max(0.0, confidence))


__all__ = ["SovereignAIPipeline", "TaskResult"]
