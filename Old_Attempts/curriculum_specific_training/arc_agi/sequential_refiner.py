"""
Sequential refinement module (TRM-style) using Shadow Copy patterns.

Implements k3d_sequential_refine() with ternary gating and per-task math core
spawning. Keeps orchestration light and sovereign (ctypes + PTX only in the
executor path).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from knowledge3d.cranium.ptx_runtime.math_core_pool import MathCorePool, get_global_math_core_pool
from knowledge3d.training.arc_agi.candidate_generator import Candidate
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor


def _ternary_sign(x: float) -> int:
    """Balanced ternary sign with small deadband."""
    if x > 0.05:
        return 1
    if x < -0.05:
        return -1
    return 0


def _should_stop(confidence_chain: List[int]) -> bool:
    """Stop if last two confidence signals are plateau (0,0) or declining (-1,-1)."""
    if len(confidence_chain) < 2:
        return False
    last_two = confidence_chain[-2:]
    return last_two == [0, 0] or last_two == [-1, -1]


def _categorize_patterns_by_tier(library: List[Dict]) -> Dict[str, List[Dict]]:
    """Deprecated: retained for backward compatibility (no-op wrapper)."""
    categorized = {"tier1": [], "tier2": [], "tier3": []}
    for entry in library:
        tier = entry.get("tier", 1)
        bucket = f"tier{tier}"
        if bucket not in categorized:
            bucket = "tier1"
        categorized[bucket].append(entry)
    return categorized


def _is_improvement(new_grid: List[List[int]], old_grid: List[List[int]]) -> bool:
    """Return True if grids differ (any change counts as improvement)."""
    if len(new_grid) != len(old_grid):
        return True
    for r_new, r_old in zip(new_grid, old_grid):
        if len(r_new) != len(r_old):
            return True
        for a, b in zip(r_new, r_old):
            if a != b:
                return True
    return False


def _score_entry(entry: Dict) -> float:
    """Prefer opcode-aware score when present; fall back to quality_score."""
    if entry is None:
        return 0.0
    return float(entry.get("quality_score_opcode", entry.get("quality_score", 0.0)) or 0.0)


def k3d_sequential_refine(
    input_grid: Sequence[Sequence[int]],
    initial_candidate: Candidate,
    shadow_copy: DualShadowCopy,
    drawing_galaxy: Optional[DrawingGalaxy],
    grammar_galaxy: Optional[GrammarGalaxy] = None,
    executor: Optional[ARCRPNExecutor] = None,
    core_pool: Optional[MathCorePool] = None,
    n: int = 6,
    T: int = 3,
) -> Tuple[List[List[int]], List[str]]:
    """
    Apply discovered Shadow Copy patterns sequentially with ternary gating.

    Interleaves Shadow/Drawing/Grammar sources round-robin. Stops early if
    confidence plateaus or declines.
    """

    pool = core_pool or get_global_math_core_pool()
    core_id: Optional[int] = None
    local_executor = executor
    if local_executor is None:
        core_id = pool.spawn_core(tier=2, reuse=True)
        local_executor = ARCRPNExecutor(pool=pool, instance_id=core_id)

    try:
        output_grid, instruction, program = initial_candidate
        current_candidate = output_grid
        applied_patterns: List[str] = [program]

        shadow_patterns = [entry for entry in shadow_copy.library if _score_entry(entry) >= 0.60][:20]

        drawing_primitives = []
        if drawing_galaxy is not None:
            rel_ids = getattr(drawing_galaxy, "rel_shapes", [])
            prop_ids = getattr(drawing_galaxy, "prop_shapes", [])
            drawing_primitives = [drawing_galaxy.shapes[sid] for sid in list(rel_ids) + list(prop_ids) if sid in drawing_galaxy.shapes]

        grammar_rules: List[Dict] = grammar_galaxy.get_high_confidence_rules(min_score=0.70) if grammar_galaxy else []

        if not (shadow_patterns or drawing_primitives or grammar_rules):
            return current_candidate, applied_patterns

        confidence_chain: List[int] = []

        for cycle in range(T):
            cycle_improved = False
            for recursion in range(n):
                source = recursion % 3  # 0=shadow,1=drawing,2=grammar
                program_to_apply = ""
                confidence = 0.5

                if source == 0 and shadow_patterns:
                    pat = shadow_patterns[recursion % len(shadow_patterns)]
                    program_to_apply = pat.get("program", "")
                    confidence = _score_entry(pat)
                elif source == 1 and drawing_primitives:
                    prim = drawing_primitives[recursion % len(drawing_primitives)]
                    payload = getattr(prim, "payload", {}) or {}
                    program_to_apply = payload.get("procedural_programs", {}).get("composition") or payload.get("rpn_program", "")
                    confidence = 0.8
                elif source == 2 and grammar_rules:
                    rule = grammar_rules[recursion % len(grammar_rules)]
                    program_to_apply = rule.get("rpn_program", "")
                    confidence = float(rule.get("quality_score", 0.7))
                else:
                    continue

                if not program_to_apply:
                    continue

                if _ternary_sign(confidence - 0.70) <= 0:
                    continue  # gate out low-confidence patterns
                try:
                    refined_grid = local_executor.execute(current_candidate, program_to_apply)
                except Exception:
                    continue
                if refined_grid is None:
                    continue
                if _is_improvement(refined_grid, current_candidate):
                    current_candidate = refined_grid
                    applied_patterns.append(program_to_apply)
                    cycle_improved = True
                    source_label = ["shadow", "drawing", "grammar"][source]
                    print(f"    [REFINER] Applied pattern from {source_label} (confidence={confidence:.2f})")
                confidence_chain.append(_ternary_sign(confidence - 0.70))

            if not cycle_improved:
                confidence_chain.append(0)
            if not cycle_improved and _should_stop(confidence_chain):
                break

        if len(applied_patterns) > 1:
            print(f"  [REFINER] Total patterns applied: {len(applied_patterns) - 1}")
        return current_candidate, applied_patterns

    finally:
        if core_id is not None:
            pool.release_core(core_id, pool=True)


def k3d_sequential_refine_adaptive(
    input_grid: Sequence[Sequence[int]],
    initial_candidate: Candidate,
    shadow_copy: DualShadowCopy,
    drawing_galaxy: Optional[DrawingGalaxy],
    grammar_galaxy: Optional[GrammarGalaxy] = None,
    executor: Optional[ARCRPNExecutor] = None,
    core_pool: Optional[MathCorePool] = None,
    n: int = 6,
    T: int = 3,
) -> Tuple[List[List[int]], List[str]]:
    """
    Adaptive refiner that routes patterns to tier1/2/3 cores based on complexity.
    """

    pool = core_pool or get_global_math_core_pool()
    tier1 = pool.spawn_core(tier=1, reuse=True)
    tier2 = pool.spawn_core(tier=2, reuse=True)
    tier3 = pool.spawn_core(tier=3, reuse=True)

    exec_cache: Dict[int, ARCRPNExecutor] = {}

    def _exec(core_id: int, grid, program):
        local_exec = exec_cache.get(core_id)
        if local_exec is None:
            local_exec = ARCRPNExecutor(pool=pool, instance_id=core_id)
            exec_cache[core_id] = local_exec
        return local_exec.execute(grid, program)

    try:
        output_grid, instruction, program = initial_candidate
        current_candidate = output_grid
        applied_patterns: List[str] = [program]
        confidence_chain: List[int] = []

        shadow_patterns = [e for e in shadow_copy.library if _score_entry(e) >= 0.60]
        categorized: Dict[str, List[Dict]] = {"tier1": [], "tier2": [], "tier3": []}
        for entry in shadow_patterns:
            tier = entry.get("tier", 1)
            bucket = f"tier{tier}"
            if bucket not in categorized:
                bucket = "tier1"
            categorized[bucket].append(entry)

        drawing_primitives = []
        if drawing_galaxy is not None:
            rel_ids = getattr(drawing_galaxy, "rel_shapes", [])
            prop_ids = getattr(drawing_galaxy, "prop_shapes", [])
            drawing_primitives = [drawing_galaxy.shapes[sid] for sid in list(rel_ids) + list(prop_ids) if sid in drawing_galaxy.shapes]

        grammar_rules: List[Dict] = grammar_galaxy.get_high_confidence_rules(min_score=0.70) if grammar_galaxy else []

        for cycle in range(T):
            cycle_improved = False

            # Tier-1 fast probes
            for pat in categorized.get("tier1", [])[: max(1, n // 3)]:
                prog = pat.get("program", "")
                if not prog:
                    continue
                try:
                    refined = _exec(tier1, current_candidate, prog)
                except Exception:
                    continue
                if refined and _is_improvement(refined, current_candidate):
                    current_candidate = refined
                    applied_patterns.append(prog)
                    cycle_improved = True
                    confidence_chain.append(_ternary_sign(_score_entry(pat) - 0.70))
                    print(f"    [REFINER] Applied pattern from shadow (tier1) (confidence={_score_entry(pat):.2f})")
                    break

            # Tier-2 if not improved
            if not cycle_improved:
                for pat in categorized.get("tier2", [])[: max(1, n // 3)]:
                    prog = pat.get("program", "")
                    if not prog:
                        continue
                    try:
                        refined = _exec(tier2, current_candidate, prog)
                    except Exception:
                        continue
                if refined and _is_improvement(refined, current_candidate):
                    current_candidate = refined
                    applied_patterns.append(prog)
                    cycle_improved = True
                    confidence_chain.append(_ternary_sign(_score_entry(pat) - 0.70))
                    print(f"    [REFINER] Applied pattern from shadow (tier2) (confidence={_score_entry(pat):.2f})")
                    break

            # Tier-3 only if still not improved
            if not cycle_improved:
                for pat in categorized.get("tier3", [])[: max(1, n // 3)]:
                    prog = pat.get("program", "")
                    if not prog:
                        continue
                    try:
                        refined = _exec(tier3, current_candidate, prog)
                    except Exception:
                        continue
                if refined and _is_improvement(refined, current_candidate):
                    current_candidate = refined
                    applied_patterns.append(prog)
                    cycle_improved = True
                    confidence_chain.append(_ternary_sign(_score_entry(pat) - 0.70))
                    print(f"    [REFINER] Applied pattern from shadow (tier3) (confidence={_score_entry(pat):.2f})")
                    break

            # Interleave Drawing/Grammar even if shadow patterns are absent
            if not cycle_improved:
                if drawing_primitives:
                    prim = drawing_primitives[cycle % len(drawing_primitives)]
                    payload = getattr(prim, "payload", {}) or {}
                    prog = payload.get("procedural_programs", {}).get("composition") or payload.get("rpn_program", "")
                    if prog:
                        try:
                            refined = _exec(tier1, current_candidate, prog)
                            if refined and _is_improvement(refined, current_candidate):
                                current_candidate = refined
                                applied_patterns.append(prog)
                                cycle_improved = True
                                print("    [REFINER] Applied pattern from drawing (fallback)")
                        except Exception:
                            pass

                if (not cycle_improved) and grammar_rules:
                    rule = grammar_rules[cycle % len(grammar_rules)]
                    prog = rule.get("rpn_program", "")
                    if prog:
                        try:
                            refined = _exec(tier2, current_candidate, prog)
                            if refined and _is_improvement(refined, current_candidate):
                                current_candidate = refined
                                applied_patterns.append(prog)
                                cycle_improved = True
                                print("    [REFINER] Applied pattern from grammar (fallback)")
                        except Exception:
                            pass

            if not cycle_improved:
                confidence_chain.append(0)
            if not cycle_improved and _should_stop(confidence_chain):
                break

        if len(applied_patterns) > 1:
            print(f"  [REFINER] Total patterns applied: {len(applied_patterns) - 1}")
        return current_candidate, applied_patterns

    finally:
        pool.release_core(tier1, pool=True)
        pool.release_core(tier2, pool=True)
        pool.release_core(tier3, pool=True)


__all__ = [
    "k3d_sequential_refine",
    "k3d_sequential_refine_adaptive",
    "_ternary_sign",
    "_is_improvement",
    "_should_stop",
]
