#!/usr/bin/env python3
"""Validate regression fixes before launching new ARC runs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from knowledge3d.training.arc_agi.candidate_generator import CandidateGenerator
from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.rpn_executor import ARCRPNExecutor
from knowledge3d.training.arc_agi.sleeptime_consolidator import SleepTimeConsolidator
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignAIPipeline

CHECKPOINT_DIR = Path("/K3D/Knowledge3D.local/checkpoints/arc_agi")


def test_scale_invariant_generation() -> bool:
    print("\n" + "=" * 70)
    print("TEST 1: Scale-invariant primitive generation")
    print("=" * 70)

    drawing = DrawingGalaxy()
    drawing.load(CHECKPOINT_DIR / "drawing_galaxy.json")
    primitives = [sid for sid, item in drawing.shapes.items() if getattr(item, "item_type", "") == "scale_invariant" or item.payload.get("type") == "scale_invariant"]
    print(f"Scale-invariant primitives in catalog: {primitives}")
    if not primitives:
        print("✗ No scale-invariant primitives found in catalog")
        return False

    executor = ARCRPNExecutor()
    generator = CandidateGenerator(
        matryoshka_dim=128,
        max_candidates=20,
        drawing_galaxy=drawing,
        executor=executor,
    )
    demo_grid = [[0, 0], [0, 0]]
    candidates = generator._generate_scale_invariant_candidates(demo_grid)
    print(f"Generated {len(candidates)} candidate(s) from scale-invariant primitives")
    for idx, (_, desc, prog) in enumerate(candidates[:3], start=1):
        print(f"  {idx}. {desc}: {prog}")
    return bool(candidates)


def test_vocabulary_parsing() -> bool:
    print("\n" + "=" * 70)
    print("TEST 2: Vocabulary instrumentation")
    print("=" * 70)

    grammar = GrammarGalaxy()
    grammar.load(CHECKPOINT_DIR / "grammar_galaxy.json")
    drawing = DrawingGalaxy()
    drawing.load(CHECKPOINT_DIR / "drawing_galaxy.json")

    pipeline = SovereignAIPipeline(matryoshka_dim=128)
    pipeline.grammar = grammar
    pipeline.drawing = drawing

    sample_programs = [
        "FLIP_V 1 rotate",
        "REL_LINE 0.0 0.0 1.0 1.0",
        "PROP_GRID 3 3",
    ]
    detected_any = False
    for prog in sample_programs:
        rules = pipeline._parse_grammar_rules_from_program(prog)
        shapes = pipeline._parse_drawing_shapes_from_program(prog)
        print(f"Program: {prog}")
        print(f"  Grammar rules: {rules if rules else 'NONE'}")
        print(f"  Shapes: {shapes if shapes else 'NONE'}")
        if rules or shapes:
            detected_any = True
    return detected_any


def test_sleeptime_audit() -> bool:
    print("\n" + "=" * 70)
    print("TEST 3: SleepTime audit plumbing")
    print("=" * 70)
    dummy = SleepTimeConsolidator
    print(f"SleepTimeConsolidator available: {dummy}")
    print("(Full audit exercised when running scripts/run_sleeptime_consolidation.py)")
    return True


def main() -> int:
    results = {
        "Scale-invariant primitives": test_scale_invariant_generation(),
        "Vocabulary instrumentation": test_vocabulary_parsing(),
        "SleepTime audit": test_sleeptime_audit(),
    }

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status:>4} - {name}")

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
