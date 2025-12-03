#!/usr/bin/env python3
"""Run SleepTime consolidation on ARC-AGI checkpoints."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_DIR = Path("/K3D/Knowledge3D.local/checkpoints/arc_agi")
DRAWING_CHECKPOINT = CHECKPOINT_DIR / "drawing_galaxy.json"
GRAMMAR_CHECKPOINT = CHECKPOINT_DIR / "grammar_galaxy.json"
SHADOW_CHECKPOINT = CHECKPOINT_DIR / "shadow_copy.json"

from knowledge3d.training.arc_agi.drawing_galaxy import DrawingGalaxy
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy
from knowledge3d.training.arc_agi.dual_shadow_copy import DualShadowCopy
from knowledge3d.training.arc_agi.sleeptime_consolidator import SleepTimeConsolidator


def main() -> None:
    print("[SLEEPTIME] Loading current state from checkpoints...")
    drawing_galaxy = DrawingGalaxy()
    grammar_galaxy = GrammarGalaxy()
    shadow_copy = DualShadowCopy(drawing_galaxy, grammar_galaxy, staged=False)

    drawing_galaxy.load(DRAWING_CHECKPOINT)
    grammar_galaxy.load(GRAMMAR_CHECKPOINT)
    shadow_copy.load(SHADOW_CHECKPOINT)

    print(f"  Drawing shapes: {len(drawing_galaxy.shapes)}")
    print(f"  Grammar rules: {len(grammar_galaxy.rules)}")
    print(f"  Shadow entries: {len(shadow_copy.library)}")

    print("\n[SLEEPTIME] Running consolidation cycle...")
    consolidator = SleepTimeConsolidator(
        shadow_copy,
        drawing_galaxy,
        grammar_galaxy,
        min_quality=0.6,
        min_uses_for_canonical=5,
        canonical_success_threshold=0.7,
    )
    stats = consolidator.consolidate()

    print("\n[SLEEPTIME] Saving consolidated checkpoints...")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    drawing_galaxy.save(DRAWING_CHECKPOINT)
    grammar_galaxy.save(GRAMMAR_CHECKPOINT)
    shadow_copy.save(SHADOW_CHECKPOINT)

    report_name = f"consolidation_report_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
    report_path = CHECKPOINT_DIR / report_name
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, indent=2)

    print(f"\n[SLEEPTIME] Consolidation complete. Report saved to: {report_path}")
    print("  Ready for Run 035.")


if __name__ == "__main__":
    main()
