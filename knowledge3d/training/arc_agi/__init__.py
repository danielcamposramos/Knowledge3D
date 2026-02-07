"""
Compatibility shim for deprecated arc_agi modules.

This keeps legacy imports working without restoring duplicate code into the
active tree. The implementation lives under Old_Attempts and is imported
via wrapper modules in this package.
"""

from __future__ import annotations

import importlib
from pathlib import Path
import sys


_ARC_ROOT = (
    Path(__file__).resolve().parents[3]
    / "Old_Attempts"
    / "curriculum_specific_training"
    / "arc_agi"
)
if _ARC_ROOT.is_dir():
    arc_path = str(_ARC_ROOT)
    if arc_path not in sys.path:
        sys.path.insert(0, arc_path)
    if arc_path not in __path__:
        __path__.append(arc_path)


_LEGACY_EXPORTS = {
    "CandidateGenerator": "candidate_generator",
    "HybridCandidateGenerator": "hybrid_generator",
    "ProgramComposer": "program_composer",
    "DualShadowCopy": "dual_shadow_copy",
    "SovereignAIPipeline": "sovereign_pipeline",
}


def __getattr__(name: str):
    module_name = _LEGACY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


__all__ = sorted(_LEGACY_EXPORTS.keys())
