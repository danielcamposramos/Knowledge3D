from __future__ import annotations

import importlib
from typing import Any, Tuple

__all__ = [
    "ARCGridProcessor",
    "CandidateGenerator",
    "ParallelCandidateGenerator",
    "CompositionalCandidateGenerator",
    "DrawingGalaxy",
    "SovereignTRMRouter",
    "RoutingCandidate",
    "ProgramComposer",
    "DualShadowCopy",
    "SovereignAIPipeline",
    "TaskResult",
]

_LAZY_IMPORTS: dict[str, Tuple[str, str]] = {
    "ARCGridProcessor": ("knowledge3d.training.arc_agi.grid_processor", "ARCGridProcessor"),
    "CandidateGenerator": ("knowledge3d.training.arc_agi.candidate_generator", "CandidateGenerator"),
    "ParallelCandidateGenerator": ("knowledge3d.training.arc_agi.parallel_generator", "ParallelCandidateGenerator"),
    "CompositionalCandidateGenerator": ("knowledge3d.training.arc_agi.compositional_generator", "CompositionalCandidateGenerator"),
    "DrawingGalaxy": ("knowledge3d.training.arc_agi.drawing_galaxy", "DrawingGalaxy"),
    "SovereignTRMRouter": ("knowledge3d.training.arc_agi.sovereign_trm_router", "SovereignTRMRouter"),
    "RoutingCandidate": ("knowledge3d.training.arc_agi.sovereign_trm_router", "RoutingCandidate"),
    "ProgramComposer": ("knowledge3d.training.arc_agi.program_composer", "ProgramComposer"),
    "DualShadowCopy": ("knowledge3d.training.arc_agi.dual_shadow_copy", "DualShadowCopy"),
    "SovereignAIPipeline": ("knowledge3d.training.arc_agi.sovereign_pipeline", "SovereignAIPipeline"),
    "TaskResult": ("knowledge3d.training.arc_agi.sovereign_pipeline", "TaskResult"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        module_name, attr = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__} has no attribute {name}")
