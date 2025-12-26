from __future__ import annotations

"""
Math benchmarks package.

Keep this module import-light: benchmark tooling is often imported in CPU-only
contexts (unit tests, docs builds), while some submodules may eagerly load GPU
bridges/galaxies. We therefore export common symbols via lazy `__getattr__`.
"""

from typing import Any

_EXPORTS = {
    "MathProceduralizer": ("knowledge3d.training.math_benchmarks.math_proceduralizer", "MathProceduralizer"),
    "MathDatasetLoader": ("knowledge3d.training.math_benchmarks.math_proceduralizer", "MathDatasetLoader"),
    "MathOutputAdapter": ("knowledge3d.training.math_benchmarks.math_output_adapter", "MathOutputAdapter"),
    "MathBenchmarkEvaluator": ("knowledge3d.training.math_benchmarks.benchmark_evaluator", "MathBenchmarkEvaluator"),
    "SovereignMathPipeline": ("knowledge3d.training.math_benchmarks.sovereign_math_pipeline", "SovereignMathPipeline"),
    "train_multimodal": ("knowledge3d.training.math_benchmarks.sovereign_math_pipeline", "train_multimodal"),
    "WordProblemSolver": ("knowledge3d.training.math_benchmarks.word_problem_solver", "WordProblemSolver"),
    "SovereignComposer": ("knowledge3d.training.math_benchmarks.sovereign_composer", "SovereignComposer"),
    "UnifiedGalaxyLoader": ("knowledge3d.training.math_benchmarks.galaxy_loader", "UnifiedGalaxyLoader"),
    "UNIFIED_GALAXY": ("knowledge3d.training.math_benchmarks.galaxy_loader", "UNIFIED_GALAXY"),
    "SymbolRegistry": ("knowledge3d.training.math_benchmarks.symbol_registry", "SymbolRegistry"),
    "SYMBOL_REGISTRY": ("knowledge3d.training.math_benchmarks.symbol_registry", "SYMBOL_REGISTRY"),
    "populate_registry_from_galaxy": ("knowledge3d.training.math_benchmarks.symbol_registry", "populate_registry_from_galaxy"),
    "BookGalaxyIngester": ("knowledge3d.training.math_benchmarks.book_galaxy_ingestion", "BookGalaxyIngester"),
    "BookGalaxyLibrary": ("knowledge3d.training.math_benchmarks.book_galaxy_library", "BookGalaxyLibrary"),
    "BookGalaxyHit": ("knowledge3d.training.math_benchmarks.book_galaxy_library", "BookGalaxyHit"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name in _EXPORTS:
        mod_path, attr = _EXPORTS[name]
        module = __import__(mod_path, fromlist=[attr])
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = sorted(_EXPORTS.keys())
