# DEPRECATED: placeholder until swarm delivers the refactored evaluator.
"""
Lightweight stub for the legacy math bench evaluator.

The original implementation lived under ``knowledge3d.tools.phase25``.  During
the tool reorganisation the full evaluator has not yet been ported, so this
module provides a tiny shim that keeps dependent scripts functional while
returning a descriptive payload.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional


def run(repos: Optional[Iterable[str]] = None, limit: int = 5) -> Dict[str, object]:
    """
    Return a stub report noting that the evaluator is pending migration.
    """
    return {
        "status": "deprecated",
        "message": "math_bench_evaluator pending refactor; no evaluation executed",
        "repos": list(repos or []),
        "limit": limit,
    }

