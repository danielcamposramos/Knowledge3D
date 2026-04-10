"""Archived transitional ARC-3 local harness.

The live ARC-3 path is `GAME_2D -> Tablet/WINE -> Knowledgeverse.execute_task()`.
This Python-only local harness was copied to `Old_Attempts/benchmarks/arc3_local.py`
and is no longer part of the active benchmark surface.
"""

from __future__ import annotations

from typing import Any


ARCHIVED_TRANSITIONAL_SURFACE = True
ARCHIVED_PATH = "Old_Attempts/benchmarks/arc3_local.py"
ARCHIVE_REASON = "arc3_local_archived_use_arc3_sdk_agent_or_tablet_boundary"


ACTION_NAMES = ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6", "ACTION7"]
ACTION_LABELS = ["Move Up", "Move Down", "Move Left", "Move Right", "Perform", "Click", "Undo"]


def _archived(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(ARCHIVE_REASON)


make_task = _archived
apply_action = _archived
grids_equal = _archived
run_game = _archived
run_local_arc3 = _archived


__all__ = [
    "ACTION_LABELS",
    "ACTION_NAMES",
    "ARCHIVED_PATH",
    "ARCHIVED_TRANSITIONAL_SURFACE",
    "ARCHIVE_REASON",
    "apply_action",
    "grids_equal",
    "make_task",
    "run_game",
    "run_local_arc3",
]
