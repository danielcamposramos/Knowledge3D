"""Single bootstrap point for always-on foundational Knowledgeverse galaxies.

This keeps foundational knowledge available as one system image instead of
scattered per-task setup paths.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

from .foundational_operations_bootstrap import populate_foundational_operations
from .objects_3d_galaxy import default_3d_objects_entries
from .reality_galaxy import default_reality_entries
from .tool_galaxy import default_tool_entries


def _existing_ids(entries: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", "")).strip()
        if entry_id:
            out.add(entry_id)
    return out


def _populate_galaxy_entries(
    galaxy_manager: Any,
    *,
    galaxy_name: str,
    entry_builder: Callable[[], list[dict[str, Any]]],
) -> dict[str, int]:
    galaxy = galaxy_manager.get_galaxy(galaxy_name)
    current_entries = list(getattr(galaxy, "entries", []))
    existing_ids = _existing_ids(current_entries)

    inserted = 0
    generated = entry_builder()
    for entry in generated:
        entry_id = str(entry.get("id", "")).strip()
        if not entry_id or entry_id in existing_ids:
            continue
        galaxy_manager.add_entry(galaxy_name, entry)
        existing_ids.add(entry_id)
        inserted += 1

    return {
        "before": len(current_entries),
        "generated": len(generated),
        "inserted": inserted,
        "after": len(existing_ids),
    }


def populate_always_on_foundational_galaxies(galaxy_manager: Any) -> dict[str, Any]:
    """Populate all foundational deterministic galaxies through one entrypoint."""
    sync_context = (
        galaxy_manager.bulk_disk_sync()
        if hasattr(galaxy_manager, "bulk_disk_sync")
        else nullcontext()
    )
    with sync_context:
        return {
            "operations": populate_foundational_operations(galaxy_manager),
            "reality": _populate_galaxy_entries(
                galaxy_manager,
                galaxy_name="Reality",
                entry_builder=default_reality_entries,
            ),
            "objects_3d": _populate_galaxy_entries(
                galaxy_manager,
                galaxy_name="3DObjects",
                entry_builder=default_3d_objects_entries,
            ),
            "tool": _populate_galaxy_entries(
                galaxy_manager,
                galaxy_name="Tool",
                entry_builder=default_tool_entries,
            ),
        }
