"""Compose foundational House rooms, props, and seed stars."""

from __future__ import annotations

from typing import Any

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge, MeshRenderResult

from .galaxy_manager import GalaxyManager
from .house_books import HOUSE_BOOKS
from .house_doors import HOUSE_DOORS
from .house_furniture import HOUSE_FURNITURE
from .house_gallery_displays import GALLERY_DISPLAYS
from .house_knowledge_tree import KNOWLEDGE_TREE_BRANCHES
from .house_memory_tablet import MEMORY_TABLET
from .house_observatory import OBSERVATORY_INSTRUMENTS
from .house_rooms import HOUSE_ROOMS
from .house_workshop_tools import WORKSHOP_TOOLS
from .seed_stars import SEED_STARS


def build_house(manager: GalaxyManager, *, galaxy_name: str = "House") -> dict[str, Any]:
    """Compose room templates, props, and seed stars into a House galaxy."""
    bridge = MeshBridge()
    room_meshes: dict[str, MeshRenderResult] = {}
    furniture_meshes: dict[str, MeshRenderResult] = {}
    door_meshes: dict[str, MeshRenderResult] = {}
    tool_meshes: dict[str, MeshRenderResult] = {}
    book_meshes: dict[str, MeshRenderResult] = {}
    display_meshes: dict[str, MeshRenderResult] = {}
    instrument_meshes: dict[str, MeshRenderResult] = {}
    tablet_meshes: dict[str, MeshRenderResult] = {}
    tree_meshes: dict[str, MeshRenderResult] = {}

    with manager.bulk_disk_sync():
        for room in HOUSE_ROOMS:
            manager.store_meaning_star(galaxy_name, room)
        for furniture in HOUSE_FURNITURE:
            manager.store_meaning_star(galaxy_name, furniture)
        for door in HOUSE_DOORS:
            manager.store_meaning_star(galaxy_name, door)
        for tool in WORKSHOP_TOOLS:
            manager.store_meaning_star(galaxy_name, tool)
        for book in HOUSE_BOOKS:
            manager.store_meaning_star(galaxy_name, book)
        for display in GALLERY_DISPLAYS:
            manager.store_meaning_star(galaxy_name, display)
        for instrument in OBSERVATORY_INSTRUMENTS:
            manager.store_meaning_star(galaxy_name, instrument)
        manager.store_meaning_star(galaxy_name, MEMORY_TABLET)
        for node in KNOWLEDGE_TREE_BRANCHES:
            manager.store_meaning_star(galaxy_name, node)
        for star in SEED_STARS:
            manager.store_meaning_star(galaxy_name, star)

    for room in HOUSE_ROOMS:
        if room.visual_rpn:
            room_meshes[room.star_id] = bridge.execute_rpn_program(room.visual_rpn)
    for furniture in HOUSE_FURNITURE:
        if furniture.visual_rpn:
            furniture_meshes[furniture.star_id] = bridge.execute_rpn_program(furniture.visual_rpn)
    for door in HOUSE_DOORS:
        if door.visual_rpn:
            door_meshes[door.star_id] = bridge.execute_rpn_program(door.visual_rpn)
    for tool in WORKSHOP_TOOLS:
        if tool.visual_rpn:
            tool_meshes[tool.star_id] = bridge.execute_rpn_program(tool.visual_rpn)
    for book in HOUSE_BOOKS:
        if book.visual_rpn:
            book_meshes[book.star_id] = bridge.execute_rpn_program(book.visual_rpn)
    for display in GALLERY_DISPLAYS:
        if display.visual_rpn:
            display_meshes[display.star_id] = bridge.execute_rpn_program(display.visual_rpn)
    for instrument in OBSERVATORY_INSTRUMENTS:
        if instrument.visual_rpn:
            instrument_meshes[instrument.star_id] = bridge.execute_rpn_program(instrument.visual_rpn)
    if MEMORY_TABLET.visual_rpn:
        tablet_meshes[MEMORY_TABLET.star_id] = bridge.execute_rpn_program(MEMORY_TABLET.visual_rpn)
    for node in KNOWLEDGE_TREE_BRANCHES:
        if node.visual_rpn:
            tree_meshes[node.star_id] = bridge.execute_rpn_program(node.visual_rpn)

    return {
        "galaxy": galaxy_name,
        "rooms": len(HOUSE_ROOMS),
        "furniture": len(HOUSE_FURNITURE),
        "doors": len(HOUSE_DOORS),
        "tools": len(WORKSHOP_TOOLS),
        "books": len(HOUSE_BOOKS),
        "displays": len(GALLERY_DISPLAYS),
        "instruments": len(OBSERVATORY_INSTRUMENTS),
        "tablet": 1,
        "tree_nodes": len(KNOWLEDGE_TREE_BRANCHES),
        "seed_stars": len(SEED_STARS),
        "room_meshes": room_meshes,
        "furniture_meshes": furniture_meshes,
        "door_meshes": door_meshes,
        "tool_meshes": tool_meshes,
        "book_meshes": book_meshes,
        "display_meshes": display_meshes,
        "instrument_meshes": instrument_meshes,
        "tablet_meshes": tablet_meshes,
        "tree_meshes": tree_meshes,
    }


__all__ = ["build_house"]
