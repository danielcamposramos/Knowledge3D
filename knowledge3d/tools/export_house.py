"""Export the procedural House as a GLB scene."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pygltflib import Node, Scene

from knowledge3d.cranium.bridges.mesh_bridge import MeshBridge
from knowledge3d.knowledgeverse.house_books import HOUSE_BOOKS
from knowledge3d.knowledgeverse.house_doors import HOUSE_DOORS
from knowledge3d.knowledgeverse.house_furniture import HOUSE_FURNITURE
from knowledge3d.knowledgeverse.house_gallery_displays import GALLERY_DISPLAYS
from knowledge3d.knowledgeverse.house_knowledge_tree import KNOWLEDGE_TREE_BRANCHES
from knowledge3d.knowledgeverse.house_memory_tablet import MEMORY_TABLET
from knowledge3d.knowledgeverse.house_nav_graph import build_house_nav_graph
from knowledge3d.knowledgeverse.house_observatory import OBSERVATORY_INSTRUMENTS
from knowledge3d.knowledgeverse.house_rooms import HOUSE_ROOMS
from knowledge3d.knowledgeverse.house_workshop_tools import WORKSHOP_TOOLS
from knowledge3d.knowledgeverse.meaning_star import MeaningCentricStar
from knowledge3d.tools.gltf_export import compose_scene, mesh_to_gltf_node


HOUSE_ROOT_PARENT = "__HOUSE_ROOT__"


def _surface_forms_payload(star: MeaningCentricStar) -> dict[str, dict[str, Any]]:
    return {
        language: surface_form.to_dict()
        for language, surface_form in star.surface_forms.items()
    }


def _star_k3d_payload(star: MeaningCentricStar, *, backend: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "star_id": star.star_id,
        "meaning_class": star.meaning_class,
        "domain": star.domain,
        "house_room": star.house_room,
        "house_position": [float(value) for value in star.house_position],
        "surface_forms": _surface_forms_payload(star),
        "behavior_rpn": star.behavior_rpn,
        "component_refs": list(star.component_refs),
        "taxonomy_refs": list(star.taxonomy_refs),
        "backend": backend,
    }
    if star.galaxy_ref:
        payload["galaxy_ref"] = star.galaxy_ref
    if star.visual_rpn:
        payload["visual_rpn"] = star.visual_rpn
    return payload


def _local_translation(star: MeaningCentricStar) -> tuple[float, float, float]:
    return (
        float(star.house_position[0]),
        float(star.house_position[1]),
        float(star.house_position[2]),
    )


def export_house_glb(
    output_path: Path,
    *,
    include_books: bool = True,
    include_tree: bool = True,
) -> dict[str, Any]:
    """Build and export the full House as a GLB file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bridge = MeshBridge()
    room_by_house_room = {room.house_room: room for room in HOUSE_ROOMS}

    ordered_nodes: list[tuple[str, str | None, Any]] = []

    for room in HOUSE_ROOMS:
        result = bridge.execute_rpn_program(room.visual_rpn or "")
        ordered_nodes.append(
            (
                room.star_id,
                None,
                mesh_to_gltf_node(
                    result.mesh,
                    name=room.star_id,
                    translation=_local_translation(room),
                    extras=_star_k3d_payload(room, backend=result.backend),
                ),
            )
        )

    for furniture in HOUSE_FURNITURE:
        result = bridge.execute_rpn_program(furniture.visual_rpn or "")
        parent = room_by_house_room.get(furniture.house_room)
        ordered_nodes.append(
            (
                furniture.star_id,
                parent.star_id if parent is not None else None,
                mesh_to_gltf_node(
                    result.mesh,
                    name=furniture.star_id,
                    translation=_local_translation(furniture),
                    extras=_star_k3d_payload(furniture, backend=result.backend),
                ),
            )
        )

    for door in HOUSE_DOORS:
        result = bridge.execute_rpn_program(door.visual_rpn or "")
        parent = room_by_house_room.get(door.house_room)
        ordered_nodes.append(
            (
                door.star_id,
                parent.star_id if parent is not None else None,
                mesh_to_gltf_node(
                    result.mesh,
                    name=door.star_id,
                    translation=_local_translation(door),
                    extras=_star_k3d_payload(door, backend=result.backend),
                ),
            )
        )

    for tool in WORKSHOP_TOOLS:
        result = bridge.execute_rpn_program(tool.visual_rpn or "")
        parent = room_by_house_room.get(tool.house_room)
        ordered_nodes.append(
            (
                tool.star_id,
                parent.star_id if parent is not None else None,
                mesh_to_gltf_node(
                    result.mesh,
                    name=tool.star_id,
                    translation=_local_translation(tool),
                    extras=_star_k3d_payload(tool, backend=result.backend),
                ),
            )
        )

    if include_books:
        for book in HOUSE_BOOKS:
            result = bridge.execute_rpn_program(book.visual_rpn or "")
            parent = room_by_house_room.get(book.house_room)
            ordered_nodes.append(
                (
                    book.star_id,
                    parent.star_id if parent is not None else None,
                    mesh_to_gltf_node(
                        result.mesh,
                        name=book.star_id,
                        translation=_local_translation(book),
                        extras=_star_k3d_payload(book, backend=result.backend),
                    ),
                ),
            )

    for display in GALLERY_DISPLAYS:
        result = bridge.execute_rpn_program(display.visual_rpn or "")
        parent = room_by_house_room.get(display.house_room)
        ordered_nodes.append(
            (
                display.star_id,
                parent.star_id if parent is not None else None,
                mesh_to_gltf_node(
                    result.mesh,
                    name=display.star_id,
                    translation=_local_translation(display),
                    extras=_star_k3d_payload(display, backend=result.backend),
                ),
            )
        )

    if include_tree:
        for node in KNOWLEDGE_TREE_BRANCHES:
            result = bridge.execute_rpn_program(node.visual_rpn or "")
            parent = room_by_house_room.get(node.house_room)
            ordered_nodes.append(
                (
                    node.star_id,
                    parent.star_id if parent is not None else None,
                    mesh_to_gltf_node(
                        result.mesh,
                        name=node.star_id,
                        translation=_local_translation(node),
                        extras=_star_k3d_payload(node, backend=result.backend),
                    ),
                )
            )

    for instrument in OBSERVATORY_INSTRUMENTS:
        result = bridge.execute_rpn_program(instrument.visual_rpn or "")
        parent = room_by_house_room.get(instrument.house_room)
        ordered_nodes.append(
            (
                instrument.star_id,
                parent.star_id if parent is not None else None,
                mesh_to_gltf_node(
                    result.mesh,
                    name=instrument.star_id,
                    translation=_local_translation(instrument),
                    extras=_star_k3d_payload(instrument, backend=result.backend),
                ),
            )
        )

    tablet_result = bridge.execute_rpn_program(MEMORY_TABLET.visual_rpn or "")
    ordered_nodes.append(
        (
            MEMORY_TABLET.star_id,
            HOUSE_ROOT_PARENT,
            mesh_to_gltf_node(
                tablet_result.mesh,
                name=MEMORY_TABLET.star_id,
                translation=_local_translation(MEMORY_TABLET),
                extras=_star_k3d_payload(MEMORY_TABLET, backend=tablet_result.backend),
            ),
        )
    )

    gltf = compose_scene([data for _, _, data in ordered_nodes], asset_generator="Knowledge3D House Export")
    nav_graph = build_house_nav_graph()

    node_index_by_star_id = {
        star_id: index
        for index, (star_id, _, _) in enumerate(ordered_nodes)
    }

    room_indices = {room.star_id: node_index_by_star_id[room.star_id] for room in HOUSE_ROOMS}
    for star_id, parent_star_id, _ in ordered_nodes:
        if parent_star_id is None:
            continue
        if parent_star_id == HOUSE_ROOT_PARENT:
            continue
        child_index = node_index_by_star_id[star_id]
        parent_index = node_index_by_star_id[parent_star_id]
        parent_node = gltf.nodes[parent_index]
        children = list(parent_node.children or [])
        children.append(child_index)
        parent_node.children = children

    root_children = [room_indices[room.star_id] for room in HOUSE_ROOMS]
    for star_id, parent_star_id, _ in ordered_nodes:
        if parent_star_id != HOUSE_ROOT_PARENT:
            continue
        root_children.append(node_index_by_star_id[star_id])

    root_index = len(gltf.nodes)
    gltf.nodes.append(
        Node(
            name="House",
            children=root_children,
            extras={
                "k3d": {
                    "star_id": "house_root",
                    "meaning_class": "house",
                    "domain": "House",
                    "surface_forms": {
                        "en": {"word_ref": "house", "char_refs": ["char_h", "char_o", "char_u", "char_s", "char_e"]},
                    },
                    "nav_graph": nav_graph.to_metadata(),
                }
            },
        )
    )
    gltf.scenes = [Scene(nodes=[root_index])]
    gltf.scene = 0
    gltf.save_binary(str(output_path))

    total_vertices = 0
    total_triangles = 0
    for _, _, data in ordered_nodes:
        mesh_index = int(data.node.mesh or 0)
        primitive = list(data.mesh.primitives or [None])[0]
        if primitive is None:
            continue
        pos_accessor = data.accessors[0]
        idx_accessor = data.accessors[3]
        total_vertices += int(pos_accessor.count or 0)
        total_triangles += int(int(idx_accessor.count or 0) / 3)

    return {
        "rooms": len(HOUSE_ROOMS),
        "furniture": len(HOUSE_FURNITURE),
        "doors": len(HOUSE_DOORS),
        "tools": len(WORKSHOP_TOOLS),
        "books": len(HOUSE_BOOKS) if include_books else 0,
        "displays": len(GALLERY_DISPLAYS),
        "instruments": len(OBSERVATORY_INSTRUMENTS),
        "tablet": 1,
        "tree_nodes": len(KNOWLEDGE_TREE_BRANCHES) if include_tree else 0,
        "nodes": len(gltf.nodes),
        "total_vertices": total_vertices,
        "total_triangles": total_triangles,
        "file_size_kb": output_path.stat().st_size / 1024.0,
        "output": str(output_path),
    }


__all__ = ["export_house_glb"]
