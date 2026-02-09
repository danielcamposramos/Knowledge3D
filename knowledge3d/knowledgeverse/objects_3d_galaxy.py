"""Knowledgeverse 3D Objects Galaxy bootstrap (procedural, additive, idempotent)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _entry(
    *,
    entry_id: str,
    name: str,
    category: str,
    rpn_program: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_meta = {
        "source": "week19_3dobjects_bootstrap",
        "bootstrap": "week19_spatial_enabler",
        "procedural": True,
        "cross_modal": ["drawing", "math", "reality"],
    }
    if metadata:
        base_meta.update(metadata)
    return {
        "id": entry_id,
        "name": name,
        "domain": "3d_objects",
        "category": category,
        "rpn_program": rpn_program,
        "tags": tags or [],
        "metadata": base_meta,
    }


def create_mesh_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(
        [
            _entry(
                entry_id="obj3d_mesh_triangle_face",
                name="Triangle Mesh Face",
                category="meshes",
                rpn_program="V1 V2 V3 MAKE_TRI_FACE",
                tags=["mesh", "triangle"],
            ),
            _entry(
                entry_id="obj3d_mesh_quad_face",
                name="Quad Mesh Face",
                category="meshes",
                rpn_program="V1 V2 V3 V4 MAKE_QUAD_FACE",
                tags=["mesh", "quad"],
            ),
            _entry(
                entry_id="obj3d_mesh_compute_normal",
                name="Face Normal from Vertices",
                category="meshes",
                rpn_program="V2 V1 SUB V3 V1 SUB CROSS NORMALIZE",
                tags=["mesh", "normal", "vector"],
            ),
        ]
    )
    for grid in range(2, 26):
        out.append(
            _entry(
                entry_id=f"obj3d_mesh_grid_{grid}x{grid}",
                name=f"Grid Mesh {grid}x{grid}",
                category="meshes_grid",
                rpn_program=f"{grid} {grid} GENERATE_GRID_MESH",
                tags=["mesh", "grid", "procedural"],
                metadata={"generative": True},
            )
        )
    for segments in range(3, 33):
        out.append(
            _entry(
                entry_id=f"obj3d_mesh_cylinder_segments_{segments}",
                name=f"Cylinder Mesh segments={segments}",
                category="meshes_parametric",
                rpn_program=f"RADIUS HEIGHT {segments} GENERATE_CYLINDER_MESH",
                tags=["mesh", "cylinder", "procedural"],
                metadata={"generative": True},
            )
        )
    return out


def create_transformation_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(
        [
            _entry(
                entry_id="obj3d_xform_translate",
                name="3D Translation Matrix",
                category="transformations",
                rpn_program="TX TY TZ MAT4_TRANSLATE",
                tags=["transform", "translation"],
            ),
            _entry(
                entry_id="obj3d_xform_scale",
                name="3D Scale Matrix",
                category="transformations",
                rpn_program="SX SY SZ MAT4_SCALE",
                tags=["transform", "scale"],
            ),
            _entry(
                entry_id="obj3d_xform_rotate_axis_angle",
                name="Axis-Angle Rotation Matrix",
                category="transformations",
                rpn_program="AX AY AZ ANGLE MAT4_ROT_AXIS_ANGLE",
                tags=["transform", "rotation"],
            ),
            _entry(
                entry_id="obj3d_xform_apply",
                name="Apply Transform to Vertex",
                category="transformations",
                rpn_program="MAT4 VEC4 MAT4_VEC4_MUL",
                tags=["transform", "matrix", "vertex"],
            ),
        ]
    )
    for axis in ("x", "y", "z"):
        for angle in range(0, 360, 10):
            out.append(
                _entry(
                    entry_id=f"obj3d_xform_rot_{axis}_{angle}",
                    name=f"Rotate {axis.upper()} {angle}deg",
                    category="transformations_rotation",
                    rpn_program=f"{angle} DEG2RAD ROT3D_{axis.upper()}",
                    tags=["transform", "rotation", f"axis_{axis}"],
                )
            )
    for fov in range(30, 121, 5):
        out.append(
            _entry(
                entry_id=f"obj3d_xform_perspective_fov_{fov}",
                name=f"Perspective Projection FOV={fov}",
                category="transformations_projection",
                rpn_program=f"{fov} DEG2RAD ASPECT Z_NEAR Z_FAR MAT4_PERSPECTIVE",
                tags=["projection", "camera"],
            )
        )
    return out


def create_spatial_query_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(
        [
            _entry(
                entry_id="obj3d_query_point_in_triangle",
                name="Point in Triangle (Barycentric)",
                category="spatial_queries",
                rpn_program="P V1 V2 V3 BARYCENTRIC_TEST",
                tags=["query", "barycentric", "collision"],
            ),
            _entry(
                entry_id="obj3d_query_ray_triangle",
                name="Ray-Triangle Intersection",
                category="spatial_queries",
                rpn_program="RAY_O RAY_D V1 V2 V3 MOLLER_TRUMBORE",
                tags=["query", "raycast", "intersection"],
            ),
            _entry(
                entry_id="obj3d_query_aabb_overlap",
                name="AABB Overlap Test",
                category="spatial_queries",
                rpn_program="AABB1 AABB2 AABB_OVERLAP_TEST",
                tags=["query", "aabb", "collision"],
            ),
            _entry(
                entry_id="obj3d_query_nearest_vertex",
                name="Nearest Vertex Search",
                category="spatial_queries",
                rpn_program="POINT VERTICES KD_NEAREST",
                tags=["query", "nearest", "kdtree"],
            ),
        ]
    )
    for radius in range(1, 31):
        out.append(
            _entry(
                entry_id=f"obj3d_query_sphere_overlap_r_{radius}",
                name=f"Sphere Overlap radius={radius}",
                category="spatial_queries_sphere",
                rpn_program=f"C1 C2 {radius} SPHERE_OVERLAP_TEST",
                tags=["query", "sphere", "collision"],
            )
        )
    for res in range(8, 73, 4):
        out.append(
            _entry(
                entry_id=f"obj3d_query_voxel_traverse_{res}",
                name=f"Voxel Traversal res={res}",
                category="spatial_queries_voxel",
                rpn_program=f"RAY GRID_{res} VOXEL_TRAVERSE",
                tags=["query", "voxel", "raycast"],
            )
        )
    return out


def create_procedural_generation_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    out.extend(
        [
            _entry(
                entry_id="obj3d_gen_cube",
                name="Generate Cube Mesh",
                category="procedural_generation",
                rpn_program="SIZE GENERATE_CUBE_VERTICES GENERATE_CUBE_FACES",
                tags=["procedural", "mesh", "cube"],
                metadata={"generative": True},
            ),
            _entry(
                entry_id="obj3d_gen_uv_sphere",
                name="Generate UV Sphere Mesh",
                category="procedural_generation",
                rpn_program="RADIUS STACKS SLICES GENERATE_UV_SPHERE",
                tags=["procedural", "mesh", "sphere"],
                metadata={"generative": True},
            ),
            _entry(
                entry_id="obj3d_gen_lathe_profile",
                name="Generate Lathe Mesh from Profile",
                category="procedural_generation",
                rpn_program="PROFILE_CURVE SEGMENTS GENERATE_LATHE",
                tags=["procedural", "mesh", "lathe"],
                metadata={"generative": True, "cross_modal": ["drawing", "character", "math"]},
            ),
        ]
    )
    for stacks in range(4, 41, 2):
        for slices in (8, 12, 16, 20, 24):
            out.append(
                _entry(
                    entry_id=f"obj3d_gen_sphere_s{stacks}_c{slices}",
                    name=f"UV Sphere stacks={stacks} slices={slices}",
                    category="procedural_generation_sphere",
                    rpn_program=f"RADIUS {stacks} {slices} GENERATE_UV_SPHERE",
                    tags=["procedural", "sphere", "mesh"],
                    metadata={"generative": True},
                )
            )
    for depth in range(1, 31):
        out.append(
            _entry(
                entry_id=f"obj3d_gen_fractal_lsystem_depth_{depth}",
                name=f"3D L-System Structure depth={depth}",
                category="procedural_generation_lsystem",
                rpn_program=f"AXIOM3D RULES3D {depth} EXPAND_LSYSTEM3D",
                tags=["procedural", "fractal", "lsystem"],
                metadata={"generative": True, "cross_modal": ["drawing", "reality", "grammar"]},
            )
        )
    return out


def default_3d_objects_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    entries.extend(create_mesh_primitives())
    entries.extend(create_transformation_primitives())
    entries.extend(create_spatial_query_primitives())
    entries.extend(create_procedural_generation_primitives())
    return entries


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n")


def bootstrap_3d_objects_galaxy(storage_root: str | Path = "../Knowledge3D.local") -> dict[str, int]:
    """Append deterministic 3DObjects entries without resetting existing data."""
    galaxies_root = Path(storage_root) / "galaxies"
    path = galaxies_root / "3DObjects.jsonl"
    existing = _read_jsonl(path)
    existing_ids = {str(row.get("id", "")) for row in existing}
    generated = default_3d_objects_entries()
    to_append = [row for row in generated if str(row.get("id", "")) and str(row.get("id", "")) not in existing_ids]
    if to_append:
        _append_jsonl(path, to_append)
    return {
        "before": len(existing),
        "generated": len(generated),
        "appended": len(to_append),
        "after": len(existing) + len(to_append),
    }

