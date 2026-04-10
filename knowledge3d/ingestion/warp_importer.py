"""Ingestion-only adapter from NVIDIA Warp scene definitions to K3D stars.

This module is explicitly outside the hot path. It is allowed to inspect Warp
Python objects and convert them into Reality Galaxy substrate before the
sovereign PTX runtime consumes them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


def _seq3(values: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if values is None:
        return default
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (tuple, list)) and len(values) >= 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    return default


def _warp_shape_to_drawing_star(shape: Any) -> str:
    geometry = getattr(shape, "geo_type", None) or getattr(shape, "type", None) or getattr(shape, "shape_type", None)
    radius = float(getattr(shape, "radius", 0.5) or 0.5)
    half_extents = _seq3(getattr(shape, "half_extents", None), (0.5, 0.5, 0.5))
    if geometry in {"sphere", 1, "GEO_SPHERE"}:
        return f"drawing_sphere_radius_{radius:.3f}".replace(".", "_")
    if geometry in {"box", 2, "GEO_BOX"}:
        return (
            f"drawing_box_{half_extents[0]:.3f}x{half_extents[1]:.3f}x{half_extents[2]:.3f}"
            .replace(".", "_")
        )
    if geometry in {"capsule", 3, "GEO_CAPSULE"}:
        half_height = float(getattr(shape, "half_height", 0.5) or 0.5)
        return f"drawing_capsule_r_{radius:.3f}_h_{half_height:.3f}".replace(".", "_")
    return "drawing_convex_hull_default"


def _warp_material_to_reality_star(material: Any) -> str:
    if material is None:
        return "physics_material_steel"
    attrs = {
        "friction_static": getattr(material, "static_friction", None),
        "friction_dynamic": getattr(material, "dynamic_friction", None),
        "restitution": getattr(material, "restitution", None),
        "density": getattr(material, "density", None),
    }
    if attrs["friction_dynamic"] is not None and float(attrs["friction_dynamic"]) < 0.08:
        return "physics_material_ice"
    if attrs["restitution"] is not None and float(attrs["restitution"]) > 0.6:
        return "physics_material_rubber"
    if attrs["density"] is not None and float(attrs["density"]) < 1200.0:
        return "physics_material_wood"
    return "physics_material_steel"


@dataclass(frozen=True)
class ImportedWarpScene:
    stars: list[dict[str, Any]]
    source_path: str | None = None


def import_warp_model(model: Any, *, source_path: str | None = None) -> ImportedWarpScene:
    bodies: Sequence[Any] = getattr(model, "bodies", None) or getattr(model, "body_q", None) or []
    shapes: Sequence[Any] = getattr(model, "shapes", None) or getattr(model, "shape_geo", None) or []
    materials: Sequence[Any] = getattr(model, "shape_materials", None) or getattr(model, "materials", None) or []
    body_masses: Sequence[float] = getattr(model, "body_mass", None) or getattr(model, "masses", None) or []
    body_vels: Sequence[Any] = getattr(model, "body_qd", None) or getattr(model, "velocities", None) or []

    stars: list[dict[str, Any]] = []
    body_count = max(len(bodies), len(body_masses), len(body_vels), len(shapes))
    for idx in range(body_count):
        body = bodies[idx] if idx < len(bodies) else None
        shape = shapes[idx] if idx < len(shapes) else None
        material = materials[idx] if idx < len(materials) else None
        mass = float(body_masses[idx]) if idx < len(body_masses) else float(getattr(body, "mass", 1.0) or 1.0)
        pos = _seq3(getattr(body, "position", None) or getattr(body, "translation", None), (0.0, 0.0, 0.0))
        vel = _seq3(body_vels[idx] if idx < len(body_vels) else getattr(body, "linear_velocity", None), (0.0, 0.0, 0.0))

        stars.append(
            {
                "star_id": f"warp_body_{idx}",
                "facet": "rigid_body",
                "material_star_id": _warp_material_to_reality_star(material),
                "shape_star_id": _warp_shape_to_drawing_star(shape),
                "physics_rpn_addr": "physics_law_default_gravity",
                "mass": mass,
                "position_x": pos[0],
                "position_y": pos[1],
                "position_z": pos[2],
                "velocity_x": vel[0],
                "velocity_y": vel[1],
                "velocity_z": vel[2],
                "is_sleeping": bool(getattr(body, "is_sleeping", False)),
                "source_path": source_path,
            }
        )
    return ImportedWarpScene(stars=stars, source_path=source_path)


def import_warp_modelbuilder(builder: Any, *, source_path: str | None = None) -> ImportedWarpScene:
    if hasattr(builder, "finalize"):
        return import_warp_model(builder.finalize(), source_path=source_path)
    return import_warp_model(builder, source_path=source_path)


def load_warp_scene_from_file(path: str | Path) -> ImportedWarpScene:
    path = Path(path)
    namespace: dict[str, Any] = {}
    exec(path.read_text(encoding="utf-8"), namespace)
    candidate = namespace.get("model") or namespace.get("builder")
    if candidate is None:
        raise ValueError(f"No `model` or `builder` object found in {path}")
    if hasattr(candidate, "finalize"):
        return import_warp_modelbuilder(candidate, source_path=str(path))
    return import_warp_model(candidate, source_path=str(path))


__all__ = [
    "ImportedWarpScene",
    "import_warp_model",
    "import_warp_modelbuilder",
    "load_warp_scene_from_file",
]
