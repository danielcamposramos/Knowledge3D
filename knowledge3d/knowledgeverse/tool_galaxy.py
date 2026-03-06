"""Procedural tool-node schema and bootstrap for always-on procedural means.

Tool-nodes are Knowledgeverse entries for reusable procedural means:

- multimodal composition recipes
- PTX-backed codec surfaces already wired into the runtime
- promotion targets for future opcode admission

They live in one always-on Tool galaxy so the system does not fragment
"verbs" across task-specific setup paths.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


_ALLOWED_MODALITIES = {
    "drawing",
    "reality",
    "audio",
    "3dobjects",
    "math",
    "grammar",
    "video",
    "signal",
}
_ALLOWED_PROMOTION_STAGES = {"recipe", "macro", "opcode_candidate", "kernel"}


def _dedupe_strs(values: tuple[str, ...] | list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        token = str(value).strip()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _math_core_profile(
    *,
    preferred_tier: int | None = None,
    tier_role: str | None = None,
    spawn_policy: str = "adaptive_reuse",
    cascade: tuple[str, ...] | list[str] = (),
    execution_model: str = "tiered_rpn",
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "spawn_policy": str(spawn_policy),
        "cascade": _dedupe_strs(list(cascade)),
        "execution_model": str(execution_model),
        "pool": "dynamic_math_core_pool",
    }
    if preferred_tier is not None:
        tier = int(preferred_tier)
        if tier not in {1, 2, 3}:
            raise ValueError(f"invalid preferred_tier: {preferred_tier}")
        profile["preferred_tier"] = tier
        profile["tier_role"] = str(tier_role or {1: "worker_worker", 2: "worker", 3: "master"}[tier])
    elif tier_role:
        profile["tier_role"] = str(tier_role)
    return profile


def _entrypoint_argument_schema(
    *,
    positional: Sequence[tuple[str, str] | Mapping[str, Any]] = (),
    required_kwargs: Sequence[tuple[str, str] | Mapping[str, Any]] = (),
    optional_kwargs: Sequence[tuple[str, str] | Mapping[str, Any]] = (),
    strict_kwargs: bool = True,
    allow_additional_positionals: bool = False,
) -> dict[str, Any]:
    def _param_rows(items: Sequence[tuple[str, str] | Mapping[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, Mapping):
                name = str(item.get("name", "")).strip()
                type_name = str(item.get("type", "")).strip() or "Any"
                aliases = _dedupe_strs([str(value) for value in item.get("aliases", [])])
                has_default = "default" in item
                default_value = item.get("default")
            else:
                name, type_name = item
                name = str(name).strip()
                type_name = str(type_name).strip() or "Any"
                aliases = []
                has_default = False
                default_value = None
            token = str(name).strip()
            if not token:
                continue
            row: dict[str, Any] = {"name": token, "type": type_name}
            if aliases:
                row["aliases"] = aliases
            if has_default:
                row["default"] = default_value
            rows.append(row)
        return rows

    return {
        "positional": _param_rows(positional),
        "required_kwargs": _param_rows(required_kwargs),
        "optional_kwargs": _param_rows(optional_kwargs),
        "strict_kwargs": bool(strict_kwargs),
        "allow_additional_positionals": bool(allow_additional_positionals),
    }


def _execution_chain_step(
    *,
    entrypoint: str,
    argument_schema: Mapping[str, Any] | None = None,
    store_as: tuple[str, ...] | list[str] = (),
    store_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "entrypoint": str(entrypoint).strip(),
        "argument_schema": dict(argument_schema or {}),
        "store_as": _dedupe_strs(list(store_as)),
        "store_fields": {str(key).strip(): str(value).strip() for key, value in (store_fields or {}).items() if str(key).strip() and str(value).strip()},
    }


def _execution_chain_preset(
    *,
    required_inputs: tuple[str, ...] | list[str],
    steps: Sequence[Mapping[str, Any]],
    return_alias: str | None = None,
    selectors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "required_inputs": _dedupe_strs(list(required_inputs)),
        "steps": [dict(step) for step in steps],
        "return_alias": str(return_alias or "").strip(),
        "selectors": {
            str(key).strip(): value
            for key, value in (selectors or {}).items()
            if str(key).strip()
        },
    }


@dataclass(frozen=True)
class ToolNode:
    """Schema for first-class procedural tools stored as galaxy knowledge."""

    tool_id: str
    name: str
    category: str
    tool_kind: str
    modalities: tuple[str, ...]
    description: str
    rpn_program: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    drawing_refs: tuple[str, ...] = ()
    reality_refs: tuple[str, ...] = ()
    object_refs: tuple[str, ...] = ()
    audio_refs: tuple[str, ...] = ()
    tool_refs: tuple[str, ...] = ()
    rule_refs: tuple[str, ...] = ()
    word_refs: tuple[str, ...] = ()
    codec_ops: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    promotion_stage: str = "recipe"
    runtime_status: str = "recipe_only"
    source: str = "week23_multimodal_tool_bootstrap"
    math_core: Mapping[str, Any] = field(default_factory=dict)
    memory_residency: str = "knowledgeverse_galaxy"
    execution_residency: str = "gpu_ptx"
    entrypoint_argument_schemas: Mapping[str, Any] = field(default_factory=dict)
    execution_chain_presets: Mapping[str, Any] = field(default_factory=dict)
    metadata_extra: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.tool_id.strip():
            raise ValueError("tool_id is required")
        if not self.name.strip():
            raise ValueError("name is required")
        if not self.category.strip():
            raise ValueError("category is required")
        if not self.tool_kind.strip():
            raise ValueError("tool_kind is required")
        if not self.rpn_program.strip():
            raise ValueError("rpn_program is required")
        if self.promotion_stage not in _ALLOWED_PROMOTION_STAGES:
            raise ValueError(f"invalid promotion_stage: {self.promotion_stage}")
        modalities = _dedupe_strs(list(self.modalities))
        if not modalities:
            raise ValueError("at least one modality is required")
        invalid = [mod for mod in modalities if mod not in _ALLOWED_MODALITIES]
        if invalid:
            raise ValueError(f"invalid modalities: {invalid}")
        if self.math_core:
            preferred = self.math_core.get("preferred_tier")
            if preferred is not None and int(preferred) not in {1, 2, 3}:
                raise ValueError(f"invalid preferred_tier: {preferred}")
        if self.entrypoint_argument_schemas:
            if not isinstance(self.entrypoint_argument_schemas, Mapping):
                raise ValueError("entrypoint_argument_schemas must be a mapping")
            for key, value in self.entrypoint_argument_schemas.items():
                token = str(key).strip()
                if not token:
                    raise ValueError("entrypoint_argument_schemas contains empty key")
                if not isinstance(value, Mapping):
                    raise ValueError(f"entrypoint schema for {token} must be a mapping")
        if self.execution_chain_presets:
            if not isinstance(self.execution_chain_presets, Mapping):
                raise ValueError("execution_chain_presets must be a mapping")

    def to_entry(self) -> dict[str, Any]:
        self.validate()
        drawing_refs = _dedupe_strs(list(self.drawing_refs))
        reality_refs = _dedupe_strs(list(self.reality_refs))
        object_refs = _dedupe_strs(list(self.object_refs))
        audio_refs = _dedupe_strs(list(self.audio_refs))
        tool_refs = _dedupe_strs(list(self.tool_refs))
        rule_refs = _dedupe_strs(list(self.rule_refs))
        word_refs = _dedupe_strs(list(self.word_refs))
        codec_ops = _dedupe_strs(list(self.codec_ops))
        tags = _dedupe_strs(list(self.tags))
        modalities = _dedupe_strs(list(self.modalities))
        component_refs = _dedupe_strs(
            drawing_refs + reality_refs + object_refs + audio_refs + tool_refs + rule_refs
        )
        math_core = dict(self.math_core)
        if "cascade" in math_core:
            math_core["cascade"] = _dedupe_strs(list(math_core.get("cascade", [])))
        if "preferred_tier" in math_core and math_core["preferred_tier"] is not None:
            math_core["preferred_tier"] = int(math_core["preferred_tier"])
        metadata = {
            "source": self.source,
            "tool_kind": self.tool_kind,
            "modalities": modalities,
            "promotion_stage": self.promotion_stage,
            "runtime_status": self.runtime_status,
            "description": self.description,
            "inputs": _dedupe_strs(list(self.inputs)),
            "outputs": _dedupe_strs(list(self.outputs)),
            "procedural": True,
            "symlink": "tool_galaxy",
            "memory_residency": self.memory_residency,
            "execution_residency": self.execution_residency,
            "cross_modal": modalities,
            "component_galaxies": {
                "Drawing": drawing_refs,
                "Reality": reality_refs,
                "3DObjects": object_refs,
                "Audio": audio_refs,
                "Tool": tool_refs,
                "Grammar": rule_refs,
            },
            "codec_ops": codec_ops,
        }
        if math_core:
            metadata["math_core"] = math_core
        if self.entrypoint_argument_schemas:
            metadata["entrypoint_argument_schemas"] = {
                str(key).strip(): dict(value)
                for key, value in self.entrypoint_argument_schemas.items()
                if str(key).strip()
            }
        if self.execution_chain_presets:
            metadata["execution_chain_presets"] = {
                str(key).strip(): dict(value)
                for key, value in self.execution_chain_presets.items()
                if str(key).strip()
            }
        metadata.update(dict(self.metadata_extra))
        return {
            "type": "tool_node",
            "id": self.tool_id,
            "name": self.name,
            "domain": "tool",
            "category": self.category,
            "pattern_type": "tool_node",
            "rpn_program": self.rpn_program,
            "procedural_programs": {"compose_rpn": self.rpn_program},
            "component_refs": component_refs,
            "drawing_refs": drawing_refs,
            "reality_refs": reality_refs,
            "object_refs": object_refs,
            "audio_refs": audio_refs,
            "tool_refs": tool_refs,
            "rule_refs": rule_refs,
            "word_refs": word_refs,
            "tags": tags,
            "metadata": metadata,
        }


def _multimodal_fusion_tools() -> list[ToolNode]:
    """Recipe-stage multimodal fusion tools."""

    return [
        ToolNode(
            tool_id="tool_fusion_contour_to_mesh_v1",
            name="2D Contour to 3D Mesh Recipe",
            category="drawing_reality_fusion",
            tool_kind="contour_extrusion",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "Recipe-stage tool that turns a 2D contour into a 3D mesh plan via lathe or "
                "extrusion, then prepares that mesh for procedural material mapping."
            ),
            rpn_program=(
                "tool_geom_profile_prep_v1 tool_geom_profile_lathe_mesh_v1 "
                "tool_geom_profile_extrude_mesh_v1 tool_geom_profile_sweep_mesh_v1 "
                "glyph_curve_transfer cubic_bezier_eval "
                "TERNARY_QUANT RESHAPE_TO_BLOCKS "
                "obj3d_mesh_compute_normal"
            ),
            inputs=("drawing_contour", "extrusion_depth", "surface_material"),
            outputs=("mesh_plan", "uv_ready_surface", "preview_outline"),
            drawing_refs=("glyph_curve_transfer", "cubic_bezier_eval"),
            object_refs=("obj3d_gen_lathe_profile", "obj3d_mesh_compute_normal", "obj3d_xform_apply"),
            tool_refs=(
                "tool_geom_profile_prep_v1",
                "tool_geom_profile_lathe_mesh_v1",
                "tool_geom_profile_extrude_mesh_v1",
                "tool_geom_profile_sweep_mesh_v1",
            ),
            reality_refs=("reality_proc_noise_o1_p0p3",),
            codec_ops=("TERNARY_QUANT", "RESHAPE_TO_BLOCKS"),
            tags=("fusion", "drawing", "3d", "extrusion", "recipe"),
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            metadata_extra={
                "promotion_targets": ["MESH_EXTRUDE", "UV_PROJECT"],
                "phase_target": "phase_1_draw_extrude_texture",
                "procedural_goal": "convert contour/profile knowledge into mesh-ready control data",
            },
        ),
        ToolNode(
            tool_id="tool_fusion_surface_material_projection_v1",
            name="Procedural Surface Material Projection Recipe",
            category="drawing_reality_fusion",
            tool_kind="material_projection",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "Recipe-stage tool that projects a procedural 2D material onto a 3D "
                "surface using UV or triplanar mapping targets."
            ),
            rpn_program=(
                "tool_paint_palette_contrastive_v1 tool_paint_gradient_cascade_v1 "
                "barycentric_raster backface_culling "
                "TERNARY_QUANT BLOCKS_TO_GRID "
                "reality_proc_noise_o1_p0p3 obj3d_mesh_compute_normal"
            ),
            inputs=("surface_mesh", "procedural_material", "projection_strategy"),
            outputs=("textured_surface", "normal_hint", "material_preview"),
            drawing_refs=("barycentric_raster", "backface_culling"),
            object_refs=("obj3d_mesh_compute_normal", "obj3d_xform_apply"),
            reality_refs=("reality_proc_noise_o1_p0p3", "reality_proc_lsystem_expand"),
            tool_refs=(
                "tool_fusion_contour_to_mesh_v1",
                "tool_paint_palette_contrastive_v1",
                "tool_paint_gradient_cascade_v1",
            ),
            codec_ops=("TERNARY_QUANT", "BLOCKS_TO_GRID"),
            tags=("fusion", "material", "surface", "triplanar", "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            entrypoint_argument_schemas={
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.project_material": _entrypoint_argument_schema(
                    positional=(
                        {"name": "mesh", "type": "SurfaceMeshLike", "aliases": ("surface_mesh",)},
                        {"name": "candidate", "type": "SurfaceMaterialCandidate", "aliases": ("procedural_material",)},
                    ),
                    optional_kwargs=(
                        {"name": "preview_size", "type": "int", "default": 64},
                        {"name": "projection_strategy", "type": "str|None", "default": None},
                    ),
                ),
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_lathe_mesh": _entrypoint_argument_schema(
                    positional=(
                        {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                    ),
                    required_kwargs=(
                        {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                        {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                    ),
                    optional_kwargs=(
                        {"name": "color", "type": "int", "default": 1},
                        {"name": "pad", "type": "int", "default": 0},
                        {"name": "segments", "type": "int", "default": 24},
                        {"name": "height_scale", "type": "float", "default": 1.0},
                        {"name": "radius_scale", "type": "float", "default": 1.0},
                        {"name": "cap_ends", "type": "bool", "default": True},
                        {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                        {"name": "preview_size", "type": "int", "default": 64},
                        {"name": "projection_strategy", "type": "str|None", "default": None},
                    ),
                ),
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_extrude_mesh": _entrypoint_argument_schema(
                    positional=(
                        {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                    ),
                    required_kwargs=(
                        {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                        {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                    ),
                    optional_kwargs=(
                        {"name": "color", "type": "int", "default": 1},
                        {"name": "pad", "type": "int", "default": 0},
                        {"name": "depth_scale", "type": "float", "default": 0.5},
                        {"name": "width_scale", "type": "float", "default": 1.0},
                        {"name": "height_scale", "type": "float", "default": 1.0},
                        {"name": "cap_ends", "type": "bool", "default": True},
                        {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                        {"name": "preview_size", "type": "int", "default": 64},
                        {"name": "projection_strategy", "type": "str|None", "default": None},
                    ),
                ),
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_sweep_mesh": _entrypoint_argument_schema(
                    positional=(
                        {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                    ),
                    required_kwargs=(
                        {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                        {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                    ),
                    optional_kwargs=(
                        {"name": "color", "type": "int", "default": 1},
                        {"name": "pad", "type": "int", "default": 0},
                        {"name": "depth_scale", "type": "float", "default": 0.5},
                        {"name": "width_scale", "type": "float", "default": 1.0},
                        {"name": "height_scale", "type": "float", "default": 1.0},
                        {"name": "cap_ends", "type": "bool", "default": True},
                        {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                        {"name": "preview_size", "type": "int", "default": 64},
                        {"name": "projection_strategy", "type": "str|None", "default": None},
                    ),
                ),
            },
            execution_chain_presets={
                "contour_material_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates"),
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_geometry_bridge."
                                "ProceduralGeometryBridge.contour_to_lathe_mesh"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                                ),
                                optional_kwargs=(
                                    {"name": "color", "type": "int", "default": 1},
                                    {"name": "pad", "type": "int", "default": 0},
                                    {"name": "segments", "type": "int", "default": 24},
                                    {"name": "height_scale", "type": "float", "default": 1.0},
                                    {"name": "radius_scale", "type": "float", "default": 1.0},
                                    {"name": "cap_ends", "type": "bool", "default": True},
                                ),
                            ),
                            store_as=("surface_mesh",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.select_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                ),
                            ),
                            store_as=("material_selection",),
                            store_fields={"selected_material": "selected"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.project_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "mesh", "type": "SurfaceMeshLike", "aliases": ("surface_mesh",)},
                                    {"name": "candidate", "type": "SurfaceMaterialCandidate", "aliases": ("selected_material",)},
                                ),
                                optional_kwargs=(
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                    ),
                    return_alias="textured_surface",
                ),
                "contour_extrude_material_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates", "geometry_mode"),
                    selectors={"geometry_mode": "extrude"},
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_geometry_bridge."
                                "ProceduralGeometryBridge.contour_to_extrude_mesh"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                                ),
                                optional_kwargs=(
                                    {"name": "color", "type": "int", "default": 1},
                                    {"name": "pad", "type": "int", "default": 0},
                                    {"name": "depth_scale", "type": "float", "default": 0.5},
                                    {"name": "width_scale", "type": "float", "default": 1.0},
                                    {"name": "height_scale", "type": "float", "default": 1.0},
                                    {"name": "cap_ends", "type": "bool", "default": True},
                                ),
                            ),
                            store_as=("surface_mesh",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.select_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                ),
                            ),
                            store_as=("material_selection",),
                            store_fields={"selected_material": "selected"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.project_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "mesh", "type": "SurfaceMeshLike", "aliases": ("surface_mesh",)},
                                    {"name": "candidate", "type": "SurfaceMaterialCandidate", "aliases": ("selected_material",)},
                                ),
                                optional_kwargs=(
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                    ),
                    return_alias="textured_surface",
                ),
                "contour_sweep_material_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates", "geometry_mode"),
                    selectors={"geometry_mode": "sweep"},
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_geometry_bridge."
                                "ProceduralGeometryBridge.contour_to_sweep_mesh"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                                ),
                                optional_kwargs=(
                                    {"name": "color", "type": "int", "default": 1},
                                    {"name": "pad", "type": "int", "default": 0},
                                    {"name": "depth_scale", "type": "float", "default": 0.5},
                                    {"name": "width_scale", "type": "float", "default": 1.0},
                                    {"name": "height_scale", "type": "float", "default": 1.0},
                                    {"name": "cap_ends", "type": "bool", "default": True},
                                ),
                            ),
                            store_as=("surface_mesh",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.select_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                ),
                            ),
                            store_as=("material_selection",),
                            store_fields={"selected_material": "selected"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.project_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "mesh", "type": "SurfaceMeshLike", "aliases": ("surface_mesh",)},
                                    {"name": "candidate", "type": "SurfaceMaterialCandidate", "aliases": ("selected_material",)},
                                ),
                                optional_kwargs=(
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                    ),
                    return_alias="textured_surface",
                ),
            },
            metadata_extra={
                "promotion_targets": ["TRIPLANAR_MAP", "UV_PROJECT"],
                "phase_target": "phase_1_draw_extrude_texture",
                "procedural_goal": "project reusable 2D procedural material onto 3D surfaces",
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.project_material",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_lathe_mesh",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_extrude_mesh",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_sweep_mesh",
                ],
                "verified_by": ["tests/test_procedural_material_bridge.py"],
            },
        ),
        ToolNode(
            tool_id="tool_signal_audio_spectrogram_v1",
            name="Audio to Spectrogram Recipe",
            category="audio_drawing_fusion",
            tool_kind="signal_projection",
            modalities=("audio", "drawing", "signal"),
            description=(
                "Recipe-stage tool that maps an audio signal into a drawable spectrogram "
                "representation for inspection, retrieval, and later synthesis."
            ),
            rpn_program=(
                "sine_wave_as_curve curve_to_waveform_map "
                "MDCT RESHAPE_TO_BLOCKS BLOCKS_TO_GRID"
            ),
            inputs=("audio_signal", "window_size", "palette"),
            outputs=("spectrogram", "signal_summary"),
            drawing_refs=("sine_wave_as_curve", "curve_to_waveform_map"),
            codec_ops=("MDCT", "RESHAPE_TO_BLOCKS", "BLOCKS_TO_GRID"),
            tags=("audio", "spectrogram", "signal", "drawing", "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            entrypoint_argument_schemas={
                "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.audio_to_spectrogram": _entrypoint_argument_schema(
                    positional=(
                        ("clip_id", "str"),
                        {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                    ),
                ),
                "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.audio_to_spectrogram_configured": _entrypoint_argument_schema(
                    positional=(
                        ("clip_id", "str"),
                        {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                    ),
                    optional_kwargs=(
                        {"name": "frame_size", "type": "int", "default": 1024},
                        {"name": "threshold", "type": "float", "default": 0.2},
                    ),
                ),
            },
            metadata_extra={
                "promotion_targets": ["FFT_FORWARD", "AUDIO_TO_SPECTROGRAM"],
                "phase_target": "phase_2_signal_fusion",
                "procedural_goal": "convert audio into a drawable, searchable block-grid spectrum",
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.audio_to_spectrogram",
                    "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.audio_to_spectrogram_configured",
                ],
                "verified_by": ["tests/test_procedural_signal_bridge.py"],
            },
        ),
        ToolNode(
            tool_id="tool_signal_spectrogram_surface_v1",
            name="Spectrogram to Surface Displacement Recipe",
            category="audio_drawing_reality_fusion",
            tool_kind="signal_surface_displacement",
            modalities=("audio", "drawing", "3dobjects", "reality", "signal"),
            description=(
                "Recipe-stage tri-modal tool that turns a spectrogram into a heightfield "
                "or surface displacement plan over a 3D mesh grid."
            ),
            rpn_program=(
                "tool_signal_audio_spectrogram_v1 "
                "DCT8 RESHAPE_TO_BLOCKS BLOCKS_TO_GRID "
                "obj3d_mesh_grid_16x16 reality_proc_noise_o1_p0p3"
            ),
            inputs=("audio_signal", "mesh_grid", "displacement_gain"),
            outputs=("heightfield_plan", "surface_mesh", "cross_modal_preview"),
            object_refs=("obj3d_mesh_grid_16x16", "obj3d_mesh_compute_normal"),
            drawing_refs=("sine_wave_as_curve",),
            reality_refs=("reality_proc_noise_o1_p0p3",),
            tool_refs=("tool_signal_audio_spectrogram_v1",),
            codec_ops=("DCT8", "RESHAPE_TO_BLOCKS", "BLOCKS_TO_GRID"),
            tags=("audio", "drawing", "reality", "heightfield", "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            entrypoint_argument_schemas={
                "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.spectrogram_to_surface": _entrypoint_argument_schema(
                    positional=(
                        {"name": "projection", "type": "SpectrogramPlan", "aliases": ("spectrogram_projection", "spectrogram")},
                    ),
                    optional_kwargs=(
                        {"name": "displacement_gain", "type": "float", "default": 0.25},
                        {"name": "time_scale", "type": "float", "default": 1.0},
                        {"name": "frequency_scale", "type": "float", "default": 1.0},
                    ),
                ),
            },
            execution_chain_presets={
                "audio_surface_chain": _execution_chain_preset(
                    required_inputs=("clip_id", "audio_signal"),
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_signal_bridge."
                                "ProceduralSignalBridge.audio_to_spectrogram_configured"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    ("clip_id", "str"),
                                    {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                                ),
                                optional_kwargs=(
                                    {"name": "frame_size", "type": "int", "default": 1024},
                                    {"name": "threshold", "type": "float", "default": 0.2},
                                ),
                            ),
                            store_as=("spectrogram_projection",),
                            store_fields={"spectrogram": "spectrogram", "signal_summary": "metadata"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_signal_bridge."
                                "ProceduralSignalBridge.spectrogram_to_surface"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "projection", "type": "SpectrogramPlan", "aliases": ("spectrogram_projection", "spectrogram")},
                                ),
                                optional_kwargs=(
                                    {"name": "displacement_gain", "type": "float", "default": 0.25},
                                    {"name": "time_scale", "type": "float", "default": 1.0},
                                    {"name": "frequency_scale", "type": "float", "default": 1.0},
                                ),
                            ),
                            store_as=("surface_mesh",),
                            store_fields={"heightfield_plan": "heightfield"},
                        ),
                    ),
                    return_alias="surface_mesh",
                ),
            },
            metadata_extra={
                "promotion_targets": ["DISPLACEMENT_MAP", "SPECTROGRAM_TO_HEIGHTFIELD"],
                "phase_target": "phase_2_signal_fusion",
                "procedural_goal": "reuse spectrogram blocks as surface displacement control data",
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_signal_bridge.ProceduralSignalBridge.spectrogram_to_surface",
                ],
                "verified_by": ["tests/test_procedural_signal_bridge.py"],
            },
        ),
        ToolNode(
            tool_id="tool_fusion_signal_surface_material_v1",
            name="Signal Surface Material Fusion Recipe",
            category="audio_drawing_reality_fusion",
            tool_kind="signal_material_projection",
            modalities=("audio", "drawing", "3dobjects", "reality", "signal"),
            description=(
                "Recipe-stage multimodal tool that carries an audio signal through "
                "spectrogram projection, surface displacement, and procedural material "
                "selection/projection as one executable route."
            ),
            rpn_program=(
                "tool_signal_audio_spectrogram_v1 "
                "tool_signal_spectrogram_surface_v1 "
                "tool_paint_palette_contrastive_v1 "
                "TERNARY_QUANT BLOCKS_TO_GRID obj3d_mesh_compute_normal"
            ),
            inputs=("audio_signal", "material_candidates", "displacement_gain"),
            outputs=("textured_surface", "material_preview", "signal_summary"),
            object_refs=("obj3d_mesh_compute_normal",),
            drawing_refs=("curve_to_waveform_map",),
            reality_refs=("reality_proc_noise_o1_p0p3",),
            tool_refs=(
                "tool_signal_audio_spectrogram_v1",
                "tool_signal_spectrogram_surface_v1",
                "tool_fusion_surface_material_projection_v1",
                "tool_paint_palette_contrastive_v1",
            ),
            codec_ops=("MDCT", "TERNARY_QUANT", "RESHAPE_TO_BLOCKS", "BLOCKS_TO_GRID"),
            tags=("audio", "signal", "surface", "material", "fusion", "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            entrypoint_argument_schemas={
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface": _entrypoint_argument_schema(
                    positional=(
                        ("clip_id", "str"),
                        {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                    ),
                    required_kwargs=(
                        {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                    ),
                    optional_kwargs=(
                        {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                        {"name": "target_material", "type": "SurfaceMaterialCandidate|None", "aliases": ("surface_material",), "default": None},
                        {"name": "frame_size", "type": "int", "default": 1024},
                        {"name": "threshold", "type": "float", "default": 0.2},
                        {"name": "displacement_gain", "type": "float", "default": 0.25},
                        {"name": "preview_size", "type": "int", "default": 64},
                        {"name": "projection_strategy", "type": "str|None", "default": None},
                    ),
                ),
                "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_projection_to_material_target": _entrypoint_argument_schema(
                    positional=(
                        {"name": "projection", "type": "SpectrogramPlan", "aliases": ("spectrogram_projection", "spectrogram")},
                    ),
                    optional_kwargs=(
                        {"name": "material_id", "type": "str", "default": "signal_target"},
                        {"name": "name", "type": "str", "default": "Signal Target Material"},
                    ),
                ),
            },
            execution_chain_presets={
                "signal_surface_material_chain": _execution_chain_preset(
                    required_inputs=("clip_id", "audio_signal", "material_candidates"),
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_signal_bridge."
                                "ProceduralSignalBridge.audio_to_spectrogram_configured"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    ("clip_id", "str"),
                                    {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                                ),
                                optional_kwargs=(
                                    {"name": "frame_size", "type": "int", "default": 1024},
                                    {"name": "threshold", "type": "float", "default": 0.2},
                                ),
                            ),
                            store_as=("spectrogram_projection",),
                            store_fields={"spectrogram": "spectrogram", "signal_summary": "metadata"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.signal_projection_to_material_target"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "projection", "type": "SpectrogramPlan", "aliases": ("spectrogram_projection", "spectrogram")},
                                ),
                                optional_kwargs=(
                                    {"name": "material_id", "type": "str", "default": "signal_target"},
                                    {"name": "name", "type": "str", "default": "Signal Target Material"},
                                ),
                            ),
                            store_as=("surface_material",),
                            store_fields={"target_material": "__self__"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_signal_bridge."
                                "ProceduralSignalBridge.spectrogram_to_surface"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "projection", "type": "SpectrogramPlan", "aliases": ("spectrogram_projection", "spectrogram")},
                                ),
                                optional_kwargs=(
                                    {"name": "displacement_gain", "type": "float", "default": 0.25},
                                    {"name": "time_scale", "type": "float", "default": 1.0},
                                    {"name": "frequency_scale", "type": "float", "default": 1.0},
                                ),
                            ),
                            store_as=("surface_mesh",),
                            store_fields={"heightfield_plan": "heightfield"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.select_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                ),
                            ),
                            store_as=("material_selection",),
                            store_fields={"selected_material": "selected"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.project_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "mesh", "type": "SurfaceMeshLike", "aliases": ("surface_mesh",)},
                                    {"name": "candidate", "type": "SurfaceMaterialCandidate", "aliases": ("selected_material",)},
                                ),
                                optional_kwargs=(
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                    ),
                    return_alias="textured_surface",
                ),
                "signal_surface_material_with_target_chain": _execution_chain_preset(
                    required_inputs=("clip_id", "audio_signal", "material_candidates", "surface_material"),
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_signal_bridge."
                                "ProceduralSignalBridge.audio_to_spectrogram_configured"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    ("clip_id", "str"),
                                    {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                                ),
                                optional_kwargs=(
                                    {"name": "frame_size", "type": "int", "default": 1024},
                                    {"name": "threshold", "type": "float", "default": 0.2},
                                ),
                            ),
                            store_as=("spectrogram_projection",),
                            store_fields={"spectrogram": "spectrogram", "signal_summary": "metadata"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_signal_bridge."
                                "ProceduralSignalBridge.spectrogram_to_surface"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "projection", "type": "SpectrogramPlan", "aliases": ("spectrogram_projection", "spectrogram")},
                                ),
                                optional_kwargs=(
                                    {"name": "displacement_gain", "type": "float", "default": 0.25},
                                    {"name": "time_scale", "type": "float", "default": 1.0},
                                    {"name": "frequency_scale", "type": "float", "default": 1.0},
                                ),
                            ),
                            store_as=("surface_mesh",),
                            store_fields={"heightfield_plan": "heightfield"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.select_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                ),
                            ),
                            store_as=("material_selection",),
                            store_fields={"selected_material": "selected"},
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.project_material"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "mesh", "type": "SurfaceMeshLike", "aliases": ("surface_mesh",)},
                                    {"name": "candidate", "type": "SurfaceMaterialCandidate", "aliases": ("selected_material",)},
                                ),
                                optional_kwargs=(
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                    ),
                    return_alias="textured_surface",
                ),
            },
            metadata_extra={
                "promotion_targets": ["SIGNAL_SURFACE_MATERIAL", "TRIPLANAR_MAP"],
                "phase_target": "phase_2_signal_fusion",
                "procedural_goal": "let signal-derived surfaces participate directly in material selection and projection",
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_projection_to_material_target",
                ],
                "verified_by": [
                    "tests/test_procedural_signal_bridge.py",
                    "tests/test_procedural_material_bridge.py",
                ],
            },
        ),
    ]


def _procedural_math_core_tools() -> list[ToolNode]:
    """Always-on Tool entries that expose the real tiered math-core runtime."""

    return [
        ToolNode(
            tool_id="tool_mathcore_tier1_scalar_worker_worker_v1",
            name="Tier-1 Scalar Worker-Worker Surface",
            category="math_core_runtime",
            tool_kind="math_core_tier",
            modalities=("math", "drawing", "signal"),
            description=(
                "Real Tier-1 lightweight RPN surface for cheap scalar, ternary, and stack work. "
                "This is the worker-worker tier used when latency matters more than expressiveness."
            ),
            rpn_program="dup swap + - * / TERNARY_QUANT",
            inputs=("scalar_stack", "ternary_threshold"),
            outputs=("cheap_scalar_result", "ternary_signal"),
            codec_ops=("TERNARY_QUANT",),
            tags=("math_core", "tier1", "worker_worker", "rpn", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_rpn_available",
            math_core=_math_core_profile(
                preferred_tier=1,
                cascade=("parallel_fanout", "local_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.lightweight_rpn.LightweightRPNEngine.execute_single",
                    "knowledge3d.cranium.bridges.tiered_rpn.TieredRPNEngine.execute_single",
                ],
                "phase_target": "always_on_foundation",
                "procedural_goal": "keep cheap scalar and ternary work on the worker-worker tier",
            },
        ),
        ToolNode(
            tool_id="tool_mathcore_tier2_vector_worker_v1",
            name="Tier-2 Vector Worker Surface",
            category="math_core_runtime",
            tool_kind="math_core_tier",
            modalities=("math", "drawing", "audio", "signal", "reality"),
            description=(
                "Real Tier-2 sovereign RPN surface for general vector, block, and mid-complexity work. "
                "This is the worker tier that most procedural geometry, signal, and paint programs should prefer."
            ),
            rpn_program="BATCH_DCT RESHAPE_TO_BLOCKS BLOCKS_TO_GRID",
            inputs=("vector_or_grid_payload", "shape_hints"),
            outputs=("mid_tier_result", "reduced_summary"),
            codec_ops=("BATCH_DCT", "RESHAPE_TO_BLOCKS", "BLOCKS_TO_GRID"),
            tags=("math_core", "tier2", "worker", "rpn", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_rpn_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.tiered_rpn.TieredRPNEngine.execute_scalar",
                    "knowledge3d.cranium.ptx_runtime.modular_rpn_engine.ModularRPNEngine.evaluate",
                ],
                "phase_target": "always_on_foundation",
                "procedural_goal": "route general procedural math to the worker tier before escalating upward",
            },
        ),
        ToolNode(
            tool_id="tool_mathcore_tier3_master_v1",
            name="Tier-3 Master Math Surface",
            category="math_core_runtime",
            tool_kind="math_core_tier",
            modalities=("math", "reality", "drawing", "audio", "signal"),
            description=(
                "Real Tier-3 advanced RPN surface for matrix, TRM-coupled, and high-complexity routines. "
                "This is the master tier and should only be used after cheaper tiers have prepared the state."
            ),
            rpn_program="TRM_MATVEC_512x1024 MATMUL_SMALL",
            inputs=("matrix_state", "summary_state"),
            outputs=("master_decision", "high_complexity_result"),
            tags=("math_core", "tier3", "master", "rpn", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_rpn_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.advanced_rpn.AdvancedRPNEngine.execute_scalar",
                    "knowledge3d.cranium.bridges.tiered_rpn.TieredRPNEngine.execute_matrix",
                ],
                "phase_target": "always_on_foundation",
                "procedural_goal": "reserve the master tier for high-complexity reductions and final commitments",
            },
        ),
        ToolNode(
            tool_id="tool_mathcore_spawn_cascade_v1",
            name="Dynamic Math-Core Spawn and Cascade Policy",
            category="math_core_runtime",
            tool_kind="math_core_allocator",
            modalities=("math", "drawing", "audio", "signal", "reality", "video"),
            description=(
                "Always-on runtime policy for dynamic math-core spawning, pooling, reuse, and worker-worker -> worker -> master cascade. "
                "Storage remains in the Knowledgeverse Tool galaxy while execution stays GPU/PTX-backed."
            ),
            rpn_program="MATH_CORE_ALLOCATE MATH_CORE_TOUCH MATH_CORE_RELEASE",
            inputs=("tier_request", "workload_shape", "gpu_capacity"),
            outputs=("core_binding", "allocation_policy", "pool_snapshot"),
            tags=("math_core", "allocator", "cascade", "pool", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_runtime_available",
            math_core=_math_core_profile(
                tier_role="allocator",
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.math_core_pool.MathCorePool.spawn_core",
                    "knowledge3d.cranium.ptx_runtime.math_core_pool.MathCorePool.release_core",
                    "knowledge3d.cranium.ptx_runtime.math_core_pool.MathCorePool.snapshot",
                    "knowledge3d.cranium.ptx_runtime.modular_rpn_engine.ModularRPNEngine.get_math_core_descriptor",
                ],
                "phase_target": "always_on_foundation",
                "procedural_goal": "make tier intent, pool reuse, and cascade policy explicit system knowledge instead of hidden runtime state",
            },
        ),
    ]


def _procedural_codec_tools() -> list[ToolNode]:
    """Kernel-backed procedural codecs already available in the PTX/RPN surface."""

    return [
        ToolNode(
            tool_id="tool_codec_ternary_blocks_v1",
            name="Ternary Block Codec Surface",
            category="procedural_codec",
            tool_kind="procedural_codec",
            modalities=("drawing", "audio", "signal", "video"),
            description=(
                "PTX-backed ternary quantization surface for compact procedural coefficients "
                "with block/grid reshaping. This is an always-on codec primitive, not a future target."
            ),
            rpn_program="TERNARY_QUANT TERNARY_DEQUANT RESHAPE_TO_BLOCKS BLOCKS_TO_GRID",
            inputs=("coefficient_field", "ternary_threshold", "block_shape"),
            outputs=("ternary_coefficients", "grid_blocks", "decoded_field"),
            codec_ops=("TERNARY_QUANT", "TERNARY_DEQUANT", "RESHAPE_TO_BLOCKS", "BLOCKS_TO_GRID"),
            tags=("codec", "ternary", "blocks", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_rpn_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "phase_target": "always_on_foundation",
                "verified_by": [
                    "knowledge3d/cranium/tests/test_ternary_codec_ops.py",
                    "knowledge3d/cranium/tests/test_rpn_codec_integration.py",
                ],
                "procedural_goal": "store compact ternary coefficient fields with deterministic rebuild",
            },
        ),
        ToolNode(
            tool_id="tool_codec_video_dct8_grid_v1",
            name="DCT8 Grid Codec Surface",
            category="procedural_codec",
            tool_kind="procedural_codec",
            modalities=("drawing", "video", "signal"),
            description=(
                "PTX-backed DCT8 block transform for image and video-style spatial frequency analysis. "
                "Use with ternary quantization for compact procedural visual storage."
            ),
            rpn_program="RESHAPE_TO_BLOCKS DCT8 TERNARY_QUANT TERNARY_DEQUANT IDCT8 BLOCKS_TO_GRID",
            inputs=("image_or_frame_grid", "block_shape", "ternary_threshold"),
            outputs=("frequency_blocks", "ternary_visual_coefficients", "reconstructed_grid"),
            codec_ops=(
                "RESHAPE_TO_BLOCKS",
                "DCT8",
                "TERNARY_QUANT",
                "TERNARY_DEQUANT",
                "IDCT8",
                "BLOCKS_TO_GRID",
            ),
            tags=("codec", "dct8", "video", "drawing", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_rpn_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "phase_target": "always_on_foundation",
                "verified_by": ["knowledge3d/cranium/tests/test_rpn_codec_integration.py"],
                "procedural_goal": "convert visual grids into compact frequency blocks and back",
            },
        ),
        ToolNode(
            tool_id="tool_codec_audio_mdct_v1",
            name="Audio MDCT Codec Surface",
            category="procedural_codec",
            tool_kind="procedural_codec",
            modalities=("audio", "signal"),
            description=(
                "PTX-backed MDCT/iMDCT surface for procedural audio analysis and reconstruction. "
                "This is the canonical audio frequency-time codec currently available in runtime."
            ),
            rpn_program="1024 BATCH_MDCT 0.1 TERNARY_QUANT TERNARY_DEQUANT 1024 IMDCT",
            inputs=("audio_frames", "frame_size_even", "ternary_threshold"),
            outputs=("mdct_coefficients", "ternary_audio_coefficients", "reconstructed_audio"),
            codec_ops=("BATCH_MDCT", "TERNARY_QUANT", "TERNARY_DEQUANT", "IMDCT"),
            tags=("codec", "audio", "mdct", "signal", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_rpn_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "phase_target": "always_on_foundation",
                "verified_by": [
                    "knowledge3d/cranium/tests/test_ternary_codec_ops.py",
                    "knowledge3d/cranium/tests/test_rpn_codec_integration.py",
                ],
                "procedural_goal": "map waveform frames into compact spectral coefficients and reconstruct them deterministically",
            },
        ),
    ]


def _procedural_paint_tools() -> list[ToolNode]:
    """Bridge-backed PTX visual tools always available to the system."""

    return [
        ToolNode(
            tool_id="tool_paint_gradient_backdrop_v1",
            name="Gradient Backdrop Paint Surface",
            category="procedural_paint",
            tool_kind="ptx_canvas_surface",
            modalities=("drawing", "video", "signal"),
            description=(
                "PTX-backed gradient backdrop surface for procedural painting. "
                "Provides linear, radial, and conic backgrounds through the canonical drawing effects bridge."
            ),
            rpn_program="GRADIENT_LINEAR GRADIENT_RADIAL GRADIENT_CONIC",
            inputs=("canvas_size", "gradient_stops", "gradient_geometry"),
            outputs=("rgba_backdrop", "procedural_background"),
            tags=("paint", "gradient", "canvas", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=1,
                cascade=("parallel_fanout",),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.linear_gradient",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.radial_gradient",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.conic_gradient",
                ],
                "phase_target": "phase_1_visual_editing",
                "verified_by": ["tests/test_drawing_effects_gpu.py"],
                "procedural_goal": "render painterly backdrops without bitmap assets",
            },
        ),
        ToolNode(
            tool_id="tool_paint_filter_stack_v1",
            name="Paint Filter Stack",
            category="procedural_paint",
            tool_kind="ptx_canvas_surface",
            modalities=("drawing", "video", "signal"),
            description=(
                "PTX-backed blur, sharpen, and invert stack for procedural canvases. "
                "These are the canonical image-editing primitives for painterly post-processing."
            ),
            rpn_program="FILTER_BLUR FILTER_SHARPEN FILTER_INVERT",
            inputs=("rgba_canvas", "blur_radius", "sharpen_amount"),
            outputs=("filtered_canvas", "stylized_canvas"),
            tags=("paint", "blur", "sharpen", "invert", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.blur_rgba",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.sharpen_rgba",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.invert_rgba",
                ],
                "phase_target": "phase_1_visual_editing",
                "verified_by": ["tests/test_drawing_effects_gpu.py"],
                "procedural_goal": "apply painterly post-processing without leaving the sovereign canvas",
            },
        ),
        ToolNode(
            tool_id="tool_paint_composite_edge_v1",
            name="Canvas Composite and Edge Ink",
            category="procedural_paint",
            tool_kind="ptx_canvas_surface",
            modalities=("drawing", "video", "signal"),
            description=(
                "PTX-backed alpha compositing and Sobel edge extraction for procedural canvases. "
                "This turns drawings plus backdrops into a stylized, inspectable paint stack."
            ),
            rpn_program="BLEND_MODE FILTER_EDGE",
            inputs=("background_canvas", "foreground_canvas"),
            outputs=("composited_canvas", "edge_map"),
            tags=("paint", "composite", "edges", "ink", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.alpha_over_rgba",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.edge_map",
                    "knowledge3d.cranium.bridges.procedural_drawing_bridge.ProceduralDrawingBridge.render_painterly_gpu",
                ],
                "phase_target": "phase_1_visual_editing",
                "verified_by": ["tests/test_drawing_effects_gpu.py"],
                "procedural_goal": "compose procedural layers and extract ink-style structure on GPU",
            },
        ),
        ToolNode(
            tool_id="tool_paint_palette_contrastive_v1",
            name="Ternary Contrastive Palette Surface",
            category="procedural_paint",
            tool_kind="ptx_canvas_surface",
            modalities=("drawing", "video", "signal"),
            description=(
                "Ternary contrastive palette selection surface. Converts compact color palettes into "
                "gradient-like ternary signatures so the system can prefer positive candidates and reject "
                "negative examples without heavier color toolchains."
            ),
            rpn_program="TERNARY_QUANT PALETTE_CONTRASTIVE_SCORE",
            inputs=("target_palette", "candidate_palette", "negative_palettes"),
            outputs=("contrastive_score", "palette_signature", "gradient_stops"),
            codec_ops=("TERNARY_QUANT",),
            tags=("paint", "palette", "contrastive", "ternary", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=1,
                cascade=("parallel_fanout",),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.palette_to_gradient_stops",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.encode_palette_signature",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.contrastive_palette_score",
                ],
                "phase_target": "phase_1_visual_editing",
                "verified_by": ["tests/test_drawing_effects_gpu.py"],
                "procedural_goal": "select compact procedural palettes with ternary positive/negative pressure",
            },
        ),
        ToolNode(
            tool_id="tool_paint_gradient_contrastive_v1",
            name="Ternary Contrastive Gradient Surface",
            category="procedural_paint",
            tool_kind="ptx_canvas_surface",
            modalities=("drawing", "video", "signal"),
            description=(
                "PTX-backed ternary contrastive gradient analysis. Encodes gradient stop deltas as ternary trends, "
                "scores candidates against positives and negatives, and supports cheap gradient comparison."
            ),
            rpn_program="TERNARY_QUANT GRADIENT_CONTRASTIVE_SCORE",
            inputs=("target_gradient", "candidate_gradient", "negative_gradients"),
            outputs=("contrastive_score", "gradient_signature"),
            codec_ops=("TERNARY_QUANT",),
            tags=("paint", "gradient", "contrastive", "ternary", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=1,
                cascade=("parallel_fanout",),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.encode_gradient_signature",
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.contrastive_gradient_score",
                ],
                "phase_target": "phase_1_visual_editing",
                "verified_by": ["tests/test_drawing_effects_gpu.py"],
                "procedural_goal": "learn and compare gradients with positive, negative, and neutral ternary signals",
            },
        ),
        ToolNode(
            tool_id="tool_paint_gradient_cascade_v1",
            name="Stacked Ternary Gradient Composer",
            category="procedural_paint",
            tool_kind="ptx_canvas_surface",
            modalities=("drawing", "video", "signal"),
            description=(
                "Composes gradients from stacked ternary layers and renders them through the sovereign PTX gradient rasterizer. "
                "This is a cheap substitute for heavier continuous color machinery when ternary layers are sufficient."
            ),
            rpn_program="TERNARY_QUANT GRADIENT_LINEAR",
            inputs=("base_stop", "position_layers", "color_layers"),
            outputs=("gradient_stops", "rgba_gradient"),
            codec_ops=("TERNARY_QUANT",),
            tags=("paint", "gradient", "cascade", "ternary", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=1,
                cascade=("parallel_fanout", "local_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_effects.DrawingEffects.linear_gradient_from_ternary_cascade",
                ],
                "phase_target": "phase_1_visual_editing",
                "verified_by": ["tests/test_drawing_effects_gpu.py"],
                "procedural_goal": "stack ternary deltas into editable gradients without bulk math frameworks",
            },
        ),
    ]


def _procedural_geometry_prep_tools() -> list[ToolNode]:
    """PTX-backed geometry-prep surfaces for contour/profile workflows."""

    return [
        ToolNode(
            tool_id="tool_geom_bbox_crop_v1",
            name="Geometry Bounding Box and Crop Surface",
            category="procedural_geometry_prep",
            tool_kind="geometry_prep",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "PTX-backed bounding-box and crop surface for contour-bearing grids. "
                "This is the canonical extraction step before profile or mesh preparation."
            ),
            rpn_program="FIND_BBOX CROP_REGION",
            inputs=("drawing_grid", "target_color", "padding"),
            outputs=("bbox", "cropped_region"),
            tags=("geometry", "bbox", "crop", "contour", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_runtime_available",
            math_core=_math_core_profile(
                preferred_tier=1,
                cascade=("parallel_fanout",),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.drawing_transform_kernels.find_bbox_gpu",
                    "knowledge3d.cranium.ptx_runtime.drawing_transform_kernels.crop_gpu",
                    "knowledge3d.cranium.ptx_runtime.geometry_prep.GeometryPrep.extract_bbox",
                ],
                "phase_target": "phase_1_draw_extrude_texture",
                "verified_by": ["tests/test_geometry_prep_gpu.py"],
                "procedural_goal": "extract the active contour region deterministically on GPU",
            },
        ),
        ToolNode(
            tool_id="tool_geom_profile_prep_v1",
            name="Contour Profile Preparation Surface",
            category="procedural_geometry_prep",
            tool_kind="profile_prep",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "PTX-backed contour profile preparation for 2D -> 3D workflows. "
                "Returns bounded profile regions and silhouette metadata for later lathe/extrusion composition."
            ),
            rpn_program="tool_geom_bbox_crop_v1 PROFILE_SILHOUETTE PREPARE_PROFILE_METADATA",
            inputs=("drawing_grid", "target_color", "padding"),
            outputs=("profile_region", "column_fill", "silhouette_hints"),
            tool_refs=("tool_geom_bbox_crop_v1",),
            object_refs=("obj3d_gen_lathe_profile", "obj3d_mesh_compute_normal"),
            tags=("geometry", "profile", "lathe", "extrusion", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_runtime_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.ptx_runtime.geometry_prep.GeometryPrep.prepare_profile",
                ],
                "phase_target": "phase_1_draw_extrude_texture",
                "verified_by": ["tests/test_geometry_prep_gpu.py"],
                "procedural_goal": "convert contour-bearing drawing regions into reusable profile metadata",
            },
        ),
        ToolNode(
            tool_id="tool_geom_profile_lathe_mesh_v1",
            name="Profile to Lathe Mesh Surface",
            category="procedural_geometry_prep",
            tool_kind="geometry_mesh_bridge",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "Bridge-stage surface that turns a prepared contour profile into deterministic "
                "lathe-style mesh vertices, faces, and normals."
            ),
            rpn_program="tool_geom_profile_prep_v1 PROFILE_TO_LATHE_MESH",
            inputs=("drawing_grid", "target_color", "segments", "radius_scale"),
            outputs=("mesh_vertices", "mesh_indices", "mesh_normals"),
            tool_refs=("tool_geom_profile_prep_v1",),
            object_refs=("obj3d_gen_lathe_profile", "obj3d_mesh_compute_normal"),
            codec_ops=("TERNARY_QUANT",),
            tags=("geometry", "lathe", "mesh", "3d", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_geometry_bridge.ProceduralGeometryBridge.contour_to_lathe_mesh",
                ],
                "phase_target": "phase_1_draw_extrude_texture",
                "verified_by": ["tests/test_procedural_geometry_bridge.py"],
                "procedural_goal": "emit deterministic mesh topology from a PTX-prepared contour profile using ternary trend reduction where neutral spans can collapse cheaply",
            },
        ),
        ToolNode(
            tool_id="tool_geom_profile_extrude_mesh_v1",
            name="Profile to Extrude Mesh Surface",
            category="procedural_geometry_prep",
            tool_kind="geometry_mesh_bridge",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "Bridge-stage surface that turns a prepared contour profile into deterministic "
                "extruded mesh vertices, faces, and normals."
            ),
            rpn_program="tool_geom_profile_prep_v1 PROFILE_TO_EXTRUDE_MESH",
            inputs=("drawing_grid", "target_color", "depth_scale", "width_scale"),
            outputs=("mesh_vertices", "mesh_indices", "mesh_normals"),
            tool_refs=("tool_geom_profile_prep_v1",),
            object_refs=("obj3d_gen_lathe_profile", "obj3d_mesh_compute_normal"),
            codec_ops=("TERNARY_QUANT",),
            tags=("geometry", "extrude", "mesh", "3d", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_geometry_bridge.ProceduralGeometryBridge.contour_to_extrude_mesh",
                ],
                "phase_target": "phase_1_draw_extrude_texture",
                "verified_by": ["tests/test_procedural_geometry_bridge.py"],
                "procedural_goal": "emit deterministic extruded mesh topology from a PTX-prepared contour profile using ternary contour motion reduction",
            },
        ),
        ToolNode(
            tool_id="tool_geom_profile_sweep_mesh_v1",
            name="Profile to Sweep Mesh Surface",
            category="procedural_geometry_prep",
            tool_kind="geometry_mesh_bridge",
            modalities=("drawing", "3dobjects", "reality"),
            description=(
                "Bridge-stage surface that turns a prepared contour profile into a deterministic "
                "centerline-following sweep mesh."
            ),
            rpn_program="tool_geom_profile_prep_v1 PROFILE_TO_SWEEP_MESH",
            inputs=("drawing_grid", "target_color", "depth_scale", "width_scale"),
            outputs=("mesh_vertices", "mesh_indices", "mesh_normals"),
            tool_refs=("tool_geom_profile_prep_v1",),
            object_refs=("obj3d_gen_lathe_profile", "obj3d_mesh_compute_normal"),
            codec_ops=("TERNARY_QUANT",),
            tags=("geometry", "sweep", "mesh", "3d", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=2,
                cascade=("parallel_fanout", "worker_reduce"),
            ),
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_geometry_bridge.ProceduralGeometryBridge.contour_to_sweep_mesh",
                ],
                "phase_target": "phase_1_draw_extrude_texture",
                "verified_by": ["tests/test_procedural_geometry_bridge.py"],
                "procedural_goal": "emit deterministic sweep mesh topology from a PTX-prepared contour profile using ternary centerline motion reduction",
            },
        ),
    ]


def _procedural_temporal_tools() -> list[ToolNode]:
    """Temporal/video tools built on the real signal/material/geometry surfaces."""

    return [
        ToolNode(
            tool_id="tool_video_temporal_preview_v1",
            name="Temporal Preview Surface",
            category="procedural_temporal_video",
            tool_kind="temporal_layering",
            modalities=("drawing", "video", "signal", "reality"),
            description=(
                "Bridge-stage temporal preview surface that turns a textured procedural "
                "surface into a deterministic frame sequence with PTX temporal deltas and "
                "coherence analysis."
            ),
            rpn_program="tool_codec_video_dct8_grid_v1 tool_mathcore_tier2_vector_worker_v1 tool_mathcore_tier3_master_v1",
            inputs=("textured_surface", "frame_count", "time_span"),
            outputs=("temporal_preview", "temporal_deltas", "coherence_summary"),
            tool_refs=("tool_codec_video_dct8_grid_v1",),
            codec_ops=("DCT8", "TERNARY_QUANT"),
            tags=("temporal", "video", "timeline", "preview", "always_on", "ptx"),
            promotion_stage="kernel",
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            entrypoint_argument_schemas={
                "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_temporal_preview": _entrypoint_argument_schema(
                    positional=(
                        {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface", "surface_plan")},
                    ),
                    optional_kwargs=(
                        {"name": "frame_count", "type": "int", "default": 4},
                        {"name": "time_span", "type": "float", "default": 1.0},
                        {"name": "feature_grid", "type": "int", "default": 8},
                        {"name": "encode_frames", "type": "bool", "default": True},
                        {"name": "codec_threshold", "type": "float", "default": 0.2},
                        {"name": "timeline_id", "type": "str|None", "default": None},
                    ),
                ),
                "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset": _entrypoint_argument_schema(
                    positional=(
                        {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface", "surface_plan")},
                    ),
                    optional_kwargs=(
                        {"name": "timeline_preset", "type": "str|None", "default": None},
                        {"name": "frame_count", "type": "int", "default": None},
                        {"name": "time_span", "type": "float", "default": None},
                        {"name": "feature_grid", "type": "int", "default": None},
                        {"name": "encode_frames", "type": "bool", "default": True},
                        {"name": "codec_threshold", "type": "float", "default": 0.2},
                        {"name": "timeline_id", "type": "str|None", "default": None},
                    ),
                ),
            },
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_temporal_preview",
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset",
                ],
                "phase_target": "phase_2_signal_fusion",
                "verified_by": ["tests/test_procedural_temporal_bridge.py"],
                "procedural_goal": "turn procedural surfaces into temporal previews with real PTX delta/coherence analysis",
                "timeline_presets": ["ui_idle", "ui_focus", "world_breathe", "world_orbit"],
            },
        ),
        ToolNode(
            tool_id="tool_fusion_surface_material_timeline_v1",
            name="Surface Material Timeline Fusion",
            category="drawing_reality_video_fusion",
            tool_kind="temporal_fusion",
            modalities=("drawing", "3dobjects", "reality", "video"),
            description=(
                "Executable temporal fusion route from contour-bearing drawings through "
                "textured 3D surfaces into a temporal preview sequence."
            ),
            rpn_program="tool_fusion_surface_material_projection_v1 tool_video_temporal_preview_v1 TERNARY_QUANT",
            inputs=("drawing_contour", "surface_material", "material_candidates", "frame_count", "timeline_preset"),
            outputs=("temporal_preview", "textured_surface", "coherence_summary"),
            tool_refs=(
                "tool_fusion_surface_material_projection_v1",
                "tool_video_temporal_preview_v1",
            ),
            codec_ops=("TERNARY_QUANT", "DCT8"),
            tags=("fusion", "temporal", "timeline", "video", "surface", "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            execution_chain_presets={
                "contour_timeline_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates"),
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.contour_to_textured_lathe_mesh"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                                ),
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "color", "type": "int", "default": 1},
                                    {"name": "pad", "type": "int", "default": 0},
                                    {"name": "segments", "type": "int", "default": 24},
                                    {"name": "height_scale", "type": "float", "default": 1.0},
                                    {"name": "radius_scale", "type": "float", "default": 1.0},
                                    {"name": "cap_ends", "type": "bool", "default": True},
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_temporal_bridge."
                                "ProceduralTemporalBridge.surface_material_to_timeline_preset"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface",)},
                                ),
                                optional_kwargs=(
                                    {"name": "timeline_preset", "type": "str|None", "default": None},
                                    {"name": "frame_count", "type": "int", "default": None},
                                    {"name": "time_span", "type": "float", "default": None},
                                    {"name": "feature_grid", "type": "int", "default": None},
                                    {"name": "encode_frames", "type": "bool", "default": True},
                                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                                    {"name": "timeline_id", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("temporal_preview",),
                        ),
                    ),
                    return_alias="temporal_preview",
                ),
                "contour_extrude_timeline_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates", "geometry_mode"),
                    selectors={"geometry_mode": "extrude"},
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.contour_to_textured_extrude_mesh"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                                ),
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "color", "type": "int", "default": 1},
                                    {"name": "pad", "type": "int", "default": 0},
                                    {"name": "depth_scale", "type": "float", "default": 0.5},
                                    {"name": "width_scale", "type": "float", "default": 1.0},
                                    {"name": "height_scale", "type": "float", "default": 1.0},
                                    {"name": "cap_ends", "type": "bool", "default": True},
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_temporal_bridge."
                                "ProceduralTemporalBridge.surface_material_to_timeline_preset"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface",)},
                                ),
                                optional_kwargs=(
                                    {"name": "timeline_preset", "type": "str|None", "default": None},
                                    {"name": "frame_count", "type": "int", "default": None},
                                    {"name": "time_span", "type": "float", "default": None},
                                    {"name": "feature_grid", "type": "int", "default": None},
                                    {"name": "encode_frames", "type": "bool", "default": True},
                                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                                    {"name": "timeline_id", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("temporal_preview",),
                        ),
                    ),
                    return_alias="temporal_preview",
                ),
                "contour_sweep_timeline_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates", "geometry_mode"),
                    selectors={"geometry_mode": "sweep"},
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.contour_to_textured_sweep_mesh"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                                ),
                                required_kwargs=(
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "color", "type": "int", "default": 1},
                                    {"name": "pad", "type": "int", "default": 0},
                                    {"name": "depth_scale", "type": "float", "default": 0.5},
                                    {"name": "width_scale", "type": "float", "default": 1.0},
                                    {"name": "height_scale", "type": "float", "default": 1.0},
                                    {"name": "cap_ends", "type": "bool", "default": True},
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_temporal_bridge."
                                "ProceduralTemporalBridge.surface_material_to_timeline_preset"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface",)},
                                ),
                                optional_kwargs=(
                                    {"name": "timeline_preset", "type": "str|None", "default": None},
                                    {"name": "frame_count", "type": "int", "default": None},
                                    {"name": "time_span", "type": "float", "default": None},
                                    {"name": "feature_grid", "type": "int", "default": None},
                                    {"name": "encode_frames", "type": "bool", "default": True},
                                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                                    {"name": "timeline_id", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("temporal_preview",),
                        ),
                    ),
                    return_alias="temporal_preview",
                ),
            },
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_temporal_preview",
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset",
                ],
                "phase_target": "phase_2_signal_fusion",
                "verified_by": [
                    "tests/test_tool_execution.py",
                    "tests/test_procedural_temporal_bridge.py",
                ],
                "procedural_goal": "carry contour-driven textured surfaces into reusable temporal routes, including named UI/world animation presets",
                "timeline_presets": ["ui_idle", "ui_focus", "world_breathe", "world_orbit"],
            },
        ),
        ToolNode(
            tool_id="tool_fusion_signal_surface_material_timeline_v1",
            name="Signal Surface Material Timeline Fusion",
            category="audio_drawing_reality_video_fusion",
            tool_kind="temporal_fusion",
            modalities=("audio", "drawing", "3dobjects", "reality", "signal", "video"),
            description=(
                "Executable temporal fusion route from audio through spectrogram, surface, "
                "material projection, and temporal preview sequence."
            ),
            rpn_program="tool_fusion_signal_surface_material_v1 tool_video_temporal_preview_v1 TERNARY_QUANT",
            inputs=("clip_id", "audio_signal", "material_candidates", "frame_count", "timeline_preset"),
            outputs=("temporal_preview", "textured_surface", "coherence_summary"),
            tool_refs=(
                "tool_fusion_signal_surface_material_v1",
                "tool_video_temporal_preview_v1",
            ),
            codec_ops=("MDCT", "TERNARY_QUANT", "DCT8"),
            tags=("fusion", "temporal", "timeline", "audio", "video", "signal", "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            execution_chain_presets={
                "signal_timeline_chain": _execution_chain_preset(
                    required_inputs=("clip_id", "audio_signal", "material_candidates"),
                    steps=(
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_material_bridge."
                                "ProceduralMaterialBridge.signal_to_textured_surface"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    ("clip_id", "str"),
                                    {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                                ),
                                required_kwargs=(
                                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                                ),
                                optional_kwargs=(
                                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                                    {"name": "target_material", "type": "SurfaceMaterialCandidate|None", "aliases": ("surface_material",), "default": None},
                                    {"name": "frame_size", "type": "int", "default": 1024},
                                    {"name": "threshold", "type": "float", "default": 0.2},
                                    {"name": "displacement_gain", "type": "float", "default": 0.25},
                                    {"name": "preview_size", "type": "int", "default": 64},
                                    {"name": "projection_strategy", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("textured_surface",),
                        ),
                        _execution_chain_step(
                            entrypoint=(
                                "knowledge3d.cranium.bridges.procedural_temporal_bridge."
                                "ProceduralTemporalBridge.surface_material_to_timeline_preset"
                            ),
                            argument_schema=_entrypoint_argument_schema(
                                positional=(
                                    {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface",)},
                                ),
                                optional_kwargs=(
                                    {"name": "timeline_preset", "type": "str|None", "default": None},
                                    {"name": "frame_count", "type": "int", "default": None},
                                    {"name": "time_span", "type": "float", "default": None},
                                    {"name": "feature_grid", "type": "int", "default": None},
                                    {"name": "encode_frames", "type": "bool", "default": True},
                                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                                    {"name": "timeline_id", "type": "str|None", "default": None},
                                ),
                            ),
                            store_as=("temporal_preview",),
                        ),
                    ),
                    return_alias="temporal_preview",
                ),
            },
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_temporal_preview",
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface",
                ],
                "phase_target": "phase_2_signal_fusion",
                "verified_by": [
                    "tests/test_tool_execution.py",
                    "tests/test_procedural_temporal_bridge.py",
                ],
                "procedural_goal": "promote signal-derived textured surfaces into temporal previews and named UI/world animation presets inside the single Tool-execution system",
                "timeline_presets": ["ui_idle", "ui_focus", "world_breathe", "world_orbit"],
            },
        ),
    ]


def _procedural_timeline_specialization_tools() -> list[ToolNode]:
    """Canonical UI/world animation routes built on top of the generic timeline layer."""

    def _timeline_step(default_preset: str) -> dict[str, Any]:
        return _execution_chain_step(
            entrypoint=(
                "knowledge3d.cranium.bridges.procedural_temporal_bridge."
                "ProceduralTemporalBridge.surface_material_to_timeline_preset"
            ),
            argument_schema=_entrypoint_argument_schema(
                positional=(
                    {"name": "surface_plan", "type": "SurfaceMaterialPlan", "aliases": ("textured_surface",)},
                ),
                optional_kwargs=(
                    {"name": "timeline_preset", "type": "str|None", "default": default_preset},
                    {"name": "frame_count", "type": "int", "default": None},
                    {"name": "time_span", "type": "float", "default": None},
                    {"name": "feature_grid", "type": "int", "default": None},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "timeline_id", "type": "str|None", "default": None},
                ),
            ),
            store_as=("temporal_preview",),
        )

    def _contour_surface_step(mode: str) -> dict[str, Any]:
        if mode == "extrude":
            entrypoint = (
                "knowledge3d.cranium.bridges.procedural_material_bridge."
                "ProceduralMaterialBridge.contour_to_textured_extrude_mesh"
            )
            optional_kwargs = (
                {"name": "color", "type": "int", "default": 1},
                {"name": "pad", "type": "int", "default": 0},
                {"name": "depth_scale", "type": "float", "default": 0.5},
                {"name": "width_scale", "type": "float", "default": 1.0},
                {"name": "height_scale", "type": "float", "default": 1.0},
                {"name": "cap_ends", "type": "bool", "default": True},
                {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                {"name": "preview_size", "type": "int", "default": 64},
                {"name": "projection_strategy", "type": "str|None", "default": None},
            )
        elif mode == "sweep":
            entrypoint = (
                "knowledge3d.cranium.bridges.procedural_material_bridge."
                "ProceduralMaterialBridge.contour_to_textured_sweep_mesh"
            )
            optional_kwargs = (
                {"name": "color", "type": "int", "default": 1},
                {"name": "pad", "type": "int", "default": 0},
                {"name": "depth_scale", "type": "float", "default": 0.5},
                {"name": "width_scale", "type": "float", "default": 1.0},
                {"name": "height_scale", "type": "float", "default": 1.0},
                {"name": "cap_ends", "type": "bool", "default": True},
                {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                {"name": "preview_size", "type": "int", "default": 64},
                {"name": "projection_strategy", "type": "str|None", "default": None},
            )
        else:
            entrypoint = (
                "knowledge3d.cranium.bridges.procedural_material_bridge."
                "ProceduralMaterialBridge.contour_to_textured_lathe_mesh"
            )
            optional_kwargs = (
                {"name": "color", "type": "int", "default": 1},
                {"name": "pad", "type": "int", "default": 0},
                {"name": "segments", "type": "int", "default": 24},
                {"name": "height_scale", "type": "float", "default": 1.0},
                {"name": "radius_scale", "type": "float", "default": 1.0},
                {"name": "cap_ends", "type": "bool", "default": True},
                {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                {"name": "preview_size", "type": "int", "default": 64},
                {"name": "projection_strategy", "type": "str|None", "default": None},
            )
        return _execution_chain_step(
            entrypoint=entrypoint,
            argument_schema=_entrypoint_argument_schema(
                positional=(
                    {"name": "grid", "type": "ndarray", "aliases": ("drawing_grid", "drawing_contour")},
                ),
                required_kwargs=(
                    {"name": "target_material", "type": "SurfaceMaterialCandidate", "aliases": ("surface_material",)},
                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                ),
                optional_kwargs=optional_kwargs,
            ),
            store_as=("textured_surface",),
        )

    def _signal_surface_step() -> dict[str, Any]:
        return _execution_chain_step(
            entrypoint=(
                "knowledge3d.cranium.bridges.procedural_material_bridge."
                "ProceduralMaterialBridge.signal_to_textured_surface"
            ),
            argument_schema=_entrypoint_argument_schema(
                positional=(
                    ("clip_id", "str"),
                    {"name": "samples", "type": "TernaryVector", "aliases": ("audio_signal",)},
                ),
                required_kwargs=(
                    {"name": "candidates", "type": "Sequence[SurfaceMaterialCandidate]", "aliases": ("material_candidates",)},
                ),
                optional_kwargs=(
                    {"name": "negative_materials", "type": "Sequence[SurfaceMaterialCandidate]", "default": []},
                    {"name": "target_material", "type": "SurfaceMaterialCandidate|None", "aliases": ("surface_material",), "default": None},
                    {"name": "frame_size", "type": "int", "default": 1024},
                    {"name": "threshold", "type": "float", "default": 0.2},
                    {"name": "displacement_gain", "type": "float", "default": 0.25},
                    {"name": "preview_size", "type": "int", "default": 64},
                    {"name": "projection_strategy", "type": "str|None", "default": None},
                ),
            ),
            store_as=("textured_surface",),
        )

    def _surface_animation_tool(
        *,
        tool_id: str,
        name: str,
        category: str,
        tool_kind: str,
        description: str,
        default_preset: str,
        preset_family: tuple[str, ...],
    ) -> ToolNode:
        preset_label = "ui" if default_preset.startswith("ui_") else "world"
        return ToolNode(
            tool_id=tool_id,
            name=name,
            category=category,
            tool_kind=tool_kind,
            modalities=("drawing", "3dobjects", "reality", "video"),
            description=description,
            rpn_program=f"tool_fusion_surface_material_projection_v1 tool_video_temporal_preview_v1 {default_preset} TERNARY_QUANT",
            inputs=("drawing_contour", "surface_material", "material_candidates", "frame_count", "timeline_preset"),
            outputs=("temporal_preview", "textured_surface", "coherence_summary"),
            tool_refs=("tool_fusion_surface_material_projection_v1", "tool_video_temporal_preview_v1"),
            codec_ops=("TERNARY_QUANT", "DCT8"),
            tags=("fusion", "temporal", "timeline", "video", preset_label, "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            execution_chain_presets={
                f"{preset_label}_lathe_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates"),
                    steps=(_contour_surface_step("lathe"), _timeline_step(default_preset)),
                    return_alias="temporal_preview",
                ),
                f"{preset_label}_extrude_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates", "geometry_mode"),
                    selectors={"geometry_mode": "extrude"},
                    steps=(_contour_surface_step("extrude"), _timeline_step(default_preset)),
                    return_alias="temporal_preview",
                ),
                f"{preset_label}_sweep_chain": _execution_chain_preset(
                    required_inputs=("drawing_contour", "surface_material", "material_candidates", "geometry_mode"),
                    selectors={"geometry_mode": "sweep"},
                    steps=(_contour_surface_step("sweep"), _timeline_step(default_preset)),
                    return_alias="temporal_preview",
                ),
            },
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_lathe_mesh",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_extrude_mesh",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.contour_to_textured_sweep_mesh",
                ],
                "phase_target": "phase_2_signal_fusion",
                "verified_by": [
                    "tests/test_tool_execution.py",
                    "tests/test_procedural_temporal_bridge.py",
                    "tests/test_tool_galaxy.py",
                ],
                "procedural_goal": description,
                "timeline_presets": list(preset_family),
                "default_timeline_preset": default_preset,
            },
        )

    def _signal_animation_tool(
        *,
        tool_id: str,
        name: str,
        category: str,
        tool_kind: str,
        description: str,
        default_preset: str,
        preset_family: tuple[str, ...],
    ) -> ToolNode:
        preset_label = "ui" if default_preset.startswith("ui_") else "world"
        return ToolNode(
            tool_id=tool_id,
            name=name,
            category=category,
            tool_kind=tool_kind,
            modalities=("audio", "drawing", "3dobjects", "reality", "signal", "video"),
            description=description,
            rpn_program=f"tool_fusion_signal_surface_material_v1 tool_video_temporal_preview_v1 {default_preset} TERNARY_QUANT",
            inputs=("clip_id", "audio_signal", "material_candidates", "frame_count", "timeline_preset"),
            outputs=("temporal_preview", "textured_surface", "coherence_summary"),
            tool_refs=("tool_fusion_signal_surface_material_v1", "tool_video_temporal_preview_v1"),
            codec_ops=("MDCT", "TERNARY_QUANT", "DCT8"),
            tags=("fusion", "temporal", "timeline", "audio", "video", preset_label, "recipe"),
            runtime_status="ptx_bridge_available",
            math_core=_math_core_profile(
                preferred_tier=3,
                cascade=("parallel_fanout", "worker_reduce", "master_commit"),
            ),
            execution_chain_presets={
                f"{preset_label}_signal_chain": _execution_chain_preset(
                    required_inputs=("clip_id", "audio_signal", "material_candidates"),
                    steps=(_signal_surface_step(), _timeline_step(default_preset)),
                    return_alias="temporal_preview",
                ),
            },
            metadata_extra={
                "entrypoints": [
                    "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.surface_material_to_timeline_preset",
                    "knowledge3d.cranium.bridges.procedural_material_bridge.ProceduralMaterialBridge.signal_to_textured_surface",
                ],
                "phase_target": "phase_2_signal_fusion",
                "verified_by": [
                    "tests/test_tool_execution.py",
                    "tests/test_procedural_temporal_bridge.py",
                    "tests/test_tool_galaxy.py",
                ],
                "procedural_goal": description,
                "timeline_presets": list(preset_family),
                "default_timeline_preset": default_preset,
            },
        )

    return [
        _surface_animation_tool(
            tool_id="tool_fusion_surface_material_ui_animation_v1",
            name="Surface Material UI Animation Fusion",
            category="drawing_reality_ui_video_fusion",
            tool_kind="ui_animation_fusion",
            description="Carry contour-driven textured surfaces into canonical UI/HUD animation routes with default UI timeline behavior.",
            default_preset="ui_idle",
            preset_family=("ui_idle", "ui_focus"),
        ),
        _surface_animation_tool(
            tool_id="tool_fusion_surface_material_world_animation_v1",
            name="Surface Material World Animation Fusion",
            category="drawing_reality_world_video_fusion",
            tool_kind="world_animation_fusion",
            description="Carry contour-driven textured surfaces into canonical world/surface animation routes with default ambient world timeline behavior.",
            default_preset="world_breathe",
            preset_family=("world_breathe", "world_orbit"),
        ),
        _signal_animation_tool(
            tool_id="tool_fusion_signal_surface_material_ui_animation_v1",
            name="Signal Surface UI Animation Fusion",
            category="audio_drawing_reality_ui_video_fusion",
            tool_kind="ui_animation_fusion",
            description="Carry audio-derived textured surfaces into canonical UI/HUD animation routes with default UI timeline behavior.",
            default_preset="ui_idle",
            preset_family=("ui_idle", "ui_focus"),
        ),
        _signal_animation_tool(
            tool_id="tool_fusion_signal_surface_material_world_animation_v1",
            name="Signal Surface World Animation Fusion",
            category="audio_drawing_reality_world_video_fusion",
            tool_kind="world_animation_fusion",
            description="Carry audio-derived textured surfaces into canonical world/surface animation routes with default ambient world timeline behavior.",
            default_preset="world_breathe",
            preset_family=("world_breathe", "world_orbit"),
        ),
    ]


def _procedural_scene_tools() -> list[ToolNode]:
    """Scene-level temporal composition built on the existing timeline layer."""

    scene_entrypoint = (
        "knowledge3d.cranium.bridges.procedural_temporal_bridge."
        "ProceduralTemporalBridge.compose_scene_timeline"
    )
    surface_scene_entrypoint = (
        "knowledge3d.cranium.bridges.procedural_temporal_bridge."
        "ProceduralTemporalBridge.surface_materials_to_scene_timeline"
    )
    house_room_scene_entrypoint = (
        "knowledge3d.cranium.bridges.procedural_temporal_bridge."
        "ProceduralTemporalBridge.execution_events_to_house_room_scene"
    )
    house_tour_scene_entrypoint = (
        "knowledge3d.cranium.bridges.procedural_temporal_bridge."
        "ProceduralTemporalBridge.execution_events_to_house_tour_scene"
    )

    generic_scene = ToolNode(
        tool_id="tool_video_temporal_scene_v1",
        name="Temporal Scene Composition",
        category="procedural_temporal_scene",
        tool_kind="temporal_scene",
        modalities=("drawing", "reality", "signal", "video"),
        description=(
            "Compose multiple temporal preview layers into one deterministic scene "
            "timeline using the PTX alpha compositing surface."
        ),
        rpn_program="tool_video_temporal_preview_v1 alpha_over_rgba TERNARY_QUANT",
        inputs=("temporal_layers", "textured_surfaces", "timeline_preset", "scene_layout"),
        outputs=("scene_timeline", "scene_coherence_summary"),
        tool_refs=("tool_video_temporal_preview_v1",),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "temporal", "video", "layering", "composition", "always_on"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            scene_entrypoint: _entrypoint_argument_schema(
                positional=(
                    {"name": "layers", "type": "Sequence[TemporalSceneLayer]", "aliases": ("temporal_layers",)},
                ),
                optional_kwargs=(
                    {"name": "canvas_width", "type": "int|None", "default": None},
                    {"name": "canvas_height", "type": "int|None", "default": None},
                    {"name": "background_rgba", "type": "ndarray|None", "default": None},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                    {"name": "scene_layout", "type": "str", "default": "overlay"},
                ),
            ),
            surface_scene_entrypoint: _entrypoint_argument_schema(
                positional=(
                    {"name": "surface_plans", "type": "Sequence[SurfaceMaterialPlan]", "aliases": ("textured_surfaces", "surface_layers")},
                ),
                optional_kwargs=(
                    {"name": "timeline_preset", "type": "str|None", "default": None},
                    {"name": "frame_count", "type": "int|None", "default": None},
                    {"name": "time_span", "type": "float|None", "default": None},
                    {"name": "feature_grid", "type": "int|None", "default": None},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_layout", "type": "str", "default": "overlay"},
                    {"name": "scene_id", "type": "str|None", "default": None},
                    {"name": "layer_opacity", "type": "Sequence[float]|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [scene_entrypoint, surface_scene_entrypoint],
            "phase_target": "phase_3_scene_layering",
            "verified_by": [
                "tests/test_procedural_temporal_bridge.py",
                "tests/test_tool_execution.py",
                "tests/test_tool_galaxy.py",
            ],
            "procedural_goal": "carry the timeline system into scene-level UI/world playback without leaving the Tool galaxy contract",
            "scene_layouts": ["overlay", "horizontal_strip", "vertical_strip", "golden_orbit"],
            "timeline_presets": ["ui_idle", "ui_focus", "world_breathe", "world_orbit"],
        },
    )

    ui_scene = ToolNode(
        tool_id="tool_fusion_surface_material_ui_scene_v1",
        name="Surface Material UI Scene Fusion",
        category="drawing_reality_ui_scene_video_fusion",
        tool_kind="ui_scene_fusion",
        modalities=("drawing", "3dobjects", "reality", "video"),
        description=(
            "Compose multiple textured procedural surfaces into a canonical UI/HUD scene "
            "timeline with default UI preset behavior."
        ),
        rpn_program="tool_fusion_surface_material_ui_animation_v1 tool_video_temporal_scene_v1 ui_idle",
        inputs=("textured_surfaces", "scene_layout"),
        outputs=("scene_timeline", "scene_coherence_summary"),
        tool_refs=("tool_fusion_surface_material_ui_animation_v1", "tool_video_temporal_scene_v1"),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "ui", "hud", "temporal", "video", "fusion"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            surface_scene_entrypoint: _entrypoint_argument_schema(
                positional=(
                    {"name": "surface_plans", "type": "Sequence[SurfaceMaterialPlan]", "aliases": ("textured_surfaces", "surface_layers")},
                ),
                optional_kwargs=(
                    {"name": "timeline_preset", "type": "str|None", "default": "ui_idle"},
                    {"name": "frame_count", "type": "int|None", "default": None},
                    {"name": "time_span", "type": "float|None", "default": None},
                    {"name": "feature_grid", "type": "int|None", "default": None},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_layout", "type": "str", "default": "overlay"},
                    {"name": "scene_id", "type": "str|None", "default": None},
                    {"name": "layer_opacity", "type": "Sequence[float]|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [surface_scene_entrypoint],
            "phase_target": "phase_3_scene_layering",
            "verified_by": [
                "tests/test_procedural_temporal_bridge.py",
                "tests/test_tool_execution.py",
                "tests/test_tool_galaxy.py",
            ],
            "procedural_goal": "make UI/HUD scene playback a canonical Tool route over the existing surface timeline substrate",
            "default_timeline_preset": "ui_idle",
            "default_scene_layout": "overlay",
            "timeline_presets": ["ui_idle", "ui_focus"],
        },
    )

    world_scene = ToolNode(
        tool_id="tool_fusion_surface_material_world_scene_v1",
        name="Surface Material World Scene Fusion",
        category="drawing_reality_world_scene_video_fusion",
        tool_kind="world_scene_fusion",
        modalities=("drawing", "3dobjects", "reality", "video"),
        description=(
            "Compose multiple textured procedural surfaces into a canonical world scene "
            "timeline with default ambient world preset behavior."
        ),
        rpn_program="tool_fusion_surface_material_world_animation_v1 tool_video_temporal_scene_v1 world_breathe",
        inputs=("textured_surfaces", "scene_layout"),
        outputs=("scene_timeline", "scene_coherence_summary"),
        tool_refs=("tool_fusion_surface_material_world_animation_v1", "tool_video_temporal_scene_v1"),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "world", "temporal", "video", "fusion"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            surface_scene_entrypoint: _entrypoint_argument_schema(
                positional=(
                    {"name": "surface_plans", "type": "Sequence[SurfaceMaterialPlan]", "aliases": ("textured_surfaces", "surface_layers")},
                ),
                optional_kwargs=(
                    {"name": "timeline_preset", "type": "str|None", "default": "world_breathe"},
                    {"name": "frame_count", "type": "int|None", "default": None},
                    {"name": "time_span", "type": "float|None", "default": None},
                    {"name": "feature_grid", "type": "int|None", "default": None},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_layout", "type": "str", "default": "golden_orbit"},
                    {"name": "scene_id", "type": "str|None", "default": None},
                    {"name": "layer_opacity", "type": "Sequence[float]|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [surface_scene_entrypoint],
            "phase_target": "phase_3_scene_layering",
            "verified_by": [
                "tests/test_procedural_temporal_bridge.py",
                "tests/test_tool_execution.py",
                "tests/test_tool_galaxy.py",
            ],
            "procedural_goal": "make world/House scene playback a canonical Tool route over the existing surface timeline substrate",
            "default_timeline_preset": "world_breathe",
            "default_scene_layout": "golden_orbit",
            "timeline_presets": ["world_breathe", "world_orbit"],
        },
    )
    replay_scene = ToolNode(
        tool_id="tool_house_replay_scene_v1",
        name="House Replay Scene Playback",
        category="house_world_replay_scene_video",
        tool_kind="replay_scene_fusion",
        modalities=("drawing", "video", "signal", "reality"),
        description=(
            "Reconstruct a deterministic House/world playback scene from replay journal "
            "entries or live ActionBuffer records using golden-ratio layer placement."
        ),
        rpn_program="tool_video_temporal_scene_v1 golden_angle action_replay_journal TERNARY_QUANT",
        inputs=("journal_path", "replay_entries", "action_buffers"),
        outputs=("scene_timeline", "scene_coherence_summary", "replay_summary"),
        tool_refs=("tool_video_temporal_scene_v1", "tool_mathcore_tier3_master_v1"),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "replay", "journal", "house", "world", "playback", "fusion"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.replay_journal_to_scene_timeline": _entrypoint_argument_schema(
                optional_kwargs=(
                    {"name": "journal_path", "type": "str|Path|None", "default": None},
                    {"name": "replay_entries", "type": "Sequence[dict]|None", "default": None},
                    {"name": "max_events", "type": "int", "default": 8},
                    {"name": "frame_count", "type": "int", "default": 6},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                    {"name": "scene_layout", "type": "str", "default": "golden_orbit"},
                ),
            ),
            "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.action_buffers_to_scene_timeline": _entrypoint_argument_schema(
                positional=(
                    {"name": "action_buffers", "type": "Sequence[ActionBuffer]", "aliases": ("action_buffers",)},
                ),
                optional_kwargs=(
                    {"name": "max_events", "type": "int", "default": 8},
                    {"name": "frame_count", "type": "int", "default": 6},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                    {"name": "scene_layout", "type": "str", "default": "golden_orbit"},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [
                "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.replay_journal_to_scene_timeline",
                "knowledge3d.cranium.bridges.procedural_temporal_bridge.ProceduralTemporalBridge.action_buffers_to_scene_timeline",
            ],
            "phase_target": "phase_3_scene_layering",
            "verified_by": [
                "tests/test_procedural_temporal_bridge.py",
                "tests/test_tool_execution.py",
                "tests/test_tool_galaxy.py",
            ],
            "procedural_goal": "connect ActionBuffer/replay-journal history into canonical House/world scene playback",
            "default_scene_layout": "golden_orbit",
            "scene_layouts": ["golden_orbit"],
        },
    )
    library_scene = ToolNode(
        tool_id="tool_house_library_scene_v1",
        name="House Library Scene Playback",
        category="house_library_scene_video",
        tool_kind="library_scene_fusion",
        modalities=("drawing", "video", "reality"),
        description="Render settled, high-confidence execution knowledge as a stable House library scene.",
        rpn_program="tool_house_replay_scene_v1 house_library overlay TERNARY_QUANT",
        inputs=("event_log_path", "execution_events", "max_events"),
        outputs=("scene_timeline", "scene_coherence_summary"),
        tool_refs=("tool_house_replay_scene_v1", "tool_video_temporal_scene_v1"),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "house", "library", "knowledge", "settled", "playback"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            house_room_scene_entrypoint: _entrypoint_argument_schema(
                optional_kwargs=(
                    {"name": "event_log_path", "type": "str|Path|None", "default": None},
                    {"name": "execution_events", "type": "Sequence[dict]|None", "default": None},
                    {"name": "room_preset", "type": "str", "default": "house_library"},
                    {"name": "max_events", "type": "int", "default": 8},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [house_room_scene_entrypoint],
            "phase_target": "phase_2d_house_room_playback",
            "default_room_preset": "house_library",
            "default_scene_layout": "overlay",
            "timeline_presets": ["ui_idle"],
        },
    )
    garden_scene = ToolNode(
        tool_id="tool_house_garden_scene_v1",
        name="House Garden Scene Playback",
        category="house_garden_scene_video",
        tool_kind="garden_scene_fusion",
        modalities=("drawing", "video", "reality"),
        description="Render active learning and curiosity-heavy execution patterns as an organic House garden scene.",
        rpn_program="tool_house_replay_scene_v1 house_garden golden_orbit TERNARY_QUANT",
        inputs=("event_log_path", "execution_events", "max_events"),
        outputs=("scene_timeline", "scene_coherence_summary"),
        tool_refs=("tool_house_replay_scene_v1", "tool_video_temporal_scene_v1"),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "house", "garden", "learning", "growth", "exploration"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            house_room_scene_entrypoint: _entrypoint_argument_schema(
                optional_kwargs=(
                    {"name": "event_log_path", "type": "str|Path|None", "default": None},
                    {"name": "execution_events", "type": "Sequence[dict]|None", "default": None},
                    {"name": "room_preset", "type": "str", "default": "house_garden"},
                    {"name": "max_events", "type": "int", "default": 8},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [house_room_scene_entrypoint],
            "phase_target": "phase_2d_house_room_playback",
            "default_room_preset": "house_garden",
            "default_scene_layout": "golden_orbit",
            "timeline_presets": ["world_breathe"],
        },
    )
    museum_scene = ToolNode(
        tool_id="tool_house_museum_scene_v1",
        name="House Museum Scene Playback",
        category="house_museum_scene_video",
        tool_kind="museum_scene_fusion",
        modalities=("drawing", "video", "reality"),
        description="Render failures and archived attempts as a museum-style contrastive lesson timeline.",
        rpn_program="tool_house_replay_scene_v1 house_museum horizontal_strip TERNARY_QUANT",
        inputs=("event_log_path", "execution_events", "max_events"),
        outputs=("scene_timeline", "scene_coherence_summary"),
        tool_refs=("tool_house_replay_scene_v1", "tool_video_temporal_scene_v1"),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "house", "museum", "history", "archive", "failures", "lessons"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            house_room_scene_entrypoint: _entrypoint_argument_schema(
                optional_kwargs=(
                    {"name": "event_log_path", "type": "str|Path|None", "default": None},
                    {"name": "execution_events", "type": "Sequence[dict]|None", "default": None},
                    {"name": "room_preset", "type": "str", "default": "house_museum"},
                    {"name": "max_events", "type": "int", "default": 8},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [house_room_scene_entrypoint],
            "phase_target": "phase_2d_house_room_playback",
            "default_room_preset": "house_museum",
            "default_scene_layout": "horizontal_strip",
            "timeline_presets": ["ui_idle"],
        },
    )
    tour_scene = ToolNode(
        tool_id="tool_house_tour_scene_v1",
        name="House Tour Scene Playback",
        category="house_tour_scene_video",
        tool_kind="tour_scene_fusion",
        modalities=("drawing", "video", "reality"),
        description="Compose library, garden, and museum execution rooms into one compound House tour scene.",
        rpn_program="tool_house_library_scene_v1 tool_house_garden_scene_v1 tool_house_museum_scene_v1 vertical_strip",
        inputs=("event_log_path", "execution_events", "max_events_per_room"),
        outputs=("scene_timeline", "scene_coherence_summary", "tour_summary"),
        tool_refs=(
            "tool_house_library_scene_v1",
            "tool_house_garden_scene_v1",
            "tool_house_museum_scene_v1",
        ),
        codec_ops=("TERNARY_QUANT", "DCT8"),
        tags=("scene", "house", "tour", "overview", "compound", "playback"),
        runtime_status="ptx_bridge_available",
        math_core=_math_core_profile(
            preferred_tier=3,
            cascade=("parallel_fanout", "worker_reduce", "master_commit"),
        ),
        entrypoint_argument_schemas={
            house_tour_scene_entrypoint: _entrypoint_argument_schema(
                optional_kwargs=(
                    {"name": "event_log_path", "type": "str|Path|None", "default": None},
                    {"name": "execution_events", "type": "Sequence[dict]|None", "default": None},
                    {"name": "max_events_per_room", "type": "int", "default": 6},
                    {"name": "feature_grid", "type": "int", "default": 8},
                    {"name": "encode_frames", "type": "bool", "default": True},
                    {"name": "codec_threshold", "type": "float", "default": 0.2},
                    {"name": "scene_id", "type": "str|None", "default": None},
                ),
            ),
        },
        metadata_extra={
            "entrypoints": [house_tour_scene_entrypoint],
            "phase_target": "phase_2d_house_room_playback",
            "tour_rooms": ["house_library", "house_garden", "house_museum"],
            "default_scene_layout": "vertical_strip",
        },
    )

    return [
        generic_scene,
        ui_scene,
        world_scene,
        replay_scene,
        library_scene,
        garden_scene,
        museum_scene,
        tour_scene,
    ]


def default_tool_entries() -> list[dict[str, Any]]:
    """All always-on Tool galaxy entries in one authoritative place."""

    tools = (
        _procedural_math_core_tools()
        + _procedural_codec_tools()
        + _procedural_paint_tools()
        + _procedural_geometry_prep_tools()
        + _multimodal_fusion_tools()
        + _procedural_temporal_tools()
        + _procedural_timeline_specialization_tools()
        + _procedural_scene_tools()
    )
    return [tool.to_entry() for tool in tools]


def default_multimodal_tool_entries() -> list[dict[str, Any]]:
    """Backward-compatible alias for older callers.

    The Tool galaxy is now unified: codec tools + multimodal recipes are seeded
    together and available in all sessions.
    """

    return default_tool_entries()


def build_tool_payload() -> list[dict[str, Any]]:
    return [{"galaxy": "Tool", "entry": entry} for entry in default_tool_entries()]


def build_multimodal_tool_payload() -> list[dict[str, Any]]:
    """Backward-compatible alias for callers using the old payload builder name."""

    return build_tool_payload()


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


def bootstrap_tool_galaxy(storage_root: str | Path = "../Knowledge3D.local") -> dict[str, int]:
    """Append deterministic tool-node entries without resetting existing data."""
    galaxies_root = Path(storage_root) / "galaxies"
    path = galaxies_root / "Tool.jsonl"
    existing = _read_jsonl(path)
    existing_ids = {str(row.get("id", "")) for row in existing}
    generated = default_tool_entries()
    to_append = [row for row in generated if str(row.get("id", "")) not in existing_ids]
    if to_append:
        _append_jsonl(path, to_append)
    return {
        "before": len(existing),
        "generated": len(generated),
        "appended": len(to_append),
        "after": len(existing) + len(to_append),
    }
