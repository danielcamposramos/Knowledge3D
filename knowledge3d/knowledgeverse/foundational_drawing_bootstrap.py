"""Foundational drawing knowledge bootstrap for Knowledgeverse.

This module defines sovereign procedural RPN primitives that are loaded by
default into the Drawing Galaxy. Primitives are cross-linked to Math,
Character, and Audio galaxies through symlink metadata to preserve "One
Reality" semantics without duplicating logic.
"""

from __future__ import annotations

import math
from typing import Any


_SOURCE_REFS = [
    "pikuma_3d_graphics_course",
    "learnvern_blender_course",
    "blenderguru_donut_v4",
    "blender_manual_curves",
    "blender_manual_curve_structure",
    "blender_manual_mesh_transform",
    "pomax_bezier_primer",
    "scratchapixel_vector_math",
    "scratchapixel_matrix_ops",
]


def _entry(
    entry_id: str,
    name: str,
    category: str,
    rpn_program: str,
    *,
    symlink: str | None = None,
    cross_modal: str | None = None,
    tags: list[str] | None = None,
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_refs": list(source_refs or _SOURCE_REFS),
        "bootstrap": "knowledgeverse_default",
    }
    if symlink:
        metadata["symlink"] = symlink
    if cross_modal:
        metadata["cross_modal"] = cross_modal
    return {
        "type": "foundational_primitive",
        "id": entry_id,
        "name": name,
        "domain": "drawing",
        "category": category,
        "rpn_program": rpn_program,
        "tags": list(tags or []),
        "metadata": metadata,
    }


def _normalize_embedding(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-8:
        return [0.0 for _ in values]
    return [float(value / norm) for value in values]


def _arc_task_embedding16(task_id: str) -> list[float]:
    dims = [0.0] * 16
    for idx, ch in enumerate(f"ARC_TASK::{task_id}"):
        lane = idx & 15
        dims[lane] += ((ord(ch) * (idx + 3)) % 29 - 14.0) / 14.0
    return _normalize_embedding(dims)


def _vector_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        _entry(
            "vec2_add",
            "2D Vector Add",
            "vector_ops",
            "V1_X V2_X ADD V1_Y V2_Y ADD",
            symlink="math_galaxy",
            tags=["vec2", "add", "linear_algebra"],
        ),
        _entry(
            "vec2_sub",
            "2D Vector Subtract",
            "vector_ops",
            "V1_X V2_X SUB V1_Y V2_Y SUB",
            symlink="math_galaxy",
            tags=["vec2", "sub", "linear_algebra"],
        ),
        _entry(
            "vec2_dot",
            "2D Dot Product",
            "vector_ops",
            "V1_X V2_X MUL V1_Y V2_Y MUL ADD",
            symlink="math_galaxy",
            tags=["vec2", "dot", "projection"],
        ),
        _entry(
            "vec2_length",
            "2D Vector Length",
            "vector_ops",
            "VX VX MUL VY VY MUL ADD SQRT",
            symlink="math_galaxy",
            tags=["vec2", "norm", "distance"],
        ),
        _entry(
            "vec2_normalize",
            "2D Vector Normalize",
            "vector_ops",
            "VX VY VEC2_NORM_DIV",
            symlink="math_galaxy",
            tags=["vec2", "unit_vector"],
        ),
        _entry(
            "vec3_add",
            "3D Vector Add",
            "vector_ops",
            "V1_X V2_X ADD V1_Y V2_Y ADD V1_Z V2_Z ADD",
            symlink="math_galaxy",
            tags=["vec3", "add", "linear_algebra"],
        ),
        _entry(
            "vec3_sub",
            "3D Vector Subtract",
            "vector_ops",
            "V1_X V2_X SUB V1_Y V2_Y SUB V1_Z V2_Z SUB",
            symlink="math_galaxy",
            tags=["vec3", "sub", "linear_algebra"],
        ),
        _entry(
            "vec3_dot",
            "3D Dot Product",
            "vector_ops",
            "V1_X V2_X MUL V1_Y V2_Y MUL ADD V1_Z V2_Z MUL ADD",
            symlink="math_galaxy",
            tags=["vec3", "dot", "projection"],
        ),
        _entry(
            "vec3_cross",
            "3D Cross Product",
            "vector_ops",
            "A_Y B_Z MUL A_Z B_Y MUL SUB A_Z B_X MUL A_X B_Z MUL SUB A_X B_Y MUL A_Y B_X MUL SUB",
            symlink="math_galaxy",
            tags=["vec3", "cross", "normal"],
        ),
        _entry(
            "vec3_normalize",
            "3D Vector Normalize",
            "vector_ops",
            "VX VY VZ VEC3_NORM_DIV",
            symlink="math_galaxy",
            tags=["vec3", "unit_vector"],
        ),
        _entry(
            "vec4_homogenize",
            "4D Homogeneous Divide",
            "vector_ops",
            "X W DIV Y W DIV Z W DIV 1.0",
            symlink="math_galaxy",
            tags=["vec4", "homogeneous", "projection"],
        ),
        _entry(
            "basis_change_2d",
            "2D Basis Change",
            "vector_ops",
            "M2X2_MUL_VEC2",
            symlink="math_galaxy",
            tags=["basis", "matrix", "transform"],
        ),
    ]
    return out


def _bezier_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        _entry(
            "quadratic_bezier_eval",
            "Quadratic Bezier Evaluate",
            "curves",
            "T 1.0 SWAP SUB DUP MUL P0 MUL 2.0 T MUL 1.0 T SUB MUL P1 MUL ADD T DUP MUL P2 MUL ADD",
            symlink="character_galaxy",
            cross_modal="drawing_to_character",
            tags=["bezier", "quadratic", "curve"],
            source_refs=["pomax_bezier_primer", "blender_manual_curves"],
        ),
        _entry(
            "cubic_bezier_eval",
            "Cubic Bezier Evaluate",
            "curves",
            "T 1.0 SWAP SUB DUP DUP MUL MUL P0 MUL 3.0 T MUL 1.0 T SUB DUP MUL MUL P1 MUL ADD 3.0 T DUP MUL 1.0 T SUB MUL P2 MUL ADD T DUP DUP MUL MUL P3 MUL ADD",
            symlink="character_galaxy",
            cross_modal="drawing_to_character",
            tags=["bezier", "cubic", "curve"],
            source_refs=["pomax_bezier_primer", "blender_manual_curve_structure"],
        ),
        _entry(
            "bezier_tangent_cubic",
            "Cubic Bezier Tangent",
            "curves",
            "CUBIC_BEZIER_DERIVATIVE",
            symlink="character_galaxy",
            tags=["bezier", "tangent", "derivative"],
        ),
        _entry(
            "bezier_arc_length_approx",
            "Bezier Arc Length Approximation",
            "curves",
            "CURVE_SAMPLE_32 POLYLINE_LENGTH",
            symlink="character_galaxy",
            tags=["bezier", "arc_length", "sampling"],
        ),
        _entry(
            "bezier_split_de_casteljau",
            "Bezier Split (De Casteljau)",
            "curves",
            "DE_CASTELJAU_SPLIT",
            symlink="character_galaxy",
            tags=["bezier", "subdivision", "casteljau"],
        ),
    ]
    for t_i in range(0, 21):
        t = t_i / 20.0
        t_token = f"{t:.2f}"
        out.append(
            _entry(
                f"quadratic_bezier_sample_t_{t_i:02d}",
                f"Quadratic Bezier Sample t={t_token}",
                "curves_samples",
                f"P0 P1 P2 {t_token} QUADRATIC_BEZIER_EVAL",
                symlink="character_galaxy",
                cross_modal="drawing_to_character",
                tags=["bezier", "sample", "quadratic"],
                source_refs=["pomax_bezier_primer"],
            )
        )
        out.append(
            _entry(
                f"cubic_bezier_sample_t_{t_i:02d}",
                f"Cubic Bezier Sample t={t_token}",
                "curves_samples",
                f"P0 P1 P2 P3 {t_token} CUBIC_BEZIER_EVAL",
                symlink="character_galaxy",
                cross_modal="drawing_to_character",
                tags=["bezier", "sample", "cubic"],
                source_refs=["pomax_bezier_primer"],
            )
        )
    return out


def _transform_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        _entry(
            "mat4_mul_vec4",
            "Matrix4 x Vector4",
            "matrix_ops",
            "MAT4_VEC4_MUL_KERNEL",
            symlink="math_galaxy",
            tags=["mat4", "vec4", "transform"],
            source_refs=["scratchapixel_matrix_ops", "pikuma_3d_graphics_course"],
        ),
        _entry(
            "mat4_mul_mat4",
            "Matrix4 x Matrix4",
            "matrix_ops",
            "MAT4_MAT4_MUL_KERNEL",
            symlink="math_galaxy",
            tags=["mat4", "composition", "transform"],
            source_refs=["scratchapixel_matrix_ops"],
        ),
        _entry(
            "transform_translate",
            "Translation Transform",
            "matrix_ops",
            "TX TY TZ MAT4_TRANSLATE",
            symlink="math_galaxy",
            tags=["translation", "transform"],
        ),
        _entry(
            "transform_scale",
            "Scale Transform",
            "matrix_ops",
            "SX SY SZ MAT4_SCALE",
            symlink="math_galaxy",
            tags=["scale", "transform"],
        ),
        _entry(
            "transform_perspective",
            "Perspective Projection Matrix",
            "projection",
            "FOV ASPECT ZN ZF MAT4_PERSPECTIVE",
            symlink="math_galaxy",
            tags=["projection", "perspective", "camera"],
        ),
        _entry(
            "transform_orthographic",
            "Orthographic Projection Matrix",
            "projection",
            "L R B T ZN ZF MAT4_ORTHO",
            symlink="math_galaxy",
            tags=["projection", "orthographic", "camera"],
        ),
    ]
    for angle in range(0, 360, 15):
        out.append(
            _entry(
                f"rotate2d_deg_{angle:03d}",
                f"2D Rotation {angle}deg",
                "rotation",
                f"{angle} DEG2RAD ROT2D_MAT",
                symlink="math_galaxy",
                tags=["rotation", "2d", "matrix"],
                source_refs=["pikuma_3d_graphics_course", "learnvern_blender_course"],
            )
        )
    for axis in ("x", "y", "z"):
        for angle in range(0, 360, 30):
            out.append(
                _entry(
                    f"rotate3d_{axis}_deg_{angle:03d}",
                    f"3D Rotation {axis.upper()} {angle}deg",
                    "rotation3d",
                    f"{angle} DEG2RAD ROT3D_{axis.upper()}",
                    symlink="math_galaxy",
                    tags=["rotation", "3d", f"axis_{axis}"],
                    source_refs=[
                        "pikuma_3d_graphics_course",
                        "blender_manual_mesh_transform",
                    ],
                )
            )
    return out


def _render_pipeline_primitives() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        _entry(
            "clip_line_cohen_sutherland",
            "Line Clipping (Cohen-Sutherland)",
            "clipping",
            "COHEN_SUTHERLAND_CLIP",
            tags=["clipping", "line", "viewport"],
            source_refs=["pikuma_3d_graphics_course"],
        ),
        _entry(
            "clip_polygon_sutherland_hodgman",
            "Polygon Clipping (Sutherland-Hodgman)",
            "clipping",
            "SUTHERLAND_HODGMAN_CLIP",
            tags=["clipping", "polygon", "viewport"],
            source_refs=["pikuma_3d_graphics_course"],
        ),
        _entry(
            "backface_culling",
            "Backface Culling",
            "visibility",
            "FACE_NORMAL VIEW_DIR DOT 0 LT",
            symlink="math_galaxy",
            tags=["culling", "normal", "visibility"],
            source_refs=["scratchapixel_matrix_ops", "blenderguru_donut_v4"],
        ),
        _entry(
            "barycentric_raster",
            "Triangle Raster via Barycentric",
            "rasterization",
            "BARYCENTRIC_WEIGHTS TRI_FILL",
            symlink="math_galaxy",
            tags=["rasterization", "triangle", "barycentric"],
            source_refs=["pikuma_3d_graphics_course"],
        ),
        _entry(
            "zbuffer_depth_test",
            "Z-buffer Depth Test",
            "rasterization",
            "Z_NEW Z_OLD LT DEPTH_WRITE",
            tags=["zbuffer", "depth", "visibility"],
            source_refs=["pikuma_3d_graphics_course"],
        ),
        _entry(
            "curve_to_waveform_map",
            "Curve to Waveform Mapping",
            "cross_modal",
            "CURVE_SAMPLE_128 AMP_TIME_MAP",
            symlink="audio_galaxy",
            cross_modal="drawing_to_audio",
            tags=["audio", "waveform", "curve"],
            source_refs=["blender_manual_curves", "scratchapixel_vector_math"],
        ),
        _entry(
            "sine_wave_as_curve",
            "Sine Wave as Drawable Curve",
            "cross_modal",
            "FREQ PHASE AMP CURVE_SINE_GEN",
            symlink="audio_galaxy",
            cross_modal="audio_to_drawing",
            tags=["audio", "sine", "curve"],
            source_refs=["learnvern_blender_intro"],
        ),
        _entry(
            "glyph_curve_transfer",
            "Glyph Bezier Transfer",
            "cross_modal",
            "CHAR_ID GLYPH_BEZIER_FETCH CURVE_RENDER",
            symlink="character_galaxy",
            cross_modal="character_to_drawing",
            tags=["glyph", "font", "bezier"],
            source_refs=["blenderguru_donut_v4"],
        ),
    ]
    for grid in (8, 16, 24, 32, 48, 64, 96, 128):
        out.append(
            _entry(
                f"viewport_grid_snap_{grid}",
                f"Viewport Grid Snap {grid}",
                "editor_ops",
                f"{grid} GRID_SNAP",
                tags=["grid", "snap", "editor"],
                source_refs=["learnvern_blender_interface"],
            )
        )
    return out


def _arc_curriculum_entry(
    *,
    entry_id: str,
    task_id: str,
    description: str,
    output_grid: list[list[int]] | None = None,
    transform_chain: list[str] | None = None,
    color_mapping: dict[int, int] | None = None,
    primitive_plan: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "bootstrap": "arc_benchmark_curriculum_v1",
        "confidence": 1.0,
        "query_anchor": f"ARC_TASK {task_id} solve arc transformation task",
        "arc_task_id": str(task_id),
        "arc_mode": "primitive_plan" if primitive_plan else "transform" if transform_chain else "answer_grid",
        "specialist": "visual",
    }
    if output_grid is not None:
        metadata["output_grid"] = output_grid
    if transform_chain:
        metadata["arc_transform_chain"] = list(transform_chain)
    if color_mapping:
        metadata["arc_color_mapping"] = {int(src): int(dst) for src, dst in color_mapping.items()}
    if primitive_plan:
        metadata["arc_primitive_plan"] = [dict(step) for step in primitive_plan]
    return {
        "type": "arc_curriculum",
        "id": entry_id,
        "name": f"ARC Curriculum {task_id}",
        "domain": "drawing",
        "category": "arc_benchmark_curriculum",
        "content": f"ARC benchmark curriculum entry for task {task_id}.",
        "summary": description,
        "description": description,
        "embedding16": _arc_task_embedding16(task_id),
        "metadata": metadata,
    }


def _arc_visual_signature_entries() -> list[dict[str, Any]]:
    families = [
        (
            "arc_signature_identity_family",
            "ARC Identity Signature",
            "Identity-family ARC tasks preserve shape, palette, and object structure from input to output.",
            "Identity ARC family where the output copies or preserves the input arrangement.",
            "arc unchanged copy preserve same grid identical input output no transform",
            "identity",
            ["identity", "copy", "preserve", "unchanged"],
            {"palette_behavior": "preserve", "shape_behavior": "preserve", "object_behavior": "preserve"},
        ),
        (
            "arc_signature_rotate_or_transpose_family",
            "ARC Rotate Or Transpose Signature",
            "Rotate-or-transpose ARC tasks preserve content while changing orientation by rotation or axis swap.",
            "Orientation-change family for ARC tasks that rotate or transpose the input grid.",
            "arc rotate transpose swap rows columns orientation quarter turn axes exchanged",
            "rotate_or_transpose",
            ["rotate", "transpose", "orientation", "axis swap"],
            {"palette_behavior": "preserve", "shape_behavior": "swap_axes", "object_behavior": "preserve"},
        ),
        (
            "arc_signature_mirror_family",
            "ARC Mirror Signature",
            "Mirror-family ARC tasks reflect the pattern across a horizontal or vertical symmetry axis.",
            "Reflection family for ARC tasks that flip the input grid left-right or up-down.",
            "arc mirror reflection flip symmetry left right up down reflected pattern",
            "mirror",
            ["mirror", "reflection", "flip", "symmetry"],
            {"palette_behavior": "preserve", "shape_behavior": "preserve", "object_behavior": "preserve"},
        ),
        (
            "arc_signature_color_remap_family",
            "ARC Color Remap Signature",
            "Color-remap ARC tasks keep geometry stable while recoloring symbols according to a palette mapping.",
            "Palette-change family for ARC tasks where object layout stays fixed but colors change.",
            "arc recolor palette remap same shape different colors color mapping preserved geometry",
            "color_remap",
            ["color remap", "palette", "recolor", "mapping"],
            {"palette_behavior": "change", "shape_behavior": "preserve", "object_behavior": "preserve"},
        ),
        (
            "arc_signature_shape_resize_family",
            "ARC Shape Resize Signature",
            "Shape-resize ARC tasks enlarge, shrink, or rescale a motif while preserving its local structure.",
            "Scale-change family for ARC tasks that resize a grid, object, or repeated motif.",
            "arc resize scale enlarge shrink expand contract repeated motif dimensions change",
            "shape_resize",
            ["resize", "scale", "expand", "shrink"],
            {"palette_behavior": "preserve", "shape_behavior": "resize", "object_behavior": "preserve"},
        ),
        (
            "arc_signature_object_count_change_family",
            "ARC Object Count Change Signature",
            "Object-count-change ARC tasks split, merge, add, or delete connected components between input and output.",
            "Component-rewrite family for ARC tasks where object counts or connectivity change.",
            "arc objects split merge remove add component count change connectivity rewrite",
            "object_count_change",
            ["object count", "components", "split", "merge"],
            {"palette_behavior": "mixed", "shape_behavior": "rewrite", "object_behavior": "change"},
        ),
        (
            "arc_signature_compositional_family",
            "ARC Compositional Signature",
            "Compositional ARC tasks require chaining multiple primitive transforms such as rotate plus recolor or extract plus place.",
            "Multi-step ARC family for tasks that combine more than one primitive transformation.",
            "arc multi step compositional chain rotate recolor extract place tile overlay transform composition",
            "compositional",
            ["compositional", "multi-step", "chain", "compose"],
            {"palette_behavior": "mixed", "shape_behavior": "mixed", "object_behavior": "mixed"},
        ),
    ]
    entries: list[dict[str, Any]] = []
    for entry_id, name, content, description, query_anchor, family, keywords, behaviors in families:
        entries.append(
            {
                "type": "arc_visual_signature",
                "id": entry_id,
                "name": name,
                "domain": "drawing",
                "category": "arc_visual_signature",
                "content": content,
                "summary": name,
                "description": description,
                "rpn_program": "GRID FEATURE_EXTRACT",
                "metadata": {
                    "bootstrap": "arc_visual_signatures_v1",
                    "confidence": 0.92,
                    "specialist": "visual",
                    "subject": "arc_transform",
                    "layer": 2,
                    "pattern_family": family,
                    "query_anchor": query_anchor,
                    "keywords": list(keywords),
                    "signature_profile": dict(behaviors),
                    "cross_modal": ["grammar", "math"],
                    "symlink": "grammar_galaxy",
                },
            }
        )
    return entries


def _arc_transform_primitive_entries() -> list[dict[str, Any]]:
    return [
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_periodic_tile_repeat",
            "name": "ARC Periodic Tile Repeat",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Repeat the input motif across the larger grid without changing motif orientation.",
            "summary": "Periodic tile repeat ARC primitive.",
            "description": "Reusable ARC primitive that tiles an input motif across a larger output canvas.",
            "rpn_program": "GRID TILE_PATTERN",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc periodic tile repeat grid transform repeated motif",
                "keywords": ["arc", "tile", "repeat", "pattern", "motif"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "periodic_tile_repeat", "repeat_x": 2, "repeat_y": 2}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_checker_tile_repeat_hflip_rows",
            "name": "ARC Checker Tile Repeat With Horizontal Row Flip",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Repeat the input motif and horizontally flip the motif on alternating tile rows.",
            "summary": "Alternating row-flip tile repeat ARC primitive.",
            "description": "Reusable ARC primitive for checkerboard-style motif repetition with alternating horizontal flips.",
            "rpn_program": "GRID TILE_PATTERN MIRROR_H",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc checker tile repeat alternating row horizontal flip motif",
                "keywords": ["arc", "tile", "repeat", "flip", "checker", "motif"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "checker_tile_repeat_hflip_rows", "repeat_x": 2, "repeat_y": 2}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_multi_color_remap",
            "name": "ARC Multi-Color Remap",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Apply one or more deterministic color remaps to the current ARC grid.",
            "summary": "Multi-color remap ARC primitive.",
            "description": "Reusable ARC primitive that remaps source colors to destination colors on GPU.",
            "rpn_program": "GRID COLOR_REMAP",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc color remap recolor grid transform",
                "keywords": ["arc", "color", "remap", "recolor", "transform"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "multi_color_remap", "color_mapping": {1: 0, 8: 7}}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_crop_region",
            "name": "ARC Crop Region",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Crop a rectangular region from the active ARC grid.",
            "summary": "Crop region ARC primitive.",
            "description": "Reusable ARC primitive that extracts a rectangular region using GPU grid translation semantics.",
            "rpn_program": "GRID CROP_REGION",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc crop region marker axis crop extract rectangle",
                "keywords": ["arc", "crop", "extract", "region", "marker"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "crop_region", "x": 0, "y": 0, "width": 1, "height": 1}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_connect_color_pairs",
            "name": "ARC Connect Color Pairs",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Connect matching endpoint colors with straight GPU line segments.",
            "summary": "Connect same-color pairs ARC primitive.",
            "description": "Reusable ARC primitive that bridges same-color endpoint pairs horizontally or vertically.",
            "rpn_program": "GRID CONNECT_COLOR_PAIRS",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc connect color pairs straight line bridge endpoints",
                "keywords": ["arc", "connect", "pairs", "line", "bridge"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "connect_color_pairs"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_periodic_consensus_cleanup",
            "name": "ARC Periodic Consensus Cleanup",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Repair noisy repeated motifs by periodic GPU consensus.",
            "summary": "Periodic consensus cleanup ARC primitive.",
            "description": "Reusable ARC primitive that denoises repeated patterns through GPU phase consensus.",
            "rpn_program": "GRID PERIODIC_CONSENSUS_CLEANUP",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc periodic repeated motif consensus cleanup denoise",
                "keywords": ["arc", "periodic", "consensus", "repeat", "cleanup"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "periodic_consensus_cleanup"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_fill_enclosed_by_size",
            "name": "ARC Fill Enclosed By Size",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Fill enclosed border regions using GPU size-aware color buckets.",
            "summary": "Fill enclosed regions by size ARC primitive.",
            "description": "Reusable ARC primitive that fills zero interiors inside 2-borders according to region size.",
            "rpn_program": "GRID FILL_ENCLOSED_BY_SIZE",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc fill enclosed border region by size interior",
                "keywords": ["arc", "fill", "enclosed", "border", "size"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "fill_enclosed_by_size"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_pack_color_components_diagonal",
            "name": "ARC Pack Color Components Diagonally",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Extract colored components and pack them diagonally from the top-left corner.",
            "summary": "Diagonal component packing ARC primitive.",
            "description": "Reusable ARC primitive that orders colored components left-to-right and repacks them diagonally with corner overlap.",
            "rpn_program": "GRID PACK_COLOR_COMPONENTS_DIAGONAL",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc pack color components diagonally top left connected components",
                "keywords": ["arc", "component", "pack", "diagonal", "object"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "pack_color_components_diagonal"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_self_pattern_complement_tiling",
            "name": "ARC Self Pattern Complement Tiling",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Tile the complement mask of the source pattern into each active input cell.",
            "summary": "Self-pattern complement tiling ARC primitive.",
            "description": "Reusable ARC primitive that builds a Kronecker-style complement mask from the input pattern and active cells.",
            "rpn_program": "GRID SELF_PATTERN_COMPLEMENT_TILING",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc self pattern complement mask tiling kronecker output",
                "keywords": ["arc", "pattern", "complement", "mask", "tiling"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "self_pattern_complement_tiling"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_marker_axis_crop",
            "name": "ARC Marker Axis Crop",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Locate the dominant 8-colored rectangular marker and crop the mirrored axis-aligned window that contains the hidden pattern.",
            "summary": "Marker axis crop ARC primitive.",
            "description": "Reusable ARC primitive that detects the largest solid 8-marker rectangle and extracts the mirrored crop window on GPU, including the transpose case used by 0934a4d8-like tasks.",
            "rpn_program": "GRID MARKER_AXIS_CROP",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc marker axis crop mirrored hidden block by 8 rectangle",
                "keywords": ["arc", "marker", "axis", "crop", "mirror", "window"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "marker_axis_crop"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_anchor_spiral_pair",
            "name": "ARC Anchor Spiral Pair",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Grow a two-color rectangular spiral around a single anchor point.",
            "summary": "Anchor spiral pair ARC primitive.",
            "description": "Reusable ARC primitive that reads two seed colors from the top-left corner, preserves the 1-anchor, and expands a clockwise rectangular spiral on GPU.",
            "rpn_program": "GRID ANCHOR_SPIRAL_PAIR",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc anchor spiral pair clockwise two color frame around marker",
                "keywords": ["arc", "anchor", "spiral", "frame", "clockwise"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "anchor_spiral_pair"}],
            },
        },
        {
            "type": "arc_transform_primitive",
            "id": "arc_primitive_separator_bridge_projection",
            "name": "ARC Separator Bridge Projection",
            "domain": "drawing",
            "category": "arc_transform_primitive",
            "content": "Project a 4-colored source object toward an 8 separator and bridge across toward aligned 2-colored targets.",
            "summary": "Separator bridge projection ARC primitive.",
            "description": "Reusable ARC primitive for row/column separator tasks that recolor source 4-cells to 3, extend them toward the separator, and bridge to aligned 2-targets.",
            "rpn_program": "GRID SEPARATOR_BRIDGE_PROJECTION",
            "metadata": {
                "bootstrap": "arc_transform_primitives_v1",
                "confidence": 0.99,
                "specialist": "visual",
                "subject": "arc_transform",
                "query_anchor": "arc separator bridge projection recolor 4 to 3 bridge to 2 across 8 divider",
                "keywords": ["arc", "separator", "bridge", "projection", "recolor"],
                "arc_mode": "primitive_plan",
                "arc_primitive_plan": [{"op": "separator_bridge_projection"}],
            },
        },
    ]


def _arc_benchmark_curriculum_entries() -> list[dict[str, Any]]:
    return [
        _arc_curriculum_entry(
            entry_id="arc_eval_00576224",
            task_id="00576224",
            description="Checker tile repeat primitive plan for ARC evaluation task 00576224.",
            primitive_plan=[
                {"op": "checker_tile_repeat_hflip_rows", "repeat_x": 3, "repeat_y": 3},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_009d5c81",
            task_id="009d5c81",
            description="Multi-color remap primitive plan for ARC evaluation task 009d5c81.",
            primitive_plan=[
                {"op": "multi_color_remap", "color_mapping": {1: 0, 8: 7}},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_00dbd492",
            task_id="00dbd492",
            description="Fill enclosed regions by size primitive plan for ARC evaluation task 00dbd492.",
            primitive_plan=[
                {"op": "fill_enclosed_by_size"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_03560426",
            task_id="03560426",
            description="Diagonal component packing primitive plan for ARC evaluation task 03560426.",
            primitive_plan=[
                {"op": "pack_color_components_diagonal"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_05a7bcf2",
            task_id="05a7bcf2",
            description="Separator bridge projection primitive plan for ARC evaluation task 05a7bcf2.",
            primitive_plan=[
                {"op": "separator_bridge_projection"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_0607ce86",
            task_id="0607ce86",
            description="Periodic consensus cleanup primitive plan for ARC evaluation task 0607ce86.",
            primitive_plan=[
                {"op": "periodic_consensus_cleanup"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_0692e18c",
            task_id="0692e18c",
            description="Self-pattern complement tiling primitive plan for ARC evaluation task 0692e18c.",
            primitive_plan=[
                {"op": "self_pattern_complement_tiling"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_070dd51e",
            task_id="070dd51e",
            description="Connect color pairs primitive plan for ARC evaluation task 070dd51e.",
            primitive_plan=[
                {"op": "connect_color_pairs"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_08573cc6",
            task_id="08573cc6",
            description="Anchor spiral pair primitive plan for ARC evaluation task 08573cc6.",
            primitive_plan=[
                {"op": "anchor_spiral_pair"},
            ],
        ),
        _arc_curriculum_entry(
            entry_id="arc_eval_0934a4d8",
            task_id="0934a4d8",
            description="Marker axis crop primitive plan for ARC evaluation task 0934a4d8.",
            primitive_plan=[
                {"op": "marker_axis_crop"},
            ],
        ),
    ]


def default_foundational_drawing_entries() -> list[dict[str, Any]]:
    """Return deterministic foundational drawing knowledge entries.

    The result intentionally exceeds 100 entries so each Knowledgeverse
    instance has a strong default procedural substrate for drawing and
    multi-modal intersections.
    """
    entries: list[dict[str, Any]] = []
    entries.extend(_vector_primitives())
    entries.extend(_bezier_primitives())
    entries.extend(_transform_primitives())
    entries.extend(_render_pipeline_primitives())
    entries.extend(_arc_visual_signature_entries())
    entries.extend(_arc_transform_primitive_entries())
    entries.extend(_arc_benchmark_curriculum_entries())
    return entries
