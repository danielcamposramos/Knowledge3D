"""Foundational drawing knowledge bootstrap for Knowledgeverse.

This module defines sovereign procedural RPN primitives that are loaded by
default into the Drawing Galaxy. Primitives are cross-linked to Math,
Character, and Audio galaxies through symlink metadata to preserve "One
Reality" semantics without duplicating logic.
"""

from __future__ import annotations

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
    return entries

