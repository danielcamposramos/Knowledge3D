"""Build the foundational Galaxy star population for ARC3 and sovereign lookup."""

from __future__ import annotations

from typing import Any

from knowledge3d.cranium.action_primitives_bootstrap import (
    ACTION_CLICK,
    ACTION_DIAGONAL_UR,
    ACTION_GRAB,
    ACTION_HOLD,
    ACTION_LOOK_AT,
    ACTION_MOVE_DOWN,
    ACTION_MOVE_LEFT,
    ACTION_MOVE_RIGHT,
    ACTION_MOVE_UP,
    ACTION_PERFORM,
    ACTION_REACH,
    ACTION_RELEASE,
    ACTION_TELEPORT,
    ACTION_UNDO,
    ACTION_USE,
    ACTION_WALK_TO,
    build_default_action_galaxy,
)
from knowledge3d.cranium.reality_nodes import RealityNode
from knowledge3d.knowledgeverse.action_embedding_loader import (
    ARC3_EXTENDED_ACTION_ATOM_IDS,
    _displacement_to_embedding,
    _get_displacement,
    _node_to_embedding,
)
from knowledge3d.knowledgeverse.foundational_drawing_bootstrap import default_foundational_drawing_entries


STAR_TYPE_ACTION = 0
STAR_TYPE_DRAWING = 1
STAR_TYPE_CHARACTER = 2
STAR_TYPE_GRAMMAR = 3
STAR_TYPE_REALITY = 4
STAR_TYPE_MATH = 5
STAR_TYPE_SPATIAL = 6

STAR_FLAG_ACTIVE = 0x01
STAR_FLAG_LEARNABLE = 0x02


def _fnv1a32(text: str) -> int:
    value = 2166136261
    for byte in str(text or "").encode("utf-8"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return int(value)


def _pad32(values: list[float]) -> list[float]:
    row = [float(value) for value in list(values or [])[:32]]
    if len(row) < 32:
        row.extend([0.0] * (32 - len(row)))
    return row[:32]


def _normalize(values: list[float]) -> list[float]:
    norm = sum(float(value) * float(value) for value in values) ** 0.5
    if norm <= 1.0e-8:
        return [0.0] * len(values)
    return [float(value) / norm for value in values]


def _hash_tokens_into_embedding(embedding: list[float], text: str, *, start: int = 8) -> list[float]:
    for token in [tok for tok in str(text or "").split() if tok]:
        bucket = start + (_fnv1a32(token) % max(1, len(embedding) - start))
        embedding[bucket] += 0.25
    return embedding


def _surface_signal(surface_forms: dict[str, str] | None) -> float:
    if not surface_forms:
        return 0.0
    return min(1.0, len(surface_forms) / 4.0)


def _reality_node_embedding(node: RealityNode | None) -> list[float]:
    if node is None:
        return [0.0] * 32
    return _pad32(_node_to_embedding(node))


def _drawing_star_embedding(entry: dict[str, Any]) -> list[float]:
    embedding = [0.0] * 32
    category = str(entry.get("category", "") or "")
    tags = [str(tag) for tag in list(entry.get("tags") or [])]
    rpn_program = str(entry.get("rpn_program", "") or "")
    metadata = dict(entry.get("metadata") or {})

    category_sig = {
        "vector_ops": 0.85,
        "curves": 0.70,
        "curves_samples": 0.65,
        "matrix_ops": 0.55,
        "projection": 0.45,
        "rotation": 0.60,
        "rotation3d": 0.75,
        "clipping": 0.35,
        "visibility": 0.25,
        "rasterization": 0.20,
        "cross_modal": 0.40,
    }
    embedding[2] = category_sig.get(category, 0.30)
    embedding[3] = min(1.0, len(rpn_program.split()) / 14.0)
    embedding[4] = 1.0 if metadata.get("symlink") == "math_galaxy" else 0.0
    embedding[5] = 1.0 if metadata.get("cross_modal") else 0.0
    embedding[6] = min(1.0, len(tags) / 6.0)
    embedding[7] = 1.0 if "rotation" in tags or "transform" in tags else 0.0
    if "translation" in tags:
        embedding[0] = 0.75
    if "scale" in tags:
        embedding[0] = -0.65
    if "rotation" in tags:
        embedding[1] = 0.9
    if "bezier" in tags:
        embedding[1] = 0.55
    _hash_tokens_into_embedding(embedding, f"{entry.get('id', '')} {entry.get('name', '')} {rpn_program} {' '.join(tags)}")
    return _normalize(embedding)


def _math_star_embedding(star_def: dict[str, Any]) -> list[float]:
    embedding = [0.0] * 32
    behavior = str(star_def.get("behavior_rpn", "") or "")
    visual = str(star_def.get("visual_rpn", "") or "")
    surface_forms = dict(star_def.get("surface_forms") or {})

    if "ADD" in behavior or "SUM" in behavior:
        embedding[0] = 0.8
    if "MUL" in behavior or "POW" in behavior:
        embedding[0] = -0.8
    if "SUB" in behavior or "DIV" in behavior:
        embedding[1] = -0.7
    if "SIN" in behavior or "COS" in behavior or "PI" in behavior:
        embedding[1] = 0.9

    embedding[2] = 0.9
    embedding[3] = min(1.0, len(behavior.split()) / 10.0)
    embedding[4] = 1.0 if visual else 0.0
    embedding[5] = 1.0 if star_def.get("_ref_ids") else 0.0
    embedding[6] = min(1.0, len(list(star_def.get("_ref_ids") or [])) / 4.0)
    embedding[7] = _surface_signal(surface_forms)

    _hash_tokens_into_embedding(embedding, f"{visual} {behavior} {' '.join(surface_forms.values())}")
    return _normalize(embedding)


def _spatial_star_embedding(star_def: dict[str, Any]) -> list[float]:
    embedding = [0.0] * 32
    meaning = str(star_def.get("meaning_rpn", "") or "")
    surface_forms = dict(star_def.get("surface_forms") or {})
    star_id = str(star_def.get("id", "") or "")

    if "symmetry" in star_id:
        embedding[0] = 0.85
    if "translation" in star_id or "pattern_repeat" in star_id:
        embedding[0] = -0.65
    if "object" in star_id or "containment" in star_id or "boundary" in star_id:
        embedding[1] = 0.75
    if "transform" in star_id or "goal" in star_id:
        embedding[1] = -0.75

    embedding[2] = -0.9
    embedding[3] = min(1.0, len(meaning.split()) / 10.0)
    embedding[4] = 1.0 if star_def.get("_ref_ids") else 0.0
    embedding[5] = min(1.0, len(list(star_def.get("_ref_ids") or [])) / 4.0)
    embedding[6] = 1.0 if "GRID" in meaning or "CELL" in meaning else 0.0
    embedding[7] = _surface_signal(surface_forms)

    _hash_tokens_into_embedding(embedding, f"{star_id} {meaning} {' '.join(surface_forms.values())}")
    return _normalize(embedding)


def _drawing_lookup() -> dict[str, dict[str, Any]]:
    return {str(entry["id"]): dict(entry) for entry in default_foundational_drawing_entries()}


def _drawing_math_stars() -> list[dict[str, Any]]:
    drawing = _drawing_lookup()
    selected_ids = [
        "vec2_add",
        "vec2_sub",
        "vec2_dot",
        "vec2_length",
        "vec2_normalize",
        "vec3_add",
        "vec3_sub",
        "vec3_dot",
        "vec3_cross",
        "vec3_normalize",
        "vec4_homogenize",
        "basis_change_2d",
        "quadratic_bezier_eval",
        "cubic_bezier_eval",
        "bezier_tangent_cubic",
        "bezier_arc_length_approx",
        "bezier_split_de_casteljau",
        "quadratic_bezier_sample_t_00",
        "quadratic_bezier_sample_t_05",
        "quadratic_bezier_sample_t_10",
        "quadratic_bezier_sample_t_15",
        "quadratic_bezier_sample_t_20",
        "cubic_bezier_sample_t_00",
        "cubic_bezier_sample_t_05",
        "cubic_bezier_sample_t_10",
        "cubic_bezier_sample_t_15",
        "cubic_bezier_sample_t_20",
        "mat4_mul_vec4",
        "mat4_mul_mat4",
        "transform_translate",
        "transform_scale",
        "transform_perspective",
        "transform_orthographic",
        "rotate2d_deg_000",
        "rotate2d_deg_090",
        "rotate2d_deg_180",
        "rotate2d_deg_270",
        "rotate3d_x_deg_090",
        "rotate3d_y_deg_090",
        "rotate3d_z_deg_090",
    ]
    refs = {
        "vec2_add": ["add_op", "grid_cell"],
        "vec2_sub": ["sub_op", "grid_cell"],
        "vec2_dot": ["mul_op", "add_op"],
        "vec2_length": ["mul_op", "add_op", "sqrt_op"],
        "vec2_normalize": ["vec2_length", "div_op"],
        "vec3_add": ["vec2_add", "add_op"],
        "vec3_sub": ["vec2_sub", "sub_op"],
        "vec3_dot": ["mul_op", "add_op"],
        "vec3_cross": ["mul_op", "sub_op"],
        "vec3_normalize": ["vec2_normalize", "div_op"],
        "vec4_homogenize": ["div_op"],
        "basis_change_2d": ["mul_op"],
        "quadratic_bezier_eval": ["mul_op", "add_op", "pow_op"],
        "cubic_bezier_eval": ["quadratic_bezier_eval", "mul_op", "pow_op"],
        "bezier_tangent_cubic": ["cubic_bezier_eval", "derivative_op"],
        "bezier_arc_length_approx": ["vec2_length", "add_op"],
        "bezier_split_de_casteljau": ["line_segment", "add_op"],
        "quadratic_bezier_sample_t_00": ["quadratic_bezier_eval"],
        "quadratic_bezier_sample_t_05": ["quadratic_bezier_eval"],
        "quadratic_bezier_sample_t_10": ["quadratic_bezier_eval"],
        "quadratic_bezier_sample_t_15": ["quadratic_bezier_eval"],
        "quadratic_bezier_sample_t_20": ["quadratic_bezier_eval"],
        "cubic_bezier_sample_t_00": ["cubic_bezier_eval"],
        "cubic_bezier_sample_t_05": ["cubic_bezier_eval"],
        "cubic_bezier_sample_t_10": ["cubic_bezier_eval"],
        "cubic_bezier_sample_t_15": ["cubic_bezier_eval"],
        "cubic_bezier_sample_t_20": ["cubic_bezier_eval"],
        "mat4_mul_vec4": ["mul_op", "add_op"],
        "mat4_mul_mat4": ["mul_op", "add_op"],
        "transform_translate": ["translate_2d", "translation_concept", ACTION_MOVE_UP],
        "transform_scale": ["scale_2d", "mul_op"],
        "transform_perspective": ["div_op", "pi_const"],
        "transform_orthographic": ["sub_op", "div_op"],
        "rotate2d_deg_000": ["rotate_2d", "sin_op", "cos_op"],
        "rotate2d_deg_090": ["rotate_2d", "sin_op", "cos_op"],
        "rotate2d_deg_180": ["rotate_2d", "sin_op", "cos_op"],
        "rotate2d_deg_270": ["rotate_2d", "sin_op", "cos_op"],
        "rotate3d_x_deg_090": ["rotate_2d", "sin_op", "cos_op"],
        "rotate3d_y_deg_090": ["rotate_2d", "sin_op", "cos_op"],
        "rotate3d_z_deg_090": ["rotate_2d", "sin_op", "cos_op"],
    }

    stars: list[dict[str, Any]] = []
    for entry_id in selected_ids:
        entry = dict(drawing[entry_id])
        stars.append(
            {
                "id": entry_id,
                "name": entry.get("name", entry_id),
                "embedding": _drawing_star_embedding(entry),
                "galaxy_id": _fnv1a32("drawing"),
                "star_type": STAR_TYPE_DRAWING,
                "_ref_ids": list(refs.get(entry_id, [])),
                "flags": STAR_FLAG_ACTIVE,
            }
        )
    # Alias line segment as a drawing concept derived from existing entry ids.
    stars.append(
        {
            "id": "line_segment",
            "name": "Line Segment",
            "embedding": _drawing_star_embedding(
                {
                    "id": "line_segment",
                    "name": "Line Segment",
                    "category": "curves",
                    "rpn_program": "P0 P1 LERP",
                    "tags": ["line", "segment", "curve"],
                    "metadata": {"symlink": "math_galaxy"},
                }
            ),
            "galaxy_id": _fnv1a32("drawing"),
            "star_type": STAR_TYPE_DRAWING,
            "_ref_ids": ["vec2_sub", "mul_op"],
            "flags": STAR_FLAG_ACTIVE,
        }
    )
    return stars


def _math_operation_stars() -> list[dict[str, Any]]:
    defs = [
        {"id": "add_op", "name": "Add", "visual_rpn": "DRAW_CROSS_PLUS", "behavior_rpn": "A B ADD", "surface_forms": {"en": "add", "pt": "somar"}, "_ref_ids": []},
        {"id": "sub_op", "name": "Subtract", "visual_rpn": "DRAW_HLINE", "behavior_rpn": "A B SUB", "surface_forms": {"en": "subtract", "pt": "subtrair"}, "_ref_ids": ["add_op"]},
        {"id": "mul_op", "name": "Multiply", "visual_rpn": "DRAW_CROSS_X", "behavior_rpn": "A B MUL", "surface_forms": {"en": "multiply", "pt": "multiplicar"}, "_ref_ids": ["add_op"]},
        {"id": "div_op", "name": "Divide", "visual_rpn": "DRAW_FRACTION_BAR", "behavior_rpn": "A B DIV", "surface_forms": {"en": "divide", "pt": "dividir"}, "_ref_ids": ["mul_op"]},
        {"id": "pow_op", "name": "Power", "visual_rpn": "DRAW_SUPERSCRIPT", "behavior_rpn": "BASE EXP POW", "surface_forms": {"en": "power", "pt": "potência"}, "_ref_ids": ["mul_op"]},
        {"id": "sqrt_op", "name": "Square Root", "visual_rpn": "DRAW_RADICAL", "behavior_rpn": "X SQRT", "surface_forms": {"en": "square root", "pt": "raiz quadrada"}, "_ref_ids": ["pow_op"]},
        {"id": "negate_op", "name": "Negate", "visual_rpn": "DRAW_HLINE", "behavior_rpn": "X -1 MUL", "surface_forms": {"en": "negate", "pt": "negar"}, "_ref_ids": ["mul_op"]},
        {"id": "abs_op", "name": "Absolute Value", "visual_rpn": "DRAW_VLINE_PAIR", "behavior_rpn": "X ABS", "surface_forms": {"en": "absolute value", "pt": "valor absoluto"}, "_ref_ids": ["negate_op"]},
        {"id": "sin_op", "name": "Sine", "visual_rpn": "DRAW_SIN_LABEL", "behavior_rpn": "THETA SIN", "surface_forms": {"en": "sine", "pt": "seno"}, "_ref_ids": ["rotate_2d"]},
        {"id": "cos_op", "name": "Cosine", "visual_rpn": "DRAW_COS_LABEL", "behavior_rpn": "THETA COS", "surface_forms": {"en": "cosine", "pt": "cosseno"}, "_ref_ids": ["rotate_2d"]},
        {"id": "pi_const", "name": "Pi", "visual_rpn": "DRAW_PI_GLYPH", "behavior_rpn": "3.14159265", "surface_forms": {"en": "pi", "pt": "pi"}, "_ref_ids": ["sin_op", "cos_op"]},
        {"id": "circle_const", "name": "Circle", "visual_rpn": "DRAW_CIRCLE_GLYPH", "behavior_rpn": "R 2 MUL PI MUL", "surface_forms": {"en": "circle", "pt": "círculo"}, "_ref_ids": ["pi_const", "mul_op"]},
        {"id": "derivative_op", "name": "Derivative", "visual_rpn": "DRAW_PARTIAL_D", "behavior_rpn": "DX F_NEXT F_PREV SUB SWAP DIV", "surface_forms": {"en": "derivative", "pt": "derivada"}, "_ref_ids": ["sub_op", "div_op"]},
        {"id": "integral_op", "name": "Integral", "visual_rpn": "DRAW_INTEGRAL", "behavior_rpn": "DX F_SUM MUL", "surface_forms": {"en": "integral", "pt": "integral"}, "_ref_ids": ["add_op", "mul_op"]},
        {"id": "sum_op", "name": "Summation", "visual_rpn": "DRAW_SIGMA", "behavior_rpn": "N 0 DO I RECALL F_EVAL ADD LOOP", "surface_forms": {"en": "sum", "pt": "somatório"}, "_ref_ids": ["add_op"]},
        {"id": "delta_op", "name": "Delta", "visual_rpn": "DRAW_DELTA", "behavior_rpn": "F_NEW F_OLD SUB", "surface_forms": {"en": "delta", "pt": "delta"}, "_ref_ids": ["sub_op"]},
    ]
    stars: list[dict[str, Any]] = []
    for definition in defs:
        definition = dict(definition)
        definition["embedding"] = _math_star_embedding(definition)
        definition["galaxy_id"] = _fnv1a32("math")
        definition["star_type"] = STAR_TYPE_MATH
        definition["flags"] = STAR_FLAG_ACTIVE
        stars.append(definition)
    return stars


def _spatial_reasoning_stars() -> list[dict[str, Any]]:
    defs = [
        {"id": "symmetry_x", "name": "X Symmetry", "meaning_rpn": "GRID FLIP_X COMPARE", "surface_forms": {"en": "x symmetry", "pt": "simetria em x"}, "_ref_ids": ["reflect_x"]},
        {"id": "symmetry_y", "name": "Y Symmetry", "meaning_rpn": "GRID FLIP_Y COMPARE", "surface_forms": {"en": "y symmetry", "pt": "simetria em y"}, "_ref_ids": ["reflect_y"]},
        {"id": "symmetry_rotate", "name": "Rotational Symmetry", "meaning_rpn": "GRID 90 ROTATE COMPARE", "surface_forms": {"en": "rotational symmetry", "pt": "simetria rotacional"}, "_ref_ids": ["rotate_2d"]},
        {"id": "translation_concept", "name": "Translation", "meaning_rpn": "OBJECT DX DY TRANSLATE", "surface_forms": {"en": "translation", "pt": "translação"}, "_ref_ids": ["translate_2d", "vec2_add"]},
        {"id": "pattern_repeat", "name": "Pattern Repeat", "meaning_rpn": "TILE OFFSET COPY", "surface_forms": {"en": "pattern repeat", "pt": "repetição de padrão"}, "_ref_ids": ["translation_concept", "add_op"]},
        {"id": "adjacency", "name": "Adjacency", "meaning_rpn": "CELL NEIGHBOR_4 CHECK", "surface_forms": {"en": "adjacency", "pt": "adjacência"}, "_ref_ids": ["vec2_sub", "abs_op"]},
        {"id": "containment", "name": "Containment", "meaning_rpn": "INNER OUTER BOUNDS_CHECK", "surface_forms": {"en": "containment", "pt": "contenção"}, "_ref_ids": ["vec2_sub", "abs_op"]},
        {"id": "color_fill", "name": "Color Fill", "meaning_rpn": "SEED COLOR FLOOD_FILL", "surface_forms": {"en": "color fill", "pt": "preenchimento de cor"}, "_ref_ids": ["adjacency"]},
        {"id": "grid_cell", "name": "Grid Cell", "meaning_rpn": "ROW COL CELL_INDEX", "surface_forms": {"en": "grid cell", "pt": "célula de grade"}, "_ref_ids": ["mul_op", "add_op"]},
        {"id": "boundary", "name": "Boundary", "meaning_rpn": "CELL NEIGHBOR_4 DIFF_COUNT", "surface_forms": {"en": "boundary", "pt": "fronteira"}, "_ref_ids": ["adjacency", "sub_op"]},
        {"id": "object_detect", "name": "Object Detect", "meaning_rpn": "GRID CONNECTED_COMPONENTS", "surface_forms": {"en": "object detection", "pt": "detecção de objeto"}, "_ref_ids": ["adjacency", "color_fill"]},
        {"id": "transform_detect", "name": "Transform Detect", "meaning_rpn": "FRAME_A FRAME_B DIFF CLASSIFY", "surface_forms": {"en": "transform detection", "pt": "detecção de transformação"}, "_ref_ids": ["delta_op", "symmetry_x"]},
        {"id": "goal_infer", "name": "Goal Infer", "meaning_rpn": "EXAMPLES PATTERN_EXTRACT GENERALIZE", "surface_forms": {"en": "goal inference", "pt": "inferência de objetivo"}, "_ref_ids": ["pattern_repeat", "transform_detect"]},
        {"id": "action_evaluate", "name": "Action Evaluate", "meaning_rpn": "STATE ACTION APPLY SCORE", "surface_forms": {"en": "action evaluation", "pt": "avaliação de ação"}, "_ref_ids": ["delta_op", "translation_concept"]},
        {"id": "explore_strategy", "name": "Explore Strategy", "meaning_rpn": "HISTORY DIVERSITY_SCORE THRESHOLD COMPARE", "surface_forms": {"en": "exploration strategy", "pt": "estratégia de exploração"}, "_ref_ids": ["sum_op", "div_op"]},
        {"id": "translate_2d", "name": "2D Translation", "meaning_rpn": "TX TY VEC2_ADD", "surface_forms": {"en": "2d translation", "pt": "translação 2d"}, "_ref_ids": ["vec2_add", ACTION_MOVE_UP, "translation_concept", "add_op"]},
        {"id": "rotate_2d", "name": "2D Rotation", "meaning_rpn": "THETA DEG2RAD ROT2D_MAT", "surface_forms": {"en": "2d rotation", "pt": "rotação 2d"}, "_ref_ids": ["sin_op", "cos_op"]},
        {"id": "scale_2d", "name": "2D Scale", "meaning_rpn": "SX SY MAT2_SCALE", "surface_forms": {"en": "2d scale", "pt": "escala 2d"}, "_ref_ids": ["mul_op"]},
        {"id": "reflect_x", "name": "Reflect X", "meaning_rpn": "Y -1 MUL Y STORE", "surface_forms": {"en": "reflect x", "pt": "refletir x"}, "_ref_ids": ["mul_op", "negate_op"]},
        {"id": "reflect_y", "name": "Reflect Y", "meaning_rpn": "X -1 MUL X STORE", "surface_forms": {"en": "reflect y", "pt": "refletir y"}, "_ref_ids": ["mul_op", "negate_op"]},
    ]
    stars: list[dict[str, Any]] = []
    for definition in defs:
        definition = dict(definition)
        definition["embedding"] = _spatial_star_embedding(definition)
        definition["galaxy_id"] = _fnv1a32("spatial")
        definition["star_type"] = STAR_TYPE_SPATIAL
        definition["flags"] = STAR_FLAG_ACTIVE
        stars.append(definition)
    return stars


def _support_action_stars(reality_galaxy: Any) -> list[dict[str, Any]]:
    ids = [
        ACTION_DIAGONAL_UR,
        ACTION_REACH,
        ACTION_GRAB,
        ACTION_HOLD,
        ACTION_RELEASE,
        ACTION_USE,
        ACTION_WALK_TO,
        ACTION_TELEPORT,
        ACTION_LOOK_AT,
    ]
    refs = {
        ACTION_DIAGONAL_UR: ["translation_concept", "translate_2d", "vec2_add"],
        ACTION_REACH: ["object_detect", "adjacency"],
        ACTION_GRAB: [ACTION_REACH, "containment"],
        ACTION_HOLD: [ACTION_GRAB, "action_evaluate"],
        ACTION_RELEASE: [ACTION_GRAB, "delta_op"],
        ACTION_USE: [ACTION_HOLD, "goal_infer"],
        ACTION_WALK_TO: ["translation_concept", "pattern_repeat", "goal_infer"],
        ACTION_TELEPORT: ["translation_concept", "delta_op"],
        ACTION_LOOK_AT: ["transform_detect", "symmetry_rotate"],
    }
    stars: list[dict[str, Any]] = []
    for node_id in ids:
        node = reality_galaxy.get_node(node_id) if reality_galaxy is not None and hasattr(reality_galaxy, "get_node") else None
        stars.append(
            {
                "id": node_id,
                "name": str(getattr(node, "node_id", node_id)).split(":")[-1],
                "embedding": _reality_node_embedding(node),
                "galaxy_id": _fnv1a32("reality"),
                "star_type": STAR_TYPE_REALITY,
                "_ref_ids": list(refs.get(node_id, [])),
                "flags": STAR_FLAG_ACTIVE,
            }
        )
    return stars


def _action_population(reality_galaxy: Any) -> list[dict[str, Any]]:
    refs = {
        ACTION_MOVE_UP: ["translate_2d", "translation_concept", "vec2_add", "add_op"],
        ACTION_MOVE_DOWN: ["translate_2d", "translation_concept", "vec2_add", "sub_op"],
        ACTION_MOVE_LEFT: ["translate_2d", "translation_concept", "vec2_add", "sub_op"],
        ACTION_MOVE_RIGHT: ["translate_2d", "translation_concept", "vec2_add", "add_op"],
        ACTION_PERFORM: ["action_evaluate", "goal_infer", ACTION_USE, ACTION_HOLD],
        ACTION_CLICK: [ACTION_REACH, ACTION_LOOK_AT, "object_detect", "goal_infer"],
        ACTION_UNDO: ["delta_op", ACTION_RELEASE, "explore_strategy", "action_evaluate"],
    }
    stars: list[dict[str, Any]] = []
    for action_id in ARC3_EXTENDED_ACTION_ATOM_IDS:
        node = reality_galaxy.get_node(action_id) if reality_galaxy is not None and hasattr(reality_galaxy, "get_node") else None
        embedding = _reality_node_embedding(node) if node is not None else _displacement_to_embedding(_get_displacement(reality_galaxy, action_id))
        stars.append(
            {
                "id": action_id,
                "name": str(getattr(node, "node_id", action_id)).split(":")[-1],
                "embedding": embedding,
                "galaxy_id": _fnv1a32("reality"),
                "star_type": STAR_TYPE_ACTION,
                "_ref_ids": list(refs.get(action_id, [])),
                "flags": STAR_FLAG_ACTIVE | STAR_FLAG_LEARNABLE,
            }
        )
    return stars


def build_foundational_galaxy_table(galaxy: Any | None = None) -> list[dict[str, Any]]:
    """Build the complete foundational Galaxy table with ARC3 actions first."""
    reality_galaxy = galaxy if galaxy is not None else build_default_action_galaxy()
    stars: list[dict[str, Any]] = []
    stars.extend(_action_population(reality_galaxy))
    stars.extend(_drawing_math_stars())
    stars.extend(_math_operation_stars())
    stars.extend(_spatial_reasoning_stars())
    stars.extend(_support_action_stars(reality_galaxy))

    id_to_index = {str(star["id"]): index for index, star in enumerate(stars)}
    for star in stars:
        ref_ids = [str(ref_id) for ref_id in list(star.pop("_ref_ids", []))[:4]]
        star["component_refs"] = [id_to_index[ref_id] for ref_id in ref_ids if ref_id in id_to_index][:4]
    return stars


__all__ = ["build_foundational_galaxy_table"]
