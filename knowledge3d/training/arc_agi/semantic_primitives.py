"""Semantic primitive definitions for ARC-AGI tasks.

These dictionaries provide a minimal, explicit mapping from natural language
tokens to structured semantic concepts used by the semantic parser and
semantic→RPN compiler.
"""

SPATIAL_SEMANTICS = {
    # Position
    "top": {"type": "position", "y": 0, "anchor": "top"},
    "bottom": {"type": "position", "y": "max", "anchor": "bottom"},
    "left": {"type": "position", "x": 0, "anchor": "left"},
    "right": {"type": "position", "x": "max", "anchor": "right"},
    "center": {"type": "position", "x": "mid", "y": "mid"},
    "corner": {"type": "position", "compound": True},
    "top-left": {"type": "position", "x": 0, "y": 0, "anchor": "corner"},
    "top-right": {"type": "position", "x": "max", "y": 0, "anchor": "corner"},
    "bottom-left": {"type": "position", "x": 0, "y": "max", "anchor": "corner"},
    "bottom-right": {"type": "position", "x": "max", "y": "max", "anchor": "corner"},

    # Direction
    "up": {"type": "direction", "dy": -1},
    "down": {"type": "direction", "dy": +1},
    "left_dir": {"type": "direction", "dx": -1},
    "right_dir": {"type": "direction", "dx": +1},
    "horizontal": {"type": "direction_axis", "axis": "horizontal"},
    "vertical": {"type": "direction_axis", "axis": "vertical"},

    # Transformation
    "rotate": {"type": "transform", "rpn_op": "rotate", "opcode": 70},
    "flip": {"type": "transform", "rpn_op": "flip", "opcode": None},
    "mirror": {"type": "transform", "rpn_op": "flip", "opcode": None},
    "scale": {"type": "transform", "rpn_op": "scale", "opcode": 71},
    "translate": {"type": "transform", "rpn_op": "translate", "opcode": 72},
    "move": {"type": "transform", "rpn_op": "translate", "opcode": 72},

    # Angle (for rotation)
    "90_degrees": {"type": "angle", "degrees": 90, "k": 1},
    "180_degrees": {"type": "angle", "degrees": 180, "k": 2},
    "270_degrees": {"type": "angle", "degrees": 270, "k": 3},
    "clockwise": {"type": "direction", "sign": -1},
    "counterclockwise": {"type": "direction", "sign": +1},
}

COLOR_SEMANTICS = {
    "black": {"type": "color", "value": 0},
    "blue": {"type": "color", "value": 1},
    "red": {"type": "color", "value": 2},
    "green": {"type": "color", "value": 3},
    "yellow": {"type": "color", "value": 4},
    "grey": {"type": "color", "value": 5},
    "gray": {"type": "color", "value": 5},
    "pink": {"type": "color", "value": 6},
    "orange": {"type": "color", "value": 7},
    "cyan": {"type": "color", "value": 8},
    "brown": {"type": "color", "value": 9},
}

SHAPE_SEMANTICS = {
    "square": {"type": "shape", "pattern": "filled_rectangle"},
    "rectangle": {"type": "shape", "pattern": "filled_rectangle"},
    "line": {"type": "shape", "pattern": "line"},
    "cross": {"type": "shape", "pattern": "cross"},
    "diagonal": {"type": "shape", "pattern": "diagonal"},
    "border": {"type": "shape", "pattern": "border"},
    "fill": {"type": "shape", "pattern": "fill_region"},
}

SIZE_SEMANTICS = {
    "largest": {"type": "size", "comparator": "max"},
    "smallest": {"type": "size", "comparator": "min"},
    "bigger": {"type": "size", "comparator": "greater"},
    "smaller": {"type": "size", "comparator": "less"},
}

ACTION_SEMANTICS = {
    "fill": {"type": "action", "rpn_op": "FILL", "opcode": 0x6B},
    "draw": {"type": "action", "rpn_op": "LINE", "opcode": 0x65},
    "move": {"type": "action", "rpn_op": "translate", "opcode": 72},
    "copy": {"type": "action", "rpn_op": "duplicate"},
    "extend": {"type": "action", "rpn_op": "extend_pattern"},
    "continue": {"type": "action", "rpn_op": "continue_sequence"},
    "repeat": {"type": "action", "rpn_op": "repeat_pattern"},
    "rotate": {"type": "action", "rpn_op": "rotate", "opcode": 70},
    "flip": {"type": "action", "rpn_op": "flip"},
    "mirror": {"type": "action", "rpn_op": "flip"},
    "recolor": {"type": "action", "rpn_op": "recolor"},
    "paint": {"type": "action", "rpn_op": "recolor"},
}

__all__ = [
    "SPATIAL_SEMANTICS",
    "COLOR_SEMANTICS",
    "SHAPE_SEMANTICS",
    "SIZE_SEMANTICS",
    "ACTION_SEMANTICS",
]
