"""Focused drawing-engine surface tests.

These tests validate the live drawing surface without pretending every declared
opcode is already production-ready in PTX. Unsupported runtime opcodes must fail
cleanly instead of silently claiming success.
"""

from __future__ import annotations

import numpy as np
import pytest

from knowledge3d.cranium.bridges.drawing_primitives_bridge import create_drawing_engine
from knowledge3d.cranium.ptx_runtime.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.ptx_runtime.rpn_opcodes import (
    OP_ATMOSPHERE_FOG,
    OP_BEZIER_EVAL,
    OP_CROSS_MODAL_LINK,
    OP_DRAW_FIELD_COEF,
    OP_DRAW_LAYER_NEW,
    OP_DRAW_REL_LINE,
    OP_LAYER_BLEND,
    OP_MARCHING_CUBES,
    OP_NURBS_EVAL,
    OP_PROCEDURAL_TEXTURE,
    OP_SHAPE_UNION,
    OP_VIGNETTE,
)

SUPPORTED_ENGINE_EXPRESSIONS = [
    "0.5 3 10 nurbs_eval",
    "0.0 32 32 32 0.5 marching_cubes",
    "5 0 10 lsystem_generate",
    "0 16 16 parametric_surface",
]

QUARANTINED_ENGINE_EXPRESSIONS = [
    "0.5 0 0 1 2 3 2 4 0 bezier_eval",
    "0 0 2 2 1 1 3 3 shape_union",
    "0 layer_blend",
    "cross_modal_link",
    "0 0.1 0.5 procedural_texture",
]


@pytest.fixture
def engine() -> ModularRPNEngine:
    inst = ModularRPNEngine()
    try:
        yield inst
    finally:
        inst.close()


def test_bridge_phase2_primitives() -> None:
    drawing_engine = create_drawing_engine()

    t_values = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    control_points = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 2.0, 4.0, 0.0], dtype=np.float32)
    bezier = drawing_engine.bridge.bezier_eval(t_values, control_points)
    assert bezier.shape == (3, 2)
    assert np.allclose(bezier[0], [0.0, 0.0])
    assert np.allclose(bezier[-1], [4.0, 0.0])

    shape_a = np.array([[0.0, 0.0, 2.0, 2.0]], dtype=np.float32)
    shape_b = np.array([[1.0, 1.0, 3.0, 3.0]], dtype=np.float32)
    union_result = drawing_engine.bridge.shape_union(shape_a, shape_b)
    intersect_result = drawing_engine.bridge.shape_intersect(shape_a, shape_b)
    assert union_result.tolist() == [[0.0, 0.0, 3.0, 3.0]]
    assert intersect_result.tolist() == [[1.0, 1.0, 2.0, 2.0]]


def test_bridge_stateful_layer_and_scene_ops() -> None:
    drawing_engine = create_drawing_engine()

    layer_a = np.ones((4, 4, 4), dtype=np.float32) * 0.25
    layer_b = np.ones((4, 4, 4), dtype=np.float32) * 0.75
    layer_a[:, :, 3] = 1.0
    layer_b[:, :, 3] = 1.0
    drawing_engine.bind_layers(layer_a, layer_b)
    drawing_engine.execute_opcode(OP_LAYER_BLEND, [1.0])
    blended = drawing_engine.get_last_layer_output()
    assert blended is not None
    assert blended.shape == layer_a.shape
    assert float(blended[0, 0, 0]) > float(layer_a[0, 0, 0])

    scene = np.ones((4, 4, 4), dtype=np.float32) * 0.5
    scene[:, :, 3] = 1.0
    drawing_engine.bind_scene(scene)
    drawing_engine.execute_opcode(OP_ATMOSPHERE_FOG, [0.8, 0.9, 1.0, 0.25])
    fogged = drawing_engine.get_last_scene_output()
    assert fogged is not None
    assert fogged.shape == scene.shape
    assert float(fogged[0, 0, 2]) >= float(scene[0, 0, 2])

    drawing_engine.bind_scene(scene)
    drawing_engine.execute_opcode(OP_VIGNETTE, [0.3, 2.0, 2.0])
    vignetted = drawing_engine.get_last_scene_output()
    assert vignetted is not None
    assert vignetted.shape == scene.shape


def test_engine_registers_drawing_opcodes(engine: ModularRPNEngine) -> None:
    expected = {
        "bezier_eval": OP_BEZIER_EVAL,
        "shape_union": OP_SHAPE_UNION,
        "rel_line": OP_DRAW_REL_LINE,
        "field_coef": OP_DRAW_FIELD_COEF,
        "layer_new": OP_DRAW_LAYER_NEW,
        "layer_blend": OP_LAYER_BLEND,
        "nurbs_eval": OP_NURBS_EVAL,
        "marching_cubes": OP_MARCHING_CUBES,
        "cross_modal_link": OP_CROSS_MODAL_LINK,
        "procedural_texture": OP_PROCEDURAL_TEXTURE,
    }
    for token, opcode in expected.items():
        assert engine.OPCODES[token] == opcode


@pytest.mark.parametrize("expr", SUPPORTED_ENGINE_EXPRESSIONS)
def test_engine_supported_runtime_subset_returns_scalars(engine: ModularRPNEngine, expr: str) -> None:
    result = engine.evaluate(expr)
    assert isinstance(result, float)


def test_engine_supported_batch_subset_returns_scalars(engine: ModularRPNEngine) -> None:
    results = engine.evaluate_batch(SUPPORTED_ENGINE_EXPRESSIONS)
    assert len(results) == len(SUPPORTED_ENGINE_EXPRESSIONS)
    assert all(isinstance(value, float) for value in results)


@pytest.mark.parametrize("expr", QUARANTINED_ENGINE_EXPRESSIONS)
def test_quarantined_runtime_ops_fail_cleanly(engine: ModularRPNEngine, expr: str) -> None:
    with pytest.raises((RuntimeError, ValueError)):
        engine.evaluate(expr)


def test_gpu_call_counter_advances_on_live_subset(engine: ModularRPNEngine) -> None:
    initial = engine.get_gpu_call_count()
    for expr in SUPPORTED_ENGINE_EXPRESSIONS[:2]:
        engine.evaluate(expr)
    assert engine.get_gpu_call_count() > initial


def test_invalid_token_still_errors(engine: ModularRPNEngine) -> None:
    with pytest.raises(ValueError):
        engine.evaluate("2 3 invalid_opcode")
