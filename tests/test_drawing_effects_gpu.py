from __future__ import annotations

import numpy as np
import pytest


def _require_gpu():
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA device not available")
    return cupy


@pytest.mark.cuda
def test_linear_gradient_endpoints():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects

    effects = DrawingEffects()
    rgba = effects.linear_gradient(
        32,
        16,
        [
            (0.0, 1.0, 0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0, 1.0, 1.0),
        ],
        x1=0.0,
        y1=0.0,
        x2=1.0,
        y2=0.0,
    )

    assert rgba.shape == (16, 32, 4)
    assert rgba[0, 0, 0] > 0.95
    assert rgba[0, 0, 2] < 0.05
    assert rgba[0, -1, 2] > 0.95
    assert rgba[0, -1, 0] < 0.05
    assert np.allclose(rgba[..., 3], 1.0, atol=1e-5)


@pytest.mark.cuda
def test_blur_sharpen_and_invert_stack():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects

    effects = DrawingEffects()
    canvas = np.zeros((33, 33, 4), dtype=np.float32)
    canvas[16, 16, :3] = 1.0
    canvas[16, 16, 3] = 1.0

    blurred = effects.blur_rgba(canvas, radius=2)
    sharpened = effects.sharpen_rgba(blurred, radius=1, amount=1.25)
    inverted = effects.invert_rgba(blurred)

    assert blurred.shape == canvas.shape
    assert sharpened.shape == canvas.shape
    assert inverted.shape == canvas.shape

    assert blurred[16, 16, 0] < canvas[16, 16, 0]
    assert float(np.max(sharpened[..., 0])) >= float(np.max(blurred[..., 0]))
    assert np.allclose(inverted[..., 3], blurred[..., 3], atol=1e-6)
    assert np.allclose(inverted[..., :3], 1.0 - blurred[..., :3], atol=1e-5)


@pytest.mark.cuda
def test_painterly_bridge_stack_and_edge_map():
    _require_gpu()
    from knowledge3d.cranium.bridges.procedural_drawing_bridge import ProceduralDrawingBridge

    bridge = ProceduralDrawingBridge(matryoshka_dim=64)
    result = bridge.render_painterly_gpu(
        "-0.75 -0.75 MOVE 0.75 0.75 LINE STROKE",
        width=64,
        height=64,
        background="radial",
        blur_radius=1,
        sharpen_amount=0.6,
    )

    assert result.rgba is not None
    assert result.rgba.shape == (64, 64, 4)
    assert float(np.min(result.rgba[..., 3])) > 0.95
    assert np.any(result.rgba[..., 0] > 0.1)

    edges = bridge.edge_map_gpu(result.rgba)
    assert edges.shape == (64, 64)
    assert float(np.max(edges)) > 0.05


@pytest.mark.cuda
def test_ternary_gradient_signature_and_contrastive_score():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects

    effects = DrawingEffects()
    target = [
        (0.0, 0.1, 0.2, 0.8, 1.0),
        (0.5, 0.6, 0.3, 0.4, 1.0),
        (1.0, 0.9, 0.7, 0.1, 1.0),
    ]
    positive = [
        (0.0, 0.12, 0.18, 0.82, 1.0),
        (0.52, 0.58, 0.32, 0.38, 1.0),
        (1.0, 0.88, 0.72, 0.08, 1.0),
    ]
    negative = [
        (0.0, 0.9, 0.8, 0.1, 1.0),
        (0.5, 0.4, 0.7, 0.5, 1.0),
        (1.0, 0.1, 0.2, 0.9, 1.0),
    ]

    signature = effects.encode_gradient_signature(target)
    assert signature.delta_trits
    for row in signature.delta_trits:
        assert set(row).issubset({-1, 0, 1})

    pos_score = effects.contrastive_gradient_score(target, positive, negative_examples=(negative,))
    neg_score = effects.contrastive_gradient_score(target, negative, negative_examples=(positive,))

    assert pos_score.score > neg_score.score
    assert pos_score.positive_similarity >= neg_score.positive_similarity


@pytest.mark.cuda
def test_ternary_palette_signature_and_contrastive_score():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects

    effects = DrawingEffects()
    target = [
        (0.1, 0.2, 0.8, 1.0),
        (0.4, 0.5, 0.9, 1.0),
        (0.9, 0.95, 1.0, 1.0),
    ]
    positive = [
        (0.12, 0.18, 0.78, 1.0),
        (0.42, 0.52, 0.88, 1.0),
        (0.92, 0.96, 1.0, 1.0),
    ]
    negative = [
        (0.8, 0.2, 0.1, 1.0),
        (0.9, 0.5, 0.3, 1.0),
        (1.0, 0.9, 0.7, 1.0),
    ]

    signature = effects.encode_palette_signature(target)
    assert signature.gradient_signature.delta_trits
    for row in signature.gradient_signature.delta_trits:
        assert set(row).issubset({-1, 0, 1})

    pos_score = effects.contrastive_palette_score(target, positive, negative_examples=(negative,))
    neg_score = effects.contrastive_palette_score(target, negative, negative_examples=(positive,))

    assert pos_score.score > neg_score.score
    assert pos_score.positive_similarity >= neg_score.positive_similarity


@pytest.mark.cuda
def test_linear_gradient_from_ternary_cascade_renders():
    _require_gpu()
    from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects

    effects = DrawingEffects()
    rgba = effects.linear_gradient_from_ternary_cascade(
        32,
        12,
        base_stop=(0.0, 0.1, 0.1, 0.1, 1.0),
        position_layers=(
            (0, 0, 0, 0),
            (0, 1, 0, -1),
        ),
        color_layers=(
            ((1, 0, -1, 0), (1, 0, -1, 0), (0, 1, 0, 0), (0, 1, 0, 0)),
            ((0, 1, 0, 0), (0, 0, 1, 0), (0, -1, 1, 0), (1, 0, 0, 0)),
        ),
        x1=0.0,
        y1=0.0,
        x2=1.0,
        y2=0.0,
    )

    assert rgba.shape == (12, 32, 4)
    assert np.allclose(rgba[..., 3], 1.0, atol=1e-5)
    assert float(np.mean(rgba[:, -1, 0])) > float(np.mean(rgba[:, 0, 0]))
