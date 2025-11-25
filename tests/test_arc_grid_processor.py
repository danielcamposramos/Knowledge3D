from __future__ import annotations

import numpy as np

from knowledge3d.training.arc_agi import ARCGridProcessor


class _TestVisualEmbedder:
    """Deterministic stub visual embedder for tests."""

    def __init__(self, dim: int):
        self.dim = dim

    def emit_fractal_features(self, raster: np.ndarray) -> np.ndarray:
        flat = np.asarray(raster, dtype=np.float32).ravel()
        if flat.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        # Simple deterministic projection: repeat / truncate pattern.
        tiled = np.resize(flat, self.dim)
        return tiled.astype(np.float32)


def _make_processor(dim: int = 32) -> ARCGridProcessor:
    return ARCGridProcessor(matryoshka_dim=dim, visual_embedder=_TestVisualEmbedder(dim))


def test_grid_to_rpn_program_basic():
    processor = _make_processor()
    grid = [
        [0, 1],
        [2, 0],
    ]
    rpn = processor.grid_to_rpn_program(grid)

    # Should contain commands for the two non-zero cells.
    assert "MOVE" in rpn
    assert "LINE" in rpn
    assert "SET_FILL_COLOR" in rpn
    # Colors 1 and 2 should appear.
    assert "1" in rpn
    assert "2" in rpn


def test_grid_to_spatial_embedding_shape_and_type():
    dim = 64
    processor = _make_processor(dim)
    grid = [
        [0, 1, 0],
        [1, 2, 1],
        [0, 1, 0],
    ]
    emb = processor.grid_to_spatial_embedding(grid)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (dim,)
    assert emb.dtype == np.float32


def test_detect_spatial_primitive_rotation_90():
    processor = _make_processor()
    grid_before = [
        [1, 0],
        [0, 0],
    ]
    grid_after = [
        [0, 1],
        [0, 0],
    ]
    result = processor.detect_spatial_primitive(grid_before, grid_after)
    assert result["primitive"] == "ROTATE_90"
    assert result["parameters"]["angle"] == 90


def test_detect_spatial_primitive_flip_horizontal():
    processor = _make_processor()
    grid_before = [
        [1, 0],
        [2, 3],
    ]
    grid_after = [
        [0, 1],
        [3, 2],
    ]
    result = processor.detect_spatial_primitive(grid_before, grid_after)
    assert result["primitive"] == "FLIP_H"


def test_detect_spatial_primitive_translation():
    processor = _make_processor()
    grid_before = [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    grid_after = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
    ]
    result = processor.detect_spatial_primitive(grid_before, grid_after)
    assert result["primitive"] == "TRANSLATE"
    assert result["parameters"]["dx"] == 0
    assert result["parameters"]["dy"] == 2

