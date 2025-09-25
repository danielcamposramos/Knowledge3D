from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from knowledge3d.cranium.ptx.modality_ops import PTXModalityOps
from knowledge3d.cranium.ptx.ptx_ops import PTX_OPS


@pytest.fixture(scope="module", autouse=True)
def ensure_ptx_available() -> None:
    try:
        PTX_OPS.text_modality("ptx smoketest")
    except RuntimeError as exc:
        pytest.skip(f"PTX modality ops unavailable: {exc}")


def _touch_file(path: Path, size: int = 128) -> Path:
    data = os.urandom(size)
    path.write_bytes(data)
    return path


def test_text_modality_confidence_range() -> None:
    result = PTX_OPS.text_modality("Honest garden tree 2025.09")
    assert isinstance(result["features"], list)
    assert len(result["features"]) == PTXModalityOps.TEXT_DIM
    assert 0.0 <= float(result["confidence"]) <= 1.0
    metrics: Dict[str, float] = result["metrics"]
    assert "length_norm" in metrics
    assert 0.0 <= metrics["length_norm"] <= 1.0


def test_image_modality_features_shape(tmp_path: Path) -> None:
    image_path = _touch_file(tmp_path / "sample_image.dat")
    result = PTX_OPS.image_modality(image_path.as_posix())
    assert len(result["features"]) == PTXModalityOps.IMAGE_DIM
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_audio_modality_features_shape(tmp_path: Path) -> None:
    audio_path = _touch_file(tmp_path / "sample_audio.dat")
    result = PTX_OPS.audio_modality(audio_path.as_posix())
    assert len(result["features"]) == PTXModalityOps.AUDIO_DIM
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_video_modality_features_shape(tmp_path: Path) -> None:
    video_path = _touch_file(tmp_path / "sample_video.dat", size=512)
    result = PTX_OPS.video_modality(video_path.as_posix())
    assert len(result["features"]) == PTXModalityOps.VIDEO_DIM
    assert 0.0 <= float(result["confidence"]) <= 1.0


def test_modality_kernels_are_deterministic() -> None:
    text = "Modal PTX determinism check"
    features_a = np.array(PTX_OPS.text_modality(text)["features"], dtype=np.float32)
    features_b = np.array(PTX_OPS.text_modality(text)["features"], dtype=np.float32)
    np.testing.assert_allclose(features_a, features_b, atol=1e-6)
