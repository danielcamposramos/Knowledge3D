import numpy as np

from knowledge3d.cranium.bridges.procedural_signal_bridge import ProceduralSignalBridge
from knowledge3d.cranium.ternary import TernaryVector


def test_audio_to_spectrogram_emits_tier2_plan_and_preview():
    samples = TernaryVector([(-1 if i % 5 == 0 else (1 if i % 2 == 0 else 0)) for i in range(1500)])
    bridge = ProceduralSignalBridge(frame_size=256, threshold=0.2)

    projection = bridge.audio_to_spectrogram("sig0", samples)

    assert projection.spectrogram.shape == (128, 6)
    assert projection.preview_rgba.shape == (128, 6, 4)
    assert np.allclose(projection.preview_rgba[projection.spectrogram > 0][:, 3], 1.0, atol=1e-5)
    assert projection.metadata["math_core_plan"]["preferred_tier"] == 2
    assert projection.metadata["frame_count"] == 6
    assert 0.0 <= projection.metadata["positive_ratio"] <= 1.0
    assert 0.0 <= projection.metadata["negative_ratio"] <= 1.0
    assert 0.0 <= projection.metadata["neutral_ratio"] <= 1.0


def test_spectrogram_to_surface_emits_mesh_with_tier3_plan():
    samples = TernaryVector([(-1 if i % 7 == 0 else (1 if i % 3 == 0 else 0)) for i in range(1024)])
    bridge = ProceduralSignalBridge(frame_size=256, threshold=0.15)
    projection = bridge.audio_to_spectrogram("sig_surface", samples)

    surface = bridge.spectrogram_to_surface(projection, displacement_gain=0.5)

    rows, cols = projection.spectrogram.shape
    assert surface.heightfield.shape == (rows, cols)
    assert surface.vertices.shape == (rows * cols, 3)
    assert surface.indices.shape == ((rows - 1) * (cols - 1) * 2, 3)
    assert surface.normals.shape == (rows * cols, 3)
    assert surface.metadata["math_core_plan"]["preferred_tier"] == 3
    assert surface.metadata["source_math_core_plan"]["preferred_tier"] == 2
    lengths = np.linalg.norm(surface.normals, axis=1)
    assert np.all(lengths > 0.0)
    assert np.allclose(lengths, 1.0, atol=1e-4)
