import numpy as np

from knowledge3d.cranium.codecs.procedural_audio import ProceduralAudioSynthesizer


def test_analyze_pure_sine():
    synth = ProceduralAudioSynthesizer(sample_rate=8000)

    t = np.linspace(0, 1, 8000, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    harmonics = synth.analyze(audio, n_harmonics=5)

    assert len(harmonics) > 0
    freq, amp, phase = harmonics[0]
    assert 435 < freq < 445
    assert amp > 0.5


def test_synthesize_roundtrip():
    synth = ProceduralAudioSynthesizer(sample_rate=8000)

    t = np.linspace(0, 1, 8000, endpoint=False)
    original = (
        np.sin(2 * np.pi * 440 * t)
        + 0.5 * np.sin(2 * np.pi * 880 * t)
        + 0.25 * np.sin(2 * np.pi * 1320 * t)
    ).astype(np.float32)

    harmonics = synth.analyze(original, n_harmonics=10)
    reconstructed = synth.synthesize(harmonics, duration_sec=1.0)

    mse = np.mean((original - reconstructed) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))
    assert psnr > 20
