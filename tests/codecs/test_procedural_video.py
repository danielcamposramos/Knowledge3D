import numpy as np

from knowledge3d.cranium.codecs.procedural_video import ProceduralVideoGenerator


def test_generate_frame_deterministic():
    gen = ProceduralVideoGenerator(width=128, height=128)
    seed = np.random.default_rng(0).standard_normal(64).astype(np.float32)

    frame1 = gen.generate_frame(seed, time_param=0.5)
    frame2 = gen.generate_frame(seed, time_param=0.5)

    assert np.array_equal(frame1, frame2)


def test_generate_frame_temporal_coherence():
    gen = ProceduralVideoGenerator(width=64, height=64)
    seed = np.random.default_rng(1).standard_normal(256).astype(np.float32)

    frame_t0 = gen.generate_frame(seed, time_param=0.0)
    frame_t1 = gen.generate_frame(seed, time_param=0.01)

    diff = np.mean(np.abs(frame_t1.astype(float) - frame_t0.astype(float)))
    assert diff < 50


def test_perlin_noise_range():
    gen = ProceduralVideoGenerator(width=32, height=32)
    u, v = np.meshgrid(np.linspace(0, 1, 32, endpoint=False), np.linspace(0, 1, 32, endpoint=False))
    noise = gen.perlin_noise(u.astype(np.float32), v.astype(np.float32), seed_params=np.random.randn(16).astype(np.float32))
    assert noise.min() >= 0.0 and noise.max() <= 1.0
