import numpy as np

from knowledge3d.cranium.codecs.ternary_video_codec import TernaryVideoCodec


def test_video_codec_roundtrip():
    codec = TernaryVideoCodec(width=64, height=64)
    seed = np.random.default_rng(0).standard_normal(128).astype(np.float32)
    frame = codec.generator.generate_frame(seed, time_param=0.2)

    encoded = codec.encode(frame, seed=seed)
    decoded = codec.decode(encoded)

    diff = np.mean(np.abs(decoded.astype(float) - frame.astype(float)))
    assert diff < 20  # tolerate small distortion from quantisation
