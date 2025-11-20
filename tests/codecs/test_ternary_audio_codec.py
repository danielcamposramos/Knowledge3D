import numpy as np

from knowledge3d.cranium.codecs.ternary_audio_codec import TernaryAudioCodec


def test_mdct_roundtrip():
    codec = TernaryAudioCodec(sample_rate=8000, frame_size=512)
    frame = np.random.randn(512).astype(np.float32)

    mdct_coeffs = codec.mdct_frame(frame)
    reconstructed = codec.imdct_frame(mdct_coeffs)

    assert np.allclose(frame, reconstructed, atol=1e-5)


def test_encode_decode_simple_tone():
    codec = TernaryAudioCodec(sample_rate=8000)

    t = np.linspace(0, 1, 8000, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    encoded = codec.encode(audio)
    reconstructed = codec.decode(encoded)

    mse = np.mean((audio - reconstructed[: len(audio)]) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-12))
    assert psnr > 25

    ratio = codec.compute_compression_ratio(len(audio) * 4, encoded)
    assert ratio > 5
