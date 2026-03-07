from knowledge3d.cranium.codecs.sovereign_ternary_audio_codec import SovereignTernaryAudioCodec
from knowledge3d.cranium.ternary import TernaryVector


def test_audio_codec_encode_decode_length_and_seed():
    samples = [(-1 if i % 3 == 0 else (0 if i % 3 == 1 else 1)) for i in range(1500)]
    vec = TernaryVector(samples)

    codec = SovereignTernaryAudioCodec(frame_size=1024, threshold=0.2)
    meta = codec.encode("clip0", vec)

    assert meta["stored_in_galaxy"]
    assert meta["seed_rpn"] == "1024 BATCH_MDCT 0.2 TERNARY_QUANT"
    assert meta["original_length"] == 1500
    assert meta["padded_length"] == 2048
    assert meta["frame_count"] == 2
    assert meta["math_core_plan"]["preferred_tier"] == 2

    out = codec.decode("clip0")
    assert len(out.to_python()) == 1500


def test_audio_codec_persists_clip_metadata():
    samples = TernaryVector([1, 0, -1, 1, 0, -1, 1, 0, -1, 1])
    codec = SovereignTernaryAudioCodec(frame_size=8, threshold=0.15)
    codec.encode("clip_meta", samples)

    seed_rpn, residual, metadata = codec.galaxy.load_frame_details("clip_meta")
    assert seed_rpn == "8 BATCH_MDCT 0.15 TERNARY_QUANT"
    assert metadata["original_length"] == 10
    assert metadata["padded_length"] == 16
    assert metadata["frame_count"] == 2
    assert metadata["math_core_plan"]["preferred_tier"] == 2
    assert len(residual.to_python()) == 8


def test_audio_codec_encode_details_returns_quantized_coeffs():
    samples = TernaryVector([1, 0, -1, 1, 0, -1, 1, 0, -1, 1])
    codec = SovereignTernaryAudioCodec(frame_size=8, threshold=0.15)
    encoded = codec.encode_details("clip_details", samples)

    assert encoded["metadata"]["seed_rpn"] == "8 BATCH_MDCT 0.15 TERNARY_QUANT"
    assert encoded["metadata"]["frame_count"] == 2
    assert encoded["quantized_coeffs"].shape == (8,)
    assert len(encoded["residual"].to_python()) == 8
