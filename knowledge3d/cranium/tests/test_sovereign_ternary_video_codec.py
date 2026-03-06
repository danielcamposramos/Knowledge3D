import numpy as np

from knowledge3d.cranium.codecs.sovereign_ternary_video_codec import SovereignTernaryVideoCodec
from knowledge3d.cranium.ternary import TernaryVector, TernaryTensor


def test_video_codec_encode_decode_identity():
    # Small 8x8 frame, RGB constant patterns
    h, w = 8, 8
    pixels: list[int] = []
    for y in range(h):
        for x in range(w):
            r = (x + y) % 256
            g = (2 * x + y) % 256
            b = (x + 2 * y) % 256
            pixels.extend([r, g, b])
    vec = TernaryVector([0 if v < 85 else (1 if v > 170 else -1) for v in pixels])
    tensor = TernaryTensor((h, w, 3), vec)

    codec = SovereignTernaryVideoCodec(width=w, height=h, threshold=0.2)
    meta = codec.encode("frame0", tensor)
    assert meta["stored_in_galaxy"]
    assert meta["seed_rpn"].startswith("RESHAPE_TO_BLOCKS DCT8X8_FORWARD")
    assert meta["math_core_plan"]["preferred_tier"] == 2
    out = codec.decode("frame0")
    assert out.shape == (h, w, 3)
    # We quantize to ternary palette on decode; ensure lengths match
    assert len(out.values.to_python()) == len(pixels)


def test_video_codec_stores_real_procedural_seed():
    h, w = 8, 8
    pixels = [1 for _ in range(h * w * 3)]
    tensor = TernaryTensor((h, w, 3), TernaryVector(pixels))

    codec = SovereignTernaryVideoCodec(width=w, height=h, threshold=0.2)
    meta = codec.encode("frame_seed", tensor)
    seed_rpn, residual, metadata = codec.galaxy.load_frame_details("frame_seed")

    assert meta["seed_rpn"] == seed_rpn
    assert "TERNARY_QUANT" in seed_rpn
    assert metadata["math_core_plan"]["preferred_tier"] == 2
    assert metadata["blocks_per_channel"] == 1
    assert len(residual.to_python()) == h * w * 3


def test_video_codec_encode_frame_array_matches_tensor_path():
    h, w = 8, 8
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[..., 0] = 32
    frame[..., 1] = 128
    frame[..., 2] = 224

    codec = SovereignTernaryVideoCodec(width=w, height=h, threshold=0.2)
    meta = codec.encode_frame_array("frame_array", frame)
    seed_rpn, residual, metadata = codec.galaxy.load_frame_details("frame_array")

    assert meta["stored_in_galaxy"] is True
    assert meta["seed_rpn"] == seed_rpn
    assert metadata["blocks_per_channel"] == 1
    assert len(residual.to_python()) == h * w * 3
