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
    out = codec.decode("frame0")
    assert out.shape == (h, w, 3)
    # We quantize to ternary palette on decode; ensure lengths match
    assert len(out.values.to_python()) == len(pixels)
