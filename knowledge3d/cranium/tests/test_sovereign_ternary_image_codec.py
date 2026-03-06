from knowledge3d.cranium.codecs.sovereign_ternary_image_codec import SovereignTernaryImageCodec
from knowledge3d.cranium.ternary import TernaryTensor, TernaryVector


def test_image_codec_encode_decode_shape_and_seed():
    h = w = 8
    pixels: list[int] = []
    for y in range(h):
        for x in range(w):
            pixels.extend([(x + y) % 256, (2 * x + y) % 256, (x + 2 * y) % 256])
    tensor = TernaryTensor((h, w, 3), TernaryVector(pixels))

    codec = SovereignTernaryImageCodec(n_coefficients=192, threshold=0.15)
    meta = codec.encode("img0", tensor)
    assert meta["stored_in_galaxy"]
    assert meta["seed_rpn"] == "VECTORDOTMAP 192 8 8"
    assert meta["math_core_plan"]["preferred_tier"] == 2
    assert meta["padded_size"] == (8, 8)

    _, _, stored_meta = codec.galaxy.load_frame_details("img0")
    assert stored_meta["math_core_plan"]["preferred_tier"] == 2
    assert stored_meta["blocks_per_channel"] == 1

    out = codec.decode("img0")
    assert out.shape == (h, w, 3)
    assert len(out.values.to_python()) == len(pixels)


def test_image_codec_non_multiple_of_8_roundtrip_shape():
    h, w = 10, 9
    pixels: list[int] = []
    for y in range(h):
        for x in range(w):
            pixels.extend([(x + y) % 256, (3 * x + y) % 256, (x + 3 * y) % 256])
    tensor = TernaryTensor((h, w, 3), TernaryVector(pixels))

    codec = SovereignTernaryImageCodec(n_coefficients=192, threshold=0.15)
    meta = codec.encode("img_pad", tensor)
    assert meta["original_size"] == (h, w)
    assert meta["padded_size"] == (16, 16)
    assert meta["math_core_plan"]["preferred_tier"] == 2

    _, _, stored_meta = codec.galaxy.load_frame_details("img_pad")
    assert stored_meta["padded_height"] == 16
    assert stored_meta["padded_width"] == 16
    assert stored_meta["blocks_per_channel"] == 4

    out = codec.decode("img_pad")
    assert out.shape == (h, w, 3)
    assert len(out.values.to_python()) == len(pixels)
