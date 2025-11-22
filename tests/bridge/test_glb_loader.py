import json
from pathlib import Path

from knowledge3d.bridge.glb_ctypes_loader import save_stars_to_glb, load_stars_from_glb


def test_glb_roundtrip(tmp_path):
    stars = [{"id": "s1", "k3d": {"foo": "bar"}}]
    glb_path = save_stars_to_glb(stars, tmp_path / "out.glb")
    loaded = load_stars_from_glb(glb_path)
    assert loaded and isinstance(loaded[0], dict)
