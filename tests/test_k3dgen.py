import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("pygltflib")

from pygltflib import GLTF2


def test_cli_generates_files(tmp_path):
    csv_path = Path("examples/sample_vectors.csv")
    gltf_out = tmp_path / "out.gltf"

    subprocess.run(
        [
            "python3",
            "-m",
            "k3dgen",
            str(csv_path),
            "--gltf",
            str(gltf_out),
            "--k",
            "2",
        ],
        check=True,
    )
    assert gltf_out.is_file()
    expected_records = sum(1 for _ in open(csv_path)) - 1

    # Verify .gltf file
    gltf = GLTF2().load(str(gltf_out))
    assert gltf.asset.version == "2.0"
    assert len(gltf.meshes) == 1
    assert len(gltf.nodes) == 1

    # Verify primitive extras (embedded variant)
    primitive = gltf.meshes[0].primitives[0]
    assert "k3dIds" in primitive.extras
    assert len(primitive.extras["k3dIds"]) == expected_records

    assert "k3d" in primitive.extras
    k3d = primitive.extras["k3d"]
    assert len(k3d["ids"]) == expected_records
    assert len(k3d["metadata"]) == expected_records
    # Payload can be arrays or bufferView indices; we require either path
    assert ("vectors" in k3d) or ("vectorsView" in k3d)
    assert ("embeddings" in k3d) or ("embeddingsView" in k3d)


def test_cli_negative_k(tmp_path):
    csv_path = Path("examples/sample_vectors.csv")
    result = subprocess.run(
        [
            "python3",
            "-m",
            "k3dgen",
            str(csv_path),
            "--k",
            "-1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "positive integer" in result.stderr
