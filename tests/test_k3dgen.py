import json
import subprocess
from pathlib import Path

from pygltflib import GLTF2


def test_cli_generates_files(tmp_path):
    csv_path = Path("examples/sample_vectors.csv")
    gltf_out = tmp_path / "out.gltf"
    k3d_out = tmp_path / "out.k3d"

    subprocess.run(
        [
            "python",
            "-m",
            "k3dgen",
            str(csv_path),
            "--gltf",
            str(gltf_out),
            "--k3d",
            str(k3d_out),
        ],
        check=True,
    )

    assert gltf_out.is_file()
    assert k3d_out.is_file()

    data = json.loads(k3d_out.read_text())
    expected = sum(1 for _ in open(csv_path)) - 1
    assert len(data) == expected
    assert all("id" in r and "vector" in r and "metadata" in r for r in data)
    assert all(len(r["vector"]) == 3 for r in data)

    gltf = GLTF2().load(str(gltf_out))
    assert gltf.asset.version == "2.0"
    assert len(gltf.meshes) == 1
    assert len(gltf.nodes) == 1
