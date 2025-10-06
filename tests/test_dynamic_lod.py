
import json
from pathlib import Path

import numpy as np
import pytest

from knowledge3d.viewer.semantic_viz import build_saliency_payload, write_saliency_manifest


@pytest.mark.cuda
def test_dynamic_lod_tuner_updates_morton(tmp_path: Path) -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:  # pragma: no cover - GPU unavailable
        pytest.skip("CUDA device not available")

    from knowledge3d.cranium.unified_fsm import UnifiedFSMContext

    fsm = UnifiedFSMContext()
    cp = cupy

    n_nodes = 2
    unified_buffer = fsm.create_unified_buffer(n_nodes)

    # Populate fused embeddings: node 0 aligns with query (high cosine), node 1 orthogonal.
    unified_buffer[:] = 0.0
    unified_buffer[0, :512] = 1.0
    unified_buffer[1, :512] = -1.0

    # Seed morton levels (uint32 view)
    morton_idx = 512 + 256 + 3
    morton_view = unified_buffer.view(cp.uint32).reshape(n_nodes, -1)
    morton_view[0, morton_idx] = np.uint32(5)
    morton_view[1, morton_idx] = np.uint32(3)

    query = np.ones(512, dtype=np.float32)
    query_gpu = cp.asarray(query)

    saliency_gpu = fsm.apply_dynamic_lod(unified_buffer, query_gpu, saliency_threshold=0.7)
    cp.cuda.runtime.deviceSynchronize()

    saliency = saliency_gpu.get()
    morton_after = morton_view[:, morton_idx].get()

    assert saliency.shape == (n_nodes, 2)
    assert saliency[0, 0] > 0.9  # high cosine should remain near 1
    assert saliency[1, 0] < 0.0  # low cosine should be negative

    # Node 0 should lower LOD (max precision), node 1 should increase (less detail)
    assert morton_after[0] == 4
    assert morton_after[1] == 4

    payload = build_saliency_payload(["node-0", "node-1"], saliency, morton_after)
    assert "extensions" in payload and "K3D_saliency" in payload["extensions"]
    assert payload["extensions"]["K3D_saliency"]["nodes"]["node-0"]["lod"] == 4

    manifest_path = tmp_path / "saliency.json"
    write_saliency_manifest(manifest_path, [0, 1], saliency, morton_after)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert str(0) in data["extensions"]["K3D_saliency"]["nodes"]
