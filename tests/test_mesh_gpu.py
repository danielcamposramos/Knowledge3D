from __future__ import annotations

from pathlib import Path

import pytest

from knowledge3d.cranium.ptx_runtime.mesh_opcodes import (
    generate_cube,
    generate_uv_sphere,
    mat4_mul,
    mat4_scale,
    mat4_translate,
)


def _gpu_bridge():
    ptx_path = Path("knowledge3d/cranium/ptx/mesh_generators.ptx")
    if not ptx_path.exists():
        pytest.skip("mesh_generators.ptx not compiled")
    try:
        from knowledge3d.cranium.bridges.sovereign_mesh_bridge import SovereignMeshBridge

        return SovereignMeshBridge()
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CUDA mesh bridge unavailable: {exc}")


def _assert_mesh_close(cpu_mesh, gpu_mesh, *, atol: float = 1e-5) -> None:
    assert len(cpu_mesh.vertices) == len(gpu_mesh.vertices)
    assert len(cpu_mesh.triangles) == len(gpu_mesh.triangles)
    for cpu_vertex, gpu_vertex in zip(cpu_mesh.vertices, gpu_mesh.vertices, strict=False):
        assert gpu_vertex == pytest.approx(cpu_vertex, abs=atol)
    for cpu_normal, gpu_normal in zip(cpu_mesh.normals, gpu_mesh.normals, strict=False):
        assert gpu_normal == pytest.approx(cpu_normal, abs=atol)


def test_gpu_cube_matches_cpu_cube():
    bridge = _gpu_bridge()
    cpu_mesh = generate_cube(1.0)
    gpu_mesh = bridge.generate_cube(1.0)
    _assert_mesh_close(cpu_mesh, gpu_mesh)


def test_gpu_sphere_matches_cpu_sphere():
    bridge = _gpu_bridge()
    cpu_mesh = generate_uv_sphere(1.0, 12, 16)
    gpu_mesh = bridge.generate_uv_sphere(1.0, 12, 16)
    _assert_mesh_close(cpu_mesh, gpu_mesh)


def test_gpu_transform_matches_cpu_transform():
    bridge = _gpu_bridge()
    source = generate_cube(1.0)
    matrix = mat4_mul(mat4_translate(2.0, 3.0, 4.0), mat4_scale(1.5, 0.5, 2.0))
    cpu_mesh = source.transformed(matrix)
    gpu_mesh = bridge.transform_mesh(source, matrix)
    _assert_mesh_close(cpu_mesh, gpu_mesh)
