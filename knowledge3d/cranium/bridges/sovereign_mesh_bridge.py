"""Sovereign GPU-backed mesh bridge for the H2 primitive subset."""

from __future__ import annotations

import ctypes
from pathlib import Path

from knowledge3d.cranium.ptx_runtime.mesh_engine import MeshRPNEngine
from knowledge3d.cranium.ptx_runtime.mesh_opcodes import (
    Matrix4,
    MeshBuffer,
    generate_icosphere as generate_icosphere_cpu,
    mat4_identity,
    mat4_mul,
    mat4_rotate_x,
    mat4_rotate_y,
    mat4_rotate_z,
    mat4_scale,
    mat4_translate,
)
from knowledge3d.cranium.sovereign import loader


_ICOSPHERE_TRIANGLES = [
    (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
    (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
    (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
    (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
]


class SovereignMeshBridge:
    """GPU-backed mesh generation via mesh_generators.ptx."""

    _GENERATOR_TOKENS = {
        "GEN_PLANE",
        "GEN_CUBE",
        "GEN_UV_SPHERE",
        "GEN_CYLINDER",
        "GEN_CONE",
        "GEN_TORUS",
        "GEN_ICOSPHERE",
    }
    _MATRIX_TOKENS = {
        "MAT4_IDENTITY",
        "MAT4_TRANSLATE",
        "MAT4_SCALE",
        "MAT4_ROTATE_X",
        "MAT4_ROTATE_Y",
        "MAT4_ROTATE_Z",
        "MAT4_MUL",
        "MAT4_APPLY",
    }

    def __init__(self) -> None:
        ptx_path = Path(__file__).parent.parent / "ptx" / "mesh_generators.ptx"
        if not ptx_path.exists():
            raise FileNotFoundError(
                f"Mesh generators PTX not found: {ptx_path}. Compile with:\n"
                "  nvcc -ptx -arch=sm_86 "
                "knowledge3d/cranium/kernels/mesh_generators.cu "
                "-o knowledge3d/cranium/ptx/mesh_generators.ptx"
            )
        self._module = loader.load_module_from_file(str(ptx_path))
        self._kernels = {
            "plane": loader.get_function(self._module, "generate_plane_vertices"),
            "cube": loader.get_function(self._module, "generate_cube_vertices"),
            "uv_sphere": loader.get_function(self._module, "generate_uv_sphere_vertices"),
            "cylinder": loader.get_function(self._module, "generate_cylinder_vertices"),
            "cone": loader.get_function(self._module, "generate_cone_vertices"),
            "torus": loader.get_function(self._module, "generate_torus_vertices"),
            "icosphere": loader.get_function(self._module, "generate_icosphere_vertices"),
            "transform": loader.get_function(self._module, "mat4_transform_vertices"),
            "grid_indices": loader.get_function(self._module, "generate_index_buffer_grid"),
            "face_normals": loader.get_function(self._module, "compute_face_normals"),
        }
        self._engine = MeshRPNEngine()

    @staticmethod
    def _is_number(token: str) -> bool:
        try:
            float(token)
            return True
        except Exception:
            return False

    def is_supported_program(self, program: str) -> bool:
        tokens = self._engine.tokenize_rpn(program)
        uppers = [token.upper() for token in tokens]
        generator_count = sum(1 for token in uppers if token in self._GENERATOR_TOKENS)
        if generator_count != 1:
            return False
        for token, upper in zip(tokens, uppers, strict=False):
            if upper in self._GENERATOR_TOKENS or upper in self._MATRIX_TOKENS:
                continue
            if not self._is_number(token):
                return False
        return True

    def execute_supported_program(self, program: str) -> MeshBuffer:
        tokens = self._engine.tokenize_rpn(program)
        stack: list[object] = []

        def pop_float(default: float = 0.0) -> float:
            if not stack:
                return float(default)
            return float(stack.pop())

        def pop_mesh() -> MeshBuffer:
            if not stack or not isinstance(stack[-1], MeshBuffer):
                raise ValueError("GPU mesh subset expected a mesh on the stack")
            return stack.pop()

        for token in tokens:
            upper = token.upper()
            if upper not in self._GENERATOR_TOKENS and upper not in self._MATRIX_TOKENS:
                stack.append(float(token))
                continue
            if upper == "MAT4_IDENTITY":
                stack.append(mat4_identity())
            elif upper == "MAT4_TRANSLATE":
                tz = pop_float()
                ty = pop_float()
                tx = pop_float()
                stack.append(mat4_translate(tx, ty, tz))
            elif upper == "MAT4_SCALE":
                sz = pop_float()
                sy = pop_float()
                sx = pop_float()
                stack.append(mat4_scale(sx, sy, sz))
            elif upper == "MAT4_ROTATE_X":
                stack.append(mat4_rotate_x(pop_float()))
            elif upper == "MAT4_ROTATE_Y":
                stack.append(mat4_rotate_y(pop_float()))
            elif upper == "MAT4_ROTATE_Z":
                stack.append(mat4_rotate_z(pop_float()))
            elif upper == "MAT4_MUL":
                b = stack.pop()
                a = stack.pop()
                stack.append(mat4_mul(a, b))
            elif upper == "MAT4_APPLY":
                matrix = stack.pop()
                mesh = pop_mesh()
                stack.append(self.transform_mesh(mesh, matrix))
            elif upper == "GEN_PLANE":
                seg_d = int(pop_float(1.0))
                seg_w = int(pop_float(1.0))
                depth = pop_float(1.0)
                width = pop_float(1.0)
                stack.append(self.generate_plane(width, depth, seg_w, seg_d))
            elif upper == "GEN_CUBE":
                stack.append(self.generate_cube(pop_float(1.0)))
            elif upper == "GEN_UV_SPHERE":
                slices = int(pop_float(16.0))
                stacks = int(pop_float(12.0))
                radius = pop_float(1.0)
                stack.append(self.generate_uv_sphere(radius, stacks, slices))
            elif upper == "GEN_CYLINDER":
                _caps = int(pop_float(1.0))
                segments = int(pop_float(16.0))
                height = pop_float(1.0)
                radius = pop_float(0.5)
                stack.append(self.generate_cylinder(radius, height, segments, 1))
            elif upper == "GEN_CONE":
                segments = int(pop_float(16.0))
                height = pop_float(1.0)
                radius = pop_float(0.5)
                stack.append(self.generate_cone(radius, height, segments))
            elif upper == "GEN_TORUS":
                minor_segments = int(pop_float(8.0))
                major_segments = int(pop_float(16.0))
                minor_radius = pop_float(0.25)
                major_radius = pop_float(1.0)
                stack.append(self.generate_torus(major_radius, minor_radius, major_segments, minor_segments))
            elif upper == "GEN_ICOSPHERE":
                subdivisions = int(pop_float(1.0))
                radius = pop_float(1.0)
                stack.append(self.generate_icosphere(radius, subdivisions))

        if not stack or not isinstance(stack[-1], MeshBuffer):
            raise ValueError("GPU mesh subset program did not produce a mesh")
        return stack[-1]

    @staticmethod
    def _alloc_float_buffer(count: int) -> tuple[loader.CUdeviceptr, ctypes.Array[ctypes.c_float]]:
        host = (ctypes.c_float * max(1, count))()
        device = loader.gpu_malloc(max(1, count) * ctypes.sizeof(ctypes.c_float))
        return device, host

    @staticmethod
    def _alloc_uint_buffer(count: int) -> tuple[loader.CUdeviceptr, ctypes.Array[ctypes.c_uint]]:
        host = (ctypes.c_uint * max(1, count))()
        device = loader.gpu_malloc(max(1, count) * ctypes.sizeof(ctypes.c_uint))
        return device, host

    @staticmethod
    def _triangles_from_indices(buffer: ctypes.Array[ctypes.c_uint], count: int) -> list[tuple[int, int, int]]:
        triangles: list[tuple[int, int, int]] = []
        for cursor in range(0, count, 3):
            triangles.append((int(buffer[cursor]), int(buffer[cursor + 1]), int(buffer[cursor + 2])))
        return triangles

    @staticmethod
    def _mesh_from_buffers(
        *,
        vertex_count: int,
        vertices: ctypes.Array[ctypes.c_float],
        normals: ctypes.Array[ctypes.c_float],
        uvs: ctypes.Array[ctypes.c_float],
        triangles: list[tuple[int, int, int]],
        metadata: dict[str, object],
    ) -> MeshBuffer:
        mesh = MeshBuffer(
            vertices=[
                (
                    float(vertices[index * 3 + 0]),
                    float(vertices[index * 3 + 1]),
                    float(vertices[index * 3 + 2]),
                )
                for index in range(vertex_count)
            ],
            triangles=triangles,
            normals=[
                (
                    float(normals[index * 3 + 0]),
                    float(normals[index * 3 + 1]),
                    float(normals[index * 3 + 2]),
                )
                for index in range(vertex_count)
            ],
            uvs=[
                (
                    float(uvs[index * 2 + 0]),
                    float(uvs[index * 2 + 1]),
                )
                for index in range(vertex_count)
            ],
            metadata=metadata,
        )
        if mesh.triangles or len(mesh.normals) != len(mesh.vertices):
            mesh.compute_missing_normals()
        return mesh

    def _copy_back_mesh(
        self,
        *,
        d_vertices: loader.CUdeviceptr,
        d_normals: loader.CUdeviceptr,
        d_uvs: loader.CUdeviceptr,
        d_indices: loader.CUdeviceptr | None,
        vertex_count: int,
        index_count: int,
        host_vertices: ctypes.Array[ctypes.c_float],
        host_normals: ctypes.Array[ctypes.c_float],
        host_uvs: ctypes.Array[ctypes.c_float],
        host_indices: ctypes.Array[ctypes.c_uint] | None,
        metadata: dict[str, object],
    ) -> MeshBuffer:
        loader.memcpy_dtoh(ctypes.cast(host_vertices, ctypes.c_void_p), d_vertices, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
        loader.memcpy_dtoh(ctypes.cast(host_normals, ctypes.c_void_p), d_normals, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
        loader.memcpy_dtoh(ctypes.cast(host_uvs, ctypes.c_void_p), d_uvs, vertex_count * 2 * ctypes.sizeof(ctypes.c_float))
        triangles: list[tuple[int, int, int]] = []
        if d_indices is not None and host_indices is not None and index_count > 0:
            loader.memcpy_dtoh(ctypes.cast(host_indices, ctypes.c_void_p), d_indices, index_count * ctypes.sizeof(ctypes.c_uint))
            triangles = self._triangles_from_indices(host_indices, index_count)
        return self._mesh_from_buffers(
            vertex_count=vertex_count,
            vertices=host_vertices,
            normals=host_normals,
            uvs=host_uvs,
            triangles=triangles,
            metadata=metadata,
        )

    def _launch_grid_indices(self, rows: int, cols: int) -> tuple[loader.CUdeviceptr, ctypes.Array[ctypes.c_uint], int]:
        quad_count = max(0, (int(rows) - 1) * (int(cols) - 1))
        index_count = quad_count * 6
        d_indices, host_indices = self._alloc_uint_buffer(index_count)
        try:
            if index_count:
                loader.launch(
                    self._kernels["grid_indices"],
                    grid=((quad_count + 255) // 256, 1, 1),
                    block=(256, 1, 1),
                    params=[
                        ctypes.c_uint64(d_indices.value),
                        ctypes.c_int(int(rows)),
                        ctypes.c_int(int(cols)),
                    ],
                )
                loader.synchronize()
            return d_indices, host_indices, index_count
        except Exception:
            loader.gpu_free(d_indices)
            raise

    def generate_plane(self, width: float, depth: float, seg_w: int, seg_d: int) -> MeshBuffer:
        seg_w = max(1, int(seg_w))
        seg_d = max(1, int(seg_d))
        vertex_count = (seg_w + 1) * (seg_d + 1)
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        d_indices, host_indices, index_count = self._launch_grid_indices(seg_d + 1, seg_w + 1)
        try:
            loader.launch(
                self._kernels["plane"],
                grid=((vertex_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_float(float(width)),
                    ctypes.c_float(float(depth)),
                    ctypes.c_int(seg_w),
                    ctypes.c_int(seg_d),
                ],
            )
            loader.synchronize()
            return self._copy_back_mesh(
                d_vertices=d_vertices,
                d_normals=d_normals,
                d_uvs=d_uvs,
                d_indices=d_indices,
                vertex_count=vertex_count,
                index_count=index_count,
                host_vertices=host_vertices,
                host_normals=host_normals,
                host_uvs=host_uvs,
                host_indices=host_indices,
                metadata={"primitive": "plane", "backend": "gpu"},
            )
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)
            loader.gpu_free(d_indices)

    def generate_cube(self, size: float) -> MeshBuffer:
        vertex_count = 24
        index_count = 36
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        d_indices, host_indices = self._alloc_uint_buffer(index_count)
        try:
            loader.launch(
                self._kernels["cube"],
                grid=(1, 1, 1),
                block=(64, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_uint64(d_indices.value),
                    ctypes.c_float(float(size)),
                ],
            )
            loader.synchronize()
            return self._copy_back_mesh(
                d_vertices=d_vertices,
                d_normals=d_normals,
                d_uvs=d_uvs,
                d_indices=d_indices,
                vertex_count=vertex_count,
                index_count=index_count,
                host_vertices=host_vertices,
                host_normals=host_normals,
                host_uvs=host_uvs,
                host_indices=host_indices,
                metadata={"primitive": "cube", "backend": "gpu"},
            )
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)
            loader.gpu_free(d_indices)

    def generate_uv_sphere(self, radius: float, stacks: int, slices: int) -> MeshBuffer:
        stacks = max(3, int(stacks))
        slices = max(3, int(slices))
        vertex_count = (stacks + 1) * (slices + 1)
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        d_indices, host_indices, index_count = self._launch_grid_indices(stacks + 1, slices + 1)
        try:
            loader.launch(
                self._kernels["uv_sphere"],
                grid=((vertex_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_float(float(radius)),
                    ctypes.c_int(stacks),
                    ctypes.c_int(slices),
                ],
            )
            loader.synchronize()
            return self._copy_back_mesh(
                d_vertices=d_vertices,
                d_normals=d_normals,
                d_uvs=d_uvs,
                d_indices=d_indices,
                vertex_count=vertex_count,
                index_count=index_count,
                host_vertices=host_vertices,
                host_normals=host_normals,
                host_uvs=host_uvs,
                host_indices=host_indices,
                metadata={"primitive": "uv_sphere", "backend": "gpu"},
            )
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)
            loader.gpu_free(d_indices)

    def generate_cylinder(self, radius: float, height: float, segments: int, caps: int = 1) -> MeshBuffer:
        segments = max(3, int(segments))
        side_cols = segments + 1
        side_vertices = side_cols * 2
        vertex_count = side_vertices + (segments * 2) + 2
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        d_side_indices, host_side_indices, side_index_count = self._launch_grid_indices(2, side_cols)
        try:
            loader.launch(
                self._kernels["cylinder"],
                grid=((vertex_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_float(float(radius)),
                    ctypes.c_float(float(height)),
                    ctypes.c_int(segments),
                ],
            )
            loader.synchronize()
            mesh = self._copy_back_mesh(
                d_vertices=d_vertices,
                d_normals=d_normals,
                d_uvs=d_uvs,
                d_indices=d_side_indices,
                vertex_count=vertex_count,
                index_count=side_index_count,
                host_vertices=host_vertices,
                host_normals=host_normals,
                host_uvs=host_uvs,
                host_indices=host_side_indices,
                metadata={"primitive": "cylinder", "backend": "gpu"},
            )
            if int(caps):
                bottom_ring = side_vertices
                bottom_center = bottom_ring + segments
                top_ring = bottom_center + 1
                top_center = top_ring + segments
                for seg in range(segments):
                    nxt = (seg + 1) % segments
                    mesh.triangles.append((bottom_center, bottom_ring + nxt, bottom_ring + seg))
                    mesh.triangles.append((top_center, top_ring + seg, top_ring + nxt))
            mesh.compute_missing_normals()
            return mesh
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)
            loader.gpu_free(d_side_indices)

    def generate_cone(self, radius: float, height: float, segments: int) -> MeshBuffer:
        segments = max(3, int(segments))
        side_cols = segments + 1
        vertex_count = side_cols + 1 + segments + 1
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        try:
            loader.launch(
                self._kernels["cone"],
                grid=((vertex_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_float(float(radius)),
                    ctypes.c_float(float(height)),
                    ctypes.c_int(segments),
                ],
            )
            loader.synchronize()
            apex = side_cols
            base_ring = side_cols + 1
            center = base_ring + segments
            triangles: list[tuple[int, int, int]] = []
            for seg in range(segments):
                triangles.append((apex, seg, seg + 1))
                nxt = (seg + 1) % segments
                triangles.append((center, base_ring + nxt, base_ring + seg))
            loader.memcpy_dtoh(ctypes.cast(host_vertices, ctypes.c_void_p), d_vertices, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_dtoh(ctypes.cast(host_normals, ctypes.c_void_p), d_normals, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_dtoh(ctypes.cast(host_uvs, ctypes.c_void_p), d_uvs, vertex_count * 2 * ctypes.sizeof(ctypes.c_float))
            return self._mesh_from_buffers(
                vertex_count=vertex_count,
                vertices=host_vertices,
                normals=host_normals,
                uvs=host_uvs,
                triangles=triangles,
                metadata={"primitive": "cone", "backend": "gpu"},
            )
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)

    def generate_torus(self, major_radius: float, minor_radius: float, major_segments: int, minor_segments: int) -> MeshBuffer:
        major_segments = max(3, int(major_segments))
        minor_segments = max(3, int(minor_segments))
        vertex_count = (major_segments + 1) * (minor_segments + 1)
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        d_indices, host_indices, index_count = self._launch_grid_indices(major_segments + 1, minor_segments + 1)
        try:
            loader.launch(
                self._kernels["torus"],
                grid=((vertex_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_float(float(major_radius)),
                    ctypes.c_float(float(minor_radius)),
                    ctypes.c_int(major_segments),
                    ctypes.c_int(minor_segments),
                ],
            )
            loader.synchronize()
            return self._copy_back_mesh(
                d_vertices=d_vertices,
                d_normals=d_normals,
                d_uvs=d_uvs,
                d_indices=d_indices,
                vertex_count=vertex_count,
                index_count=index_count,
                host_vertices=host_vertices,
                host_normals=host_normals,
                host_uvs=host_uvs,
                host_indices=host_indices,
                metadata={"primitive": "torus", "backend": "gpu"},
            )
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)
            loader.gpu_free(d_indices)

    def generate_icosphere(self, radius: float, subdivisions: int) -> MeshBuffer:
        if int(subdivisions) > 0:
            mesh = generate_icosphere_cpu(radius, subdivisions)
            mesh.metadata["backend"] = "cpu_fallback"
            return mesh

        vertex_count = 12
        d_vertices, host_vertices = self._alloc_float_buffer(vertex_count * 3)
        d_normals, host_normals = self._alloc_float_buffer(vertex_count * 3)
        d_uvs, host_uvs = self._alloc_float_buffer(vertex_count * 2)
        try:
            loader.launch(
                self._kernels["icosphere"],
                grid=(1, 1, 1),
                block=(32, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value),
                    ctypes.c_uint64(d_uvs.value),
                    ctypes.c_float(float(radius)),
                    ctypes.c_int(int(subdivisions)),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(host_vertices, ctypes.c_void_p), d_vertices, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_dtoh(ctypes.cast(host_normals, ctypes.c_void_p), d_normals, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_dtoh(ctypes.cast(host_uvs, ctypes.c_void_p), d_uvs, vertex_count * 2 * ctypes.sizeof(ctypes.c_float))
            return self._mesh_from_buffers(
                vertex_count=vertex_count,
                vertices=host_vertices,
                normals=host_normals,
                uvs=host_uvs,
                triangles=list(_ICOSPHERE_TRIANGLES),
                metadata={"primitive": "icosphere", "backend": "gpu"},
            )
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_normals)
            loader.gpu_free(d_uvs)

    def transform_mesh(self, mesh: MeshBuffer, matrix: Matrix4) -> MeshBuffer:
        vertex_count = len(mesh.vertices)
        if vertex_count == 0:
            return mesh.clone()
        host_vertices = (ctypes.c_float * (vertex_count * 3))(
            *[float(component) for vertex in mesh.vertices for component in vertex]
        )
        if mesh.normals:
            host_normals = (ctypes.c_float * (vertex_count * 3))(
                *[float(component) for normal in mesh.normals for component in normal]
            )
            normals_count = vertex_count * 3
        else:
            host_normals = None
            normals_count = 0
        matrix_host = (ctypes.c_float * 16)(*[float(value) for value in matrix])
        d_vertices = loader.gpu_malloc(vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
        d_matrix = loader.gpu_malloc(16 * ctypes.sizeof(ctypes.c_float))
        d_normals = loader.CUdeviceptr(0)
        try:
            loader.memcpy_htod(d_vertices, ctypes.cast(host_vertices, ctypes.c_void_p), vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
            loader.memcpy_htod(d_matrix, ctypes.cast(matrix_host, ctypes.c_void_p), 16 * ctypes.sizeof(ctypes.c_float))
            if host_normals is not None:
                d_normals = loader.gpu_malloc(normals_count * ctypes.sizeof(ctypes.c_float))
                loader.memcpy_htod(d_normals, ctypes.cast(host_normals, ctypes.c_void_p), normals_count * ctypes.sizeof(ctypes.c_float))
            loader.launch(
                self._kernels["transform"],
                grid=((vertex_count + 255) // 256, 1, 1),
                block=(256, 1, 1),
                params=[
                    ctypes.c_uint64(d_vertices.value),
                    ctypes.c_uint64(d_normals.value if host_normals is not None else 0),
                    ctypes.c_uint64(d_matrix.value),
                    ctypes.c_int(vertex_count),
                ],
            )
            loader.synchronize()
            loader.memcpy_dtoh(ctypes.cast(host_vertices, ctypes.c_void_p), d_vertices, vertex_count * 3 * ctypes.sizeof(ctypes.c_float))
            if host_normals is not None:
                loader.memcpy_dtoh(ctypes.cast(host_normals, ctypes.c_void_p), d_normals, normals_count * ctypes.sizeof(ctypes.c_float))
            transformed = mesh.clone()
            transformed.vertices = [
                (
                    float(host_vertices[index * 3 + 0]),
                    float(host_vertices[index * 3 + 1]),
                    float(host_vertices[index * 3 + 2]),
                )
                for index in range(vertex_count)
            ]
            if host_normals is not None:
                transformed.normals = [
                    (
                        float(host_normals[index * 3 + 0]),
                        float(host_normals[index * 3 + 1]),
                        float(host_normals[index * 3 + 2]),
                    )
                    for index in range(vertex_count)
                ]
            transformed.metadata["backend"] = "gpu"
            return transformed
        finally:
            loader.gpu_free(d_vertices)
            loader.gpu_free(d_matrix)
            if int(d_normals.value) != 0:
                loader.gpu_free(d_normals)


__all__ = ["SovereignMeshBridge"]
