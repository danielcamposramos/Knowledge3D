"""Stdlib-only mesh primitives and H1 host-side 3D construction helpers.

This module defines the collision-safe Phase H1 opcode surface together with a
small procedural mesh runtime. These helpers deliberately stay out of the
reasoning hot path; they are used by host-side construction, ingestion, and
tests until the GPU mesh path is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from typing import Iterable

from .rpn_opcodes import (
    OP_CSG_INTERSECT,
    OP_CSG_SUBTRACT,
    OP_CSG_UNION,
    OP_EXTRUDE,
    OP_FACE_NORMAL,
    OP_GEN_CONE,
    OP_GEN_CUBE,
    OP_GEN_CYLINDER,
    OP_GEN_ICOSPHERE,
    OP_GEN_PLANE,
    OP_GEN_TORUS,
    OP_GEN_UV_SPHERE,
    OP_LATHE,
    OP_MAT4_APPLY,
    OP_MAT4_IDENTITY,
    OP_MAT4_MUL,
    OP_MAT4_ROTATE_X,
    OP_MAT4_ROTATE_Y,
    OP_MAT4_ROTATE_Z,
    OP_MAT4_SCALE,
    OP_MAT4_TRANSLATE,
    OP_MESH_BEGIN,
    OP_MESH_END,
    OP_NORMAL3,
    OP_QUAD_FACE,
    OP_TRI_FACE,
    OP_UV2,
    OP_VERTEX3,
)

Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Matrix4 = tuple[float, ...]


def _f(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _i(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _v3(x: object, y: object, z: object) -> Vector3:
    return (_f(x), _f(y), _f(z))


def _v2(x: object, y: object) -> Vector2:
    return (_f(x), _f(y))


def _sub3(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add3(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _mul3(a: Vector3, scalar: float) -> Vector3:
    return (a[0] * scalar, a[1] * scalar, a[2] * scalar)


def _dot3(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _length3(a: Vector3) -> float:
    return math.sqrt(_dot3(a, a))


def _normalize3(a: Vector3) -> Vector3:
    norm = _length3(a)
    if norm <= 1e-8:
        return (0.0, 0.0, 1.0)
    inv = 1.0 / norm
    return (a[0] * inv, a[1] * inv, a[2] * inv)


def _bounds(vertices: Iterable[Vector3]) -> tuple[Vector3, Vector3]:
    verts = list(vertices)
    if not verts:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    xs = [vertex[0] for vertex in verts]
    ys = [vertex[1] for vertex in verts]
    zs = [vertex[2] for vertex in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def _lerp2(a: Vector2, b: Vector2, t: float) -> Vector2:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


@dataclass
class Path2D:
    """Simple polyline path used by EXTRUDE and LATHE."""

    points: list[Vector2] = field(default_factory=list)
    closed: bool = False

    def move_to(self, point: Vector2) -> None:
        self.points = [point]
        self.closed = False

    def line_to(self, point: Vector2) -> None:
        if not self.points:
            self.points = [point]
            return
        self.points.append(point)

    def quad_to(self, control: Vector2, point: Vector2, segments: int = 12) -> None:
        if not self.points:
            self.points = [point]
            return
        start = self.points[-1]
        for step in range(1, max(segments, 2) + 1):
            t = step / float(max(segments, 2))
            a = _lerp2(start, control, t)
            b = _lerp2(control, point, t)
            self.points.append(_lerp2(a, b, t))

    def cubic_to(
        self,
        control_1: Vector2,
        control_2: Vector2,
        point: Vector2,
        segments: int = 18,
    ) -> None:
        if not self.points:
            self.points = [point]
            return
        start = self.points[-1]
        for step in range(1, max(segments, 2) + 1):
            t = step / float(max(segments, 2))
            mt = 1.0 - t
            x = (
                mt * mt * mt * start[0]
                + 3.0 * mt * mt * t * control_1[0]
                + 3.0 * mt * t * t * control_2[0]
                + t * t * t * point[0]
            )
            y = (
                mt * mt * mt * start[1]
                + 3.0 * mt * mt * t * control_1[1]
                + 3.0 * mt * t * t * control_2[1]
                + t * t * t * point[1]
            )
            self.points.append((x, y))

    def arc_to(
        self,
        center: Vector2,
        radius: float,
        start_angle: float,
        end_angle: float,
        segments: int = 24,
    ) -> None:
        if radius <= 0.0:
            return
        if not self.points:
            self.points = []
        step_count = max(segments, 3)
        for step in range(step_count + 1):
            t = step / float(step_count)
            angle = start_angle + (end_angle - start_angle) * t
            self.points.append(
                (
                    center[0] + math.cos(angle) * radius,
                    center[1] + math.sin(angle) * radius,
                )
            )

    def close(self) -> None:
        if len(self.points) >= 3:
            self.closed = True


@dataclass
class MeshBuffer:
    """Simple mesh buffer matching the H1 vertex/index surface."""

    vertices: list[Vector3] = field(default_factory=list)
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    normals: list[Vector3] = field(default_factory=list)
    uvs: list[Vector2] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)

    def clone(self) -> "MeshBuffer":
        return MeshBuffer(
            vertices=list(self.vertices),
            triangles=list(self.triangles),
            normals=list(self.normals),
            uvs=list(self.uvs),
            metadata=dict(self.metadata),
        )

    def append_vertex(
        self,
        vertex: Vector3,
        *,
        normal: Vector3 | None = None,
        uv: Vector2 | None = None,
    ) -> int:
        index = len(self.vertices)
        self.vertices.append(tuple(vertex))
        if normal is not None:
            while len(self.normals) < index:
                self.normals.append((0.0, 0.0, 1.0))
            self.normals.append(tuple(normal))
        if uv is not None:
            while len(self.uvs) < index:
                self.uvs.append((0.0, 0.0))
            self.uvs.append(tuple(uv))
        return index

    def append_triangle(self, i0: int, i1: int, i2: int) -> None:
        self.triangles.append((int(i0), int(i1), int(i2)))

    def append_quad(self, i0: int, i1: int, i2: int, i3: int) -> None:
        self.triangles.append((int(i0), int(i1), int(i2)))
        self.triangles.append((int(i0), int(i2), int(i3)))

    def append_mesh(self, other: "MeshBuffer") -> None:
        base = len(self.vertices)
        self.vertices.extend(other.vertices)
        self.triangles.extend((a + base, b + base, c + base) for a, b, c in other.triangles)
        self.normals.extend(other.normals)
        self.uvs.extend(other.uvs)

    def transformed(self, matrix: Matrix4) -> "MeshBuffer":
        transformed = self.clone()
        transformed.vertices = [apply_matrix_to_vertex(matrix, vertex) for vertex in self.vertices]
        if self.normals:
            transformed.normals = [_normalize3(apply_matrix_to_direction(matrix, normal)) for normal in self.normals]
        return transformed

    def compute_missing_normals(self) -> None:
        if not self.vertices:
            self.normals = []
            return
        accum = [(0.0, 0.0, 0.0) for _ in self.vertices]
        for i0, i1, i2 in self.triangles:
            if min(i0, i1, i2) < 0 or max(i0, i1, i2) >= len(self.vertices):
                continue
            v0 = self.vertices[i0]
            v1 = self.vertices[i1]
            v2 = self.vertices[i2]
            face = _normalize3(_cross3(_sub3(v1, v0), _sub3(v2, v0)))
            accum[i0] = _add3(accum[i0], face)
            accum[i1] = _add3(accum[i1], face)
            accum[i2] = _add3(accum[i2], face)
        self.normals = [_normalize3(value) for value in accum]

    def position_bounds(self) -> tuple[Vector3, Vector3]:
        if not self.vertices:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [vertex[0] for vertex in self.vertices]
        ys = [vertex[1] for vertex in self.vertices]
        zs = [vertex[2] for vertex in self.vertices]
        return (
            (min(xs), min(ys), min(zs)),
            (max(xs), max(ys), max(zs)),
        )

    def to_gltf_bytes(self) -> tuple[bytes, bytes, bytes, bytes]:
        """Return packed position/normal/uv/index buffers for GLTF export."""
        export_mesh = self.clone()
        if export_mesh.vertices and len(export_mesh.normals) != len(export_mesh.vertices):
            export_mesh.compute_missing_normals()
        if len(export_mesh.uvs) != len(export_mesh.vertices):
            export_mesh.uvs = list(export_mesh.uvs[: len(export_mesh.vertices)])
            while len(export_mesh.uvs) < len(export_mesh.vertices):
                export_mesh.uvs.append((0.0, 0.0))

        position_bytes = b"".join(
            struct.pack("<fff", float(vertex[0]), float(vertex[1]), float(vertex[2]))
            for vertex in export_mesh.vertices
        )
        normal_bytes = b"".join(
            struct.pack("<fff", float(normal[0]), float(normal[1]), float(normal[2]))
            for normal in export_mesh.normals
        )
        uv_bytes = b"".join(
            struct.pack("<ff", float(uv[0]), float(uv[1]))
            for uv in export_mesh.uvs
        )
        index_bytes = b"".join(
            struct.pack("<III", int(face[0]), int(face[1]), int(face[2]))
            for face in export_mesh.triangles
        )
        return position_bytes, normal_bytes, uv_bytes, index_bytes

    def to_dict(self) -> dict[str, object]:
        return {
            "vertices": [list(vertex) for vertex in self.vertices],
            "triangles": [list(face) for face in self.triangles],
            "normals": [list(normal) for normal in self.normals],
            "uvs": [list(uv) for uv in self.uvs],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "MeshBuffer":
        payload = dict(payload or {})
        return cls(
            vertices=[_v3(*(vertex or (0.0, 0.0, 0.0))) for vertex in payload.get("vertices", [])],
            triangles=[
                (_i(face[0]), _i(face[1]), _i(face[2]))
                for face in payload.get("triangles", [])
                if isinstance(face, (list, tuple)) and len(face) >= 3
            ],
            normals=[_v3(*(normal or (0.0, 0.0, 1.0))) for normal in payload.get("normals", [])],
            uvs=[_v2(*(uv or (0.0, 0.0))) for uv in payload.get("uvs", [])],
            metadata=dict(payload.get("metadata", {}) or {}),
        )


def mat4_identity() -> Matrix4:
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_translate(tx: float, ty: float, tz: float) -> Matrix4:
    return (
        1.0, 0.0, 0.0, tx,
        0.0, 1.0, 0.0, ty,
        0.0, 0.0, 1.0, tz,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_scale(sx: float, sy: float, sz: float) -> Matrix4:
    return (
        sx, 0.0, 0.0, 0.0,
        0.0, sy, 0.0, 0.0,
        0.0, 0.0, sz, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_rotate_x(angle_rad: float) -> Matrix4:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, c, -s, 0.0,
        0.0, s, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_rotate_y(angle_rad: float) -> Matrix4:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
        c, 0.0, s, 0.0,
        0.0, 1.0, 0.0, 0.0,
        -s, 0.0, c, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_rotate_z(angle_rad: float) -> Matrix4:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return (
        c, -s, 0.0, 0.0,
        s, c, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def mat4_mul(a: Matrix4, b: Matrix4) -> Matrix4:
    out = [0.0] * 16
    for row in range(4):
        for col in range(4):
            out[row * 4 + col] = sum(a[row * 4 + inner] * b[inner * 4 + col] for inner in range(4))
    return tuple(out)


def apply_matrix_to_vertex(matrix: Matrix4, vertex: Vector3) -> Vector3:
    x, y, z = vertex
    w = 1.0
    xp = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3] * w
    yp = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7] * w
    zp = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11] * w
    wp = matrix[12] * x + matrix[13] * y + matrix[14] * z + matrix[15] * w
    if abs(wp) > 1e-8 and abs(wp - 1.0) > 1e-8:
        inv = 1.0 / wp
        return (xp * inv, yp * inv, zp * inv)
    return (xp, yp, zp)


def apply_matrix_to_direction(matrix: Matrix4, direction: Vector3) -> Vector3:
    x, y, z = direction
    return (
        matrix[0] * x + matrix[1] * y + matrix[2] * z,
        matrix[4] * x + matrix[5] * y + matrix[6] * z,
        matrix[8] * x + matrix[9] * y + matrix[10] * z,
    )


def generate_plane(width: float, depth: float, segments_w: int, segments_d: int) -> MeshBuffer:
    mesh = MeshBuffer(metadata={"primitive": "plane"})
    seg_w = max(1, int(segments_w))
    seg_d = max(1, int(segments_d))
    half_w = float(width) / 2.0
    half_d = float(depth) / 2.0
    for dz in range(seg_d + 1):
        vz = -half_d + depth * (dz / float(seg_d))
        for dx in range(seg_w + 1):
            vx = -half_w + width * (dx / float(seg_w))
            mesh.append_vertex((vx, 0.0, vz), normal=(0.0, 1.0, 0.0), uv=(dx / float(seg_w), dz / float(seg_d)))
    row = seg_w + 1
    for dz in range(seg_d):
        for dx in range(seg_w):
            i0 = dz * row + dx
            i1 = i0 + 1
            i2 = i0 + row + 1
            i3 = i0 + row
            mesh.append_quad(i0, i1, i2, i3)
    mesh.compute_missing_normals()
    return mesh


def generate_cube(size: float) -> MeshBuffer:
    half = float(size) / 2.0
    corners = [
        (-half, -half, -half),
        (half, -half, -half),
        (half, half, -half),
        (-half, half, -half),
        (-half, -half, half),
        (half, -half, half),
        (half, half, half),
        (-half, half, half),
    ]
    faces = [
        ((0, 1, 2, 3), (0.0, 0.0, -1.0)),
        ((4, 5, 6, 7), (0.0, 0.0, 1.0)),
        ((0, 4, 7, 3), (-1.0, 0.0, 0.0)),
        ((1, 5, 6, 2), (1.0, 0.0, 0.0)),
        ((3, 2, 6, 7), (0.0, 1.0, 0.0)),
        ((0, 1, 5, 4), (0.0, -1.0, 0.0)),
    ]
    mesh = MeshBuffer(metadata={"primitive": "cube"})
    for indices, normal in faces:
        start = len(mesh.vertices)
        uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for offset, corner_index in enumerate(indices):
            mesh.append_vertex(corners[corner_index], normal=normal, uv=uvs[offset])
        mesh.append_quad(start, start + 1, start + 2, start + 3)
    mesh.compute_missing_normals()
    return mesh


def generate_uv_sphere(radius: float, stacks: int, slices: int) -> MeshBuffer:
    mesh = MeshBuffer(metadata={"primitive": "uv_sphere"})
    stack_count = max(3, int(stacks))
    slice_count = max(3, int(slices))
    for stack in range(stack_count + 1):
        phi = math.pi * stack / float(stack_count)
        y = math.cos(phi)
        ring_radius = math.sin(phi)
        for slc in range(slice_count + 1):
            theta = 2.0 * math.pi * slc / float(slice_count)
            x = math.cos(theta) * ring_radius
            z = math.sin(theta) * ring_radius
            normal = _normalize3((x, y, z))
            mesh.append_vertex(
                _mul3(normal, radius),
                normal=normal,
                uv=(slc / float(slice_count), stack / float(stack_count)),
            )
    row = slice_count + 1
    for stack in range(stack_count):
        for slc in range(slice_count):
            i0 = stack * row + slc
            i1 = i0 + 1
            i2 = i0 + row + 1
            i3 = i0 + row
            mesh.append_quad(i0, i1, i2, i3)
    mesh.compute_missing_normals()
    return mesh


def generate_cylinder(radius: float, height: float, segments: int, caps: int = 1) -> MeshBuffer:
    mesh = MeshBuffer(metadata={"primitive": "cylinder"})
    segment_count = max(3, int(segments))
    half_h = float(height) / 2.0
    top_ring: list[int] = []
    bottom_ring: list[int] = []
    for seg in range(segment_count):
        angle = 2.0 * math.pi * seg / float(segment_count)
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        normal = _normalize3((x, 0.0, z))
        bottom_ring.append(mesh.append_vertex((x, -half_h, z), normal=normal, uv=(seg / float(segment_count), 0.0)))
        top_ring.append(mesh.append_vertex((x, half_h, z), normal=normal, uv=(seg / float(segment_count), 1.0)))
    for seg in range(segment_count):
        nxt = (seg + 1) % segment_count
        mesh.append_quad(bottom_ring[seg], bottom_ring[nxt], top_ring[nxt], top_ring[seg])
    if int(caps):
        bottom_center = mesh.append_vertex((0.0, -half_h, 0.0), normal=(0.0, -1.0, 0.0), uv=(0.5, 0.5))
        top_center = mesh.append_vertex((0.0, half_h, 0.0), normal=(0.0, 1.0, 0.0), uv=(0.5, 0.5))
        for seg in range(segment_count):
            nxt = (seg + 1) % segment_count
            mesh.append_triangle(bottom_center, bottom_ring[nxt], bottom_ring[seg])
            mesh.append_triangle(top_center, top_ring[seg], top_ring[nxt])
    mesh.compute_missing_normals()
    return mesh


def generate_cone(radius: float, height: float, segments: int) -> MeshBuffer:
    mesh = MeshBuffer(metadata={"primitive": "cone"})
    segment_count = max(3, int(segments))
    half_h = float(height) / 2.0
    apex = mesh.append_vertex((0.0, half_h, 0.0), normal=(0.0, 1.0, 0.0), uv=(0.5, 1.0))
    ring: list[int] = []
    for seg in range(segment_count):
        angle = 2.0 * math.pi * seg / float(segment_count)
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        ring.append(mesh.append_vertex((x, -half_h, z), uv=(seg / float(segment_count), 0.0)))
    center = mesh.append_vertex((0.0, -half_h, 0.0), normal=(0.0, -1.0, 0.0), uv=(0.5, 0.5))
    for seg in range(segment_count):
        nxt = (seg + 1) % segment_count
        mesh.append_triangle(apex, ring[seg], ring[nxt])
        mesh.append_triangle(center, ring[nxt], ring[seg])
    mesh.compute_missing_normals()
    return mesh


def generate_torus(major_radius: float, minor_radius: float, major_segments: int, minor_segments: int) -> MeshBuffer:
    mesh = MeshBuffer(metadata={"primitive": "torus"})
    major_count = max(3, int(major_segments))
    minor_count = max(3, int(minor_segments))
    for major in range(major_count):
        theta = 2.0 * math.pi * major / float(major_count)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for minor in range(minor_count):
            phi = 2.0 * math.pi * minor / float(minor_count)
            cos_p = math.cos(phi)
            sin_p = math.sin(phi)
            x = (major_radius + minor_radius * cos_p) * cos_t
            y = minor_radius * sin_p
            z = (major_radius + minor_radius * cos_p) * sin_t
            nx = cos_p * cos_t
            ny = sin_p
            nz = cos_p * sin_t
            mesh.append_vertex((x, y, z), normal=_normalize3((nx, ny, nz)), uv=(major / float(major_count), minor / float(minor_count)))
    for major in range(major_count):
        for minor in range(minor_count):
            i0 = major * minor_count + minor
            i1 = major * minor_count + (minor + 1) % minor_count
            i2 = ((major + 1) % major_count) * minor_count + (minor + 1) % minor_count
            i3 = ((major + 1) % major_count) * minor_count + minor
            mesh.append_quad(i0, i1, i2, i3)
    mesh.compute_missing_normals()
    return mesh


def generate_icosphere(radius: float, subdivisions: int) -> MeshBuffer:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = [
        (-1.0, t, 0.0), (1.0, t, 0.0), (-1.0, -t, 0.0), (1.0, -t, 0.0),
        (0.0, -1.0, t), (0.0, 1.0, t), (0.0, -1.0, -t), (0.0, 1.0, -t),
        (t, 0.0, -1.0), (t, 0.0, 1.0), (-t, 0.0, -1.0), (-t, 0.0, 1.0),
    ]
    vertices = [_normalize3(vertex) for vertex in base]
    triangles = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(i0: int, i1: int) -> int:
        key = (min(i0, i1), max(i0, i1))
        if key in midpoint_cache:
            return midpoint_cache[key]
        vert = _normalize3(_mul3(_add3(vertices[i0], vertices[i1]), 0.5))
        vertices.append(vert)
        index = len(vertices) - 1
        midpoint_cache[key] = index
        return index

    for _ in range(max(0, int(subdivisions))):
        refined: list[tuple[int, int, int]] = []
        for i0, i1, i2 in triangles:
            a = midpoint(i0, i1)
            b = midpoint(i1, i2)
            c = midpoint(i2, i0)
            refined.extend(
                [
                    (i0, a, c),
                    (i1, b, a),
                    (i2, c, b),
                    (a, b, c),
                ]
            )
        triangles = refined
    mesh = MeshBuffer(metadata={"primitive": "icosphere"})
    for vertex in vertices:
        normal = _normalize3(vertex)
        u = 0.5 + math.atan2(normal[2], normal[0]) / (2.0 * math.pi)
        v = 0.5 - math.asin(max(-1.0, min(1.0, normal[1]))) / math.pi
        mesh.append_vertex(_mul3(normal, radius), normal=normal, uv=(u, v))
    mesh.triangles.extend(triangles)
    mesh.compute_missing_normals()
    return mesh


def extrude_path(path: Path2D, depth: float) -> MeshBuffer:
    if len(path.points) < 3:
        return MeshBuffer(metadata={"primitive": "extrude", "empty": True})
    points = list(path.points)
    if path.closed and points[0] != points[-1]:
        points.append(points[0])
    loop = points[:-1] if path.closed and len(points) > 1 else points
    mesh = MeshBuffer(metadata={"primitive": "extrude"})
    half_depth = float(depth) / 2.0
    front = [mesh.append_vertex((x, y, half_depth), uv=(x, y)) for x, y in loop]
    back = [mesh.append_vertex((x, y, -half_depth), uv=(x, y)) for x, y in loop]
    for idx in range(1, len(loop) - 1):
        mesh.append_triangle(front[0], front[idx], front[idx + 1])
        mesh.append_triangle(back[0], back[idx + 1], back[idx])
    edge_count = len(loop)
    for idx in range(edge_count):
        nxt = (idx + 1) % edge_count
        mesh.append_quad(front[idx], front[nxt], back[nxt], back[idx])
    mesh.compute_missing_normals()
    return mesh


def lathe_path(path: Path2D, segments: int) -> MeshBuffer:
    if len(path.points) < 2:
        return MeshBuffer(metadata={"primitive": "lathe", "empty": True})
    mesh = MeshBuffer(metadata={"primitive": "lathe"})
    profile = list(path.points)
    segment_count = max(3, int(segments))
    for seg in range(segment_count):
        theta = 2.0 * math.pi * seg / float(segment_count)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        for radius, height in profile:
            x = radius * cos_t
            z = radius * sin_t
            normal = _normalize3((cos_t, 0.0, sin_t))
            mesh.append_vertex((x, height, z), normal=normal, uv=(seg / float(segment_count), height))
    ring = len(profile)
    for seg in range(segment_count):
        next_seg = (seg + 1) % segment_count
        for row in range(ring - 1):
            i0 = seg * ring + row
            i1 = seg * ring + row + 1
            i2 = next_seg * ring + row + 1
            i3 = next_seg * ring + row
            mesh.append_quad(i0, i1, i2, i3)
    mesh.compute_missing_normals()
    return mesh


def _box_mesh(min_corner: Vector3, max_corner: Vector3) -> MeshBuffer:
    size = max(
        max_corner[0] - min_corner[0],
        max_corner[1] - min_corner[1],
        max_corner[2] - min_corner[2],
    )
    cube = generate_cube(size)
    cube_bounds_min, cube_bounds_max = _bounds(cube.vertices)
    current_center = (
        (cube_bounds_min[0] + cube_bounds_max[0]) / 2.0,
        (cube_bounds_min[1] + cube_bounds_max[1]) / 2.0,
        (cube_bounds_min[2] + cube_bounds_max[2]) / 2.0,
    )
    target_center = (
        (min_corner[0] + max_corner[0]) / 2.0,
        (min_corner[1] + max_corner[1]) / 2.0,
        (min_corner[2] + max_corner[2]) / 2.0,
    )
    scale = (
        (max_corner[0] - min_corner[0]) / max(size, 1e-8),
        (max_corner[1] - min_corner[1]) / max(size, 1e-8),
        (max_corner[2] - min_corner[2]) / max(size, 1e-8),
    )
    transform = mat4_mul(
        mat4_translate(
            target_center[0] - current_center[0],
            target_center[1] - current_center[1],
            target_center[2] - current_center[2],
        ),
        mat4_scale(scale[0], scale[1], scale[2]),
    )
    return cube.transformed(transform)


def csg_union(mesh_a: MeshBuffer, mesh_b: MeshBuffer) -> MeshBuffer:
    merged = mesh_a.clone()
    merged.append_mesh(mesh_b)
    merged.metadata["csg"] = "union"
    merged.compute_missing_normals()
    return merged


def csg_intersect(mesh_a: MeshBuffer, mesh_b: MeshBuffer) -> MeshBuffer:
    a_min, a_max = _bounds(mesh_a.vertices)
    b_min, b_max = _bounds(mesh_b.vertices)
    min_corner = (max(a_min[0], b_min[0]), max(a_min[1], b_min[1]), max(a_min[2], b_min[2]))
    max_corner = (min(a_max[0], b_max[0]), min(a_max[1], b_max[1]), min(a_max[2], b_max[2]))
    if min_corner[0] >= max_corner[0] or min_corner[1] >= max_corner[1] or min_corner[2] >= max_corner[2]:
        return MeshBuffer(metadata={"csg": "intersect", "empty": True})
    result = _box_mesh(min_corner, max_corner)
    result.metadata["csg"] = "intersect"
    return result


def csg_subtract(mesh_a: MeshBuffer, mesh_b: MeshBuffer) -> MeshBuffer:
    outer_min, outer_max = _bounds(mesh_a.vertices)
    inner_min, inner_max = _bounds(mesh_b.vertices)
    overlap_min = (
        max(outer_min[0], inner_min[0]),
        max(outer_min[1], inner_min[1]),
        max(outer_min[2], inner_min[2]),
    )
    overlap_max = (
        min(outer_max[0], inner_max[0]),
        min(outer_max[1], inner_max[1]),
        min(outer_max[2], inner_max[2]),
    )
    if overlap_min[0] >= overlap_max[0] or overlap_min[1] >= overlap_max[1] or overlap_min[2] >= overlap_max[2]:
        result = mesh_a.clone()
        result.metadata["csg"] = "subtract_no_overlap"
        return result
    slabs: list[MeshBuffer] = []
    x0, y0, z0 = outer_min
    x1, y1, z1 = outer_max
    ix0, iy0, iz0 = overlap_min
    ix1, iy1, iz1 = overlap_max
    boxes = [
        ((x0, y0, z0), (ix0, y1, z1)),
        ((ix1, y0, z0), (x1, y1, z1)),
        ((ix0, y0, z0), (ix1, iy0, z1)),
        ((ix0, iy1, z0), (ix1, y1, z1)),
        ((ix0, iy0, z0), (ix1, iy1, iz0)),
        ((ix0, iy0, iz1), (ix1, iy1, z1)),
    ]
    for min_corner, max_corner in boxes:
        if min_corner[0] < max_corner[0] and min_corner[1] < max_corner[1] and min_corner[2] < max_corner[2]:
            slabs.append(_box_mesh(min_corner, max_corner))
    if not slabs:
        return MeshBuffer(metadata={"csg": "subtract_empty", "empty": True})
    result = slabs[0]
    for slab in slabs[1:]:
        result = csg_union(result, slab)
    result.metadata["csg"] = "subtract"
    return result


MESH_TOKEN_TO_OPCODE = {
    "VERTEX3": OP_VERTEX3,
    "NORMAL3": OP_NORMAL3,
    "UV2": OP_UV2,
    "TRI_FACE": OP_TRI_FACE,
    "QUAD_FACE": OP_QUAD_FACE,
    "FACE_NORMAL": OP_FACE_NORMAL,
    "MESH_BEGIN": OP_MESH_BEGIN,
    "MESH_END": OP_MESH_END,
    "MAT4_IDENTITY": OP_MAT4_IDENTITY,
    "MAT4_TRANSLATE": OP_MAT4_TRANSLATE,
    "MAT4_SCALE": OP_MAT4_SCALE,
    "MAT4_ROTATE_X": OP_MAT4_ROTATE_X,
    "MAT4_ROTATE_Y": OP_MAT4_ROTATE_Y,
    "MAT4_ROTATE_Z": OP_MAT4_ROTATE_Z,
    "MAT4_MUL": OP_MAT4_MUL,
    "MAT4_APPLY": OP_MAT4_APPLY,
    "GEN_PLANE": OP_GEN_PLANE,
    "GEN_CUBE": OP_GEN_CUBE,
    "GEN_UV_SPHERE": OP_GEN_UV_SPHERE,
    "GEN_CYLINDER": OP_GEN_CYLINDER,
    "GEN_CONE": OP_GEN_CONE,
    "GEN_TORUS": OP_GEN_TORUS,
    "GEN_ICOSPHERE": OP_GEN_ICOSPHERE,
    "CSG_UNION": OP_CSG_UNION,
    "CSG_SUBTRACT": OP_CSG_SUBTRACT,
    "CSG_INTERSECT": OP_CSG_INTERSECT,
    "EXTRUDE": OP_EXTRUDE,
    "LATHE": OP_LATHE,
}

MESH_OPCODE_TO_TOKEN = {opcode: token for token, opcode in MESH_TOKEN_TO_OPCODE.items()}
MESH_TOKENS = frozenset(MESH_TOKEN_TO_OPCODE.keys())


__all__ = [
    "Matrix4",
    "MeshBuffer",
    "MESH_OPCODE_TO_TOKEN",
    "MESH_TOKEN_TO_OPCODE",
    "MESH_TOKENS",
    "Path2D",
    "Vector2",
    "Vector3",
    "apply_matrix_to_direction",
    "apply_matrix_to_vertex",
    "csg_intersect",
    "csg_subtract",
    "csg_union",
    "extrude_path",
    "generate_cone",
    "generate_cube",
    "generate_cylinder",
    "generate_icosphere",
    "generate_plane",
    "generate_torus",
    "generate_uv_sphere",
    "lathe_path",
    "mat4_identity",
    "mat4_mul",
    "mat4_rotate_x",
    "mat4_rotate_y",
    "mat4_rotate_z",
    "mat4_scale",
    "mat4_translate",
]
