"""Host-side RPN mesh executor for Phase H1 construction primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mesh_opcodes import (
    MESH_TOKEN_TO_OPCODE,
    MESH_TOKENS,
    MeshBuffer,
    Path2D,
    csg_intersect,
    csg_subtract,
    csg_union,
    extrude_path,
    generate_cone,
    generate_cube,
    generate_cylinder,
    generate_icosphere,
    generate_plane,
    generate_torus,
    generate_uv_sphere,
    lathe_path,
    mat4_identity,
    mat4_mul,
    mat4_rotate_x,
    mat4_rotate_y,
    mat4_rotate_z,
    mat4_scale,
    mat4_translate,
)


@dataclass
class MeshExecutionResult:
    mesh: MeshBuffer
    opcodes: list[int]
    tokens: list[str]


class MeshRPNEngine:
    """Evaluate H1 construction programs without touching the GPU path."""

    TOKENS = MESH_TOKENS

    def tokenize_rpn(self, expression: str) -> list[str]:
        return [token for token in str(expression).replace("×", "*").replace("÷", "/").split() if token]

    def compile_tokens(self, tokens: list[str]) -> tuple[list[int], list[float]]:
        opcodes: list[int] = []
        scalars: list[float] = []
        for token in tokens:
            upper = token.upper()
            if upper in MESH_TOKEN_TO_OPCODE:
                opcodes.append(MESH_TOKEN_TO_OPCODE[upper])
                continue
            try:
                scalars.append(float(token))
            except ValueError:
                continue
        return opcodes, scalars

    def is_mesh_expression(self, expression: str | list[str]) -> bool:
        tokens = expression if isinstance(expression, list) else self.tokenize_rpn(expression)
        return any(token.upper() in self.TOKENS for token in tokens)

    def evaluate(self, expression: str) -> MeshBuffer:
        return self.evaluate_with_trace(expression).mesh

    def evaluate_with_trace(self, expression: str) -> MeshExecutionResult:
        tokens = self.tokenize_rpn(expression)
        opcodes, _ = self.compile_tokens(tokens)
        stack: list[Any] = []
        active_meshes: list[MeshBuffer] = []
        path = Path2D()

        def current_mesh() -> MeshBuffer | None:
            return active_meshes[-1] if active_meshes else None

        def pop_float(default: float = 0.0) -> float:
            if not stack:
                return float(default)
            return float(stack.pop())

        def pop_mesh() -> MeshBuffer:
            if stack and isinstance(stack[-1], MeshBuffer):
                return stack.pop()
            mesh = current_mesh()
            if mesh is None:
                raise ValueError("Mesh operation requires a mesh on the stack or an active mesh buffer")
            return mesh

        def push_mesh(mesh: MeshBuffer) -> None:
            if active_meshes:
                active_meshes[-1] = mesh
            else:
                stack.append(mesh)

        for token in tokens:
            upper = token.upper()
            if upper not in self.TOKENS:
                try:
                    stack.append(float(token))
                    continue
                except ValueError:
                    pass
                if upper == "MOVE":
                    y = pop_float()
                    x = pop_float()
                    path.move_to((x, y))
                    continue
                if upper == "LINE":
                    y = pop_float()
                    x = pop_float()
                    path.line_to((x, y))
                    continue
                if upper == "QUAD":
                    y = pop_float()
                    x = pop_float()
                    cy = pop_float()
                    cx = pop_float()
                    path.quad_to((cx, cy), (x, y))
                    continue
                if upper == "CUBIC":
                    y = pop_float()
                    x = pop_float()
                    c2y = pop_float()
                    c2x = pop_float()
                    c1y = pop_float()
                    c1x = pop_float()
                    path.cubic_to((c1x, c1y), (c2x, c2y), (x, y))
                    continue
                if upper == "ARC":
                    cy = pop_float()
                    cx = pop_float()
                    end_angle = pop_float()
                    start_angle = pop_float()
                    radius = pop_float()
                    path.arc_to((cx, cy), radius, start_angle, end_angle)
                    continue
                if upper == "CLOSE":
                    path.close()
                    continue
                raise ValueError(f"Unknown mesh token: {token}")

            if upper == "MESH_BEGIN":
                active_meshes.append(MeshBuffer())
            elif upper == "MESH_END":
                if not active_meshes:
                    raise ValueError("MESH_END without MESH_BEGIN")
                mesh = active_meshes.pop()
                mesh.compute_missing_normals()
                stack.append(mesh)
            elif upper == "VERTEX3":
                mesh = current_mesh()
                if mesh is None:
                    raise ValueError("VERTEX3 requires an active MESH_BEGIN buffer")
                z = pop_float()
                y = pop_float()
                x = pop_float()
                mesh.append_vertex((x, y, z))
            elif upper == "NORMAL3":
                mesh = current_mesh()
                if mesh is None:
                    raise ValueError("NORMAL3 requires an active MESH_BEGIN buffer")
                z = pop_float()
                y = pop_float()
                x = pop_float()
                mesh.normals.append((x, y, z))
            elif upper == "UV2":
                mesh = current_mesh()
                if mesh is None:
                    raise ValueError("UV2 requires an active MESH_BEGIN buffer")
                v = pop_float()
                u = pop_float()
                mesh.uvs.append((u, v))
            elif upper == "TRI_FACE":
                mesh = current_mesh()
                if mesh is None:
                    raise ValueError("TRI_FACE requires an active MESH_BEGIN buffer")
                i2 = int(pop_float())
                i1 = int(pop_float())
                i0 = int(pop_float())
                mesh.append_triangle(i0, i1, i2)
            elif upper == "QUAD_FACE":
                mesh = current_mesh()
                if mesh is None:
                    raise ValueError("QUAD_FACE requires an active MESH_BEGIN buffer")
                i3 = int(pop_float())
                i2 = int(pop_float())
                i1 = int(pop_float())
                i0 = int(pop_float())
                mesh.append_quad(i0, i1, i2, i3)
            elif upper == "FACE_NORMAL":
                mesh = current_mesh()
                if mesh is None:
                    raise ValueError("FACE_NORMAL requires an active MESH_BEGIN buffer")
                i2 = int(pop_float())
                i1 = int(pop_float())
                i0 = int(pop_float())
                if max(i0, i1, i2) < len(mesh.vertices):
                    v0 = mesh.vertices[i0]
                    v1 = mesh.vertices[i1]
                    v2 = mesh.vertices[i2]
                    edge_1 = (v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2])
                    edge_2 = (v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2])
                    normal = (
                        edge_1[1] * edge_2[2] - edge_1[2] * edge_2[1],
                        edge_1[2] * edge_2[0] - edge_1[0] * edge_2[2],
                        edge_1[0] * edge_2[1] - edge_1[1] * edge_2[0],
                    )
                    mesh.normals.append(normal)
            elif upper == "MAT4_IDENTITY":
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
                push_mesh(mesh.transformed(matrix))
            elif upper == "GEN_PLANE":
                seg_d = int(pop_float(1.0))
                seg_w = int(pop_float(1.0))
                depth = pop_float(1.0)
                width = pop_float(1.0)
                push_mesh(generate_plane(width, depth, seg_w, seg_d))
            elif upper == "GEN_CUBE":
                push_mesh(generate_cube(pop_float(1.0)))
            elif upper == "GEN_UV_SPHERE":
                slices = int(pop_float(16.0))
                stacks = int(pop_float(12.0))
                radius = pop_float(1.0)
                push_mesh(generate_uv_sphere(radius, stacks, slices))
            elif upper == "GEN_CYLINDER":
                caps = int(pop_float(1.0))
                segments = int(pop_float(16.0))
                height = pop_float(1.0)
                radius = pop_float(0.5)
                push_mesh(generate_cylinder(radius, height, segments, caps))
            elif upper == "GEN_CONE":
                segments = int(pop_float(16.0))
                height = pop_float(1.0)
                radius = pop_float(0.5)
                push_mesh(generate_cone(radius, height, segments))
            elif upper == "GEN_TORUS":
                minor_segments = int(pop_float(8.0))
                major_segments = int(pop_float(16.0))
                minor_radius = pop_float(0.25)
                major_radius = pop_float(1.0)
                push_mesh(generate_torus(major_radius, minor_radius, major_segments, minor_segments))
            elif upper == "GEN_ICOSPHERE":
                subdivisions = int(pop_float(1.0))
                radius = pop_float(1.0)
                push_mesh(generate_icosphere(radius, subdivisions))
            elif upper == "CSG_UNION":
                mesh_b = pop_mesh()
                mesh_a = pop_mesh()
                push_mesh(csg_union(mesh_a, mesh_b))
            elif upper == "CSG_SUBTRACT":
                mesh_b = pop_mesh()
                mesh_a = pop_mesh()
                push_mesh(csg_subtract(mesh_a, mesh_b))
            elif upper == "CSG_INTERSECT":
                mesh_b = pop_mesh()
                mesh_a = pop_mesh()
                push_mesh(csg_intersect(mesh_a, mesh_b))
            elif upper == "EXTRUDE":
                depth = pop_float(1.0)
                push_mesh(extrude_path(path, depth))
            elif upper == "LATHE":
                segments = int(pop_float(16.0))
                push_mesh(lathe_path(path, segments))

        final_mesh = None
        for item in reversed(stack):
            if isinstance(item, MeshBuffer):
                final_mesh = item
                break
        if final_mesh is None and active_meshes:
            final_mesh = active_meshes[-1]
        if final_mesh is None:
            raise ValueError("Mesh program did not produce a mesh")
        final_mesh.compute_missing_normals()
        return MeshExecutionResult(mesh=final_mesh, opcodes=opcodes, tokens=tokens)


__all__ = ["MeshExecutionResult", "MeshRPNEngine"]
