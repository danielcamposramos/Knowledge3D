"""Deterministic bridge from contour-bearing drawings to mesh-ready geometry.

The bridge stays honest about the current substrate:
- contour/profile extraction is PTX-backed through GeometryPrep
- topology assembly is deterministic host orchestration until a dedicated mesh
  generation kernel exists
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass

import numpy as np

from knowledge3d.cranium.codecs.ternary_codec_ops import TernaryCodecOps
from knowledge3d.cranium.ptx_runtime.geometry_prep import GeometryPrep, PreparedProfile
from knowledge3d.cranium.ptx_runtime.math_core_pool import get_global_math_core_pool


@dataclass(frozen=True)
class MeshPlan:
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    profile: PreparedProfile
    metadata: dict[str, object]


def _resolve_math_core_plan(preferred_tier: int, work_items: int) -> dict[str, object]:
    pool = get_global_math_core_pool()
    snapshot = pool.snapshot()
    max_cores = max(1, int(snapshot.get("max_cores", 1)))
    active = max(0, int(snapshot.get("active", 0)))
    available = max(1, max_cores - min(active, max_cores - 1))
    work = max(1, int(work_items))
    tier = int(preferred_tier)
    if tier <= 1:
        fanout = min(work, max(1, available // 4))
        cascade = ["parallel_fanout", "local_reduce"]
    elif tier == 2:
        fanout = min(work, max(1, available // 8))
        cascade = ["parallel_fanout", "worker_reduce"]
    else:
        fanout = min(work, max(1, available // 16))
        cascade = ["parallel_fanout", "worker_reduce", "master_commit"]
    batch_size = max(1, (work + fanout - 1) // fanout)
    return {
        "preferred_tier": tier,
        "tier_role": pool.describe_tier(tier),
        "work_items": work,
        "fanout": int(fanout),
        "batch_size": int(batch_size),
        "cascade": cascade,
        "pool_snapshot": snapshot,
    }


def _compute_vertex_normals(
    vertices: np.ndarray,
    indices: np.ndarray,
    *,
    execution_plan: dict[str, object] | None = None,
) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float32)
    tris = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
    verts = np.asarray(vertices, dtype=np.float32)
    batch_size = int((execution_plan or {}).get("batch_size", max(1, tris.shape[0])))
    for start in range(0, tris.shape[0], batch_size):
        tri_batch = tris[start:start + batch_size]
        v0 = verts[tri_batch[:, 0]]
        v1 = verts[tri_batch[:, 1]]
        v2 = verts[tri_batch[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0).astype(np.float32, copy=False)
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True).astype(np.float32, copy=False)
        face_normals = np.divide(
            face_normals,
            lengths,
            out=np.zeros_like(face_normals, dtype=np.float32),
            where=lengths > 1e-8,
        )
        np.add.at(normals, tri_batch[:, 0], face_normals)
        np.add.at(normals, tri_batch[:, 1], face_normals)
        np.add.at(normals, tri_batch[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-8] = 1.0
    return (normals / lengths).astype(np.float32, copy=False)


def _safe_normalize_2d(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length <= 1e-8:
        return fallback.astype(np.float32, copy=False)
    return (vector / length).astype(np.float32, copy=False)


class ProceduralGeometryBridge:
    """Always-on 2D -> 3D bridge built on PTX geometry preparation."""

    _WARMED_PID: int | None = None
    _WARMUP_REPORT: dict[str, float | int | str] | None = None
    _TERNARY_OPS: TernaryCodecOps | None = None

    def __init__(self) -> None:
        self.geometry_prep = GeometryPrep()

    @classmethod
    def _ternary_ops(cls) -> TernaryCodecOps:
        if cls._TERNARY_OPS is None:
            cls._TERNARY_OPS = TernaryCodecOps()
        return cls._TERNARY_OPS

    def prepare_profile(self, grid: np.ndarray, color: int = 0, *, pad: int = 0) -> PreparedProfile:
        return self.geometry_prep.prepare_profile(grid, color=color, pad=pad)

    def _select_rows(
        self,
        primary_values: np.ndarray,
        *,
        active_rows: list[int],
        delta_threshold: float,
        secondary_values: tuple[np.ndarray, ...] = (),
    ) -> tuple[list[int], list[int], list[list[int]]]:
        primary_delta = np.diff(primary_values, prepend=primary_values[0]).astype(np.float32, copy=False)
        primary_trits = self._ternary_ops().quantize(primary_delta.tolist(), threshold=delta_threshold)
        secondary_trits: list[list[int]] = []
        for values in secondary_values:
            delta = np.diff(values, prepend=values[0]).astype(np.float32, copy=False)
            secondary_trits.append(
                self._ternary_ops().quantize(delta.tolist(), threshold=delta_threshold)
            )

        if not active_rows:
            return [], primary_trits, secondary_trits

        active_first = active_rows[0]
        active_last = active_rows[-1]
        selected_rows: list[int] = []
        for idx in active_rows:
            neighborhood = [primary_trits]
            neighborhood.extend(secondary_trits)
            prev_active = False
            cur_active = False
            next_active = False
            for trits in neighborhood:
                prev_trit = trits[idx - 1] if idx > 0 else trits[idx]
                cur_trit = trits[idx]
                next_trit = trits[idx + 1] if idx + 1 < len(trits) else trits[idx]
                prev_active = prev_active or (prev_trit != 0)
                cur_active = cur_active or (cur_trit != 0)
                next_active = next_active or (next_trit != 0)
            if idx in {active_first, active_last} or cur_active or prev_active or next_active:
                selected_rows.append(idx)

        if not selected_rows:
            selected_rows = [active_first, active_last] if active_first != active_last else [active_first]
        return sorted(set(selected_rows)), primary_trits, secondary_trits

    def contour_to_lathe_mesh(
        self,
        grid: np.ndarray,
        color: int = 0,
        *,
        pad: int = 0,
        segments: int = 24,
        height_scale: float = 1.0,
        radius_scale: float = 1.0,
        cap_ends: bool = True,
    ) -> MeshPlan:
        if segments < 3:
            raise ValueError("segments must be >= 3")

        profile = self.prepare_profile(grid, color=color, pad=pad)
        if profile.is_empty:
            raise ValueError("profile is empty")

        row_fill = np.asarray(profile.row_fill, dtype=np.float32)
        region_height = int(profile.region.shape[0])
        region_width = int(profile.region.shape[1])
        if region_height < 1 or region_width < 1:
            raise ValueError("profile region is empty")

        if region_height == 1:
            y_coords = np.array([0.0], dtype=np.float32)
        else:
            y_coords = np.linspace(0.5, -0.5, region_height, dtype=np.float32) * float(height_scale)
        radii = (row_fill / max(1.0, float(region_width))) * float(radius_scale)

        # Use ternary profile deltas to collapse long neutral stretches.
        delta_threshold = max(1.0 / max(1.0, float(region_width)), 0.08)
        active_rows = [idx for idx, fill in enumerate(row_fill.tolist()) if float(fill) > 0.0]
        if not active_rows:
            raise ValueError("profile has no active rows")

        selected_rows, row_delta_trits, _secondary = self._select_rows(
            radii,
            active_rows=active_rows,
            delta_threshold=delta_threshold,
        )
        y_rows = y_coords[selected_rows]
        radius_rows = radii[selected_rows]

        theta = (2.0 * math.pi * np.arange(segments, dtype=np.float32)) / float(segments)
        cos_theta = np.cos(theta).astype(np.float32, copy=False)
        sin_theta = np.sin(theta).astype(np.float32, copy=False)
        x = radius_rows[:, None] * cos_theta[None, :]
        y = np.broadcast_to(y_rows[:, None], x.shape)
        z = radius_rows[:, None] * sin_theta[None, :]
        vertices_np = np.stack((x, y, z), axis=-1).astype(np.float32, copy=False).reshape(-1, 3)

        row_count = len(selected_rows)
        seg_ids = np.arange(segments, dtype=np.uint32)
        next_seg = ((seg_ids + 1) % segments).astype(np.uint32, copy=False)
        if row_count > 1:
            row_ids = np.arange(row_count - 1, dtype=np.uint32)[:, None]
            row_base = row_ids * np.uint32(segments)
            next_base = (row_ids + 1) * np.uint32(segments)
            tri1 = np.stack(
                (row_base + seg_ids, next_base + seg_ids, next_base + next_seg),
                axis=-1,
            ).reshape(-1, 3)
            tri2 = np.stack(
                (row_base + seg_ids, next_base + next_seg, row_base + next_seg),
                axis=-1,
            ).reshape(-1, 3)
            indices_np = np.concatenate((tri1, tri2), axis=0).astype(np.uint32, copy=False)
        else:
            indices_np = np.empty((0, 3), dtype=np.uint32)

        if cap_ends:
            top_center = vertices_np.shape[0]
            bottom_center = vertices_np.shape[0] + 1
            cap_vertices = np.asarray(
                [
                    [0.0, float(y_rows[0]), 0.0],
                    [0.0, float(y_rows[-1]), 0.0],
                ],
                dtype=np.float32,
            )
            vertices_np = np.concatenate((vertices_np, cap_vertices), axis=0)
            top_cap = np.stack(
                (
                    np.full(segments, top_center, dtype=np.uint32),
                    seg_ids,
                    next_seg,
                ),
                axis=-1,
            )
            base = np.uint32((row_count - 1) * segments)
            bottom_cap = np.stack(
                (
                    np.full(segments, bottom_center, dtype=np.uint32),
                    base + next_seg,
                    base + seg_ids,
                ),
                axis=-1,
            )
            indices_np = np.concatenate((indices_np, top_cap, bottom_cap), axis=0)

        normal_plan = _resolve_math_core_plan(2, int(indices_np.shape[0]))
        normals_np = _compute_vertex_normals(vertices_np, indices_np, execution_plan=normal_plan)

        metadata: dict[str, object] = {
            "mesh_kind": "lathe",
            "segments": int(segments),
            "height_scale": float(height_scale),
            "radius_scale": float(radius_scale),
            "cap_ends": bool(cap_ends),
            "profile_bbox": [profile.bbox.min_y, profile.bbox.min_x, profile.bbox.max_y, profile.bbox.max_x],
            "profile_shape": [int(profile.region.shape[0]), int(profile.region.shape[1])],
            "profile_density": float(profile.mask_density),
            "source_shape": [int(profile.source_shape[0]), int(profile.source_shape[1])],
            "ternary_delta_threshold": float(delta_threshold),
            "row_delta_trits": [int(v) for v in row_delta_trits],
            "profile_rows_used": [int(v) for v in selected_rows],
            "profile_rows_total": int(region_height),
            "math_core_plan": normal_plan,
        }
        return MeshPlan(
            vertices=vertices_np,
            indices=indices_np,
            normals=normals_np,
            profile=profile,
            metadata=metadata,
        )

    def contour_to_extrude_mesh(
        self,
        grid: np.ndarray,
        color: int = 0,
        *,
        pad: int = 0,
        depth_scale: float = 0.5,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
        cap_ends: bool = True,
    ) -> MeshPlan:
        profile = self.prepare_profile(grid, color=color, pad=pad)
        if profile.is_empty:
            raise ValueError("profile is empty")

        region_height = int(profile.region.shape[0])
        region_width = int(profile.region.shape[1])
        if region_height < 1 or region_width < 1:
            raise ValueError("profile region is empty")

        left = np.asarray(profile.left_contour_smoothed, dtype=np.float32)
        right = np.asarray(profile.right_contour_smoothed, dtype=np.float32)
        row_fill = np.asarray(profile.row_fill, dtype=np.float32)
        valid_mask = (left >= 0.0) & (right >= left)
        active_rows = [idx for idx, valid in enumerate(valid_mask.tolist()) if bool(valid) and float(row_fill[idx]) > 0.0]
        if not active_rows:
            raise ValueError("profile has no active rows")

        widths = np.where(valid_mask, np.maximum(0.0, right - left + 1.0), 0.0).astype(np.float32, copy=False)
        delta_threshold = max(1.0 / max(1.0, float(region_width)), 0.08)
        selected_rows, width_delta_trits, edge_delta_trits = self._select_rows(
            widths,
            active_rows=active_rows,
            delta_threshold=delta_threshold,
            secondary_values=(left, right),
        )
        left_delta_trits, right_delta_trits = edge_delta_trits

        if region_height == 1:
            y_coords = np.array([0.0], dtype=np.float32)
        else:
            y_coords = np.linspace(0.5, -0.5, region_height, dtype=np.float32) * float(height_scale)

        denom = max(1.0, float(region_width - 1))
        half_depth = float(depth_scale) * 0.5
        rows_np = np.asarray(selected_rows, dtype=np.int32)
        left_x = ((left[rows_np] / denom) - 0.5) * float(width_scale)
        right_x = ((right[rows_np] / denom) - 0.5) * float(width_scale)
        y = y_coords[rows_np]
        vertices_np = np.stack(
            (
                np.stack((left_x, y, np.full_like(y, -half_depth)), axis=-1),
                np.stack((right_x, y, np.full_like(y, -half_depth)), axis=-1),
                np.stack((right_x, y, np.full_like(y, half_depth)), axis=-1),
                np.stack((left_x, y, np.full_like(y, half_depth)), axis=-1),
            ),
            axis=1,
        ).astype(np.float32, copy=False).reshape(-1, 3)

        row_count = len(selected_rows)
        if row_count > 1:
            row_ids = np.arange(row_count - 1, dtype=np.uint32)[:, None]
            base = row_ids * np.uint32(4)
            nxt = (row_ids + 1) * np.uint32(4)
            tri_blocks = [
                np.stack((base + 0, base + 1, nxt + 1), axis=-1),
                np.stack((base + 0, nxt + 1, nxt + 0), axis=-1),
                np.stack((base + 3, nxt + 2, base + 2), axis=-1),
                np.stack((base + 3, nxt + 3, nxt + 2), axis=-1),
                np.stack((base + 1, base + 2, nxt + 2), axis=-1),
                np.stack((base + 1, nxt + 2, nxt + 1), axis=-1),
                np.stack((base + 3, base + 0, nxt + 0), axis=-1),
                np.stack((base + 3, nxt + 0, nxt + 3), axis=-1),
            ]
            indices_np = np.concatenate(tri_blocks, axis=0).reshape(-1, 3).astype(np.uint32, copy=False)
        else:
            indices_np = np.empty((0, 3), dtype=np.uint32)

        if cap_ends and row_count >= 1:
            top = 0
            bottom = (row_count - 1) * 4
            caps = np.asarray(
                [
                    [top + 0, top + 1, top + 2],
                    [top + 0, top + 2, top + 3],
                    [bottom + 3, bottom + 2, bottom + 1],
                    [bottom + 3, bottom + 1, bottom + 0],
                ],
                dtype=np.uint32,
            )
            indices_np = np.concatenate((indices_np, caps), axis=0)

        normal_plan = _resolve_math_core_plan(2, int(indices_np.shape[0]))
        normals_np = _compute_vertex_normals(vertices_np, indices_np, execution_plan=normal_plan)

        metadata: dict[str, object] = {
            "mesh_kind": "extrude",
            "depth_scale": float(depth_scale),
            "width_scale": float(width_scale),
            "height_scale": float(height_scale),
            "cap_ends": bool(cap_ends),
            "profile_bbox": [profile.bbox.min_y, profile.bbox.min_x, profile.bbox.max_y, profile.bbox.max_x],
            "profile_shape": [int(profile.region.shape[0]), int(profile.region.shape[1])],
            "profile_density": float(profile.mask_density),
            "source_shape": [int(profile.source_shape[0]), int(profile.source_shape[1])],
            "ternary_delta_threshold": float(delta_threshold),
            "width_delta_trits": [int(v) for v in width_delta_trits],
            "left_delta_trits": [int(v) for v in left_delta_trits],
            "right_delta_trits": [int(v) for v in right_delta_trits],
            "profile_rows_used": [int(v) for v in selected_rows],
            "profile_rows_total": int(region_height),
            "math_core_plan": normal_plan,
        }
        return MeshPlan(
            vertices=vertices_np,
            indices=indices_np,
            normals=normals_np,
            profile=profile,
            metadata=metadata,
        )

    def contour_to_sweep_mesh(
        self,
        grid: np.ndarray,
        color: int = 0,
        *,
        pad: int = 0,
        depth_scale: float = 0.4,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
        cap_ends: bool = True,
    ) -> MeshPlan:
        profile = self.prepare_profile(grid, color=color, pad=pad)
        if profile.is_empty:
            raise ValueError("profile is empty")

        region_height = int(profile.region.shape[0])
        region_width = int(profile.region.shape[1])
        if region_height < 1 or region_width < 1:
            raise ValueError("profile region is empty")

        left = np.asarray(profile.left_contour_smoothed, dtype=np.float32)
        right = np.asarray(profile.right_contour_smoothed, dtype=np.float32)
        row_fill = np.asarray(profile.row_fill, dtype=np.float32)
        valid_mask = (left >= 0.0) & (right >= left)
        active_rows = [idx for idx, valid in enumerate(valid_mask.tolist()) if bool(valid) and float(row_fill[idx]) > 0.0]
        if not active_rows:
            raise ValueError("profile has no active rows")

        widths = np.where(valid_mask, np.maximum(0.0, right - left + 1.0), 0.0).astype(np.float32, copy=False)
        centers = np.where(valid_mask, (left + right) * 0.5, 0.0).astype(np.float32, copy=False)
        delta_threshold = max(1.0 / max(1.0, float(region_width)), 0.08)
        selected_rows, width_delta_trits, secondary_trits = self._select_rows(
            widths,
            active_rows=active_rows,
            delta_threshold=delta_threshold,
            secondary_values=(centers,),
        )
        center_delta_trits = secondary_trits[0]

        if region_height == 1:
            y_coords = np.array([0.0], dtype=np.float32)
        else:
            y_coords = np.linspace(0.5, -0.5, region_height, dtype=np.float32) * float(height_scale)

        denom = max(1.0, float(region_width - 1))
        half_depth = float(depth_scale) * 0.5
        rows_np = np.asarray(selected_rows, dtype=np.int32)
        center_x = ((centers[rows_np] / denom) - 0.5) * float(width_scale)
        y = y_coords[rows_np]
        half_widths = (widths[rows_np] / max(1.0, float(region_width))) * float(width_scale) * 0.5
        centers_xy = np.stack((center_x, y), axis=-1).astype(np.float32, copy=False)

        row_count = centers_xy.shape[0]
        fallback_tangent = np.array([0.0, -1.0], dtype=np.float32)
        if row_count == 1:
            tangents = np.repeat(fallback_tangent[None, :], 1, axis=0)
        else:
            tangents = np.zeros_like(centers_xy, dtype=np.float32)
            tangents[0] = centers_xy[1] - centers_xy[0]
            tangents[-1] = centers_xy[-1] - centers_xy[-2]
            if row_count > 2:
                tangents[1:-1] = centers_xy[2:] - centers_xy[:-2]
            tangent_lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
            tangents = np.divide(
                tangents,
                tangent_lengths,
                out=np.repeat(fallback_tangent[None, :], row_count, axis=0).astype(np.float32, copy=False),
                where=tangent_lengths > 1e-8,
            )
        normals_xy = np.stack((-tangents[:, 1], tangents[:, 0]), axis=-1).astype(np.float32, copy=False)
        normal_lengths = np.linalg.norm(normals_xy, axis=1, keepdims=True)
        normals_xy = np.divide(
            normals_xy,
            normal_lengths,
            out=np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), row_count, axis=0),
            where=normal_lengths > 1e-8,
        )
        offsets = normals_xy * half_widths[:, None]
        left_xy = centers_xy - offsets
        right_xy = centers_xy + offsets
        vertices_np = np.stack(
            (
                np.column_stack((left_xy, np.full(row_count, -half_depth, dtype=np.float32))),
                np.column_stack((right_xy, np.full(row_count, -half_depth, dtype=np.float32))),
                np.column_stack((right_xy, np.full(row_count, half_depth, dtype=np.float32))),
                np.column_stack((left_xy, np.full(row_count, half_depth, dtype=np.float32))),
            ),
            axis=1,
        ).astype(np.float32, copy=False).reshape(-1, 3)

        if row_count > 1:
            row_ids = np.arange(row_count - 1, dtype=np.uint32)[:, None]
            base = row_ids * np.uint32(4)
            nxt = (row_ids + 1) * np.uint32(4)
            tri_blocks = [
                np.stack((base + 0, base + 1, nxt + 1), axis=-1),
                np.stack((base + 0, nxt + 1, nxt + 0), axis=-1),
                np.stack((base + 3, nxt + 2, base + 2), axis=-1),
                np.stack((base + 3, nxt + 3, nxt + 2), axis=-1),
                np.stack((base + 1, base + 2, nxt + 2), axis=-1),
                np.stack((base + 1, nxt + 2, nxt + 1), axis=-1),
                np.stack((base + 3, base + 0, nxt + 0), axis=-1),
                np.stack((base + 3, nxt + 0, nxt + 3), axis=-1),
            ]
            indices_np = np.concatenate(tri_blocks, axis=0).reshape(-1, 3).astype(np.uint32, copy=False)
        else:
            indices_np = np.empty((0, 3), dtype=np.uint32)

        if cap_ends and row_count >= 1:
            top = 0
            bottom = (row_count - 1) * 4
            caps = np.asarray(
                [
                    [top + 0, top + 1, top + 2],
                    [top + 0, top + 2, top + 3],
                    [bottom + 3, bottom + 2, bottom + 1],
                    [bottom + 3, bottom + 1, bottom + 0],
                ],
                dtype=np.uint32,
            )
            indices_np = np.concatenate((indices_np, caps), axis=0)

        normal_plan = _resolve_math_core_plan(2, int(indices_np.shape[0]))
        normals_np = _compute_vertex_normals(vertices_np, indices_np, execution_plan=normal_plan)

        metadata: dict[str, object] = {
            "mesh_kind": "sweep",
            "depth_scale": float(depth_scale),
            "width_scale": float(width_scale),
            "height_scale": float(height_scale),
            "cap_ends": bool(cap_ends),
            "profile_bbox": [profile.bbox.min_y, profile.bbox.min_x, profile.bbox.max_y, profile.bbox.max_x],
            "profile_shape": [int(profile.region.shape[0]), int(profile.region.shape[1])],
            "profile_density": float(profile.mask_density),
            "source_shape": [int(profile.source_shape[0]), int(profile.source_shape[1])],
            "ternary_delta_threshold": float(delta_threshold),
            "width_delta_trits": [int(v) for v in width_delta_trits],
            "center_delta_trits": [int(v) for v in center_delta_trits],
            "profile_rows_used": [int(v) for v in selected_rows],
            "profile_rows_total": int(region_height),
            "math_core_plan": normal_plan,
        }
        return MeshPlan(
            vertices=vertices_np,
            indices=indices_np,
            normals=normals_np,
            profile=profile,
            metadata=metadata,
        )

    def warmup_runtime(self) -> dict[str, float | int | str]:
        current_pid = os.getpid()
        if self.__class__._WARMED_PID == current_pid and self.__class__._WARMUP_REPORT is not None:
            return dict(self.__class__._WARMUP_REPORT)

        sample = np.zeros((24, 24), dtype=np.int32)
        sample[5:19, 6:11] = 1

        total_start = time.perf_counter()
        prep_start = time.perf_counter()
        profile = self.prepare_profile(sample, color=1, pad=1)
        prep_ms = (time.perf_counter() - prep_start) * 1000.0

        mesh_start = time.perf_counter()
        lathe_plan = self.contour_to_lathe_mesh(sample, color=1, pad=1, segments=12)
        lathe_ms = (time.perf_counter() - mesh_start) * 1000.0

        extrude_start = time.perf_counter()
        extrude_plan = self.contour_to_extrude_mesh(sample, color=1, pad=1, depth_scale=0.4)
        extrude_ms = (time.perf_counter() - extrude_start) * 1000.0
        sweep_start = time.perf_counter()
        sweep_plan = self.contour_to_sweep_mesh(sample, color=1, pad=1, depth_scale=0.4)
        sweep_ms = (time.perf_counter() - sweep_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        report: dict[str, float | int | str] = {
            "status": "ready",
            "pid": current_pid,
            "profile_warmup_ms": prep_ms,
            "lathe_warmup_ms": lathe_ms,
            "extrude_warmup_ms": extrude_ms,
            "sweep_warmup_ms": sweep_ms,
            "total_warmup_ms": total_ms,
            "warm_profile_nonzero": int(profile.nonzero_count),
            "warm_vertex_count": int(lathe_plan.vertices.shape[0]),
            "warm_triangle_count": int(lathe_plan.indices.shape[0]),
            "warm_extrude_vertex_count": int(extrude_plan.vertices.shape[0]),
            "warm_extrude_triangle_count": int(extrude_plan.indices.shape[0]),
            "warm_sweep_vertex_count": int(sweep_plan.vertices.shape[0]),
            "warm_sweep_triangle_count": int(sweep_plan.indices.shape[0]),
        }
        self.__class__._WARMED_PID = current_pid
        self.__class__._WARMUP_REPORT = dict(report)
        return report


__all__ = ["MeshPlan", "ProceduralGeometryBridge"]
