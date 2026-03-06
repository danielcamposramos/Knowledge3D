"""Deterministic procedural material bridge for 3D surface projection.

The bridge stays honest about the current substrate:
- PTX-backed drawing effects generate material previews and normal hints
- ternary contrastive logic selects compact palette/material candidates
- host code performs deterministic surface projection until a dedicated
  UV/triplanar kernel is justified by repetition
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

import numpy as np

from knowledge3d.cranium.bridges.procedural_geometry_bridge import MeshPlan, ProceduralGeometryBridge
from knowledge3d.cranium.bridges.procedural_signal_bridge import ProceduralSignalBridge, SpectrogramPlan
from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects
from knowledge3d.cranium.ptx_runtime.math_core_pool import get_global_math_core_pool
from knowledge3d.cranium.ptx_runtime.material_projection_kernels import MaterialProjectionKernels
from knowledge3d.cranium.ternary import TernaryVector


class SurfaceMeshLike(Protocol):
    vertices: np.ndarray
    normals: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SurfaceMaterialCandidate:
    material_id: str
    name: str
    gradient_type: str = "linear"
    gradient_stops: tuple[tuple[float, float, float, float, float], ...] = ()
    palette: tuple[tuple[float, float, float, float], ...] = ()
    base_stop: tuple[float, float, float, float, float] | None = None
    position_layers: tuple[tuple[int, ...], ...] = ()
    color_layers: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()
    projection_strategy: str = "triplanar"
    tiling: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MaterialSelection:
    selected: SurfaceMaterialCandidate
    selected_stops: tuple[tuple[float, float, float, float, float], ...]
    score_table: tuple[dict[str, Any], ...]
    target_stops: tuple[tuple[float, float, float, float, float], ...]
    math_core_plan: dict[str, Any]


@dataclass(frozen=True)
class SurfaceMaterialPlan:
    mesh: SurfaceMeshLike
    selected_material: SurfaceMaterialCandidate
    material_preview: np.ndarray
    normal_hint: np.ndarray
    vertex_rgba: np.ndarray
    projection_weights: np.ndarray
    metadata: dict[str, Any]


def _resolve_material_math_core_plan(preferred_tier: int, work_items: int) -> dict[str, Any]:
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


class ProceduralMaterialBridge:
    """Always-on material selection and projection bridge."""

    _WARMED_PID: int | None = None
    _WARMUP_REPORT: dict[str, float | int | str] | None = None

    def __init__(self) -> None:
        self.effects = DrawingEffects()
        self.projection_kernels = MaterialProjectionKernels()
        self.geometry_bridge = ProceduralGeometryBridge()
        self.signal_bridge = ProceduralSignalBridge()

    def material_to_stops(
        self,
        candidate: SurfaceMaterialCandidate,
    ) -> tuple[tuple[float, float, float, float, float], ...]:
        if candidate.gradient_stops:
            return tuple(self._normalize_stops(candidate.gradient_stops))
        if candidate.palette:
            return tuple(self.effects.palette_to_gradient_stops(candidate.palette))
        if candidate.base_stop is not None and candidate.position_layers and candidate.color_layers:
            return tuple(
                self.effects.ternary_gradient.compose_stops_from_cascade(
                    base_stop=candidate.base_stop,
                    position_layers=candidate.position_layers,
                    color_layers=candidate.color_layers,
                )
            )
        raise ValueError(f"material candidate {candidate.material_id} has no procedural color definition")

    def select_material(
        self,
        *,
        target_material: SurfaceMaterialCandidate,
        candidates: Sequence[SurfaceMaterialCandidate],
        negative_materials: Sequence[SurfaceMaterialCandidate] = (),
    ) -> MaterialSelection:
        if not candidates:
            raise ValueError("at least one candidate material is required")
        target_stops = self.material_to_stops(target_material)
        negative_stops = [self.material_to_stops(item) for item in negative_materials]
        selection_plan = _resolve_material_math_core_plan(2, len(candidates))

        scored: list[tuple[float, SurfaceMaterialCandidate, tuple[tuple[float, float, float, float, float], ...], dict[str, Any]]] = []
        batch_size = int(selection_plan["batch_size"])
        for start in range(0, len(candidates), batch_size):
            candidate_batch = candidates[start:start + batch_size]
            for candidate in candidate_batch:
                candidate_stops = self.material_to_stops(candidate)
                gradient_score = self.effects.contrastive_gradient_score(
                    target_stops,
                    candidate_stops,
                    negative_examples=negative_stops,
                )
                score_row = {
                    "material_id": candidate.material_id,
                    "name": candidate.name,
                    "score": float(gradient_score.score),
                    "positive_similarity": float(gradient_score.positive_similarity),
                    "negative_penalty": float(gradient_score.negative_penalty),
                    "projection_strategy": candidate.projection_strategy,
                }
                scored.append((float(gradient_score.score), candidate, candidate_stops, score_row))

        scored.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
        _best_score, best_candidate, best_stops, _best_row = scored[0]
        return MaterialSelection(
            selected=best_candidate,
            selected_stops=tuple(best_stops),
            score_table=tuple(item[3] for item in scored),
            target_stops=tuple(target_stops),
            math_core_plan=selection_plan,
        )

    def render_material_preview(
        self,
        candidate: SurfaceMaterialCandidate,
        *,
        width: int = 64,
        height: int = 64,
    ) -> np.ndarray:
        stops = self.material_to_stops(candidate)
        gradient_type = str(candidate.gradient_type or "linear").lower()
        if candidate.base_stop is not None and candidate.position_layers and candidate.color_layers:
            return self.effects.linear_gradient_from_ternary_cascade(
                width,
                height,
                base_stop=candidate.base_stop,
                position_layers=candidate.position_layers,
                color_layers=candidate.color_layers,
                x1=0.0,
                y1=0.0,
                x2=1.0,
                y2=1.0,
            )
        if gradient_type == "radial":
            return self.effects.radial_gradient(width, height, stops, cx=0.5, cy=0.5, radius=0.72)
        if gradient_type == "conic":
            return self.effects.conic_gradient(width, height, stops, cx=0.5, cy=0.5, start_angle=0.0)
        return self.effects.linear_gradient(width, height, stops, x1=0.0, y1=0.0, x2=1.0, y2=1.0)

    def project_material(
        self,
        mesh: SurfaceMeshLike,
        candidate: SurfaceMaterialCandidate,
        *,
        preview_size: int = 64,
        projection_strategy: str | None = None,
    ) -> SurfaceMaterialPlan:
        preview = self.render_material_preview(candidate, width=preview_size, height=preview_size)
        normal_hint = self.effects.edge_map(preview)
        strategy = str(projection_strategy or candidate.projection_strategy or "triplanar").lower()
        tiling = max(0.1, float(candidate.tiling))

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.normals, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("mesh vertices must have shape [n,3]")
        if normals.shape != vertices.shape:
            raise ValueError("mesh normals must match vertex shape")

        mins = np.min(vertices, axis=0)
        maxs = np.max(vertices, axis=0)
        extents = np.maximum(maxs - mins, 1e-6)
        if strategy == "triplanar":
            projection_plan = _resolve_material_math_core_plan(2, int(vertices.shape[0]) * 3)
            yz = self.projection_kernels.sample_preview(
                preview,
                vertices[:, [1, 2]],
                mins[[1, 2]],
                extents[[1, 2]],
                tiling,
            )
            xz = self.projection_kernels.sample_preview(
                preview,
                vertices[:, [0, 2]],
                mins[[0, 2]],
                extents[[0, 2]],
                tiling,
            )
            xy = self.projection_kernels.sample_preview(
                preview,
                vertices[:, [0, 1]],
                mins[[0, 1]],
                extents[[0, 1]],
                tiling,
            )
        elif strategy == "planar_xy":
            projection_plan = _resolve_material_math_core_plan(1, int(vertices.shape[0]))
            xy = self.projection_kernels.sample_preview(
                preview,
                vertices[:, [0, 1]],
                mins[[0, 1]],
                extents[[0, 1]],
                tiling,
            )
            yz = xz = None
        elif strategy == "planar_xz":
            projection_plan = _resolve_material_math_core_plan(1, int(vertices.shape[0]))
            xz = self.projection_kernels.sample_preview(
                preview,
                vertices[:, [0, 2]],
                mins[[0, 2]],
                extents[[0, 2]],
                tiling,
            )
            yz = xy = None
        else:
            projection_plan = _resolve_material_math_core_plan(1, int(vertices.shape[0]))
            yz = self.projection_kernels.sample_preview(
                preview,
                vertices[:, [1, 2]],
                mins[[1, 2]],
                extents[[1, 2]],
                tiling,
            )
            xz = xy = None

        if strategy == "planar_xy":
            weights = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            weights[:, 2] = 1.0
            assert xy is not None
            vertex_rgba = xy
        elif strategy == "planar_xz":
            weights = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            weights[:, 1] = 1.0
            assert xz is not None
            vertex_rgba = xz
        elif strategy == "planar_yz":
            weights = np.zeros((vertices.shape[0], 3), dtype=np.float32)
            weights[:, 0] = 1.0
            assert yz is not None
            vertex_rgba = yz
        else:
            weights = np.abs(normals).astype(np.float32, copy=False)
            weight_sum = np.sum(weights, axis=1, keepdims=True)
            weight_sum[weight_sum < 1e-6] = 1.0
            weights = weights / weight_sum
            assert yz is not None and xz is not None and xy is not None
            vertex_rgba = self.projection_kernels.blend_triplanar(yz, xz, xy, weights)

        metadata = dict(mesh.metadata)
        metadata.update(
            {
                "material_id": candidate.material_id,
                "material_name": candidate.name,
                "projection_strategy": strategy,
                "preview_size": int(preview_size),
                "tiling": float(tiling),
                "projection_weight_means": [float(v) for v in np.mean(weights, axis=0).tolist()],
                "math_core_plan": projection_plan,
            }
        )
        return SurfaceMaterialPlan(
            mesh=mesh,
            selected_material=candidate,
            material_preview=preview,
            normal_hint=normal_hint,
            vertex_rgba=np.clip(vertex_rgba, 0.0, 1.0).astype(np.float32, copy=False),
            projection_weights=weights.astype(np.float32, copy=False),
            metadata=metadata,
        )

    def signal_projection_to_material_target(
        self,
        projection: SpectrogramPlan,
        *,
        material_id: str = "signal_target",
        name: str = "Signal Target Material",
    ) -> SurfaceMaterialCandidate:
        positive = float(projection.metadata.get("positive_ratio", 0.0))
        negative = float(projection.metadata.get("negative_ratio", 0.0))
        neutral = float(projection.metadata.get("neutral_ratio", 0.0))
        palette = (
            (
                0.08 + 0.06 * neutral,
                0.08 + 0.12 * neutral,
                0.10 + 0.45 * negative,
                1.0,
            ),
            (
                0.12 + 0.28 * neutral,
                0.16 + 0.26 * neutral,
                0.22 + 0.18 * neutral,
                1.0,
            ),
            (
                0.22 + 0.65 * positive,
                0.18 + 0.48 * positive,
                0.12 + 0.10 * positive,
                1.0,
            ),
        )
        return SurfaceMaterialCandidate(
            material_id=material_id,
            name=name,
            gradient_type="linear",
            palette=palette,
            projection_strategy="triplanar",
            tiling=max(1.0, float(projection.metadata.get("frame_count", 1)) / 8.0),
            metadata={
                "source": "signal_projection",
                "positive_ratio": positive,
                "negative_ratio": negative,
                "neutral_ratio": neutral,
            },
        )

    def signal_to_textured_surface(
        self,
        clip_id: str,
        samples: TernaryVector,
        *,
        candidates: Sequence[SurfaceMaterialCandidate],
        negative_materials: Sequence[SurfaceMaterialCandidate] = (),
        target_material: SurfaceMaterialCandidate | None = None,
        frame_size: int = 1024,
        threshold: float = 0.2,
        displacement_gain: float = 0.25,
        preview_size: int = 64,
        projection_strategy: str | None = None,
    ) -> SurfaceMaterialPlan:
        signal_bridge = ProceduralSignalBridge(frame_size=frame_size, threshold=threshold)
        projection = signal_bridge.audio_to_spectrogram(clip_id, samples)
        surface = signal_bridge.spectrogram_to_surface(
            projection,
            displacement_gain=displacement_gain,
        )
        target = target_material or self.signal_projection_to_material_target(projection)
        selection = self.select_material(
            target_material=target,
            candidates=candidates,
            negative_materials=negative_materials,
        )
        plan = self.project_material(
            surface,
            selection.selected,
            preview_size=preview_size,
            projection_strategy=projection_strategy,
        )
        metadata = dict(plan.metadata)
        metadata.update(
            {
                "clip_id": clip_id,
                "signal_frame_size": int(frame_size),
                "signal_threshold": float(threshold),
                "signal_projection_summary": {
                    "frame_count": int(projection.metadata.get("frame_count", 0)),
                    "frequency_bins": int(projection.metadata.get("frequency_bins", 0)),
                    "positive_ratio": float(projection.metadata.get("positive_ratio", 0.0)),
                    "negative_ratio": float(projection.metadata.get("negative_ratio", 0.0)),
                    "neutral_ratio": float(projection.metadata.get("neutral_ratio", 0.0)),
                },
                "signal_math_core_plan": dict(projection.metadata.get("math_core_plan", {})),
                "signal_surface_math_core_plan": dict(surface.metadata.get("math_core_plan", {})),
                "target_material_stops": [list(row) for row in selection.target_stops],
                "selected_material_stops": [list(row) for row in selection.selected_stops],
                "material_score_table": list(selection.score_table),
                "selection_math_core_plan": selection.math_core_plan,
            }
        )
        return SurfaceMaterialPlan(
            mesh=plan.mesh,
            selected_material=plan.selected_material,
            material_preview=plan.material_preview,
            normal_hint=plan.normal_hint,
            vertex_rgba=plan.vertex_rgba,
            projection_weights=plan.projection_weights,
            metadata=metadata,
        )

    def contour_to_textured_lathe_mesh(
        self,
        grid: np.ndarray,
        *,
        color: int = 0,
        pad: int = 0,
        segments: int = 24,
        height_scale: float = 1.0,
        radius_scale: float = 1.0,
        cap_ends: bool = True,
        target_material: SurfaceMaterialCandidate,
        candidates: Sequence[SurfaceMaterialCandidate],
        negative_materials: Sequence[SurfaceMaterialCandidate] = (),
        preview_size: int = 64,
        projection_strategy: str | None = None,
    ) -> SurfaceMaterialPlan:
        mesh = self.geometry_bridge.contour_to_lathe_mesh(
            grid,
            color=color,
            pad=pad,
            segments=segments,
            height_scale=height_scale,
            radius_scale=radius_scale,
            cap_ends=cap_ends,
        )
        return self._project_selected_material(
            mesh,
            target_material=target_material,
            candidates=candidates,
            negative_materials=negative_materials,
            preview_size=preview_size,
            projection_strategy=projection_strategy,
        )

    def contour_to_textured_extrude_mesh(
        self,
        grid: np.ndarray,
        *,
        color: int = 0,
        pad: int = 0,
        depth_scale: float = 0.5,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
        cap_ends: bool = True,
        target_material: SurfaceMaterialCandidate,
        candidates: Sequence[SurfaceMaterialCandidate],
        negative_materials: Sequence[SurfaceMaterialCandidate] = (),
        preview_size: int = 64,
        projection_strategy: str | None = None,
    ) -> SurfaceMaterialPlan:
        mesh = self.geometry_bridge.contour_to_extrude_mesh(
            grid,
            color=color,
            pad=pad,
            depth_scale=depth_scale,
            width_scale=width_scale,
            height_scale=height_scale,
            cap_ends=cap_ends,
        )
        return self._project_selected_material(
            mesh,
            target_material=target_material,
            candidates=candidates,
            negative_materials=negative_materials,
            preview_size=preview_size,
            projection_strategy=projection_strategy,
        )

    def contour_to_textured_sweep_mesh(
        self,
        grid: np.ndarray,
        *,
        color: int = 0,
        pad: int = 0,
        depth_scale: float = 0.5,
        width_scale: float = 1.0,
        height_scale: float = 1.0,
        cap_ends: bool = True,
        target_material: SurfaceMaterialCandidate,
        candidates: Sequence[SurfaceMaterialCandidate],
        negative_materials: Sequence[SurfaceMaterialCandidate] = (),
        preview_size: int = 64,
        projection_strategy: str | None = None,
    ) -> SurfaceMaterialPlan:
        mesh = self.geometry_bridge.contour_to_sweep_mesh(
            grid,
            color=color,
            pad=pad,
            depth_scale=depth_scale,
            width_scale=width_scale,
            height_scale=height_scale,
            cap_ends=cap_ends,
        )
        return self._project_selected_material(
            mesh,
            target_material=target_material,
            candidates=candidates,
            negative_materials=negative_materials,
            preview_size=preview_size,
            projection_strategy=projection_strategy,
        )

    def _project_selected_material(
        self,
        mesh: SurfaceMeshLike,
        *,
        target_material: SurfaceMaterialCandidate,
        candidates: Sequence[SurfaceMaterialCandidate],
        negative_materials: Sequence[SurfaceMaterialCandidate],
        preview_size: int,
        projection_strategy: str | None,
    ) -> SurfaceMaterialPlan:
        selection = self.select_material(
            target_material=target_material,
            candidates=candidates,
            negative_materials=negative_materials,
        )
        plan = self.project_material(
            mesh,
            selection.selected,
            preview_size=preview_size,
            projection_strategy=projection_strategy,
        )
        metadata = dict(plan.metadata)
        metadata.update(
            {
                "target_material_stops": [list(row) for row in selection.target_stops],
                "selected_material_stops": [list(row) for row in selection.selected_stops],
                "material_score_table": list(selection.score_table),
                "selection_math_core_plan": selection.math_core_plan,
            }
        )
        return SurfaceMaterialPlan(
            mesh=plan.mesh,
            selected_material=plan.selected_material,
            material_preview=plan.material_preview,
            normal_hint=plan.normal_hint,
            vertex_rgba=plan.vertex_rgba,
            projection_weights=plan.projection_weights,
            metadata=metadata,
        )

    def warmup_runtime(self) -> dict[str, float | int | str]:
        current_pid = os.getpid()
        if self.__class__._WARMED_PID == current_pid and self.__class__._WARMUP_REPORT is not None:
            return dict(self.__class__._WARMUP_REPORT)

        grid = np.zeros((24, 24), dtype=np.int32)
        grid[5:19, 8:13] = 1
        target = SurfaceMaterialCandidate(
            material_id="warmup_target",
            name="Warmup Target",
            palette=((0.12, 0.18, 0.72, 1.0), (0.65, 0.72, 0.95, 1.0), (0.9, 0.95, 1.0, 1.0)),
        )
        candidates = (
            target,
            SurfaceMaterialCandidate(
                material_id="warmup_negative",
                name="Warmup Negative",
                palette=((0.72, 0.18, 0.12, 1.0), (0.95, 0.72, 0.65, 1.0), (1.0, 0.95, 0.9, 1.0)),
            ),
        )

        total_start = time.perf_counter()
        plan = self.contour_to_textured_lathe_mesh(
            grid,
            color=1,
            pad=1,
            segments=12,
            target_material=target,
            candidates=candidates,
            negative_materials=(candidates[1],),
            preview_size=32,
        )
        total_ms = (time.perf_counter() - total_start) * 1000.0
        report: dict[str, float | int | str] = {
            "status": "ready",
            "pid": current_pid,
            "total_warmup_ms": total_ms,
            "warm_vertex_count": int(plan.mesh.vertices.shape[0]),
            "warm_triangle_count": int(plan.mesh.indices.shape[0]),
            "warm_material_id": plan.selected_material.material_id,
            "warm_preview_size": int(plan.material_preview.shape[0]),
        }
        self.__class__._WARMED_PID = current_pid
        self.__class__._WARMUP_REPORT = dict(report)
        return report

    def _normalize_stops(
        self,
        stops: Sequence[Sequence[float]],
    ) -> list[tuple[float, float, float, float, float]]:
        arr = np.ascontiguousarray(np.asarray(stops, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[1] != 5:
            raise ValueError("gradient_stops must have shape [n,5] as [pos,r,g,b,a]")
        order = np.argsort(arr[:, 0], kind="stable")
        arr = arr[order]
        arr[:, 0] = np.clip(arr[:, 0], 0.0, 1.0)
        arr[:, 1:] = np.clip(arr[:, 1:], 0.0, 1.0)
        return [tuple(float(v) for v in row.tolist()) for row in arr]

__all__ = [
    "MaterialSelection",
    "ProceduralMaterialBridge",
    "SurfaceMaterialCandidate",
    "SurfaceMaterialPlan",
]
