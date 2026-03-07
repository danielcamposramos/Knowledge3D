"""Deterministic temporal/video bridge over existing procedural surfaces.

The bridge stays honest about the current substrate:
- frame generation uses the existing procedural video generator
- temporal deltas use the sovereign temporal reasoning PTX bridge
- temporal coherence uses the sovereign world-model PTX bridge
- optional frame encoding uses the sovereign ternary video codec

This keeps temporal/video as a first-class executable route in the Tool galaxy
without inventing a second orchestration model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from knowledge3d.cranium.actions import ActionBuffer, ActionType
from knowledge3d.cranium.bridges.procedural_material_bridge import SurfaceMaterialCandidate, SurfaceMaterialPlan
from knowledge3d.cranium.bridges.sovereign_bridges import TemporalReasoning, WorldModelBridge
from knowledge3d.cranium.codecs.sovereign_ternary_video_codec import SovereignTernaryVideoCodec
from knowledge3d.cranium.ptx_runtime.drawing_effects import DrawingEffects
from knowledge3d.cranium.ptx_runtime.temporal_frame_kernels import TemporalFrameKernels
from knowledge3d.cranium.ptx_runtime.math_core_pool import get_global_math_core_pool
from knowledge3d.cranium.ptx_runtime.temporal_preset_kernels import TemporalPresetKernels
from knowledge3d.cranium.ternary import TernaryTensor, TernaryVector
from knowledge3d.knowledgeverse.execution_events import ternary_quantize_quality


@dataclass(frozen=True)
class TemporalPreviewPlan:
    surface_plan: SurfaceMaterialPlan
    frames: np.ndarray
    frame_features: np.ndarray
    temporal_deltas: np.ndarray
    coherence_scores: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class TemporalSceneLayer:
    layer_id: str
    preview_plan: TemporalPreviewPlan
    x: int = 0
    y: int = 0
    opacity: float = 1.0
    z_index: int = 0
    visible: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TemporalScenePlan:
    layers: tuple[TemporalSceneLayer, ...]
    frames: np.ndarray
    frame_features: np.ndarray
    temporal_deltas: np.ndarray
    coherence_scores: np.ndarray
    metadata: dict[str, Any]


def _resolve_temporal_math_core_plan(preferred_tier: int, work_items: int) -> dict[str, Any]:
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


class ProceduralTemporalBridge:
    """Always-on temporal/video bridge over procedural surface plans."""

    _TIMELINE_PRESETS: dict[str, dict[str, Any]] = {
        "ui_idle": {
            "domain": "ui",
            "frame_count": 6,
            "time_span": 0.8,
            "feature_grid": 8,
            "curve": "ease_in_out",
            "effect": "ui_idle",
            "loop": True,
        },
        "ui_focus": {
            "domain": "ui",
            "frame_count": 8,
            "time_span": 0.65,
            "feature_grid": 8,
            "curve": "pulse",
            "effect": "ui_focus",
            "loop": True,
        },
        "world_breathe": {
            "domain": "world",
            "frame_count": 10,
            "time_span": 1.4,
            "feature_grid": 10,
            "curve": "ease_in_out",
            "effect": "world_breathe",
            "loop": True,
        },
        "world_orbit": {
            "domain": "world",
            "frame_count": 12,
            "time_span": 1.6,
            "feature_grid": 10,
            "curve": "orbit",
            "effect": "world_orbit",
            "loop": True,
        },
    }

    def __init__(self) -> None:
        self.temporal_reasoning = TemporalReasoning()
        self.world_model = WorldModelBridge()
        self.effects = DrawingEffects()
        self.frame_kernels = TemporalFrameKernels()
        self.preset_kernels = TemporalPresetKernels()
        self._video_codec_cache: dict[tuple[int, int, float], SovereignTernaryVideoCodec] = {}
        self._preview_plan_cache: dict[tuple[Any, ...], TemporalPreviewPlan] = {}
        self._surface_scene_cache: dict[tuple[Any, ...], TemporalScenePlan] = {}
        self._house_room_scene_cache: dict[tuple[Any, ...], TemporalScenePlan] = {}
        self._house_tour_scene_cache: dict[tuple[Any, ...], TemporalScenePlan] = {}

    def surface_material_to_temporal_preview(
        self,
        surface_plan: SurfaceMaterialPlan,
        *,
        frame_count: int = 4,
        time_span: float = 1.0,
        feature_grid: int = 8,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        timeline_id: str | None = None,
    ) -> TemporalPreviewPlan:
        frame_count = max(1, int(frame_count))
        feature_grid = max(2, int(feature_grid))
        span = max(0.0, float(time_span))

        preview = np.asarray(surface_plan.material_preview, dtype=np.float32)
        if preview.ndim != 3 or preview.shape[2] != 4:
            raise ValueError("surface_plan.material_preview must have shape [H,W,4]")

        height = int(preview.shape[0])
        width = int(preview.shape[1])
        seed = self._derive_seed(surface_plan)
        cache_key = self._preview_plan_cache_key(
            surface_plan=surface_plan,
            preset_key="__linear__",
            frame_count=frame_count,
            time_span=span,
            feature_grid=feature_grid,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            timeline_id=timeline_id,
            seed=seed,
            width=width,
            height=height,
        )
        cached = self._preview_plan_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            return cached
        time_points = self._time_points(frame_count=frame_count, time_span=span, curve="linear")
        frames = self._generate_frames(seed, width=width, height=height, time_points=time_points)
        plan = self._build_preview_plan(
            surface_plan=surface_plan,
            frames=frames,
            feature_grid=feature_grid,
            time_span=span,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            timeline_id=timeline_id,
            extra_metadata={},
        )
        if cache_key is not None:
            self._preview_plan_cache[cache_key] = plan
        return plan

    def surface_material_to_timeline_preset(
        self,
        surface_plan: SurfaceMaterialPlan,
        *,
        timeline_preset: str | None = None,
        frame_count: int | None = None,
        time_span: float | None = None,
        feature_grid: int | None = None,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        timeline_id: str | None = None,
    ) -> TemporalPreviewPlan:
        preset_key = str(timeline_preset or "").strip().lower()
        if not preset_key:
            return self.surface_material_to_temporal_preview(
                surface_plan,
                frame_count=int(frame_count) if frame_count is not None else 4,
                time_span=float(time_span) if time_span is not None else 1.0,
                feature_grid=int(feature_grid) if feature_grid is not None else 8,
                encode_frames=encode_frames,
                codec_threshold=codec_threshold,
                timeline_id=timeline_id,
            )
        preset = dict(self._TIMELINE_PRESETS.get(preset_key, {}))
        if not preset:
            raise ValueError(f"unsupported timeline preset: {timeline_preset}")

        preview = np.asarray(surface_plan.material_preview, dtype=np.float32)
        if preview.ndim != 3 or preview.shape[2] != 4:
            raise ValueError("surface_plan.material_preview must have shape [H,W,4]")

        resolved_frame_count = int(frame_count if frame_count is not None else preset["frame_count"])
        resolved_time_span = float(time_span if time_span is not None else preset["time_span"])
        resolved_feature_grid = int(feature_grid if feature_grid is not None else preset["feature_grid"])
        height = int(preview.shape[0])
        width = int(preview.shape[1])
        seed = self._derive_seed(surface_plan)
        cache_key = self._preview_plan_cache_key(
            surface_plan=surface_plan,
            preset_key=preset_key,
            frame_count=resolved_frame_count,
            time_span=resolved_time_span,
            feature_grid=resolved_feature_grid,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            timeline_id=timeline_id,
            seed=seed,
            width=width,
            height=height,
        )
        cached = self._preview_plan_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            return cached
        time_points = self._time_points(
            frame_count=resolved_frame_count,
            time_span=resolved_time_span,
            curve=str(preset["curve"]),
        )
        base_frames = self._generate_frames(seed, width=width, height=height, time_points=time_points)
        frames = self._apply_timeline_preset(
            frames=base_frames,
            surface_plan=surface_plan,
            preset_key=preset_key,
            time_points=time_points,
        )
        plan = self._build_preview_plan(
            surface_plan=surface_plan,
            frames=frames,
            feature_grid=resolved_feature_grid,
            time_span=resolved_time_span,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            timeline_id=timeline_id,
            extra_metadata={
                "timeline_preset": preset_key,
                "timeline_domain": str(preset["domain"]),
                "timeline_curve": str(preset["curve"]),
                "timeline_effect": str(preset["effect"]),
                "timeline_loop": bool(preset["loop"]),
            },
        )
        if cache_key is not None:
            self._preview_plan_cache[cache_key] = plan
        return plan

    def compose_scene_timeline(
        self,
        layers: list[TemporalSceneLayer] | tuple[TemporalSceneLayer, ...],
        *,
        canvas_width: int | None = None,
        canvas_height: int | None = None,
        background_rgba: np.ndarray | None = None,
        feature_grid: int = 8,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        scene_id: str | None = None,
        scene_layout: str = "overlay",
    ) -> TemporalScenePlan:
        normalized_layers = self._normalize_scene_layers(layers)
        if not normalized_layers:
            raise ValueError("at least one visible temporal scene layer is required")
        quality_layers = self._apply_quality_weighting(normalized_layers)

        frame_count = max(int(layer.preview_plan.frames.shape[0]) for layer in quality_layers)
        default_height = max(
            int(layer.preview_plan.frames.shape[1] + max(0, layer.y))
            for layer in quality_layers
        )
        default_width = max(
            int(layer.preview_plan.frames.shape[2] + max(0, layer.x))
            for layer in quality_layers
        )
        height = max(1, int(canvas_height if canvas_height is not None else default_height))
        width = max(1, int(canvas_width if canvas_width is not None else default_width))
        base_rgba = self._scene_background_rgba(background_rgba, height=height, width=width)

        ordered_layers = tuple(
            sorted(
                quality_layers,
                key=lambda row: (
                    int((row.metadata or {}).get("quality_sort_bias", 0)),
                    int(row.z_index),
                    str(row.layer_id),
                ),
            )
        )
        scene_math_core_plan = _resolve_temporal_math_core_plan(
            2,
            frame_count * max(1, len(ordered_layers)) * max(1, width * height),
        )
        delta_plan = _resolve_temporal_math_core_plan(2, int(frame_count * feature_grid * feature_grid * 4))
        coherence_plan = _resolve_temporal_math_core_plan(3, int(frame_count * feature_grid * feature_grid * 4))

        frame_rows: list[np.ndarray] = []
        for frame_idx in range(frame_count):
            composed = base_rgba.copy()
            for layer in ordered_layers:
                frame = layer.preview_plan.frames[frame_idx % int(layer.preview_plan.frames.shape[0])]
                layer_rgba = self._place_layer_rgba(
                    frame=frame,
                    opacity=float(layer.opacity),
                    canvas_height=height,
                    canvas_width=width,
                    x=int(layer.x),
                    y=int(layer.y),
                )
                composed = self.effects.alpha_over_rgba(composed, layer_rgba)
            frame_rows.append(np.clip(np.rint(composed * 255.0), 0.0, 255.0).astype(np.uint8, copy=False))

        frames = np.stack(frame_rows, axis=0).astype(np.uint8, copy=False)
        frame_features = np.stack(
            [self._scene_frame_to_features(frame, feature_grid=feature_grid) for frame in frames],
            axis=0,
        ).astype(np.float32, copy=False)
        temporal_deltas = np.asarray(
            self.temporal_reasoning.compute_deltas(frame_features.astype(np.float32, copy=False)),
            dtype=np.float32,
        )
        coherence_scores = np.asarray(
            self.world_model.compute_temporal_coherence(
                frame_features.reshape(-1).astype(np.float32, copy=False),
                n_frames=int(frame_features.shape[0]),
                feature_dim=int(frame_features.shape[1]),
            ),
            dtype=np.float32,
        )
        coherence_scores = np.nan_to_num(
            coherence_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

        encoded_frames: list[dict[str, Any]] = []
        if encode_frames and width % 8 == 0 and height % 8 == 0:
            codec = self._video_codec_for(width=width, height=height, threshold=codec_threshold)
            prefix = str(scene_id or self._default_scene_id(ordered_layers))
            for idx, frame in enumerate(frames):
                frame_id = f"{prefix}_frame_{idx:03d}"
                meta = codec.encode_frame_array(frame_id, frame[..., :3])
                encoded_frames.append(
                    {
                        "frame_id": frame_id,
                        "seed_rpn": str(meta.get("seed_rpn", "")),
                        "math_core_plan": dict(meta.get("math_core_plan", {})),
                    }
                )

        timeline_domains = sorted(
            {
                str(layer.preview_plan.metadata.get("timeline_domain", "")).strip()
                for layer in ordered_layers
                if str(layer.preview_plan.metadata.get("timeline_domain", "")).strip()
            }
        )
        metadata = {
            "frame_count": int(frames.shape[0]),
            "frame_shape": [height, width, 4],
            "feature_grid": int(feature_grid),
            "feature_dim": int(frame_features.shape[1]),
            "layer_count": len(ordered_layers),
            "layer_ids": [str(layer.layer_id) for layer in ordered_layers],
            "layer_offsets": [[int(layer.x), int(layer.y)] for layer in ordered_layers],
            "layer_opacity": [float(layer.opacity) for layer in ordered_layers],
            "layer_quality_signals": [
                float((layer.metadata or {}).get("quality_signal", layer.preview_plan.metadata.get("quality_signal", 0.65)))
                for layer in ordered_layers
            ],
            "layer_ternary_quality": [
                int((layer.metadata or {}).get("ternary_quality", layer.preview_plan.metadata.get("ternary_quality", 0)))
                for layer in ordered_layers
            ],
            "scene_layout": str(scene_layout),
            "timeline_domains": timeline_domains,
            "scene_domain": timeline_domains[0] if len(timeline_domains) == 1 else "mixed",
            "overall_coherence": float(np.mean(coherence_scores)) if coherence_scores.size else 0.0,
            "coherence_variance": float(np.var(coherence_scores)) if coherence_scores.size else 0.0,
            "temporal_delta_mean_abs": float(np.mean(np.abs(temporal_deltas))) if temporal_deltas.size else 0.0,
            "scene_math_core_plan": scene_math_core_plan,
            "delta_math_core_plan": delta_plan,
            "coherence_math_core_plan": coherence_plan,
            "codec_enabled": bool(encoded_frames),
            "encoded_frames": encoded_frames,
            "quality_weighting": "ternary_contrastive",
            "house_room_preset": self._house_room_preset(
                float(np.mean([
                    float((layer.metadata or {}).get("quality_signal", layer.preview_plan.metadata.get("quality_signal", 0.65)))
                    for layer in ordered_layers
                ]))
                if ordered_layers else 0.0
            ),
        }
        return TemporalScenePlan(
            layers=ordered_layers,
            frames=frames,
            frame_features=frame_features,
            temporal_deltas=temporal_deltas,
            coherence_scores=coherence_scores,
            metadata=metadata,
        )

    def surface_materials_to_scene_timeline(
        self,
        surface_plans: list[SurfaceMaterialPlan] | tuple[SurfaceMaterialPlan, ...],
        *,
        timeline_preset: str | None = None,
        frame_count: int | None = None,
        time_span: float | None = None,
        feature_grid: int | None = None,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        scene_layout: str = "overlay",
        scene_id: str | None = None,
        layer_opacity: list[float] | tuple[float, ...] | None = None,
    ) -> TemporalScenePlan:
        plans = list(surface_plans)
        if not plans:
            raise ValueError("at least one surface plan is required")

        layout = str(scene_layout or "overlay").strip().lower()
        cache_key = self._surface_scene_cache_key(
            plans=plans,
            timeline_preset=timeline_preset,
            frame_count=frame_count,
            time_span=time_span,
            feature_grid=feature_grid,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_layout=layout,
            scene_id=scene_id,
            layer_opacity=layer_opacity,
        )
        cached = self._surface_scene_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            return cached
        offsets = self._scene_layout_offsets(plans, layout)
        opacity_values = list(layer_opacity or [])
        layers: list[TemporalSceneLayer] = []
        for idx, surface_plan in enumerate(plans):
            preview = self.surface_material_to_timeline_preset(
                surface_plan,
                timeline_preset=timeline_preset,
                frame_count=frame_count,
                time_span=time_span,
                feature_grid=feature_grid,
                encode_frames=False,
                codec_threshold=codec_threshold,
                timeline_id=None if scene_id is None else f"{scene_id}_layer_{idx:02d}",
            )
            x, y = offsets[idx]
            opacity = float(opacity_values[idx]) if idx < len(opacity_values) else 1.0
            layers.append(
                TemporalSceneLayer(
                    layer_id=f"scene_layer_{idx:02d}_{surface_plan.selected_material.material_id}",
                    preview_plan=preview,
                    x=int(x),
                    y=int(y),
                    opacity=opacity,
                    z_index=idx,
                    metadata={
                        "source_material_id": str(surface_plan.selected_material.material_id),
                        "source_mesh_kind": str(surface_plan.mesh.metadata.get("mesh_kind", "")),
                        "quality_signal": float(
                            surface_plan.metadata.get(
                                "quality_signal",
                                preview.metadata.get("overall_coherence", 0.65),
                            )
                        ),
                        "ternary_quality": int(
                            surface_plan.metadata.get(
                                "ternary_quality",
                                ternary_quantize_quality(
                                    float(
                                        surface_plan.metadata.get(
                                            "quality_signal",
                                            preview.metadata.get("overall_coherence", 0.65),
                                        )
                                    )
                                ),
                            )
                        ),
                    },
                )
            )

        scene = self.compose_scene_timeline(
            layers,
            feature_grid=int(feature_grid or 8),
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
            scene_layout=layout,
        )
        if cache_key is not None:
            self._surface_scene_cache[cache_key] = scene
        return scene

    def replay_journal_to_scene_timeline(
        self,
        *,
        journal_path: str | Path | None = None,
        replay_entries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        max_events: int = 8,
        frame_count: int = 6,
        feature_grid: int = 8,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        scene_id: str | None = None,
        scene_layout: str = "golden_orbit",
    ) -> TemporalScenePlan:
        entries = self._load_replay_entries(journal_path=journal_path, replay_entries=replay_entries)
        if not entries:
            raise ValueError("replay journal is empty")
        selected = entries[-max(1, int(max_events)) :]
        scene_grammars = self._detect_scene_grammars(entries)
        surface_plans = [self._event_to_surface_plan(entry) for entry in selected]
        offsets = self._scene_layout_offsets(surface_plans, str(scene_layout or "golden_orbit"))
        layers: list[TemporalSceneLayer] = []
        for idx, (entry, surface_plan) in enumerate(zip(selected, surface_plans)):
            preview = self.surface_material_to_timeline_preset(
                surface_plan,
                timeline_preset=self._action_timeline_preset(str(entry.get("action_type", ""))),
                frame_count=int(frame_count),
                feature_grid=int(feature_grid),
                encode_frames=False,
                codec_threshold=codec_threshold,
                timeline_id=None if scene_id is None else f"{scene_id}_replay_{idx:02d}",
            )
            x, y = offsets[idx]
            layers.append(
                TemporalSceneLayer(
                    layer_id=f"replay_layer_{idx:02d}_{str(entry.get('action_type', 'NO_ACTION')).lower()}",
                    preview_plan=preview,
                    x=int(x),
                    y=int(y),
                    opacity=float(np.clip(float(entry.get("final_confidence", entry.get("raw_confidence", 0.75)) or 0.75), 0.3, 1.0)),
                    z_index=idx,
                    metadata={
                        "replay_entry": dict(entry),
                        "quality_signal": float(
                            entry.get(
                                "quality_signal",
                                entry.get("final_confidence", entry.get("raw_confidence", 0.65)) or 0.65,
                            )
                        ),
                        "ternary_quality": int(
                            entry.get(
                                "ternary_quality",
                                ternary_quantize_quality(
                                    float(
                                        entry.get(
                                            "quality_signal",
                                            entry.get("final_confidence", entry.get("raw_confidence", 0.65)) or 0.65,
                                        )
                                    )
                                ),
                            )
                        ),
                    },
                )
            )

        scene = self.compose_scene_timeline(
            layers,
            feature_grid=int(feature_grid),
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
            scene_layout=str(scene_layout or "golden_orbit"),
        )
        metadata = dict(scene.metadata)
        metadata.update(
            {
                "replay_count": len(selected),
                "replay_action_types": [str(entry.get("action_type", "")).strip() for entry in selected],
                "journal_path": str(journal_path) if journal_path is not None else None,
                "replay_source": "journal",
                "scene_grammars": scene_grammars,
                "house_room_preset": self._house_room_preset(
                    float(np.mean([
                        float(entry.get("quality_signal", entry.get("final_confidence", entry.get("raw_confidence", 0.65)) or 0.65))
                        for entry in selected
                    ]))
                    if selected else 0.0
                ),
            }
        )
        return TemporalScenePlan(
            layers=scene.layers,
            frames=scene.frames,
            frame_features=scene.frame_features,
            temporal_deltas=scene.temporal_deltas,
            coherence_scores=scene.coherence_scores,
            metadata=metadata,
        )

    def action_buffers_to_scene_timeline(
        self,
        action_buffers: list[ActionBuffer] | tuple[ActionBuffer, ...],
        *,
        max_events: int = 8,
        frame_count: int = 6,
        feature_grid: int = 8,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        scene_id: str | None = None,
        scene_layout: str = "golden_orbit",
    ) -> TemporalScenePlan:
        entries = self._action_buffers_to_entries(action_buffers)
        scene = self.replay_journal_to_scene_timeline(
            replay_entries=entries,
            max_events=max_events,
            frame_count=frame_count,
            feature_grid=feature_grid,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
            scene_layout=scene_layout,
        )
        metadata = dict(scene.metadata)
        metadata["replay_source"] = "action_buffers"
        return TemporalScenePlan(
            layers=scene.layers,
            frames=scene.frames,
            frame_features=scene.frame_features,
            temporal_deltas=scene.temporal_deltas,
            coherence_scores=scene.coherence_scores,
            metadata=metadata,
        )

    def execution_events_to_house_room_scene(
        self,
        *,
        event_log_path: str | Path | None = None,
        execution_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        room_preset: str = "house_library",
        max_events: int = 8,
        feature_grid: int = 8,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        scene_id: str | None = None,
    ) -> TemporalScenePlan:
        entries = self._load_execution_entries(
            event_log_path=event_log_path,
            execution_events=execution_events,
        )
        behavior = self._room_scene_behavior(room_preset)
        cache_key = self._house_room_scene_cache_key(
            entries=entries,
            room_preset=str(room_preset),
            max_events=max_events,
            feature_grid=feature_grid,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
        )
        cached = self._house_room_scene_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            return cached
        selected = self._select_house_room_entries(
            entries,
            room_preset=str(room_preset),
            max_events=max_events,
        )
        if not selected:
            raise ValueError(f"no execution events available for room preset: {room_preset}")

        surface_plans = [self._event_to_surface_plan(entry) for entry in selected]
        offsets = self._scene_layout_offsets(surface_plans, str(behavior["scene_layout"]))
        layers: list[TemporalSceneLayer] = []
        for idx, (entry, surface_plan) in enumerate(zip(selected, surface_plans)):
            quality = float(entry.get("quality_signal", 0.65) or 0.65)
            ternary_quality = int(entry.get("ternary_quality", ternary_quantize_quality(quality)))
            curiosity = self._event_curiosity_signal(entry)
            preview = self.surface_material_to_timeline_preset(
                surface_plan,
                timeline_preset=str(behavior["timeline_preset"]),
                frame_count=int(behavior["frame_count"]),
                feature_grid=int(feature_grid),
                encode_frames=False,
                codec_threshold=codec_threshold,
                timeline_id=None if scene_id is None else f"{scene_id}_{room_preset}_{idx:02d}",
            )
            x, y = offsets[idx]
            opacity = float(np.clip(float(behavior["opacity_base"]) + 0.18 * quality + 0.12 * curiosity, 0.28, 1.0))
            z_index = idx + int(self._room_branch_depth(room_preset, quality, curiosity))
            layers.append(
                TemporalSceneLayer(
                    layer_id=f"{room_preset}_{idx:02d}_{self._event_token(entry).lower()}",
                    preview_plan=preview,
                    x=int(x),
                    y=int(y),
                    opacity=opacity,
                    z_index=z_index,
                    metadata={
                        "room_preset": str(room_preset),
                        "execution_event": dict(entry),
                        "quality_signal": quality,
                        "ternary_quality": ternary_quality,
                        "curiosity_signal": float(curiosity),
                        "fractal_branch_depth": int(self._room_branch_depth(room_preset, quality, curiosity)),
                        "contrastive_lesson": bool(int(entry.get("outcome", 0) or 0) < 0),
                    },
                )
            )

        scene = self.compose_scene_timeline(
            layers,
            feature_grid=int(feature_grid),
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
            scene_layout=str(behavior["scene_layout"]),
        )
        metadata = dict(scene.metadata)
        metadata.update(
            {
                "replay_source": "execution_events",
                "house_room_preset": str(room_preset),
                "room_behavior": dict(behavior),
                "event_count": len(selected),
                "source_event_count": len(entries),
                "event_tokens": [self._event_token(entry) for entry in selected],
            }
        )
        plan = TemporalScenePlan(
            layers=scene.layers,
            frames=scene.frames,
            frame_features=scene.frame_features,
            temporal_deltas=scene.temporal_deltas,
            coherence_scores=scene.coherence_scores,
            metadata=metadata,
        )
        if cache_key is not None:
            self._house_room_scene_cache[cache_key] = plan
        return plan

    def execution_events_to_house_tour_scene(
        self,
        *,
        event_log_path: str | Path | None = None,
        execution_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        max_events_per_room: int = 6,
        feature_grid: int = 8,
        encode_frames: bool = True,
        codec_threshold: float = 0.2,
        scene_id: str | None = None,
    ) -> TemporalScenePlan:
        entries = self._load_execution_entries(
            event_log_path=event_log_path,
            execution_events=execution_events,
        )
        if not entries:
            raise ValueError("execution event log is empty")
        cache_key = self._house_tour_scene_cache_key(
            entries=entries,
            max_events_per_room=max_events_per_room,
            feature_grid=feature_grid,
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
        )
        cached = self._house_tour_scene_cache.get(cache_key) if cache_key is not None else None
        if cached is not None:
            return cached

        room_specs = (
            ("house_library", 1.0, 0.0),
            ("house_garden", 0.88, 14.0),
            ("house_museum", 0.74, 28.0),
        )
        room_layers: list[TemporalSceneLayer] = []
        for z_index, (room_preset, opacity, y_offset) in enumerate(room_specs):
            try:
                room_scene = self.execution_events_to_house_room_scene(
                    execution_events=entries,
                    room_preset=room_preset,
                    max_events=max_events_per_room,
                    feature_grid=feature_grid,
                    encode_frames=False,
                    codec_threshold=codec_threshold,
                    scene_id=None if scene_id is None else f"{scene_id}_{room_preset}",
                )
            except ValueError:
                continue
            room_layers.append(
                self._scene_plan_to_layer(
                    room_scene,
                    layer_id=f"{room_preset}_tour_layer",
                    x=0,
                    y=int(y_offset),
                    opacity=float(opacity),
                    z_index=z_index,
                    metadata={
                        "room_preset": room_preset,
                        "tour_role": room_preset.replace("house_", ""),
                        "quality_signal": float(room_scene.metadata.get("overall_coherence", 0.65)),
                        "ternary_quality": int(
                            ternary_quantize_quality(float(room_scene.metadata.get("overall_coherence", 0.65)))
                        ),
                    },
                )
            )

        if not room_layers:
            raise ValueError("house tour requires at least one populated room layer")

        scene = self.compose_scene_timeline(
            room_layers,
            feature_grid=int(feature_grid),
            encode_frames=encode_frames,
            codec_threshold=codec_threshold,
            scene_id=scene_id,
            scene_layout="vertical_strip",
        )
        metadata = dict(scene.metadata)
        metadata.update(
            {
                "replay_source": "execution_events",
                "house_room_preset": "house_tour",
                "tour_rooms": [str((layer.metadata or {}).get("room_preset", "")) for layer in room_layers],
                "tour_layer_count": len(room_layers),
            }
        )
        plan = TemporalScenePlan(
            layers=scene.layers,
            frames=scene.frames,
            frame_features=scene.frame_features,
            temporal_deltas=scene.temporal_deltas,
            coherence_scores=scene.coherence_scores,
            metadata=metadata,
        )
        if cache_key is not None:
            self._house_tour_scene_cache[cache_key] = plan
        return plan

    def _build_preview_plan(
        self,
        *,
        surface_plan: SurfaceMaterialPlan,
        frames: np.ndarray,
        feature_grid: int,
        time_span: float,
        encode_frames: bool,
        codec_threshold: float,
        timeline_id: str | None,
        extra_metadata: dict[str, Any],
    ) -> TemporalPreviewPlan:
        frame_count = int(frames.shape[0])
        frame_features = np.stack(
            [self._frame_to_features(frame, feature_grid=feature_grid) for frame in frames],
            axis=0,
        ).astype(np.float32, copy=False)

        temporal_plan = _resolve_temporal_math_core_plan(2, int(frame_features.size))
        delta_plan = _resolve_temporal_math_core_plan(2, int(frame_features.size))
        coherence_plan = _resolve_temporal_math_core_plan(3, int(frame_features.size))

        temporal_deltas = np.asarray(
            self.temporal_reasoning.compute_deltas(frame_features.astype(np.float32, copy=False)),
            dtype=np.float32,
        )
        coherence_scores = np.asarray(
            self.world_model.compute_temporal_coherence(
                frame_features.reshape(-1).astype(np.float32, copy=False),
                n_frames=frame_count,
                feature_dim=int(frame_features.shape[1]),
            ),
            dtype=np.float32,
        )
        coherence_scores = np.nan_to_num(
            coherence_scores,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

        height = int(frames.shape[1])
        width = int(frames.shape[2])
        encoded_frames: list[dict[str, Any]] = []
        if encode_frames and width % 8 == 0 and height % 8 == 0:
            codec = self._video_codec_for(width=width, height=height, threshold=codec_threshold)
            prefix = str(timeline_id or self._default_timeline_id(surface_plan))
            for idx, frame in enumerate(frames):
                frame_id = f"{prefix}_frame_{idx:03d}"
                meta = codec.encode_frame_array(frame_id, frame)
                encoded_frames.append(
                    {
                        "frame_id": frame_id,
                        "seed_rpn": str(meta.get("seed_rpn", "")),
                        "math_core_plan": dict(meta.get("math_core_plan", {})),
                    }
                )

        metadata = {
            "frame_count": frame_count,
            "frame_shape": [height, width, 3],
            "feature_grid": int(feature_grid),
            "feature_dim": int(frame_features.shape[1]),
            "time_span": float(time_span),
            "overall_coherence": float(np.mean(coherence_scores)) if coherence_scores.size else 0.0,
            "coherence_variance": float(np.var(coherence_scores)) if coherence_scores.size else 0.0,
            "temporal_delta_mean_abs": float(np.mean(np.abs(temporal_deltas))) if temporal_deltas.size else 0.0,
            "temporal_math_core_plan": temporal_plan,
            "delta_math_core_plan": delta_plan,
            "coherence_math_core_plan": coherence_plan,
            "source_material_id": str(surface_plan.selected_material.material_id),
            "source_projection_strategy": str(surface_plan.metadata.get("projection_strategy", "")),
            "source_mesh_kind": str(surface_plan.mesh.metadata.get("mesh_kind", "")),
            "encoded_frames": encoded_frames,
            "codec_enabled": bool(encoded_frames),
        }
        metadata.update(dict(extra_metadata))
        if "clip_id" in surface_plan.metadata:
            metadata["clip_id"] = str(surface_plan.metadata.get("clip_id", ""))

        return TemporalPreviewPlan(
            surface_plan=surface_plan,
            frames=frames,
            frame_features=frame_features,
            temporal_deltas=temporal_deltas,
            coherence_scores=coherence_scores,
            metadata=metadata,
        )

    def _generate_frames(
        self,
        seed: np.ndarray,
        *,
        width: int,
        height: int,
        time_points: np.ndarray,
    ) -> np.ndarray:
        return self.frame_kernels.generate_frames(
            seed,
            width=width,
            height=height,
            time_points=np.asarray(time_points, dtype=np.float32),
        ).astype(np.uint8, copy=False)

    def _time_points(
        self,
        *,
        frame_count: int,
        time_span: float,
        curve: str,
    ) -> np.ndarray:
        count = max(1, int(frame_count))
        span = max(0.0, float(time_span))
        base = np.linspace(0.0, 1.0, count, dtype=np.float32)
        curve_name = str(curve).strip().lower()
        if curve_name == "ease_in_out":
            shaped = (0.5 - 0.5 * np.cos(base * np.pi)).astype(np.float32, copy=False)
        elif curve_name == "pulse":
            shaped = np.square(np.sin(base * np.pi)).astype(np.float32, copy=False)
        elif curve_name == "orbit":
            shaped = np.mod(base * 1.25, 1.0).astype(np.float32, copy=False)
        else:
            shaped = base
        points = shaped * span
        if span > 1.0:
            points = np.mod(points, 1.0)
        return points.astype(np.float32, copy=False)

    def _video_codec_for(self, *, width: int, height: int, threshold: float) -> SovereignTernaryVideoCodec:
        key = (int(width), int(height), float(threshold))
        codec = self._video_codec_cache.get(key)
        if codec is None:
            codec = SovereignTernaryVideoCodec(width=key[0], height=key[1], threshold=key[2])
            self._video_codec_cache[key] = codec
        return codec

    def _apply_timeline_preset(
        self,
        *,
        frames: np.ndarray,
        surface_plan: SurfaceMaterialPlan,
        preset_key: str,
        time_points: np.ndarray,
    ) -> np.ndarray:
        preview = np.asarray(surface_plan.material_preview, dtype=np.float32)
        overlay_rgba = self._resize_rgba_to_match(
            preview,
            height=int(frames.shape[1]),
            width=int(frames.shape[2]),
        )
        return self.preset_kernels.apply_preset(
            frames,
            overlay_rgba,
            preset_key=preset_key,
            time_points=np.asarray(time_points, dtype=np.float32),
            projection_weights=np.asarray(surface_plan.projection_weights, dtype=np.float32),
            normal_hint=np.asarray(surface_plan.normal_hint, dtype=np.float32),
        )

    def _resize_rgba_to_match(self, rgba: np.ndarray, *, height: int, width: int) -> np.ndarray:
        src = np.asarray(rgba, dtype=np.float32)
        if src.shape[0] == height and src.shape[1] == width:
            return src.astype(np.float32, copy=False)
        y_idx = np.linspace(0, src.shape[0] - 1, height, dtype=np.int32)
        x_idx = np.linspace(0, src.shape[1] - 1, width, dtype=np.int32)
        return src[np.ix_(y_idx, x_idx)].astype(np.float32, copy=False)

    def _preview_plan_cache_key(
        self,
        *,
        surface_plan: SurfaceMaterialPlan,
        preset_key: str,
        frame_count: int,
        time_span: float,
        feature_grid: int,
        encode_frames: bool,
        codec_threshold: float,
        timeline_id: str | None,
        seed: np.ndarray,
        width: int,
        height: int,
    ) -> tuple[Any, ...] | None:
        if encode_frames and timeline_id is not None:
            return None
        seed_hash = hashlib.sha1(np.ascontiguousarray(seed, dtype=np.float32).tobytes()).hexdigest()
        return (
            str(surface_plan.selected_material.material_id),
            str(surface_plan.mesh.metadata.get("mesh_kind", "")),
            seed_hash,
            str(preset_key),
            int(frame_count),
            round(float(time_span), 6),
            int(feature_grid),
            bool(encode_frames),
            round(float(codec_threshold), 6),
            int(width),
            int(height),
        )

    def _house_room_scene_cache_key(
        self,
        *,
        entries: list[dict[str, Any]],
        room_preset: str,
        max_events: int,
        feature_grid: int,
        encode_frames: bool,
        codec_threshold: float,
        scene_id: str | None,
    ) -> tuple[Any, ...] | None:
        if encode_frames and scene_id is not None:
            return None
        return (
            "house_room",
            self._execution_entries_digest(entries),
            str(room_preset or "house_library").strip().lower(),
            max(1, int(max_events)),
            max(2, int(feature_grid)),
            bool(encode_frames),
            round(float(codec_threshold), 6),
        )

    def _house_tour_scene_cache_key(
        self,
        *,
        entries: list[dict[str, Any]],
        max_events_per_room: int,
        feature_grid: int,
        encode_frames: bool,
        codec_threshold: float,
        scene_id: str | None,
    ) -> tuple[Any, ...] | None:
        if encode_frames and scene_id is not None:
            return None
        return (
            "house_tour",
            self._execution_entries_digest(entries),
            max(1, int(max_events_per_room)),
            max(2, int(feature_grid)),
            bool(encode_frames),
            round(float(codec_threshold), 6),
        )

    def _surface_scene_cache_key(
        self,
        *,
        plans: list[SurfaceMaterialPlan],
        timeline_preset: str | None,
        frame_count: int | None,
        time_span: float | None,
        feature_grid: int | None,
        encode_frames: bool,
        codec_threshold: float,
        scene_layout: str,
        scene_id: str | None,
        layer_opacity: list[float] | tuple[float, ...] | None,
    ) -> tuple[Any, ...] | None:
        if encode_frames and scene_id is not None:
            return None
        return (
            "surface_scene",
            tuple(id(plan) for plan in plans),
            str(timeline_preset or "").strip().lower(),
            None if frame_count is None else int(frame_count),
            None if time_span is None else round(float(time_span), 6),
            None if feature_grid is None else int(feature_grid),
            bool(encode_frames),
            round(float(codec_threshold), 6),
            str(scene_layout or "overlay").strip().lower(),
            tuple(round(float(v), 6) for v in (layer_opacity or ())),
        )

    def _execution_entries_digest(self, entries: list[dict[str, Any]]) -> str:
        compact_rows: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            compact_rows.append(
                {
                    "tool_id": str(entry.get("tool_id", "")).strip(),
                    "action_type": str(entry.get("action_type", "")).strip(),
                    "outcome": int(entry.get("outcome", entry.get("ternary_quality", 0)) or 0),
                    "ternary_quality": int(entry.get("ternary_quality", 0) or 0),
                    "quality_signal": round(float(entry.get("quality_signal", 0.0) or 0.0), 6),
                    "curiosity": round(float(entry.get("curiosity", 0.0) or 0.0), 6),
                    "timestamp": int(entry.get("timestamp_us", entry.get("timestamp", 0)) or 0),
                }
            )
        payload = json.dumps(compact_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()

    def _derive_seed(self, surface_plan: SurfaceMaterialPlan) -> np.ndarray:
        mesh = surface_plan.mesh
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.normals, dtype=np.float32)
        vertex_rgba = np.asarray(surface_plan.vertex_rgba, dtype=np.float32)
        weights = np.asarray(surface_plan.projection_weights, dtype=np.float32)
        preview = np.asarray(surface_plan.material_preview, dtype=np.float32).reshape(-1, 4)
        normal_hint = np.asarray(surface_plan.normal_hint, dtype=np.float32).reshape(-1)

        chunks: list[np.ndarray] = []
        for array in (vertices, normals, vertex_rgba, weights, preview):
            chunks.append(np.mean(array, axis=0, dtype=np.float32))
            chunks.append(np.std(array, axis=0, dtype=np.float32))
        chunks.append(
            np.asarray(
                [float(np.mean(normal_hint)), float(np.std(normal_hint))],
                dtype=np.float32,
            )
        )

        seed = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
        if seed.size < 64:
            pad = np.linspace(0.0, 1.0, 64 - seed.size, dtype=np.float32)
            seed = np.concatenate((seed, pad), axis=0)
        elif seed.size > 64:
            seed = seed[:64]
        return seed.astype(np.float32, copy=False)

    def _frame_to_features(self, frame: np.ndarray, *, feature_grid: int) -> np.ndarray:
        rgb = np.asarray(frame, dtype=np.float32)
        h, w, _ = rgb.shape
        y_idx = np.linspace(0, h - 1, feature_grid, dtype=np.int32)
        x_idx = np.linspace(0, w - 1, feature_grid, dtype=np.int32)
        sampled = rgb[np.ix_(y_idx, x_idx)]
        return (sampled.reshape(-1) / 255.0).astype(np.float32, copy=False)

    def _scene_frame_to_features(self, frame: np.ndarray, *, feature_grid: int) -> np.ndarray:
        rgba = np.asarray(frame, dtype=np.float32)
        h, w, _ = rgba.shape
        y_idx = np.linspace(0, h - 1, feature_grid, dtype=np.int32)
        x_idx = np.linspace(0, w - 1, feature_grid, dtype=np.int32)
        sampled = rgba[np.ix_(y_idx, x_idx)]
        return (sampled.reshape(-1) / 255.0).astype(np.float32, copy=False)

    def _frame_to_ternary_tensor(self, frame: np.ndarray) -> TernaryTensor:
        rgb = np.asarray(frame, dtype=np.int32)
        h, w, c = rgb.shape
        if c != 3:
            raise ValueError("frame must have shape [H,W,3]")
        ternary = np.where(rgb < 85, 0, np.where(rgb > 170, 1, -1)).astype(np.int32, copy=False)
        return TernaryTensor((h, w, 3), TernaryVector([int(v) for v in ternary.reshape(-1).tolist()]))

    def _default_timeline_id(self, surface_plan: SurfaceMaterialPlan) -> str:
        digest = hashlib.sha1(
            np.asarray(surface_plan.vertex_rgba, dtype=np.float32).tobytes()
            + str(surface_plan.selected_material.material_id).encode("utf-8")
        ).hexdigest()[:12]
        return f"timeline_{digest}"

    def _normalize_scene_layers(
        self,
        layers: list[TemporalSceneLayer] | tuple[TemporalSceneLayer, ...],
    ) -> list[TemporalSceneLayer]:
        out: list[TemporalSceneLayer] = []
        for idx, layer in enumerate(layers):
            if not isinstance(layer, TemporalSceneLayer):
                raise ValueError(f"scene layer at index {idx} is not a TemporalSceneLayer")
            if not bool(layer.visible):
                continue
            out.append(layer)
        return out

    def _apply_quality_weighting(
        self,
        layers: list[TemporalSceneLayer],
    ) -> list[TemporalSceneLayer]:
        weighted: list[TemporalSceneLayer] = []
        for layer in layers:
            metadata = dict(layer.metadata or {})
            quality_signal = float(
                metadata.get(
                    "quality_signal",
                    layer.preview_plan.metadata.get("quality_signal", layer.preview_plan.metadata.get("overall_coherence", 0.65)),
                )
            )
            ternary_quality = int(
                metadata.get(
                    "ternary_quality",
                    layer.preview_plan.metadata.get("ternary_quality", ternary_quantize_quality(quality_signal)),
                )
            )
            if ternary_quality > 0:
                opacity_scale = 1.0
                sort_bias = 1
            elif ternary_quality < 0:
                opacity_scale = 0.55
                sort_bias = -1
            else:
                opacity_scale = 0.85
                sort_bias = 0
            metadata.update(
                {
                    "quality_signal": float(np.clip(quality_signal, 0.0, 1.0)),
                    "ternary_quality": int(ternary_quality),
                    "quality_sort_bias": int(sort_bias),
                    "effective_opacity": float(np.clip(layer.opacity * opacity_scale, 0.05, 1.0)),
                }
            )
            weighted.append(
                replace(
                    layer,
                    opacity=float(metadata["effective_opacity"]),
                    metadata=metadata,
                )
            )
        return weighted

    def _scene_background_rgba(
        self,
        background_rgba: np.ndarray | None,
        *,
        height: int,
        width: int,
    ) -> np.ndarray:
        if background_rgba is None:
            return np.zeros((height, width, 4), dtype=np.float32)
        bg = np.asarray(background_rgba, dtype=np.float32)
        if bg.shape != (height, width, 4):
            raise ValueError("background_rgba must match [H,W,4] scene canvas")
        return np.clip(bg, 0.0, 1.0).astype(np.float32, copy=False)

    def _place_layer_rgba(
        self,
        *,
        frame: np.ndarray,
        opacity: float,
        canvas_height: int,
        canvas_width: int,
        x: int,
        y: int,
    ) -> np.ndarray:
        pixels = np.asarray(frame, dtype=np.float32)
        if pixels.ndim != 3 or pixels.shape[2] not in {3, 4}:
            raise ValueError("scene layer frames must have shape [H,W,3] or [H,W,4]")
        rgba = np.zeros((canvas_height, canvas_width, 4), dtype=np.float32)
        h = int(pixels.shape[0])
        w = int(pixels.shape[1])
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(canvas_width, x0 + w)
        y1 = min(canvas_height, y0 + h)
        if x0 >= x1 or y0 >= y1:
            return rgba
        crop_w = x1 - x0
        crop_h = y1 - y0
        rgba[y0:y1, x0:x1, :3] = np.clip(pixels[:crop_h, :crop_w, :3] / 255.0, 0.0, 1.0)
        if pixels.shape[2] == 4:
            alpha = np.clip(pixels[:crop_h, :crop_w, 3] / 255.0, 0.0, 1.0)
            rgba[y0:y1, x0:x1, 3] = alpha * np.clip(float(opacity), 0.0, 1.0)
        else:
            rgba[y0:y1, x0:x1, 3] = np.clip(float(opacity), 0.0, 1.0)
        return rgba

    def _scene_layout_offsets(
        self,
        plans: list[SurfaceMaterialPlan],
        layout: str,
    ) -> list[tuple[int, int]]:
        normalized = str(layout or "overlay").strip().lower()
        if normalized == "golden_orbit":
            sizes = [
                (int(np.asarray(plan.material_preview).shape[1]), int(np.asarray(plan.material_preview).shape[0]))
                for plan in plans
            ]
            return self._golden_scene_offsets(sizes)
        offsets: list[tuple[int, int]] = []
        cursor_x = 0
        cursor_y = 0
        for plan in plans:
            preview = np.asarray(plan.material_preview)
            height = int(preview.shape[0])
            width = int(preview.shape[1])
            offsets.append((cursor_x, cursor_y))
            if normalized == "horizontal_strip":
                cursor_x += width
            elif normalized == "vertical_strip":
                cursor_y += height
        return offsets

    def _default_scene_id(self, layers: tuple[TemporalSceneLayer, ...]) -> str:
        digest = hashlib.sha1()
        for layer in layers:
            digest.update(str(layer.layer_id).encode("utf-8"))
            digest.update(np.asarray(layer.preview_plan.frames, dtype=np.uint8).tobytes()[:4096])
        return f"scene_{digest.hexdigest()[:12]}"

    def _load_replay_entries(
        self,
        *,
        journal_path: str | Path | None,
        replay_entries: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    ) -> list[dict[str, Any]]:
        if replay_entries is not None:
            return [dict(row) for row in replay_entries if isinstance(row, dict)]
        if journal_path is None:
            raise ValueError("journal_path or replay_entries is required")
        path = Path(journal_path)
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    def _load_execution_entries(
        self,
        *,
        event_log_path: str | Path | None,
        execution_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    ) -> list[dict[str, Any]]:
        if execution_events is not None:
            return [dict(row) for row in execution_events if isinstance(row, dict)]
        if event_log_path is None:
            raise ValueError("event_log_path or execution_events is required")
        path = Path(event_log_path)
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
        return rows

    def _action_buffers_to_entries(
        self,
        action_buffers: list[ActionBuffer] | tuple[ActionBuffer, ...],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, buffer in enumerate(action_buffers):
            if not isinstance(buffer, ActionBuffer):
                raise ValueError(f"action buffer at index {idx} is not an ActionBuffer")
            rows.append(
                {
                    "timestamp": idx,
                    "action_type": buffer.get_action_type().name,
                    "raw_confidence": float(buffer.get_confidence()),
                    "final_confidence": float(buffer.get_confidence()),
                    "curiosity": float(buffer.get_curiosity()),
                    "quality_signal": float(buffer.get_confidence()),
                    "ternary_quality": int(ternary_quantize_quality(float(buffer.get_confidence()))),
                }
            )
        return rows

    def _event_token(self, event: Mapping[str, Any]) -> str:
        tool_id = str(event.get("tool_id", "")).strip()
        if tool_id:
            token = tool_id.replace("tool_", "").replace("_v1", "")
            return token.upper() or "TOOL_EVENT"
        action_type = str(event.get("action_type", "")).strip()
        if action_type:
            return action_type.upper()
        return "NO_ACTION"

    def _event_curiosity_signal(self, event: Mapping[str, Any]) -> float:
        if "curiosity" in event:
            try:
                return float(np.clip(float(event.get("curiosity", 0.0) or 0.0), 0.0, 1.0))
            except Exception:
                return 0.0
        quality = float(event.get("quality_signal", 0.65) or 0.65)
        ternary_quality = int(event.get("ternary_quality", ternary_quantize_quality(quality)))
        if ternary_quality == 0:
            return float(np.clip(1.0 - abs(quality - 0.55) * 2.0, 0.35, 1.0))
        if ternary_quality > 0:
            return 0.18
        return 0.32

    def _room_scene_behavior(self, room_preset: str) -> dict[str, Any]:
        normalized = str(room_preset or "house_library").strip().lower()
        if normalized == "house_garden":
            return {
                "timeline_preset": "world_breathe",
                "scene_layout": "golden_orbit",
                "frame_count": 7,
                "opacity_base": 0.52,
            }
        if normalized == "house_museum":
            return {
                "timeline_preset": "ui_idle",
                "scene_layout": "horizontal_strip",
                "frame_count": 6,
                "opacity_base": 0.44,
            }
        return {
            "timeline_preset": "ui_idle",
            "scene_layout": "overlay",
            "frame_count": 5,
            "opacity_base": 0.66,
        }

    def _room_branch_depth(self, room_preset: str, quality: float, curiosity: float) -> int:
        normalized = str(room_preset or "house_library").strip().lower()
        if normalized == "house_garden":
            return int(np.clip(2 + round(quality * 4.0 + curiosity * 2.0), 2, 7))
        if normalized == "house_museum":
            return int(np.clip(1 + round((1.0 - quality) * 3.0), 1, 4))
        return int(np.clip(1 + round(quality * 2.0), 1, 3))

    def _select_house_room_entries(
        self,
        entries: list[dict[str, Any]],
        *,
        room_preset: str,
        max_events: int,
    ) -> list[dict[str, Any]]:
        normalized = str(room_preset or "house_library").strip().lower()
        rows = [dict(entry) for entry in entries if isinstance(entry, dict)]
        if normalized == "house_garden":
            selected = [
                row for row in rows
                if int(row.get("outcome", row.get("ternary_quality", 0)) or 0) >= 0
                and float(row.get("quality_signal", 0.0) or 0.0) >= 0.40
            ]
            selected.sort(
                key=lambda row: (
                    self._event_curiosity_signal(row),
                    float(row.get("quality_signal", 0.0) or 0.0),
                    int(row.get("timestamp_us", row.get("timestamp", 0)) or 0),
                ),
                reverse=True,
            )
        elif normalized == "house_museum":
            selected = [
                row for row in rows
                if int(row.get("outcome", row.get("ternary_quality", 0)) or 0) <= 0
                or float(row.get("quality_signal", 1.0) or 1.0) < 0.40
            ]
            selected.sort(
                key=lambda row: (
                    -int(row.get("outcome", row.get("ternary_quality", 0)) or 0),
                    int(row.get("timestamp_us", row.get("timestamp", 0)) or 0),
                ),
            )
        else:
            selected = [
                row for row in rows
                if int(row.get("outcome", row.get("ternary_quality", 0)) or 0) > 0
                and float(row.get("quality_signal", 0.0) or 0.0) >= 0.70
            ]
            selected.sort(
                key=lambda row: (
                    float(row.get("quality_signal", 0.0) or 0.0),
                    int(row.get("timestamp_us", row.get("timestamp", 0)) or 0),
                ),
                reverse=True,
            )
        return selected[: max(1, int(max_events))]

    def _scene_plan_to_layer(
        self,
        scene: TemporalScenePlan,
        *,
        layer_id: str,
        x: int,
        y: int,
        opacity: float,
        z_index: int,
        metadata: dict[str, Any],
    ) -> TemporalSceneLayer:
        if not scene.layers:
            raise ValueError("scene plan must contain at least one layer")
        preview = TemporalPreviewPlan(
            surface_plan=scene.layers[0].preview_plan.surface_plan,
            frames=np.asarray(scene.frames, dtype=np.uint8),
            frame_features=np.asarray(scene.frame_features, dtype=np.float32),
            temporal_deltas=np.asarray(scene.temporal_deltas, dtype=np.float32),
            coherence_scores=np.asarray(scene.coherence_scores, dtype=np.float32),
            metadata=dict(scene.metadata),
        )
        return TemporalSceneLayer(
            layer_id=str(layer_id),
            preview_plan=preview,
            x=int(x),
            y=int(y),
            opacity=float(opacity),
            z_index=int(z_index),
            metadata=dict(metadata),
        )

    def _event_to_surface_plan(
        self,
        event: dict[str, Any],
        *,
        preview_size: int = 32,
    ) -> SurfaceMaterialPlan:
        action_name = self._event_token(event)
        confidence = float(
            event.get(
                "quality_signal",
                event.get("final_confidence", event.get("raw_confidence", 0.5)) or 0.5,
            )
        )
        curiosity = self._event_curiosity_signal(event)
        palette, gradient_type = self._action_palette_for_event(action_name)
        candidate = SurfaceMaterialCandidate(
            material_id=f"replay_{action_name.lower()}",
            name=f"Replay {action_name}",
            palette=palette,
            gradient_type=gradient_type,
            projection_strategy="planar_xy",
            metadata={
                "action_type": action_name,
                "confidence": confidence,
                "curiosity": curiosity,
                "tool_id": str(event.get("tool_id", "")).strip(),
            },
        )
        preview = self.render_action_preview(candidate, preview_size=preview_size)
        normal_hint = self.effects.edge_map(preview)
        mean_rgba = np.mean(preview.reshape(-1, 4), axis=0, dtype=np.float32)
        vertex_rgba = np.tile(mean_rgba.reshape(1, 4), (4, 1)).astype(np.float32, copy=False)
        projection_weights = np.ones((4, 3), dtype=np.float32)
        mesh = SimpleNamespace(
            vertices=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            normals=np.asarray(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            metadata={
                "mesh_kind": "replay_event_card",
                "action_type": action_name,
            },
        )
        return SurfaceMaterialPlan(
            mesh=mesh,
            selected_material=candidate,
            material_preview=preview,
            normal_hint=normal_hint,
            vertex_rgba=vertex_rgba,
            projection_weights=projection_weights,
            metadata={
                "replay_event": dict(event),
                "clip_id": f"replay_{action_name.lower()}",
                "action_type": action_name,
                "confidence": confidence,
                "curiosity": curiosity,
                "quality_signal": float(event.get("quality_signal", confidence)),
                "ternary_quality": int(
                    event.get(
                        "ternary_quality",
                        ternary_quantize_quality(float(event.get("quality_signal", confidence))),
                    )
                ),
            },
        )

    def render_action_preview(
        self,
        candidate: SurfaceMaterialCandidate,
        *,
        preview_size: int,
    ) -> np.ndarray:
        stops = self.effects.palette_to_gradient_stops(candidate.palette)
        gradient_type = str(candidate.gradient_type or "linear").lower()
        if gradient_type == "radial":
            return self.effects.radial_gradient(preview_size, preview_size, stops, cx=0.5, cy=0.5, radius=0.72)
        if gradient_type == "conic":
            return self.effects.conic_gradient(preview_size, preview_size, stops, cx=0.5, cy=0.5, start_angle=0.0)
        return self.effects.linear_gradient(preview_size, preview_size, stops, x1=0.0, y1=0.0, x2=1.0, y2=1.0)

    def _action_palette_for_event(
        self,
        action_name: str,
    ) -> tuple[tuple[tuple[float, float, float, float], ...], str]:
        if action_name in {"NAV_MOVE", "NAV_LOOK"}:
            return (
                (
                    (0.08, 0.16, 0.62, 1.0),
                    (0.22, 0.52, 0.90, 1.0),
                    (0.84, 0.96, 1.0, 1.0),
                ),
                "linear",
            )
        if action_name == "DIALOGUE":
            return (
                (
                    (0.62, 0.34, 0.08, 1.0),
                    (0.95, 0.74, 0.24, 1.0),
                    (1.0, 0.96, 0.84, 1.0),
                ),
                "radial",
            )
        if action_name == "WRITE_MEM":
            return (
                (
                    (0.08, 0.32, 0.12, 1.0),
                    (0.24, 0.68, 0.30, 1.0),
                    (0.88, 1.0, 0.90, 1.0),
                ),
                "linear",
            )
        if action_name == "UPDATE_TABLET":
            return (
                (
                    (0.10, 0.28, 0.24, 1.0),
                    (0.26, 0.72, 0.64, 1.0),
                    (0.92, 1.0, 0.98, 1.0),
                ),
                "conic",
            )
        return (
            (
                (0.18, 0.18, 0.18, 1.0),
                (0.50, 0.50, 0.50, 1.0),
                (0.92, 0.92, 0.92, 1.0),
            ),
            "linear",
        )

    def _action_timeline_preset(self, action_name: str) -> str:
        token = str(action_name or "").strip().upper()
        if token in {"NAV_MOVE", "NAV_LOOK"}:
            return "world_orbit"
        if token in {"WRITE_MEM"}:
            return "world_breathe"
        if token in {"DIALOGUE", "UPDATE_TABLET"}:
            return "ui_focus"
        return "ui_idle"

    def _golden_scene_offsets(
        self,
        sizes: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        try:
            from knowledge3d.tools.test_scripts.garden_fractal_rpn import compute_golden_angle_rpn

            golden_angle = float(compute_golden_angle_rpn())
        except Exception:
            golden_angle = 2.399963229728653
        phi = 1.618033988749895
        points: list[tuple[float, float]] = []
        radius_base = max(max(width for width, _height in sizes), max(height for _width, height in sizes), 1)
        for idx, (width, height) in enumerate(sizes):
            radius = radius_base * (0.3 + 0.18 * np.sqrt(idx + 1)) * (1.0 + idx / max(1, len(sizes)) / phi)
            angle = golden_angle * idx
            points.append((float(np.cos(angle) * radius), float(np.sin(angle) * radius)))
        min_x = min(point[0] for point in points)
        min_y = min(point[1] for point in points)
        offsets: list[tuple[int, int]] = []
        for (x, y), (width, height) in zip(points, sizes):
            offsets.append(
                (
                    int(round(x - min_x)),
                    int(round(y - min_y)),
                )
            )
        return offsets

    def _detect_scene_grammars(
        self,
        entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        tokens = [
            self._event_token(entry)
            for entry in entries
            if self._event_token(entry)
        ]
        if len(tokens) < 2:
            return []
        counts: dict[tuple[str, ...], int] = {}
        for size in (2, 3):
            if len(tokens) < size:
                continue
            for idx in range(0, len(tokens) - size + 1):
                window = tuple(tokens[idx:idx + size])
                counts[window] = counts.get(window, 0) + 1
        grammars: list[dict[str, Any]] = []
        for sequence, count in sorted(counts.items(), key=lambda item: (-item[1], len(item[0]), item[0])):
            if count < 3:
                continue
            qualities = [
                float(entry.get("quality_signal", entry.get("final_confidence", entry.get("raw_confidence", 0.65)) or 0.65))
                for entry in entries
                if self._event_token(entry) in sequence
            ]
            avg_quality = float(np.mean(np.asarray(qualities, dtype=np.float32))) if qualities else 0.0
            grammar_id = "scene_grammar_" + "_".join(sequence[:3]).lower()
            grammars.append(
                {
                    "grammar_id": grammar_id,
                    "sequence": list(sequence),
                    "count": int(count),
                    "quality_signal": avg_quality,
                    "ternary_quality": int(ternary_quantize_quality(avg_quality)),
                    "target_room": self._house_room_preset(avg_quality),
                }
            )
        return grammars

    def _house_room_preset(self, quality_signal: float) -> str:
        quality = float(np.clip(quality_signal, 0.0, 1.0))
        if quality >= 0.7:
            return "house_library"
        if quality >= 0.4:
            return "house_garden"
        return "house_museum"


__all__ = [
    "ProceduralTemporalBridge",
    "TemporalPreviewPlan",
    "TemporalSceneLayer",
    "TemporalScenePlan",
]
