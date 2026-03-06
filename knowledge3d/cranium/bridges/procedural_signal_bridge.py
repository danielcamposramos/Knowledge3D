"""Deterministic bridge from audio signals to drawable spectrograms and surfaces.

The bridge stays honest about the current substrate:
- spectral transforms are PTX-backed through the sovereign ternary audio codec
- preview coloring is PTX-backed through the signal visualization runtime
- heightfield mesh assembly remains deterministic host orchestration until a
  dedicated signal-surface kernel is justified
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from knowledge3d.cranium.codecs.sovereign_ternary_audio_codec import SovereignTernaryAudioCodec
from knowledge3d.cranium.ptx_runtime.math_core_pool import get_global_math_core_pool
from knowledge3d.cranium.ptx_runtime.signal_visualization_kernels import SignalVisualizationKernels
from knowledge3d.cranium.ternary import TernaryVector


@dataclass(frozen=True)
class SpectrogramPlan:
    spectrogram: np.ndarray
    preview_rgba: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class SignalSurfacePlan:
    heightfield: np.ndarray
    vertices: np.ndarray
    indices: np.ndarray
    normals: np.ndarray
    metadata: dict[str, object]


def _resolve_signal_math_core_plan(preferred_tier: int, work_items: int) -> dict[str, object]:
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


class ProceduralSignalBridge:
    """Always-on signal bridge built on the sovereign PTX codec substrate."""

    def __init__(self, frame_size: int = 1024, threshold: float = 0.2) -> None:
        self.frame_size = int(frame_size)
        self.threshold = float(threshold)
        self.codec = SovereignTernaryAudioCodec(frame_size=self.frame_size, threshold=self.threshold)
        self.visualization = SignalVisualizationKernels()

    def audio_to_spectrogram(
        self,
        clip_id: str,
        samples: TernaryVector,
    ) -> SpectrogramPlan:
        meta = self.codec.encode(clip_id, samples)
        _seed_rpn, residual, stored_meta = self.codec.galaxy.load_frame_details(clip_id)
        frame_count = int(stored_meta["frame_count"])
        bins = self.frame_size // 2
        spectrogram = np.asarray(residual.to_python(), dtype=np.int32).reshape(frame_count, bins).T
        preview_rgba = self.visualization.spectrogram_to_rgba(spectrogram)
        total = max(1, int(spectrogram.size))
        metadata = {
            "clip_id": clip_id,
            "seed_rpn": meta["seed_rpn"],
            "frame_size": self.frame_size,
            "threshold": self.threshold,
            "frame_count": frame_count,
            "frequency_bins": bins,
            "positive_ratio": float(np.count_nonzero(spectrogram > 0) / total),
            "negative_ratio": float(np.count_nonzero(spectrogram < 0) / total),
            "neutral_ratio": float(np.count_nonzero(spectrogram == 0) / total),
            "math_core_plan": meta["math_core_plan"],
        }
        return SpectrogramPlan(
            spectrogram=spectrogram,
            preview_rgba=preview_rgba,
            metadata=metadata,
        )

    def audio_to_spectrogram_configured(
        self,
        clip_id: str,
        samples: TernaryVector,
        *,
        frame_size: int = 1024,
        threshold: float = 0.2,
    ) -> SpectrogramPlan:
        bridge = ProceduralSignalBridge(frame_size=frame_size, threshold=threshold)
        return bridge.audio_to_spectrogram(clip_id, samples)

    def spectrogram_to_surface(
        self,
        projection: SpectrogramPlan,
        *,
        displacement_gain: float = 0.25,
        time_scale: float = 1.0,
        frequency_scale: float = 1.0,
    ) -> SignalSurfacePlan:
        spectrogram = np.asarray(projection.spectrogram, dtype=np.float32)
        rows, cols = spectrogram.shape
        surface_plan = _resolve_signal_math_core_plan(preferred_tier=3, work_items=rows * cols)
        heightfield = (spectrogram * float(displacement_gain)).astype(np.float32, copy=False)

        x_coords = np.linspace(-0.5, 0.5, cols, dtype=np.float32) * float(time_scale)
        z_coords = np.linspace(0.5, -0.5, rows, dtype=np.float32) * float(frequency_scale)
        xx, zz = np.meshgrid(x_coords, z_coords, indexing="xy")
        vertices = np.stack((xx, heightfield, zz), axis=-1).astype(np.float32, copy=False).reshape(-1, 3)

        if rows > 1 and cols > 1:
            row_ids = np.arange(rows - 1, dtype=np.uint32)[:, None]
            col_ids = np.arange(cols - 1, dtype=np.uint32)[None, :]
            base = row_ids * np.uint32(cols) + col_ids
            tri1 = np.stack((base, base + 1, base + np.uint32(cols)), axis=-1).reshape(-1, 3)
            tri2 = np.stack(
                (base + 1, base + np.uint32(cols) + 1, base + np.uint32(cols)),
                axis=-1,
            ).reshape(-1, 3)
            indices = np.concatenate((tri1, tri2), axis=0).astype(np.uint32, copy=False)
        else:
            indices = np.empty((0, 3), dtype=np.uint32)

        normals = _compute_vertex_normals(vertices, indices, execution_plan=surface_plan)
        metadata = {
            "rows": int(rows),
            "cols": int(cols),
            "displacement_gain": float(displacement_gain),
            "math_core_plan": surface_plan,
            "source_math_core_plan": dict(projection.metadata.get("math_core_plan", {})),
            "positive_ratio": projection.metadata.get("positive_ratio", 0.0),
            "negative_ratio": projection.metadata.get("negative_ratio", 0.0),
            "neutral_ratio": projection.metadata.get("neutral_ratio", 0.0),
        }
        return SignalSurfacePlan(
            heightfield=heightfield,
            vertices=vertices,
            indices=indices,
            normals=normals,
            metadata=metadata,
        )

__all__ = ["ProceduralSignalBridge", "SignalSurfacePlan", "SpectrogramPlan"]
