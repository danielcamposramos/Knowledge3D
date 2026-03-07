"""Deterministic bridge from audio signals to drawable spectrograms and surfaces.

The bridge stays honest about the current substrate:
- spectral transforms are PTX-backed through the sovereign ternary audio codec
- preview coloring is PTX-backed through the signal visualization runtime
- heightfield vertex/normal generation is PTX-backed through signal surface kernels
- surface topology assembly remains deterministic host orchestration until a
  dedicated signal-topology kernel is justified
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np

from knowledge3d.cranium.codecs.sovereign_ternary_audio_codec import SovereignTernaryAudioCodec
from knowledge3d.cranium.ptx_runtime.math_core_pool import get_global_math_core_pool
from knowledge3d.cranium.ptx_runtime.signal_surface_kernels import SignalSurfaceKernels
from knowledge3d.cranium.ptx_runtime.signal_visualization_kernels import SignalVisualizationKernels
from knowledge3d.cranium.ternary import TernaryVector


@dataclass(frozen=True)
class SpectrogramPlan:
    spectrogram: np.ndarray
    preview_rgba: np.ndarray | None
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


class ProceduralSignalBridge:
    """Always-on signal bridge built on the sovereign PTX codec substrate."""

    _CONFIGURED_CACHE: dict[tuple[int, int, float], "ProceduralSignalBridge"] = {}

    def __init__(self, frame_size: int = 1024, threshold: float = 0.2) -> None:
        self.frame_size = int(frame_size)
        self.threshold = float(threshold)
        self.codec = SovereignTernaryAudioCodec(frame_size=self.frame_size, threshold=self.threshold)
        self.visualization = SignalVisualizationKernels()
        self.surface_kernels = SignalSurfaceKernels()

    @classmethod
    def for_config(cls, *, frame_size: int = 1024, threshold: float = 0.2) -> "ProceduralSignalBridge":
        key = (os.getpid(), int(frame_size), float(threshold))
        bridge = cls._CONFIGURED_CACHE.get(key)
        if bridge is None:
            bridge = cls(frame_size=frame_size, threshold=threshold)
            cls._CONFIGURED_CACHE[key] = bridge
        return bridge

    def audio_to_spectrogram(
        self,
        clip_id: str,
        samples: TernaryVector,
        *,
        build_preview: bool = True,
    ) -> SpectrogramPlan:
        encoded = self.codec.encode_details(clip_id, samples)
        meta = dict(encoded["metadata"])
        quantized = np.asarray(encoded["quantized_coeffs"], dtype=np.int32)
        frame_count = int(meta["frame_count"])
        bins = self.frame_size // 2
        spectrogram = quantized.reshape(frame_count, bins).T
        preview_rgba = self.visualization.spectrogram_to_rgba(spectrogram) if build_preview else None
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
        build_preview: bool = True,
    ) -> SpectrogramPlan:
        bridge = self.for_config(frame_size=frame_size, threshold=threshold)
        return bridge.audio_to_spectrogram(clip_id, samples, build_preview=build_preview)

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
        vertices = self.surface_kernels.heightfield_to_vertices(
            heightfield,
            time_scale=float(time_scale),
            frequency_scale=float(frequency_scale),
        )

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

        normals = self.surface_kernels.heightfield_to_normals(
            heightfield,
            time_scale=float(time_scale),
            frequency_scale=float(frequency_scale),
        )
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
