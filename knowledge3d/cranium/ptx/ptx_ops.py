from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np  # type: ignore

from knowledge3d.cranium.phase10.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.phase10.text_to_3d_generator import TextTo3DGenerator
from knowledge3d.cranium.ptx.galaxy_buffer import GalaxyGPUMemory
from knowledge3d.cranium.ptx.geometry_ops import PTXGeometrySession
from knowledge3d.cranium.ptx.modality_ops import PTXModalityOps


class PTXOps:
    """Utility wrapper exposing GPU PTX helpers to higher-level components."""

    def __init__(self) -> None:
        self._rpn_engine = ModularRPNEngine()
        self._shape_generator = TextTo3DGenerator()
        self._geometry_session = PTXGeometrySession()
        self._modality_ops = PTXModalityOps()

    # ------------------------------------------------------------------
    def evaluate_rpn(self, expression: str, variables: Optional[Dict[str, float]] = None) -> float:
        """Evaluate an RPN expression on the GPU and return the first scalar result."""
        result = self._rpn_engine.evaluate(expression, variables=variables)
        if isinstance(result, np.ndarray):
            if result.size == 0:
                raise RuntimeError("RPN engine returned empty result vector")
            return float(result.ravel()[0])
        return float(result)

    # ------------------------------------------------------------------
    _MODALITY_EXPRESSIONS: Dict[str, str] = {
        "text": "length_norm 0.25 * mean_norm 0.25 * + std_norm 0.2 * + hist_entropy 0.15 * + vowel_ratio 0.15 * + sigmoid",
        "audio": "abs_mean 0.2 * rms 0.3 * + energy 0.3 * + band_uniformity 0.2 * + sigmoid",
        "image": "brightness_std 0.3 * saturation_std 0.3 * + colorfulness 0.2 * + dynamic_range 0.2 * + sigmoid",
        "video": "motion_mean 0.3 * motion_std 0.2 * + brightness_std 0.2 * + saturation_std 0.1 * + hist_entropy 0.2 * + sigmoid",
    }

    def text_modality(self, text: str) -> Dict[str, Any]:
        features, metrics = self._modality_ops.text_features(text)
        return self._prepare_modality_response("text", features, metrics)

    def audio_modality(self, path: str) -> Dict[str, Any]:
        features, metrics = self._modality_ops.audio_features(path)
        return self._prepare_modality_response("audio", features, metrics)

    def image_modality(self, path: str) -> Dict[str, Any]:
        features, metrics = self._modality_ops.image_features(path)
        return self._prepare_modality_response("image", features, metrics)

    def video_modality(self, path: str) -> Dict[str, Any]:
        features, metrics = self._modality_ops.video_features(path)
        return self._prepare_modality_response("video", features, metrics)

    def _prepare_modality_response(
        self,
        modality: str,
        features: np.ndarray,
        metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        expression = self._MODALITY_EXPRESSIONS.get(modality)
        confidence = 0.5
        if expression:
            confidence = self.evaluate_rpn(expression, variables=metrics)
            confidence = max(0.0, min(1.0, confidence))
        return {
            "features": [float(x) for x in features.reshape(-1)],
            "metrics": {k: float(v) for k, v in metrics.items()},
            "confidence": confidence,
        }

    def generate_shape(self, prompt: str, vertex_count: int = 32, shape_hint: Optional[int] = None) -> str:
        """Generate a GLB path for a prompt-driven shape using the PTX geometry kernel."""
        # The TextTo3DGenerator internally hashes the prompt to determine shape semantics.
        shape_path = self._shape_generator.generate_3d_from_text(prompt)
        return shape_path

    def last_generated_shape(self) -> Optional[Dict[str, Any]]:
        return getattr(self._shape_generator, "last_generation", None)

    # ------------------------------------------------------------------
    # Geometry session helpers -----------------------------------------
    def geometry_load_scene(self, glb_path: str, *, embedding_json: Optional[str] = None) -> None:
        """Load a GLB + optional embedding JSON into GPU memory."""

        self._geometry_session.load_scene(glb_path, embedding_json=embedding_json)

    def geometry_apply_transform(
        self,
        mesh_index: int,
        matrix: np.ndarray,
        *,
        recalc_normals: bool = False,
    ) -> None:
        """Apply a transform to a mesh primitive in the loaded scene."""

        self._geometry_session.apply_transform(mesh_index, matrix, recalc_normals=recalc_normals)

    def geometry_apply_transform_for_mesh(
        self,
        mesh_id: int,
        matrix: np.ndarray,
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        """Apply a transform matrix to every primitive of a glTF mesh ID."""

        self._geometry_session.apply_transform_for_mesh(
            mesh_id,
            matrix,
            primitive_index=primitive_index,
            recalc_normals=recalc_normals,
        )

    def geometry_apply_transform_for_mesh(
        self,
        mesh_id: int,
        matrix: np.ndarray,
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        """Apply a transform matrix to every primitive of a glTF mesh ID."""

        self._geometry_session.apply_transform_for_mesh(
            mesh_id,
            matrix,
            primitive_index=primitive_index,
            recalc_normals=recalc_normals,
        )

    def geometry_translate_mesh(
        self,
        mesh_id: int,
        translation: np.ndarray,
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        """Translate a glTF mesh ID by the given XYZ vector."""

        self._geometry_session.translate_mesh(
            mesh_id,
            translation,
            primitive_index=primitive_index,
            recalc_normals=recalc_normals,
        )

    def geometry_scale_mesh(
        self,
        mesh_id: int,
        scale: np.ndarray,
        *,
        primitive_index: Optional[int] = None,
        recalc_normals: bool = False,
    ) -> None:
        """Scale a glTF mesh ID by the given XYZ vector."""

        self._geometry_session.scale_mesh(
            mesh_id,
            scale,
            primitive_index=primitive_index,
            recalc_normals=recalc_normals,
        )

    def geometry_recalc_normals(self, mesh_index: int) -> None:
        """Recalculate normals on a mesh primitive in the loaded scene."""

        self._geometry_session.recalc_normals(mesh_index)

    def geometry_blend_embedding(self, node_index: int, embedding: np.ndarray, alpha: float) -> None:
        """Blend an embedding into the loaded scene's embedding buffer."""

        self._geometry_session.blend_embedding(node_index, embedding, alpha)

    def geometry_normalize_embedding(self, node_index: int) -> None:
        """Normalize an embedding vector in-place."""

        self._geometry_session.normalize_embedding(node_index)

    def geometry_save(
        self,
        *,
        target_glb: Optional[str] = None,
        target_embeddings: Optional[str] = None,
    ) -> None:
        """Persist any dirty geometry/embedding buffers back to disk."""

        self._geometry_session.save(target_glb=target_glb, target_embeddings=target_embeddings)

    def geometry_release(self) -> None:
        """Release GPU resources associated with the loaded scene."""

        self._geometry_session.close()

    @property
    def geometry_memory(self) -> GalaxyGPUMemory:
        """Return the current GalaxyGPUMemory for callers that need raw access."""

        return self._geometry_session.galaxy_memory

    def embedding_cosine_similarity(self, query_vector: np.ndarray) -> np.ndarray:
        return self._geometry_session.cosine_similarity(query_vector)

    def embedding_cosine_topk(
        self,
        query_vector: np.ndarray,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._geometry_session.cosine_topk(query_vector, k)

    # ------------------------------------------------------------------
    @staticmethod
    def format_numeric(value: float, precision: int = 6) -> str:
        arr = np.array([value], dtype=np.float64)
        return np.array2string(arr, precision=precision, suppress_small=True)[1:-1]


# Shared singleton to avoid re-initialising GPU contexts repeatedly.
PTX_OPS = PTXOps()
