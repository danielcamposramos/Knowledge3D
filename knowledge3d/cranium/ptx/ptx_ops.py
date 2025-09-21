from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore

from knowledge3d.cranium.phase10.modular_rpn_engine import ModularRPNEngine
from knowledge3d.cranium.phase10.text_to_3d_generator import TextTo3DGenerator
from knowledge3d.cranium.ptx.galaxy_buffer import GalaxyGPUMemory
from knowledge3d.cranium.ptx.geometry_ops import PTXGeometrySession


class PTXOps:
    """Utility wrapper exposing GPU PTX helpers to higher-level components."""

    def __init__(self) -> None:
        self._rpn_engine = ModularRPNEngine()
        self._shape_generator = TextTo3DGenerator()
        self._geometry_session = PTXGeometrySession()

    # ------------------------------------------------------------------
    def evaluate_rpn(self, expression: str) -> float:
        """Evaluate an RPN expression on the GPU and return the first scalar result."""
        result = self._rpn_engine.evaluate(expression)
        return float(result)

    def generate_shape(self, prompt: str, vertex_count: int = 32, shape_hint: Optional[int] = None) -> str:
        """Generate a GLB path for a prompt-driven shape using the PTX geometry kernel."""
        # The TextTo3DGenerator internally hashes the prompt to determine shape semantics.
        shape_path = self._shape_generator.generate_3d_from_text(prompt)
        return shape_path

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

    def geometry_recalc_normals(self, mesh_index: int) -> None:
        """Recalculate normals on a mesh primitive in the loaded scene."""

        self._geometry_session.recalc_normals(mesh_index)

    def geometry_blend_embedding(self, node_index: int, embedding: np.ndarray, alpha: float) -> None:
        """Blend an embedding into the loaded scene's embedding buffer."""

        self._geometry_session.blend_embedding(node_index, embedding, alpha)

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

    # ------------------------------------------------------------------
    @staticmethod
    def format_numeric(value: float, precision: int = 6) -> str:
        arr = np.array([value], dtype=np.float64)
        return np.array2string(arr, precision=precision, suppress_small=True)[1:-1]


# Shared singleton to avoid re-initialising GPU contexts repeatedly.
PTX_OPS = PTXOps()
