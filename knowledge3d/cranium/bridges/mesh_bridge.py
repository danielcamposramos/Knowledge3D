"""Host bridge for Phase H1 mesh construction programs."""

from __future__ import annotations

from dataclasses import dataclass

from knowledge3d.cranium.ptx_runtime.mesh_engine import MeshRPNEngine
from knowledge3d.cranium.ptx_runtime.mesh_opcodes import MeshBuffer


@dataclass
class MeshRenderResult:
    mesh: MeshBuffer
    program: str
    token_count: int
    backend: str = "cpu"


class MeshBridge:
    """Thin bridge around the host mesh runtime.

    This mirrors the existence of the drawing bridge without touching the
    sovereign benchmark hot path. Mesh generation remains available for asset
    construction, ingestion, and tests.
    """

    def __init__(self) -> None:
        self.engine = MeshRPNEngine()
        self._sovereign = None
        try:
            from .sovereign_mesh_bridge import SovereignMeshBridge

            self._sovereign = SovereignMeshBridge()
        except Exception:
            self._sovereign = None

    def execute_rpn_program(self, program: str) -> MeshRenderResult:
        result = None
        backend = "cpu"
        if self._sovereign is not None and self._sovereign.is_supported_program(program):
            try:
                mesh = self._sovereign.execute_supported_program(program)
                tokens = self.engine.tokenize_rpn(program)
                return MeshRenderResult(
                    mesh=mesh,
                    program=program,
                    token_count=len(tokens),
                    backend="gpu",
                )
            except Exception:
                backend = "cpu_fallback"
        result = self.engine.evaluate_with_trace(program)
        return MeshRenderResult(
            mesh=result.mesh,
            program=program,
            token_count=len(result.tokens),
            backend=backend,
        )

    def is_mesh_program(self, program: str) -> bool:
        return self.engine.is_mesh_expression(program)


__all__ = ["MeshBridge", "MeshRenderResult"]
