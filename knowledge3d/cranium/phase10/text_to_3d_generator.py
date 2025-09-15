from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore


class TextTo3DGenerator:
    def __init__(self, material_dir: str = "viewer/public/house/materialized_objects"):
        self.material_dir = Path(material_dir)
        self.material_dir.mkdir(parents=True, exist_ok=True)

    def generate_3d_from_text(self, text: str, honesty_threshold: float = 0.7) -> str:
        """Generate 3D shape metadata from text using mock single‑head logic.

        Returns a JSON path now; Phase 10.9 replaces with actual GLB writing.
        """
        emb = self._mock_text_embedding(text)
        honesty = self._calculate_honesty_score(emb)
        if honesty < float(honesty_threshold):
            raise ValueError(f"🚫 Text too dishonest (score: {honesty:.2f}) — cannot generate 3D.")

        shape = self._predict_shape_type(emb)
        vertices, faces = self._generate_shape_geometry(shape, emb)

        gid = f"shape_{shape}_{int(datetime.now().timestamp())}"
        out = self.material_dir / f"{gid}.json"  # will be .glb in Phase 10.9
        data: Dict[str, Any] = {
            "type": "generated_3d_shape",
            "name": f"{shape.capitalize()} from: '{text[:30]}...'",
            "created_at": datetime.now().isoformat(),
            "honesty_score": float(honesty),
            "embedding": emb.tolist(),
            "shape_type": shape,
            "vertex_count": int(len(vertices)),
            "face_count": int(len(faces)),
            "zone_placement": "Zone 5 (Knowledge Garden)",
            "vertices_preview": (vertices[:5].tolist() if len(vertices) > 5 else vertices.tolist()),
            "ptx_kernel_used": f"generate_{shape}_kernel",
        }
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"🌀 Generated {shape} for text: '{text}' → {out}")
        return str(out)

    def _mock_text_embedding(self, text: str, dim: int = 512) -> np.ndarray:
        import hashlib
        hv = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        v = np.array([((hv >> (i * 8)) & 0xFF) for i in range(dim)], dtype=np.float32)
        v = (v - 128.0) / 128.0
        return v

    def _calculate_honesty_score(self, emb: np.ndarray) -> float:
        h = 0.5 + 0.5 * np.sin(float(np.mean(emb)))
        return float(np.clip(h, 0.0, 1.0))

    def _predict_shape_type(self, emb: np.ndarray) -> str:
        s = float(np.sum(emb * 1000.0))
        idx = int(abs(s)) % 5
        shapes = ["tetrahedron", "cube", "octahedron", "icosahedron", "dodecahedron"]
        return shapes[idx]

    def _generate_shape_geometry(self, shape: str, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = 1.0 + float(np.linalg.norm(emb[:3]))
        if shape == "tetrahedron":
            vertices = np.array(
                [[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], dtype=np.float32
            ) * scale
            faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.uint32)
            return vertices, faces
        if shape == "cube":
            vertices = np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float32,
            ) * scale
            faces = np.array(
                [
                    [0, 1, 2],
                    [0, 2, 3],
                    [4, 5, 6],
                    [4, 6, 7],
                    [0, 1, 5],
                    [0, 5, 4],
                    [2, 3, 7],
                    [2, 7, 6],
                    [1, 2, 6],
                    [1, 6, 5],
                    [0, 3, 7],
                    [0, 7, 4],
                ],
                dtype=np.uint32,
            )
            return vertices, faces
        if shape == "octahedron":
            vertices = np.array(
                [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.float32
            ) * scale
            faces = np.array(
                [
                    [0, 2, 4],
                    [0, 4, 3],
                    [0, 3, 5],
                    [0, 5, 2],
                    [1, 4, 2],
                    [1, 3, 4],
                    [1, 5, 3],
                    [1, 2, 5],
                ],
                dtype=np.uint32,
            )
            return vertices, faces
        if shape == "icosahedron":
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            base = np.array(
                [
                    [-1, phi, 0],
                    [1, phi, 0],
                    [-1, -phi, 0],
                    [1, -phi, 0],
                    [0, -1, phi],
                    [0, 1, phi],
                    [0, -1, -phi],
                    [0, 1, -phi],
                    [phi, 0, -1],
                    [phi, 0, 1],
                    [-phi, 0, -1],
                    [-phi, 0, 1],
                ],
                dtype=np.float32,
            )
            base = base * (scale / np.linalg.norm([1, phi, 0]))
            faces = np.array(
                [
                    [0, 11, 5],
                    [0, 5, 1],
                    [0, 1, 7],
                    [0, 7, 10],
                    [0, 10, 11],
                    [1, 5, 9],
                    [5, 11, 4],
                    [11, 10, 2],
                    [10, 7, 6],
                    [7, 1, 8],
                    [3, 9, 4],
                    [3, 4, 2],
                    [3, 2, 6],
                    [3, 6, 8],
                    [3, 8, 9],
                    [4, 9, 5],
                    [2, 4, 11],
                    [6, 2, 10],
                    [8, 6, 7],
                    [9, 8, 1],
                ],
                dtype=np.uint32,
            )
            return base, faces
        if shape == "dodecahedron":
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            vertices = np.array(
                [
                    [-1, -1, -1],
                    [-1, -1, 1],
                    [-1, 1, -1],
                    [-1, 1, 1],
                    [1, -1, -1],
                    [1, -1, 1],
                    [1, 1, -1],
                    [1, 1, 1],
                    [0, -phi, -1 / phi],
                    [0, -phi, 1 / phi],
                    [0, phi, -1 / phi],
                    [0, phi, 1 / phi],
                    [-1 / phi, 0, -phi],
                    [1 / phi, 0, -phi],
                    [-1 / phi, 0, phi],
                    [1 / phi, 0, phi],
                    [-phi, -1 / phi, 0],
                    [phi, -1 / phi, 0],
                    [-phi, 1 / phi, 0],
                    [phi, 1 / phi, 0],
                ],
                dtype=np.float32,
            )
            vertices = vertices * (scale / np.linalg.norm([1, phi, 1.0 / phi]))
            # Use triangle fan approximation for metadata; real GLB will use pentagonal faces later
            faces = np.array(
                [
                    [0, 1, 9],
                    [0, 9, 8],
                    [4, 5, 6],
                    [4, 6, 0],
                    [7, 6, 5],
                    [7, 5, 3],
                    [11, 10, 6],
                    [12, 13, 10],
                ],
                dtype=np.uint32,
            )
            return vertices, faces
        # Fallback: simple icosahedron (no subdivision)
        return self._generate_icosa_fallback(scale)

    def _generate_icosa_fallback(self, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        vertices = np.array(
            [
                [-1, phi, 0],
                [1, phi, 0],
                [-1, -phi, 0],
                [1, -phi, 0],
                [0, -1, phi],
                [0, 1, phi],
                [0, -1, -phi],
                [0, 1, -phi],
                [phi, 0, -1],
                [phi, 0, 1],
                [-phi, 0, -1],
                [-phi, 0, 1],
            ],
            dtype=np.float32,
        )
        vertices = vertices * (scale / np.linalg.norm([1, phi, 0]))
        faces = np.array(
            [
                [0, 11, 5],
                [0, 5, 1],
                [0, 1, 7],
                [0, 7, 10],
                [0, 10, 11],
                [1, 5, 9],
                [5, 11, 4],
                [11, 10, 2],
                [10, 7, 6],
                [7, 1, 8],
                [3, 9, 4],
                [3, 4, 2],
                [3, 2, 6],
                [3, 6, 8],
                [3, 8, 9],
                [4, 9, 5],
                [2, 4, 11],
                [6, 2, 10],
                [8, 6, 7],
                [9, 8, 1],
            ],
            dtype=np.uint32,
        )
        return vertices, faces

