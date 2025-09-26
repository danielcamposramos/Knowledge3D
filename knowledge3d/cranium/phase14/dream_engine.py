from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np  # type: ignore
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Asset  # type: ignore


class DreamEngine:
    """Generates dream shapes from internal state via honesty‑weighted embedding drift.

    Produces actual GLBs with extras.k3d and places them conceptually in Zone 6 (Dream Chamber).
    """

    def __init__(self, galaxy_path: str, material_dir: str):
        self.galaxy_path = Path(galaxy_path)
        self.material_dir = Path(material_dir)
        self.material_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Embeddings ----------
    def load_galaxy_embeddings(self) -> np.ndarray:
        try:
            gltf = GLTF2().load(str(self.galaxy_path))
            embs: List[np.ndarray] = []
            for n in (gltf.nodes or []):
                extras = getattr(n, 'extras', None)
                if hasattr(extras, 'to_dict'):
                    try:
                        extras = extras.to_dict()
                    except Exception:
                        extras = dict(extras)
                if isinstance(extras, dict):
                    k3d = extras.get('k3d') if isinstance(extras.get('k3d'), dict) else None
                    if isinstance(k3d, dict) and str(k3d.get('type') or '').lower() == 'star':
                        emb = k3d.get('embedding')
                        if isinstance(emb, list) and emb:
                            try:
                                embs.append(np.asarray(emb, dtype=np.float32))
                            except Exception:
                                pass
            if not embs:
                return np.random.randn(16, 512).astype(np.float32)
            # pad to same length
            d = max(len(e) for e in embs)
            out: List[np.ndarray] = []
            for e in embs:
                if len(e) < d:
                    ee = np.zeros((d,), dtype=np.float32)
                    ee[: len(e)] = e
                    out.append(ee)
                else:
                    out.append(e[:d])
            return np.vstack(out)
        except Exception:
            return np.random.randn(16, 512).astype(np.float32)

    def _drift_honesty(self, cand: np.ndarray, origin: np.ndarray) -> float:
        diff = cand - origin
        if diff.size == 0:
            return 0.85
        scaled_dist = float(np.linalg.norm(diff) / (np.sqrt(diff.size) + 1e-6))
        base = float(np.exp(-0.5 * (scaled_dist / 1.5) ** 2))
        noise = float(np.random.normal(0.0, 0.05))
        honesty = base * 1.15 + 0.05 + noise
        honesty = max(0.1, min(1.0, honesty))
        return float(honesty)

    def generate_dream_embedding(self, base: np.ndarray, honesty_bias: float = 0.7) -> np.ndarray:
        if base.size == 0:
            return np.random.randn(512).astype(np.float32)
        start = base[np.random.randint(0, base.shape[0])].copy()
        d = start.shape[0]
        steps = np.random.randint(3, 8)
        for _ in range(steps):
            drift = np.random.randn(d).astype(np.float32) * 0.12
            cand = start + drift
            if self._drift_honesty(cand, start) >= honesty_bias:
                start = cand
        return start.astype(np.float32)

    # ---------- Geometry ----------
    def _shape_geometry(self, shape: str, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = 1.0 + float(np.linalg.norm(emb[:3]) * 0.3)
        if shape == 'torus':
            v = []
            R, r = 1.0 * scale, 0.5 * scale
            for i in range(24):
                for j in range(16):
                    u = i * 2 * np.pi / 24
                    t = j * 2 * np.pi / 16
                    x = (R + r * np.cos(t)) * np.cos(u)
                    y = (R + r * np.cos(t)) * np.sin(u)
                    z = r * np.sin(t)
                    v.append([x, y, z])
            v = np.asarray(v, dtype=np.float32)
            f = []
            for i in range(23):
                for j in range(15):
                    a = i * 16 + j
                    b = a + 1
                    c = (i + 1) * 16 + j
                    d2 = c + 1
                    f.append([a, b, c]); f.append([b, d2, c])
            return v, np.asarray(f, dtype=np.uint32)
        if shape == 'sphere':
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            v = np.array([
                [-1, phi, 0],[1, phi, 0],[-1,-phi, 0],[1,-phi, 0],
                [0,-1, phi],[0, 1, phi],[0,-1,-phi],[0, 1,-phi],
                [phi, 0,-1],[phi, 0, 1],[-phi,0,-1],[-phi,0, 1]
            ], dtype=np.float32)
            v = v * (scale / np.linalg.norm([1, phi, 0]))
            f = np.array([
                [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
                [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
                [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
                [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1]
            ], dtype=np.uint32)
            return v, f
        # reuse simpler shapes
        from ..phase10.text_to_3d_generator import TextTo3DGenerator  # type: ignore
        gen = TextTo3DGenerator()
        v, f = gen._generate_shape_geometry(shape, emb)
        return v * (scale / (np.max(np.abs(v)) if np.max(np.abs(v)) > 0 else 1.0)), f

    def _write_glb(self, glb_path: Path, vertices: np.ndarray, faces: np.ndarray, extras: Dict[str, Any]) -> None:
        vbytes = vertices.astype(np.float32).tobytes()
        ib = faces.reshape(-1).astype(np.uint32)
        ibytes = ib.tobytes()
        blob = vbytes + ibytes
        gltf = GLTF2(asset=Asset(generator="k3d-dream-engine"), scenes=[Scene(nodes=[0])], scene=0)
        gltf.buffers.append(Buffer(byteLength=len(blob)))
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=len(vbytes), target=34962))
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(vbytes), byteLength=len(ibytes), target=34963))
        gltf.accessors.append(Accessor(bufferView=0, byteOffset=0, componentType=5126, count=int(vertices.shape[0]), type="VEC3", min=vertices.min(axis=0).astype(float).tolist(), max=vertices.max(axis=0).astype(float).tolist()))
        gltf.accessors.append(Accessor(bufferView=1, byteOffset=0, componentType=5125, count=int(ib.size), type="SCALAR"))
        gltf.meshes.append(Mesh(primitives=[Primitive(attributes={"POSITION": 0}, indices=1, mode=4)]))
        gltf.nodes.append(Node(mesh=0, extras={"k3d": extras}))
        try:
            gltf.set_binary_blob(blob)
            gltf.save_binary(str(glb_path))
        except Exception:
            import base64
            uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")
            gltf.buffers[0].uri = uri
            gltf.save(str(glb_path.with_suffix('.gltf')))

    # ---------- Dream ----------
    def dream(self, num_dreams: int = 3) -> List[Dict[str, Any]]:
        print("🌌 Initiating Dream Sequence...")
        base = self.load_galaxy_embeddings()
        out: List[Dict[str, Any]] = []
        for _ in range(int(num_dreams)):
            try:
                d_emb = self.generate_dream_embedding(base)
                # choose shape
                shape_types = ["tetrahedron","cube","octahedron","icosahedron","dodecahedron","sphere","torus"]
                idx = int(abs(float(np.sum(d_emb[:3] * 1000.0)))) % len(shape_types)
                shape = shape_types[idx]
                v, f = self._shape_geometry(shape, d_emb)
                h = float(self._drift_honesty(d_emb, np.zeros_like(d_emb)))
                did = f"dream_{shape}_{int(np.sum(d_emb * 1000)) % 1000000}_{int(datetime.now().timestamp())}"
                glb_path = self.material_dir / f"{did}.glb"
                extras = {
                    "type": "dream_shape",
                    "name": f"Dream: {shape} from drift",
                    "created_at": datetime.now().isoformat(),
                    "honesty_score": h,
                    "embedding": d_emb.astype(float).tolist(),
                    "shape_type": shape,
                    "vertex_count": int(v.shape[0]),
                    "face_count": int(f.shape[0]),
                    "zone_placement": "Zone 6 (Dream Chamber)",
                    "source": "internal_dream",
                }
                self._write_glb(glb_path, v, f, extras)
                out.append({"path": str(glb_path), "data": extras})
            except Exception as e:
                print(f"⚠️  Dream failed: {e}")
        return out
