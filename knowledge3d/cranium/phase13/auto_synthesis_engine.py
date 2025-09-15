from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Asset  # type: ignore


class AutoSynthesisEngine:
    """Autonomous Meaning Synthesis during sleep.

    - Loads stars (id, embedding, honesty) from Galaxy GLB (extras.k3d)
    - Finds high-similarity, high-honesty pairs
    - Fuses embeddings and materializes a synthesized shape GLB (Zone 5)
    - Emits a companion rays GLB via RayBundleGenerator when available
    """

    def __init__(self, house_path: str, galaxy_path: str, material_dir: str):
        self.house_path = Path(house_path)
        self.galaxy_path = Path(galaxy_path)
        self.material_dir = Path(material_dir)
        self.material_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Loaders ----------
    def load_galaxy(self) -> List[Dict[str, Any]]:
        try:
            gltf = GLTF2().load(str(self.galaxy_path))
        except Exception:
            return []
        stars: List[Dict[str, Any]] = []
        for node in (gltf.nodes or []):
            extras = getattr(node, 'extras', None)
            if hasattr(extras, 'to_dict'):
                try:
                    extras = extras.to_dict()
                except Exception:
                    extras = dict(extras)
            if not isinstance(extras, dict):
                continue
            k3d = extras.get('k3d') if isinstance(extras.get('k3d'), dict) else None
            if not isinstance(k3d, dict):
                continue
            if str(k3d.get('type') or '').lower() != 'star':
                continue
            emb = k3d.get('embedding')
            if isinstance(emb, list) and emb:
                try:
                    emb_arr = np.asarray(emb, dtype=np.float32)
                except Exception:
                    continue
                stars.append({
                    'id': k3d.get('id') or node.name or f'star_{len(stars)}',
                    'embedding': emb_arr,
                    'honesty_score': float(k3d.get('honesty_score', 0.5)),
                    'position': list(k3d.get('position', [0.0, 0.0, 0.0])),
                })
        return stars

    # ---------- Similarity ----------
    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.size == 0 or b.size == 0:
            return 0.0
        an = a / (np.linalg.norm(a) + 1e-9)
        bn = b / (np.linalg.norm(b) + 1e-9)
        return float(np.dot(an, bn))

    def find_synthesis_candidates(
        self,
        stars: List[Dict[str, Any]],
        similarity_threshold: float = 0.7,
        honesty_threshold: float = 0.6,
        limit: int = 16,
    ) -> List[Dict[str, Any]]:
        cand: List[Dict[str, Any]] = []
        n = len(stars)
        for i in range(n):
            for j in range(i + 1, n):
                s = self._cosine_sim(stars[i]['embedding'], stars[j]['embedding'])
                avg_h = 0.5 * (float(stars[i]['honesty_score']) + float(stars[j]['honesty_score']))
                if s >= similarity_threshold and avg_h >= honesty_threshold:
                    cand.append({
                        'star_a': stars[i],
                        'star_b': stars[j],
                        'similarity': s,
                        'avg_honesty': avg_h,
                    })
                if len(cand) >= limit:
                    return cand
        return cand

    # ---------- Geometry ----------
    def _shape_geometry(self, shape: str, emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        scale = 1.0 + float(np.linalg.norm(emb[:3]))
        if shape == 'tetrahedron':
            v = np.array([[1,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]], dtype=np.float32) * scale
            f = np.array([[0,1,2],[0,2,3],[0,3,1],[1,3,2]], dtype=np.uint32)
            return v, f
        if shape == 'cube':
            v = np.array([
                [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                [-1,-1, 1],[1,-1, 1],[1,1, 1],[-1,1, 1],
            ], dtype=np.float32) * scale
            f = np.array([
                [0,1,2],[0,2,3],[4,5,6],[4,6,7],
                [0,1,5],[0,5,4],[2,3,7],[2,7,6],
                [1,2,6],[1,6,5],[0,3,7],[0,7,4]
            ], dtype=np.uint32)
            return v, f
        if shape == 'octahedron':
            v = np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=np.float32) * scale
            f = np.array([[0,2,4],[0,4,3],[0,3,5],[0,5,2],[1,4,2],[1,3,4],[1,5,3],[1,2,5]], dtype=np.uint32)
            return v, f
        if shape == 'icosahedron':
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
        # dodecahedron or fallback: noisy sphere approx
        rng = np.random.default_rng(42)
        v = rng.normal(0.0, 1.0, size=(48, 3)).astype(np.float32)
        v = v * (scale / (np.linalg.norm(v[:1])+1e-6))
        # simple triangulation fan
        f = []
        for i in range(1, v.shape[0]-1):
            f.append([0, i, i+1])
        return v, np.asarray(f, dtype=np.uint32)

    def _write_glb(self, glb_path: Path, vertices: np.ndarray, faces: np.ndarray, extras: Dict[str, Any]) -> None:
        vbytes = vertices.astype(np.float32).tobytes()
        ib = faces.reshape(-1).astype(np.uint32)
        ibytes = ib.tobytes()
        blob = vbytes + ibytes
        gltf = GLTF2(asset=Asset(generator="k3d-auto-synthesis"), scenes=[Scene(nodes=[0])], scene=0)
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

    def synthesize_new_shape(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        emb_a = np.asarray(candidate['star_a']['embedding'], dtype=np.float32)
        emb_b = np.asarray(candidate['star_b']['embedding'], dtype=np.float32)
        fused = (emb_a + emb_b) / 2.0
        shape_types = ["tetrahedron", "cube", "octahedron", "icosahedron", "dodecahedron"]
        sidx = int(abs(float(np.sum(fused[:3] * 1000.0)))) % len(shape_types)
        shape = shape_types[sidx]
        v, f = self._shape_geometry(shape, fused)
        sid = f"synth_{shape}_{candidate['star_a']['id']}_{candidate['star_b']['id']}_{int(datetime.now().timestamp())}"
        out = self.material_dir / f"{sid}.glb"
        extras = {
            "type": "synthesized_shape",
            "name": f"Synthesis: {candidate['star_a']['id']} + {candidate['star_b']['id']}",
            "created_at": datetime.now().isoformat(),
            "honesty_score": float(candidate['avg_honesty']),
            "embedding": fused.astype(float).tolist(),
            "shape_type": shape,
            "vertex_count": int(v.shape[0]),
            "face_count": int(f.shape[0]),
            "source_stars": [candidate['star_a']['id'], candidate['star_b']['id']],
            "similarity": float(candidate['similarity']),
            "zone_placement": "Zone 5 (Knowledge Garden)",
        }
        self._write_glb(out, v, f, extras)
        # Auto-generate rays for synthesized shape
        try:
            from ..phase10.ray_bundle_generator import RayBundleGenerator  # type: ignore
            rb = RayBundleGenerator()
            rb.generate_rays_from_shape(str(out), modality="text", honesty_score=float(candidate['avg_honesty']))
        except Exception:
            pass
        return {"path": str(out), "data": extras}

    def run_synthesis(self, max_items: int = 5) -> List[Dict[str, Any]]:
        stars = self.load_galaxy()
        if not stars:
            return []
        cands = self.find_synthesis_candidates(stars)
        results: List[Dict[str, Any]] = []
        for c in cands[:max_items]:
            try:
                res = self.synthesize_new_shape(c)
                results.append(res)
            except Exception:
                continue
        return results

