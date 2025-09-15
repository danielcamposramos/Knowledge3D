from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np  # type: ignore
from pygltflib import GLTF2, GLTF2 as _GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Asset  # type: ignore


class RayBundleGenerator:
    def __init__(self) -> None:
        self.modality_colors: Dict[str, List[float]] = {
            "text": [0.0, 0.0, 1.0],
            "image": [0.0, 1.0, 0.0],
            "audio": [1.0, 0.0, 0.0],
            "video": [1.0, 1.0, 0.0],
            "3d": [1.0, 0.0, 1.0],
            "spatial": [0.0, 1.0, 1.0],
            "chat": [0.5, 0.5, 0.5],
        }

    def _load_vertices_from_glb(self, glb_path: str) -> np.ndarray:
        try:
            gltf = GLTF2().load(glb_path)
            # Assume first accessor is POSITION (matches our writer)
            bv = gltf.bufferViews[glTFIndex(0)] if gltf.bufferViews else None  # type: ignore
        except Exception:
            gltf = None
        # Fallback: mock points if parse fails
        if gltf is None or not gltf.accessors or not gltf.bufferViews:
            return np.random.randn(64, 3).astype(np.float32)
        try:
            acc = gltf.accessors[0]
            if acc.type != "VEC3":
                return np.random.randn(64, 3).astype(np.float32)
            # Read binary blob via embedded buffer
            blob = gltf.binary_blob()
            start = gltf.bufferViews[acc.bufferView].byteOffset or 0
            length = (acc.count * 3 * 4)
            arr = np.frombuffer(blob[start:start+length], dtype=np.float32)
            return arr.reshape(acc.count, 3)
        except Exception:
            return np.random.randn(64, 3).astype(np.float32)

    def generate_rays_from_shape(self, glb_path: str, modality: str = "text", honesty_score: float | None = None, n_per_vertex: int = 1) -> str:
        verts = self._load_vertices_from_glb(glb_path)
        # Try to get embedding preview and honesty from source shape extras
        emb_preview: list[float] | None = None
        try:
            gltf = GLTF2().load(glb_path)
            if gltf.nodes and getattr(gltf.nodes[0], 'extras', None):
                extras = gltf.nodes[0].extras
                if hasattr(extras, 'to_dict'):
                    extras = extras.to_dict()
                if isinstance(extras, dict):
                    k3d = extras.get('k3d') if isinstance(extras.get('k3d'), dict) else None
                    if isinstance(k3d, dict):
                        if honesty_score is None and isinstance(k3d.get('honesty_score'), (int, float)):
                            honesty_score = float(k3d.get('honesty_score'))
                        emb = k3d.get('embedding')
                        if isinstance(emb, list) and emb:
                            emb_preview = [float(x) for x in emb[:8]]
        except Exception:
            pass
        color = self.modality_colors.get(modality, [1.0, 1.0, 1.0])
        rng = np.random.default_rng()
        # Prepare line segments as pairs of points
        points: list[float] = []
        colors: list[float] = []
        indices: list[int] = []
        thicknesses: list[float] = []
        vi = 0
        for v in verts:
            origin = v.astype(float)
            for _ in range(max(1, int(n_per_vertex))):
                direction = (v + rng.normal(0.0, 0.1, size=3)).astype(float)
                end = origin + (direction - origin)
                points.extend(origin.tolist())
                points.extend(end.tolist())
                colors.extend(color)
                colors.extend(color)
                indices.extend([vi, vi + 1])
                entropy = float(abs(rng.normal(0.5, 0.3)))
                thickness = max(0.01, min(0.1, entropy * 0.05))
                thicknesses.append(thickness)
                vi += 2
        # Write GLB with LINES primitive and COLOR_0 attribute; store thickness in extras.k3d
        out_glb = Path(glb_path).with_name(f"rays_{Path(glb_path).stem}.glb")
        self._write_lines_glb(out_glb, np.array(points, dtype=np.float32), np.array(indices, dtype=np.uint32), np.array(colors, dtype=np.float32), {
            "type": "ray_bundle",
            "source_shape": str(Path(glb_path)),
            "modality": modality,
            "ray_count": len(indices) // 2,
            "ray_thickness": thicknesses,
            "honesty_score": float(honesty_score) if honesty_score is not None else None,
            "embedding_preview": emb_preview,
        })
        return str(out_glb)

    def _write_lines_glb(self, path: Path, positions: np.ndarray, indices: np.ndarray, colors: np.ndarray, k3d_extras: Dict[str, Any]) -> None:
        # positions flat array; colors flat array
        vbytes = positions.tobytes()
        cbytes = colors.tobytes()
        ibytes = indices.tobytes()
        blob = vbytes + cbytes + ibytes
        gltf = GLTF2(asset=Asset(generator="k3d-ray-bundles"), scenes=[Scene(nodes=[0])], scene=0)
        gltf.buffers.append(Buffer(byteLength=len(blob)))
        # BufferViews: positions, colors, indices
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=0, byteLength=len(vbytes), target=34962))  # ARRAY_BUFFER
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(vbytes), byteLength=len(cbytes), target=34962))
        gltf.bufferViews.append(BufferView(buffer=0, byteOffset=len(vbytes)+len(cbytes), byteLength=len(ibytes), target=34963))  # ELEMENT_ARRAY_BUFFER
        # Accessors
        count = positions.size // 3
        gltf.accessors.append(Accessor(bufferView=0, byteOffset=0, componentType=5126, count=int(count), type="VEC3"))
        gltf.accessors.append(Accessor(bufferView=1, byteOffset=0, componentType=5126, count=int(count), type="VEC3"))
        gltf.accessors.append(Accessor(bufferView=2, byteOffset=0, componentType=5125, count=int(indices.size), type="SCALAR"))
        # Mesh primitive as LINES
        prim = Primitive(attributes={"POSITION": 0, "COLOR_0": 1}, indices=2, mode=1)
        gltf.meshes.append(Mesh(primitives=[prim]))
        gltf.nodes.append(Node(mesh=0, extras={"k3d": k3d_extras}))
        # Save GLB
        try:
            gltf.set_binary_blob(blob)
            gltf.save_binary(str(path))
        except Exception:
            # fallback .gltf with DataURI
            import base64
            uri = "data:application/octet-stream;base64," + base64.b64encode(blob).decode("ascii")
            gltf.buffers[0].uri = uri
            gltf.save(str(path.with_suffix('.gltf')))
