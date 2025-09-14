from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict

from pygltflib import GLTF2  # type: ignore


@dataclass
class Sample:
    id: str
    filepath: str
    geometry: Dict
    embedding: List[float]
    media_types: List[str]
    shape_type: str


class SampleLoader:
    def __init__(self, sample_dir: str):
        self.sample_dir = Path(sample_dir)
        self.loaded_samples: List[Sample] = []

    def load_samples(self, pattern: str = "*.glb") -> List[Dict]:
        out: List[Dict] = []
        for fp in sorted(self.sample_dir.glob(pattern)):
            s = self.load_sample(fp)
            if s:
                out.append({
                    'id': s.id,
                    'filepath': s.filepath,
                    'geometry': s.geometry,
                    'embedding': s.embedding,
                    'media_types': s.media_types,
                    'shape_type': s.shape_type,
                })
        return out

    def _read_bytes(self, glb: GLTF2, view_index: int) -> bytes:
        bv = glb.bufferViews[view_index]
        b = glb.buffers[bv.buffer]
        blob = glb.binary_blob() or b''
        start = (bv.byteOffset or 0)
        end = start + (bv.byteLength or 0)
        return blob[start:end]

    def _floats_from_view(self, glb: GLTF2, view_index: int) -> List[float]:
        bs = self._read_bytes(glb, view_index)
        if not bs:
            return []
        n = len(bs) // 4
        return list(struct.unpack('<' + 'f'*n, bs))

    def _ints_from_view(self, glb: GLTF2, view_index: int, comp_size: int = 4, fmt: str = 'I') -> List[int]:
        bs = self._read_bytes(glb, view_index)
        if not bs:
            return []
        n = len(bs) // comp_size
        return list(struct.unpack('<' + fmt*n, bs))

    def load_sample(self, filepath: Path) -> Optional[Sample]:
        try:
            glb = GLTF2().load_binary(str(filepath))
        except Exception:
            return None
        if not glb.meshes:
            return None
        # Choose a primitive: prefer one with embeddingsView; fallback to first
        prim = None
        prim_mesh_index = 0
        for mi, m in enumerate(glb.meshes or []):
            for pr in (m.primitives or []):
                k3d = (pr.extras or {}).get('k3d') if pr.extras else None
                if isinstance(k3d, dict) and isinstance(k3d.get('embeddingsView'), int):
                    prim = pr
                    prim_mesh_index = mi
                    break
            if prim is not None:
                break
        if prim is None:
            # fallback to first primitive of first mesh
            m0 = glb.meshes[0]
            if not m0.primitives:
                return None
            prim = m0.primitives[0]
            prim_mesh_index = 0
        # POSITION accessor index (dict or Attributes)
        if not prim.attributes:
            return None
        if isinstance(prim.attributes, dict):
            if 'POSITION' not in prim.attributes:
                return None
            pos_ai = prim.attributes['POSITION']
        else:
            if not hasattr(prim.attributes, 'POSITION'):
                return None
            pos_ai = prim.attributes.POSITION
        pos_acc = glb.accessors[pos_ai]
        pos_view = pos_acc.bufferView
        vertices = self._floats_from_view(glb, pos_view)
        indices: List[int] = []
        if prim.indices is not None:
            idx_acc = glb.accessors[prim.indices]
            idx_view = idx_acc.bufferView
            if idx_acc.componentType == 5123:
                indices = self._ints_from_view(glb, idx_view, comp_size=2, fmt='H')
            elif idx_acc.componentType == 5125:
                indices = self._ints_from_view(glb, idx_view, comp_size=4, fmt='I')
        geom = {
            'vertices': vertices,
            'indices': indices,
            'vertex_count': len(vertices) // 3,
            'face_count': len(indices) // 3,
        }
        # Extract embedding from chosen primitive (if present)
        embedding: List[float] = []
        k3d = (prim.extras or {}).get('k3d') if prim.extras else None
        if isinstance(k3d, dict):
            ev = k3d.get('embeddingsView')
            if isinstance(ev, int):
                embedding = self._floats_from_view(glb, ev)
        # Infer shape/media from extras or geometry
        md = prim.extras or {}
        shape = None
        if isinstance(md, dict):
            ws = md.get('k3d_workshop')
            if isinstance(ws, dict) and isinstance(ws.get('shape_type'), str):
                shape = ws['shape_type']
            bt = md.get('k3d_bathtub')
            if shape is None and isinstance(bt, dict):
                fk = bt.get('furniture_kind')
                if isinstance(fk, str):
                    fk_l = fk.lower()
                    if fk_l in { 'display_case' }:
                        shape = 'icosahedron'
                    elif fk_l in { 'bookshelf', 'workbench', 'gray_shell', 'pillar', 'lamp' }:
                        shape = 'cube'
        if shape is None:
            shape = self.infer_shape_type(geom)
        media = self.infer_media_types(geom, embedding)
        return Sample(id=filepath.stem, filepath=str(filepath), geometry=geom, embedding=embedding, media_types=media, shape_type=shape)

    def infer_shape_type(self, geometry: Dict) -> str:
        v = int(geometry.get('vertex_count') or 0)
        if v in (4, 12):
            return 'tetrahedron' if v == 4 else 'icosahedron'
        if v == 8:
            return 'cube'
        return 'unknown'

    def infer_media_types(self, geometry: Dict, embedding: List[float]) -> List[str]:
        # Heuristic: tetrahedron → text, many vertices → image/model
        v = int(geometry.get('vertex_count') or 0)
        if v == 4:
            return ['text']
        if v >= 100:
            return ['image']
        return []
