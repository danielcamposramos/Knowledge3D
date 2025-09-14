from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

from pygltflib import (
    GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor, Material, PbrMetallicRoughness
)


def _align4(n: int) -> int: return (n + 3) & ~3


def _load_zone_coords(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            pass
    # default layout
    return {
        'house_version': '1.0',
        'center_position': [0.0, 0.0, 0.0],
        'zones': {
            'library': { 'position': [-30.0, 0.0, 0.0], 'rotation':[0.0,0.0,0.0], 'description': 'West wing — books, shelves' },
            'garden':  { 'position': [ 30.0, 0.0, 0.0], 'rotation':[0.0,0.0,0.0], 'description': 'East wing — knowledge trees' },
            'workshop':{ 'position': [  0.0, 0.0, 30.0], 'rotation':[0.0,0.0,0.0], 'description': 'North — manipulation lab' },
            'bathtub': { 'position': [  0.0, 0.0,-30.0], 'rotation':[0.0,0.0,0.0], 'description': 'South — sleep chamber' },
            'living_room': { 'position': [0.0, 0.0, 0.0], 'rotation':[0.0,0.0,0.0], 'description': 'Center — sofa + screens' },
        }
    }


class HouseMasterRenderer:
    def __init__(self, master_layout_path: str, zone_objects_dir: str, zone_coords_path: Optional[str] = None) -> None:
        self.master_layout_path = Path(master_layout_path)
        self.zone_objects_dir = Path(zone_objects_dir)
        repo = Path(__file__).resolve().parents[2]
        self.zone_coords_path = Path(zone_coords_path) if zone_coords_path else (repo / 'viewer' / 'public' / 'house' / 'zone_coordinates.json')
        self._zones = _load_zone_coords(self.zone_coords_path)

    def _zone_position(self, zone: str) -> Tuple[float, float, float]:
        z = (self._zones.get('zones') or {}).get(zone) or {}
        p = z.get('position') or [0.0, 0.0, 0.0]
        return float(p[0]), float(p[1]), float(p[2])

    def render_house_master(self, output_path: str) -> str:
        master = GLTF2().load_binary(str(self.master_layout_path))
        base_blob = master.binary_blob() or b''
        chunks = [base_blob]
        offset = len(base_blob)

        # ensure structures
        master.buffers = master.buffers or [Buffer(byteLength=0)]
        if not master.scenes:
            master.scenes = [Scene(nodes=[])]
            master.scene = 0

        def append_bytes(bs: bytes):
            nonlocal offset
            start = offset; chunks.append(bs); offset += len(bs)
            pad = _align4(offset) - offset
            if pad: chunks.append(b"\x00"*pad); offset += pad
            return start, len(bs)

        # iterate zone dirs
        for zone in ["library","garden","workshop","bathtub","living_room"]:
            zdir = self.zone_objects_dir / zone
            if not zdir.exists() or not zdir.is_dir():
                continue
            for p in sorted(zdir.glob('*.glb')):
                try:
                    obj = GLTF2().load_binary(str(p))
                    oblob = obj.binary_blob() or b''
                    if not obj.meshes or not obj.meshes[0].primitives:
                        continue
                    prim0 = obj.meshes[0].primitives[0]
                    # POSITION accessor
                    # POSITION accessor index (support dict or Attributes)
                    if not prim0.attributes:
                        continue
                    if isinstance(prim0.attributes, dict):
                        if 'POSITION' not in prim0.attributes:
                            continue
                        pos_ai = prim0.attributes['POSITION']
                    else:
                        if not hasattr(prim0.attributes, 'POSITION'):
                            continue
                        pos_ai = prim0.attributes.POSITION
                    pos_acc = obj.accessors[pos_ai]
                    pos_bv = obj.bufferViews[pos_acc.bufferView]
                    p_start = pos_bv.byteOffset or 0
                    p_len = pos_bv.byteLength or 0
                    pos_bytes = oblob[p_start:p_start+p_len]
                    # Indices
                    idx_bytes = b''; idx_acc = None
                    if prim0.indices is not None:
                        idx_acc = obj.accessors[prim0.indices]
                        idx_bv = obj.bufferViews[idx_acc.bufferView]
                        i_start = idx_bv.byteOffset or 0
                        i_len = idx_bv.byteLength or 0
                        idx_bytes = oblob[i_start:i_start+i_len]

                    # Optional embeddings view
                    emb_bytes = b''; emb_dims = 0
                    k3d = (prim0.extras or {}).get('k3d') if prim0.extras else None
                    if isinstance(k3d, dict):
                        ev = k3d.get('embeddingsView')
                        if isinstance(ev, int) and ev < len(obj.bufferViews):
                            ebv = obj.bufferViews[ev]
                            e_start = ebv.byteOffset or 0
                            e_len = ebv.byteLength or 0
                            emb_bytes = oblob[e_start:e_start+e_len]
                            emb_dims = int(k3d.get('embeddingDims') or 0)

                    # append bytes
                    p_off, p_len2 = append_bytes(pos_bytes)
                    i_off, i_len2 = (append_bytes(idx_bytes) if idx_bytes else (offset, 0))
                    e_off, e_len2 = (append_bytes(emb_bytes) if emb_bytes else (offset, 0))

                    # buffer views
                    if master.bufferViews is None: master.bufferViews = []
                    bvi = len(master.bufferViews)
                    master.bufferViews.append(BufferView(buffer=0, byteOffset=p_off, byteLength=p_len2, target=34962))
                    if idx_bytes:
                        master.bufferViews.append(BufferView(buffer=0, byteOffset=i_off, byteLength=i_len2, target=34963))
                    if emb_bytes:
                        master.bufferViews.append(BufferView(buffer=0, byteOffset=e_off, byteLength=e_len2, target=34962))

                    # accessors (mirror types)
                    if master.accessors is None: master.accessors = []
                    ai = len(master.accessors)
                    master.accessors.append(Accessor(bufferView=bvi + 0, componentType=pos_acc.componentType, count=pos_acc.count, type=pos_acc.type))
                    if idx_bytes and idx_acc:
                        master.accessors.append(Accessor(bufferView=bvi + 1, componentType=idx_acc.componentType, count=idx_acc.count, type=idx_acc.type))

                    # primitive
                    if master.materials is None: master.materials = []
                    if not master.materials:
                        master.materials.append(Material(pbrMetallicRoughness=PbrMetallicRoughness(metallicFactor=0.0, roughnessFactor=0.6)))
                    prim = Primitive(); prim.attributes = {'POSITION': ai}; prim.indices = (ai+1) if idx_bytes else None; prim.mode = 4; prim.material = 0
                    # merge extras, annotate zone
                    prim.extras = prim0.extras or {}
                    if not isinstance(prim.extras, dict): prim.extras = {}
                    prim.extras['k3d_zone'] = {'name': zone}

                    if master.meshes is None: master.meshes = []
                    mesh_index = len(master.meshes)
                    master.meshes.append(Mesh(primitives=[prim]))

                    if master.nodes is None: master.nodes = []
                    node = Node(mesh=mesh_index, name=p.stem)
                    x,y,z = self._zone_position(zone)
                    node.translation = [float(x), float(y), float(z)]
                    master.nodes.append(node)
                    if not master.scenes:
                        master.scenes = [Scene(nodes=[])]; master.scene = 0
                    if master.scenes[0].nodes is None: master.scenes[0].nodes = []
                    master.scenes[0].nodes.append(len(master.nodes)-1)
                except Exception:
                    continue

        # finalize
        total_blob = b''.join(chunks)
        master.buffers[0].byteLength = len(total_blob)
        master.set_binary_blob(total_blob)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        master.save_binary(str(out))
        return str(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument('--master', default=str(repo / 'viewer' / 'public' / 'house' / 'house_master.glb'))
    ap.add_argument('--zones', default=str(repo / 'viewer' / 'public' / 'house'))
    ap.add_argument('--out', default=str(repo / 'viewer' / 'public' / 'house' / 'house_master_assembled.glb'))
    ap.add_argument('--coords', default=str(repo / 'viewer' / 'public' / 'house' / 'zone_coordinates.json'))
    args = ap.parse_args()
    r = HouseMasterRenderer(args.master, args.zones, zone_coords_path=args.coords)
    print('Wrote', r.render_house_master(args.out))


if __name__ == '__main__':
    main()
