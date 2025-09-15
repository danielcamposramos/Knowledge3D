from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np  # type: ignore
import struct
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor


class MinimalGLBSynthesizer:
    def __init__(self, base_dir: str = 'viewer/public/samples') -> None:
        self.base = Path(base_dir)

    def synthesize_stage_glbs(self, stage: int, count: int = 10) -> int:
        stage_dir = self.base / f'stage{stage}'
        stage_dir.mkdir(parents=True, exist_ok=True)
        added = 0
        for i in range(int(count)):
            if stage == 2:
                shape_type = 'triangular_prism' if (i % 2 == 0) else 'pentagonal_prism'
                media = ['text','image'] if (i % 2 == 0) else ['text','audio']
                dim = 128
            elif stage == 3:
                shape_type = 'truncated_icosahedron' if (i % 2 == 0) else 'snub_dodecahedron'
                media = ['text','image','audio']
                dim = 256
            elif stage == 4:
                shape_type = 'hypersphere_projection' if (i % 2 == 0) else 'fractal_tree'
                media = ['text','image','audio','video','spatial']
                dim = 512
            else:
                break
            out = stage_dir / f'stage{stage}_sample_{i:03d}.glb'
            emb = self._gen_embedding(dim, seed=1000+stage*100+i)
            self._write_minimal_glb(out, shape_type, media, emb)
            print(f"✅ Synthesized {out}")
            added += 1
        return added

    def _gen_embedding(self, dim: int, seed: int) -> List[float]:
        rng = np.random.default_rng(seed)
        v = rng.normal(0.0, 1.0, size=int(dim)).astype('float32')
        # L2 normalize
        n = float(np.linalg.norm(v) + 1e-9)
        return (v / n).tolist()

    def _write_minimal_glb(self, out_path: Path, shape_type: str, media_types: List[str], embedding: List[float]) -> None:
        verts = [0.0,0.0,0.0,  1.0,0.0,0.0,  0.0,1.0,0.0]
        idx = [0,1,2]
        pos_bytes = struct.pack('<' + 'f'*len(verts), *verts)
        idx_bytes = struct.pack('<' + 'I'*len(idx), *idx)
        emb_bytes = struct.pack('<' + 'f'*len(embedding), *embedding)
        def align4(n: int) -> int: return (n + 3) & ~3
        off = 0; chunks = []
        p_off = off; chunks.append(pos_bytes); off += len(pos_bytes); pad = align4(off)-off; chunks.append(b"\x00"*pad); off += pad
        i_off = off; chunks.append(idx_bytes); off += len(idx_bytes); pad = align4(off)-off; chunks.append(b"\x00"*pad); off += pad
        e_off = off; chunks.append(emb_bytes); off += len(emb_bytes); pad = align4(off)-off; chunks.append(b"\x00"*pad); off += pad
        blob = b''.join(chunks)

        glb = GLTF2()
        glb.buffers = [Buffer(byteLength=len(blob))]
        glb.bufferViews = [
            BufferView(buffer=0, byteOffset=p_off, byteLength=len(pos_bytes), target=34962),
            BufferView(buffer=0, byteOffset=i_off, byteLength=len(idx_bytes), target=34963),
            BufferView(buffer=0, byteOffset=e_off, byteLength=len(emb_bytes), target=34962),
        ]
        glb.accessors = [
            Accessor(bufferView=0, componentType=5126, count=len(verts)//3, type='VEC3'),
            Accessor(bufferView=1, componentType=5125, count=len(idx), type='SCALAR'),
        ]
        prim = Primitive()
        prim.attributes = {'POSITION': 0}
        prim.indices = 1
        prim.mode = 4
        prim.extras = {
            'k3d': {
                'embeddingsView': 2,
                'embeddingDims': len(embedding),
                'media_types': media_types,
                'shape_type': shape_type,
            },
            'k3d_workshop': {
                'shape_type': shape_type,
            }
        }
        mesh = Mesh(primitives=[prim])
        glb.meshes = [mesh]
        glb.nodes = [Node(mesh=0, name=out_path.stem)]
        glb.scenes = [Scene(nodes=[0])]
        glb.scene = 0
        glb.set_binary_blob(blob)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        glb.save_binary(str(out_path))


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--count', type=int, default=10)
    args = ap.parse_args()
    syn = MinimalGLBSynthesizer()
    print('Added', syn.synthesize_stage_glbs(int(args.stage), int(args.count)))


if __name__ == '__main__':  # pragma: no cover
    main()

