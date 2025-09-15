from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List
import struct

import requests  # type: ignore
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Buffer, BufferView, Accessor


class DatasetDownloader:
    def __init__(self, sources: List[str], target_dir: str = 'viewer/public/datasets/open_source'):
        self.sources = list(sources or [])
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def download_and_convert(self) -> int:
        added = 0
        for url in self.sources:
            try:
                fp = self.download_dataset(url)
                glb = self.convert_to_glb(fp)
                if self.validate_glb(glb):
                    print(f"✅ Added {glb}")
                    added += 1
                else:
                    print(f"❌ Invalid GLB {glb}")
            except Exception as e:
                print(f"❌ Failed {url}: {e}")
        return added

    def download_dataset(self, url: str) -> Path:
        name = url.split('/')[-1] or 'dataset.bin'
        out = self.target_dir / name
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        out.write_bytes(r.content)
        return out

    def convert_to_glb(self, path: Path) -> Path:
        data = path.read_bytes()
        h = hashlib.sha256(data).digest()
        emb = [(h[i % len(h)] / 255.0) - 0.5 for i in range(128)]
        out = path.with_suffix('.glb')
        self._create_glb_with_embedding(out, emb)
        return out

    def _create_glb_with_embedding(self, out_path: Path, embedding: List[float]) -> None:
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
                'media_types': ['text'],
                'shape_type': 'tetrahedron',
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

    def validate_glb(self, path: Path) -> bool:
        try:
            g = GLTF2().load_binary(str(path))
            m = g.meshes[0].primitives[0]
            ex = m.extras or {}
            k3d = ex.get('k3d') if isinstance(ex, dict) else None
            return isinstance(k3d, dict) and isinstance(k3d.get('embeddingsView'), int)
        except Exception:
            return False


def main():  # pragma: no cover
    import argparse
    from .dataset_locator import DatasetLocator  # type: ignore
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', default='.')
    ap.add_argument('--target', default='viewer/public/datasets/open_source')
    args = ap.parse_args()
    loc = DatasetLocator(args.repo)
    srcs = loc.find_dataset_sources()
    dl = DatasetDownloader(srcs, args.target)
    n = dl.download_and_convert()
    print(f"Downloaded/converted {n} datasets")


if __name__ == '__main__':  # pragma: no cover
    main()

