from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Set

from pygltflib import GLTF2  # type: ignore


def _extract_media_types(glb: GLTF2) -> Set[str]:
    mt: Set[str] = set()
    # Scene-level
    try:
        for sc in (glb.scenes or []):
            if getattr(sc, 'extras', None) and isinstance(sc.extras, dict):
                k3d = sc.extras.get('k3d')
                if isinstance(k3d, dict):
                    m = k3d.get('media_types')
                    if isinstance(m, list):
                        mt.update([str(x).lower() for x in m])
    except Exception:
        pass
    # Primitive-level
    try:
        for m in (glb.meshes or []):
            for p in (m.primitives or []):
                if getattr(p, 'extras', None) and isinstance(p.extras, dict):
                    k3d = p.extras.get('k3d')
                    if isinstance(k3d, dict):
                        m = k3d.get('media_types')
                        if isinstance(m, list):
                            mt.update([str(x).lower() for x in m])
    except Exception:
        pass
    return mt


def _extract_shape_type(glb: GLTF2) -> Optional[str]:
    # Prefer primitive-level workshop extras
    try:
        for m in (glb.meshes or []):
            for p in (m.primitives or []):
                if getattr(p, 'extras', None) and isinstance(p.extras, dict):
                    ws = p.extras.get('k3d_workshop')
                    if isinstance(ws, dict):
                        st = ws.get('shape_type')
                        if isinstance(st, str) and st:
                            return st.lower()
    except Exception:
        pass
    # Scene-level (less common)
    try:
        for sc in (glb.scenes or []):
            if getattr(sc, 'extras', None) and isinstance(sc.extras, dict):
                k3d = sc.extras.get('k3d')
                if isinstance(k3d, dict):
                    st = k3d.get('shape_type')
                    if isinstance(st, str) and st:
                        return st.lower()
    except Exception:
        pass
    return None


def _has_embeddings(glb: GLTF2) -> bool:
    try:
        for m in (glb.meshes or []):
            for p in (m.primitives or []):
                if getattr(p, 'extras', None) and isinstance(p.extras, dict):
                    k3d = p.extras.get('k3d')
                    if isinstance(k3d, dict) and isinstance(k3d.get('embeddingsView'), int):
                        return True
    except Exception:
        pass
    return False


def _embedding_dims(glb: GLTF2) -> int:
    try:
        for m in (glb.meshes or []):
            for p in (m.primitives or []):
                if getattr(p, 'extras', None) and isinstance(p.extras, dict):
                    k3d = p.extras.get('k3d')
                    if isinstance(k3d, dict):
                        dims = k3d.get('embeddingDims')
                        if isinstance(dims, int) and dims > 0:
                            return int(dims)
    except Exception:
        pass
    return 0


class StageDatasetExpander:
    def __init__(self, source_dirs: List[str], target_dir: str):
        self.source_dirs = [str(Path(d)) for d in source_dirs]
        self.target_dir = Path(target_dir)

    # --- Stage filters ---
    def _stage1(self, st: Optional[str], mt: Set[str]) -> bool:
        return (st in {'tetrahedron','cube','octahedron'}) and (len(mt) <= 1)

    def _stage2(self, st: Optional[str], mt: Set[str]) -> bool:
        return (st in {'triangular_prism','pentagonal_prism'}) and (len(mt) == 2)

    def _stage3(self, st: Optional[str], mt: Set[str]) -> bool:
        return (st in {'truncated_icosahedron','snub_dodecahedron','icosahedron'}) and (len(mt) >= 3)

    def _stage4(self, st: Optional[str], mt: Set[str]) -> bool:
        return st in {'hypersphere_projection','fractal_tree'}

    def _validate(self, fp: Path) -> Tuple[bool, Optional[str], Set[str]]:
        try:
            g = GLTF2().load_binary(str(fp))
        except Exception:
            return (False, None, set())
        if not _has_embeddings(g):
            return (False, None, set())
        st = _extract_shape_type(g)
        mt = _extract_media_types(g)
        return (True, st, mt)

    def expand_stage_dataset(self, stage: int, limit: int = 10) -> int:
        if stage not in (1,2,3,4):
            raise ValueError(f"Invalid stage: {stage}")
        stage_dir = self.target_dir / f"stage{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in stage_dir.glob('*.glb')}
        added = 0
        for src in self.source_dirs:
            for fp in sorted(Path(src).glob('**/*.glb')):
                if fp.name in existing:
                    continue
                ok, st, mt = self._validate(fp)
                if not ok:
                    continue
                if stage == 1 and not self._stage1(st, mt):
                    continue
                if stage == 2 and not self._stage2(st, mt):
                    continue
                if stage == 3 and not self._stage3(st, mt):
                    continue
                if stage == 4 and not self._stage4(st, mt):
                    continue
                try:
                    shutil.copyfile(fp, stage_dir / fp.name)
                    print(f"✅ Added {fp.name} to stage{stage}")
                    added += 1
                    if added >= int(limit):
                        return added
                except Exception:
                    continue
        # Fallback: if strict filter produced none, relax to embedding-dims heuristic for modalities
        if added == 0 and stage in (2,3):
            for src in self.source_dirs:
                for fp in sorted(Path(src).glob('**/*.glb')):
                    if fp.name in existing:
                        continue
                    try:
                        g = GLTF2().load_binary(str(fp))
                    except Exception:
                        continue
                    if not _has_embeddings(g):
                        continue
                    dims = _embedding_dims(g)
                    # crude heuristic: 2-mod ~ 64..160; 3+-mod >= 192
                    if stage == 2 and not (64 <= dims <= 160):
                        continue
                    if stage == 3 and not (dims >= 192):
                        continue
                    try:
                        shutil.copyfile(fp, stage_dir / fp.name)
                        print(f"✅ [fallback] Added {fp.name} to stage{stage} (dims={dims})")
                        added += 1
                        if added >= int(limit):
                            return added
                    except Exception:
                        continue
        # Try open-source bucket if still short
        if added < int(limit):
            print(f"🔍 Expanding Stage {stage} with open-source bucket")
            osdir = Path('viewer/public/datasets/open_source')
            for fp in sorted(osdir.glob('*.glb')):
                if fp.name in existing:
                    continue
                ok, st, mt = self._validate(fp)
                if not ok:
                    continue
                if stage == 1 and not self._stage1(st, mt):
                    continue
                if stage == 2 and not self._stage2(st, mt):
                    continue
                if stage == 3 and not self._stage3(st, mt):
                    continue
                if stage == 4 and not self._stage4(st, mt):
                    continue
                try:
                    shutil.copyfile(fp, stage_dir / fp.name)
                    print(f"✅ [open-source] Added {fp.name} to stage{stage}")
                    added += 1
                    if added >= int(limit):
                        break
                except Exception:
                    continue
        print(f"📊 Added {added} samples to stage{stage}")
        return added


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--target', default='viewer/public/samples')
    ap.add_argument('--stage', type=int, required=True)
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()
    ex = StageDatasetExpander(args.sources, args.target)
    n = ex.expand_stage_dataset(int(args.stage), limit=int(args.limit))
    print(f"Done. Added {n} for stage{args.stage}")


if __name__ == '__main__':  # pragma: no cover
    main()
