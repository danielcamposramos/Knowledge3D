from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple, Set

from pygltflib import GLTF2  # type: ignore


def _has_embeddings(glb: GLTF2) -> bool:
    try:
        for m in (glb.meshes or []):
            for p in (m.primitives or []):
                k3d = (p.extras or {}).get('k3d') if p.extras else None
                if isinstance(k3d, dict) and isinstance(k3d.get('embeddingsView'), int):
                    return True
    except Exception:
        pass
    return False


def _media_types_from_glb(glb: GLTF2) -> Set[str]:
    types: Set[str] = set()
    # Scene-level extras
    try:
        for sc in (glb.scenes or []):
            if getattr(sc, 'extras', None):
                k3d = sc.extras.get('k3d') if isinstance(sc.extras, dict) else None
                if isinstance(k3d, dict):
                    mt = k3d.get('media_types')
                    if isinstance(mt, list):
                        types.update([str(x).lower() for x in mt])
    except Exception:
        pass
    # Primitive-level extras
    try:
        for m in (glb.meshes or []):
            for p in (m.primitives or []):
                if getattr(p, 'extras', None):
                    k3d = p.extras.get('k3d') if isinstance(p.extras, dict) else None
                    if isinstance(k3d, dict):
                        mt = k3d.get('media_types')
                        if isinstance(mt, list):
                            types.update([str(x).lower() for x in mt])
    except Exception:
        pass
    # Heuristic fallback by embedding presence
    if not types and _has_embeddings(glb):
        types.add('embedding')
    return types


class DatasetExpander:
    def __init__(self, source_dirs: List[str], target_dir: str, limit: int = 10) -> None:
        self.source_dirs = [str(Path(d)) for d in source_dirs]
        self.target_dir = Path(target_dir)
        self.limit = int(limit)

    def validate_glb(self, filepath: Path) -> bool:
        try:
            g = GLTF2().load_binary(str(filepath))
            return _has_embeddings(g)
        except Exception:
            return False

    def expand_dataset(self) -> int:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in self.target_dir.glob('*.glb')}
        added = 0
        for sdir in self.source_dirs:
            pdir = Path(sdir)
            if not pdir.exists():
                continue
            for fp in sorted(pdir.glob('**/*.glb')):
                if fp.name in existing:
                    continue
                if not self.validate_glb(fp):
                    continue
                dst = self.target_dir / fp.name
                try:
                    shutil.copyfile(fp, dst)
                    print(f"✅ Added {fp.name} to {self.target_dir}")
                    added += 1
                    if added >= self.limit:
                        return added
                except Exception:
                    continue
        print(f"📊 Added {added} new samples to {self.target_dir}")
        return added


class MeaningCentricDatasetExpander:
    def __init__(self, source_dirs: List[str], target_dir: str, limit: int = 10) -> None:
        self.source_dirs = [str(Path(d)) for d in source_dirs]
        self.target_dir = Path(target_dir)
        self.limit = int(limit)

    def _score(self, filepath: Path) -> Tuple[float, Set[str]]:
        try:
            g = GLTF2().load_binary(str(filepath))
            if not _has_embeddings(g):
                return (0.0, set())
            mt = _media_types_from_glb(g)
            return (float(len(mt)), mt)
        except Exception:
            return (0.0, set())

    def expand_dataset(self) -> int:
        self.target_dir.mkdir(parents=True, exist_ok=True)
        existing = {p.name for p in self.target_dir.glob('*.glb')}
        cands: List[Tuple[Path, float, Set[str]]] = []
        for sdir in self.source_dirs:
            pdir = Path(sdir)
            if not pdir.exists():
                continue
            for fp in sorted(pdir.glob('**/*.glb')):
                if fp.name in existing:
                    continue
                score, mt = self._score(fp)
                if score > 0:
                    cands.append((fp, score, mt))
        cands.sort(key=lambda t: (t[1], t[0].name), reverse=True)
        added = 0
        for fp, score, mt in cands[: self.limit]:
            dst = self.target_dir / fp.name
            try:
                shutil.copyfile(fp, dst)
                print(f"✅ Added {fp.name} (diversity: {score:.2f}, media={sorted(mt)}) to {self.target_dir}")
                added += 1
            except Exception:
                continue
        print(f"📊 Added {added} new samples to {self.target_dir}")
        return added


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', nargs='+', required=True)
    ap.add_argument('--target', default='viewer/public/samples')
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--mode', choices=['simple','meaning'], default='meaning')
    args = ap.parse_args()
    if args.mode == 'meaning':
        de = MeaningCentricDatasetExpander(args.sources, args.target, limit=args.limit)
    else:
        de = DatasetExpander(args.sources, args.target, limit=args.limit)
    n = de.expand_dataset()
    print(f"Done. Added {n} samples.")


if __name__ == '__main__':  # pragma: no cover
    main()
