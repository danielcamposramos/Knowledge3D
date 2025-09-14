from __future__ import annotations

from typing import List

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from .sample_loader import SampleLoader  # type: ignore
from ...cranium.phase9.shape_recognizer import ShapeRecognizer  # type: ignore


class TrainingPipeline:
    def __init__(self, sample_loader: SampleLoader):
        self.sample_loader = sample_loader
        self.shape_recognizer = ShapeRecognizer()
        if torch is not None and isinstance(self.shape_recognizer, object) and hasattr(self.shape_recognizer, 'parameters'):
            self.optimizer = torch.optim.Adam(self.shape_recognizer.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()  # type: ignore
        else:
            self.optimizer = None
            self.criterion = None

    def create_dataset(self, samples: List[dict]):
        if torch is None:
            raise RuntimeError('PyTorch not available')
        # Build supervised pairs from samples with known shape_type
        shape_to_idx = {
            'tetrahedron': 0, 'cube': 1, 'octahedron': 2, 'icosahedron': 3, 'dodecahedron': 4,
            'triangular_prism': 5, 'pentagonal_prism': 6, 'rhombic_dodecahedron': 7,
            'truncated_icosahedron': 8, 'snub_dodecahedron': 9, 'great_rhombicuboctahedron': 10,
            'omnitruncated_icosahedron': 11, 'hypersphere_projection': 12
        }
        X: List[List[float]] = []
        y: List[int] = []
        # Normalize to 256 dims by pad/trim
        def _norm(v: List[float], d: int = 256) -> List[float]:
            vv = list(v or [])
            if len(vv) < d: vv = vv + [0.0]*(d-len(vv))
            if len(vv) > d: vv = vv[:d]
            return vv
        for s in samples:
            emb = s.get('embedding') or []
            shp = s.get('shape_type') or 'unknown'
            if not emb or shp not in shape_to_idx:
                continue
            X.append(_norm([float(x) for x in emb]))
            y.append(shape_to_idx[shp])
        if not X:
            raise RuntimeError('No labeled samples with embeddings')
        import torch as _t
        return TensorDataset(_t.tensor(X, dtype=_t.float32), _t.tensor(y, dtype=_t.long))

    def train_shape_recognizer(self, sample_pattern: str = '*.glb', epochs: int = 10, batch_size: int = 16) -> None:
        if torch is None:
            raise RuntimeError('PyTorch not available')
        samples = self.sample_loader.load_samples(pattern=sample_pattern)
        if not samples:
            raise RuntimeError('No samples found')
        ds = self.create_dataset(samples)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        for ep in range(int(epochs)):
            total = 0.0
            for xb, yb in dl:
                self.optimizer.zero_grad()  # type: ignore
                logits = self.shape_recognizer(xb)
                loss = self.criterion(logits, yb)  # type: ignore
                loss.backward()
                self.optimizer.step()  # type: ignore
                total += float(loss.item())
            if ep % 2 == 0:
                print(f"Epoch {ep}: loss={total/max(1,len(dl)):.4f}")
        # Save
        out = Path('viewer/public/models')
        out.mkdir(parents=True, exist_ok=True)
        import torch as _t
        _t.save(self.shape_recognizer.state_dict(), str(out / 'shape_recognizer.pth'))
        print('Saved model to', out / 'shape_recognizer.pth')


if __name__ == '__main__':  # pragma: no cover
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='viewer/public/samples')
    ap.add_argument('--pattern', default='*.glb')
    ap.add_argument('--epochs', type=int, default=10)
    args = ap.parse_args()
    sl = SampleLoader(args.samples)
    tp = TrainingPipeline(sl)
    tp.train_shape_recognizer(sample_pattern=args.pattern, epochs=args.epochs)

