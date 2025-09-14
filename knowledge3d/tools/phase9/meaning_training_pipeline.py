from __future__ import annotations

from typing import List
from pathlib import Path

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


class MeaningCentricTrainingPipeline:
    def __init__(self, sample_loader: SampleLoader):
        self.sample_loader = sample_loader
        self.shape_recognizer = ShapeRecognizer()
        if torch is not None and hasattr(self.shape_recognizer, 'parameters'):
            self.optimizer = torch.optim.Adam(self.shape_recognizer.parameters(), lr=0.001)
            self.criterion = nn.CrossEntropyLoss()  # type: ignore
        else:
            self.optimizer = None
            self.criterion = None

    def _norm(self, v: List[float], d: int = 256) -> List[float]:
        vv = list(v or [])
        if len(vv) < d: vv = vv + [0.0]*(d-len(vv))
        if len(vv) > d: vv = vv[:d]
        return vv

    def create_dataset(self, samples: List[dict]):
        if torch is None:
            raise RuntimeError('PyTorch not available')
        shape_to_idx = {
            'tetrahedron': 0, 'cube': 1, 'octahedron': 2, 'icosahedron': 3, 'dodecahedron': 4,
            'triangular_prism': 5, 'pentagonal_prism': 6, 'rhombic_dodecahedron': 7,
            'truncated_icosahedron': 8, 'snub_dodecahedron': 9, 'great_rhombicuboctahedron': 10,
            'omnitruncated_icosahedron': 11, 'hypersphere_projection': 12
        }
        X: List[List[float]] = []
        y: List[int] = []
        for s in samples:
            emb = s.get('embedding') or []
            shp = s.get('shape_type') or 'unknown'
            if not emb or shp not in shape_to_idx:
                continue
            X.append(self._norm([float(x) for x in emb]))
            y.append(shape_to_idx[shp])
        if not X:
            raise RuntimeError('No labeled samples with embeddings')
        import torch as _t
        return TensorDataset(_t.tensor(X, dtype=_t.float32), _t.tensor(y, dtype=_t.long))

    def meaning_centric_loss(self, logits, embeddings, labels):
        ce = self.criterion(logits, labels)  # type: ignore
        emb = nn.functional.normalize(embeddings, dim=1)
        cos = emb @ emb.T
        probs = nn.functional.softmax(logits, dim=1)
        diffs = (probs.unsqueeze(1) - probs.unsqueeze(0)) ** 2
        meaning = (diffs.mean(dim=2) * cos).mean()
        return ce + 0.1 * meaning

    def train(self, epochs: int = 50, batch_size: int = 16, pattern: str = '*.glb', out_path: str = 'viewer/public/models/shape_recognizer_final.pth') -> None:
        if torch is None:
            raise RuntimeError('PyTorch not available')
        samples = self.sample_loader.load_samples(pattern=pattern)
        ds = self.create_dataset(samples)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = self.shape_recognizer
        try:
            model.to(device)  # type: ignore[attr-defined]
        except Exception:
            pass
        for ep in range(int(epochs)):
            total = 0.0
            for xb, yb in dl:
                xb = xb.to(device)
                yb = yb.to(device)
                self.optimizer.zero_grad()  # type: ignore
                logits = model(xb)  # type: ignore[misc]
                loss = self.meaning_centric_loss(logits, xb, yb)
                loss.backward()
                self.optimizer.step()  # type: ignore
                total += float(loss.item())
            if ep % 5 == 0:
                print(f"Epoch {ep}: loss={total/max(1,len(dl)):.4f}")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        import torch as _t
        _t.save(model.state_dict(), out_path)
        print('Saved model to', out_path)


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', default='viewer/public/samples')
    ap.add_argument('--pattern', default='*.glb')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--out', default='viewer/public/models/shape_recognizer_final.pth')
    args = ap.parse_args()
    sl = SampleLoader(args.samples)
    tp = MeaningCentricTrainingPipeline(sl)
    tp.train(epochs=int(args.epochs), pattern=args.pattern, out_path=args.out)


if __name__ == '__main__':  # pragma: no cover
    main()
