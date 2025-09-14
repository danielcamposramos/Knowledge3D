from __future__ import annotations

from typing import List
from pathlib import Path

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

from .sample_loader import SampleLoader  # type: ignore
from ...cranium.phase9.shape_recognizer import ShapeRecognizer  # type: ignore


class MeaningCentricModelValidator:
    def __init__(self, model_path: str):
        if torch is None:
            raise RuntimeError('PyTorch not available')
        self.model = ShapeRecognizer()
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        self.model.eval()

    def validate(self, test_samples: List[dict]) -> float:
        correct = 0
        total = len(test_samples)
        for s in test_samples:
            y_pred = self.model.predict_shape(s.get('embedding') or [])
            y_true = s.get('shape_type') or 'unknown'
            ok = (y_pred == y_true)
            correct += int(ok)
            print(f"🔮 {s.get('id')}: Pred {y_pred}, True {y_true}")
        acc = (correct / total) if total else 0.0
        print(f"🎯 Accuracy: {acc:.2%}")
        return acc


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='viewer/public/models/shape_recognizer_final.pth')
    ap.add_argument('--samples', default='viewer/public/samples')
    ap.add_argument('--limit', type=int, default=10)
    args = ap.parse_args()
    sl = SampleLoader(args.samples)
    all_s = sl.load_samples('*.glb')
    tests = all_s[-int(args.limit):] if all_s else []
    val = MeaningCentricModelValidator(args.model)
    val.validate(tests)


if __name__ == '__main__':  # pragma: no cover
    main()
