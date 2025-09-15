from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List, Dict

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None

from .book_processor import BookProcessor  # type: ignore
from ...cranium.phase10.thinking_tag_embedder import ThinkingTagEmbedder  # type: ignore


class ThinkingTagTrainer:
    def __init__(self, book_processor: BookProcessor, input_dim: int = 512):
        self.book_processor = book_processor
        self.input_dim = int(input_dim)
        self.tag_embedder = ThinkingTagEmbedder(input_dim=self.input_dim)
        if torch is not None:
            self.optimizer = torch.optim.Adam(self.tag_embedder.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()  # type: ignore
        else:
            self.optimizer = None
            self.criterion = None

    def train(self, epochs: int = 50, limit: int | None = None, out_model_path: str | None = None, out_tags_path: str | None = None) -> None:
        if torch is None:
            print('❌ PyTorch unavailable for thinking tag training')
            return
        # Process books via Ollama
        tags = self.book_processor.process_books(limit=limit)
        if not tags:
            print('❌ No books processed')
            return
        # Save raw tags
        out_books = Path('viewer/public/books')
        out_books.mkdir(parents=True, exist_ok=True)
        (out_books / 'thinking_tags.json').write_text(json.dumps(tags, ensure_ascii=False, indent=2), encoding='utf-8')
        # Build dataset
        ds, tag_names = self.create_dataset(tags)
        dl = DataLoader(ds, batch_size=16, shuffle=True)
        # Save tag names
        out_models = Path('viewer/public/models')
        out_models.mkdir(parents=True, exist_ok=True)
        tags_out = Path(out_tags_path) if out_tags_path else (out_models / 'tag_names.json')
        tags_out.write_text(json.dumps(tag_names, ensure_ascii=False, indent=2), encoding='utf-8')
        # Train
        for ep in range(int(epochs)):
            total = 0.0
            for xb, yb in dl:
                self.optimizer.zero_grad()  # type: ignore
                logits = self.tag_embedder(xb)
                loss = self.criterion(logits, yb)  # type: ignore
                loss.backward()
                self.optimizer.step()  # type: ignore
                total += float(loss.item())
            if ep % 10 == 0:
                print(f'Epoch {ep}: loss={total/max(1,len(dl)):.4f}')
        # Save weights
        model_out = Path(out_model_path) if out_model_path else (out_models / 'thinking_tag_embedder.pth')
        torch.save(self.tag_embedder.state_dict(), str(model_out))
        print(f'✅ Thinking tag embedder trained and saved to {model_out}')

    def create_dataset(self, thinking_tags: List[Dict]) -> tuple[TensorDataset, List[str]]:
        import torch as _t
        X: List[List[float]] = []
        Y: List[List[float]] = []
        tag_to_idx = self._build_tag_index(thinking_tags)
        for rec in thinking_tags:
            emb = self._hash_embedding(rec.get('source_path') or '')
            lab = self._label_vector(rec.get('thinking_tags') or [], tag_to_idx)
            X.append(emb)
            Y.append(lab)
        ds = TensorDataset(_t.tensor(X, dtype=_t.float32), _t.tensor(Y, dtype=_t.float32))
        tag_names = [None] * len(tag_to_idx)
        for k, i in tag_to_idx.items():
            tag_names[i] = k
        return ds, [t for t in tag_names if t is not None]

    def _build_tag_index(self, recs: List[Dict]) -> Dict[str, int]:
        tags: Dict[str, int] = {}
        for r in recs:
            for t in (r.get('thinking_tags') or []):
                if t not in tags:
                    tags[t] = len(tags)
        return tags

    def _hash_embedding(self, path: str) -> List[float]:
        p = Path(path)
        try:
            data = p.read_bytes()
        except Exception:
            data = p.as_posix().encode('utf-8')
        h = hashlib.sha256(data).digest()
        v = [((h[i % len(h)]) / 255.0) - 0.5 for i in range(self.input_dim)]
        return v

    def _label_vector(self, tags: List[str], tag_to_idx: Dict[str, int]) -> List[float]:
        y = [0.0] * len(tag_to_idx)
        for t in tags:
            i = tag_to_idx.get(t)
            if i is not None:
                y[i] = 1.0
        return y


def main():  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--books', default='/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/How to think/JSON')
    ap.add_argument('--model', default='exaone3.5:latest')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--output_model', default=None)
    ap.add_argument('--output_tags', default=None)
    args = ap.parse_args()
    bp = BookProcessor(args.books, ollama_model=args.model)
    tr = ThinkingTagTrainer(bp)
    tr.train(epochs=int(args.epochs), limit=args.limit, out_model_path=args.output_model, out_tags_path=args.output_tags)


if __name__ == '__main__':  # pragma: no cover
    main()
