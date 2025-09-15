from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np  # type: ignore

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

from ...tools.phase9.sample_loader import SampleLoader  # type: ignore
from ..phase9.shape_recognizer import ShapeRecognizer  # type: ignore
from .thinking_tag_embedder import ThinkingTagEmbedder  # type: ignore
import json as _json


class SelfReflectionEngine:
    def __init__(self, model_path: str, samples_root: str = 'viewer/public/samples', tag_model_path: str | None = None, tag_names_path: str | None = None):
        self.samples_root = Path(samples_root)
        self.loader = SampleLoader(str(self.samples_root))
        self.model = ShapeRecognizer()
        if torch is not None and hasattr(self.model, 'load_state_dict'):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # type: ignore
                self.model.eval()  # type: ignore
            except Exception:
                pass
        # Optional thinking-tag embedder
        self._tag_names: list[str] = []
        self._tag_embedder: ThinkingTagEmbedder | None = None
        try:
            if tag_model_path and tag_names_path and torch is not None:
                self._tag_embedder = ThinkingTagEmbedder()
                self._tag_embedder.load_state_dict(torch.load(tag_model_path, map_location='cpu'))  # type: ignore
                self._tag_embedder.eval()  # type: ignore
                self._tag_names = list(_json.loads(Path(tag_names_path).read_text(encoding='utf-8')))
        except Exception:
            self._tag_embedder = None

    def reflect_on_prediction(self, sample: dict, true_shape: str) -> str:
        pred_shape = self.model.predict_shape(sample.get('embedding') or [])
        if pred_shape == true_shape:
            # If tag embedder present, include tags
            if self._tag_embedder and self._tag_names:
                tags = self._tag_embedder.predict_thinking_tags(sample.get('embedding') or [], self._tag_names)
                if tags:
                    return f"✅ Correct prediction: {pred_shape}. DEEP Thinking tags: {', '.join(tags)}"
            return f"✅ Correct prediction: {pred_shape}. Embedding cluster matches training data."
        sim = self._similarity_to_true_cluster(sample.get('embedding') or [], true_shape)
        if sim > 0.8:
            return f"⚠️ Wrong but close: Predicted {pred_shape}, True {true_shape}. Embedding similarity to true cluster: {sim:.2f}"
        return f"❌ Wrong prediction: Predicted {pred_shape}, True {true_shape}. Low embedding similarity to true cluster: {sim:.2f}"

    def _similarity_to_true_cluster(self, emb: List[float], true_shape: str) -> float:
        if not emb:
            return 0.0
        # Collect training-like samples by shape
        xs = []
        for fp in self.samples_root.rglob('*.glb'):
            try:
                s = self.loader.load_sample(fp)
                if s and str(s.shape_type).lower() == str(true_shape).lower():
                    xs.append(s.embedding)
            except Exception:
                continue
        if not xs:
            return 0.0
        e = np.asarray(emb, dtype=float)
        e = e / (np.linalg.norm(e) + 1e-9)
        sims = []
        for v in xs:
            v = np.asarray(v, dtype=float)
            v = v / (np.linalg.norm(v) + 1e-9)
            sims.append(float(np.dot(e, v)))
        return float(np.mean(sims)) if sims else 0.0
