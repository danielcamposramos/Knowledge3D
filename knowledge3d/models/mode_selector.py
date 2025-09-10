from __future__ import annotations

"""
Mode Selector — learned routing between compose and compose_generate modes.
Trains on outcome logs to predict optimal mode given query + contexts.
"""

import json
import pickle
from pathlib import Path
from typing import List

import numpy as np  # noqa: F401  (kept for potential future features)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class ModeSelector:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=1000,
                        stop_words="english",
                        ngram_range=(1, 2),
                    ),
                ),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ]
        )
        self.trained = False

    def _featurize(self, query: str, contexts: List[str]) -> str:
        ctx_text = " ".join([str(c or "") for c in contexts[:4]])
        return f"{str(query or '').strip()} {ctx_text}".strip()

    def predict(self, query: str, contexts: List[str]) -> int:
        """Returns 0 for compose, 1 for compose_generate"""
        if not self.trained:
            return 0
        feature_text = self._featurize(query, contexts)
        try:
            pred = self.pipeline.predict([feature_text])[0]
            return int(pred)
        except Exception:
            return 0

    def train(self, dataset_path: Path, test_size: float = 0.2) -> None:
        print(f"Loading training data from {dataset_path}")
        X: List[str] = []
        y: List[int] = []
        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        q = str(rec.get("question") or "")
                        ctxs = rec.get("contexts") or []
                        mode = str(rec.get("mode") or "compose").strip()
                        label = 1 if mode == "compose_generate" else 0
                        X.append(self._featurize(q, ctxs))
                        y.append(label)
                    except Exception:
                        continue
        except OSError as e:
            print(f"Error reading dataset: {e}")
            return
        if len(X) < 10:
            print("Insufficient training data (<10 rows)")
            return
        print(f"Training on {len(X)} samples")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=float(test_size), random_state=42
        )
        self.pipeline.fit(X_train, y_train)
        self.trained = True
        y_pred = self.pipeline.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        print("\nClassification Report:")
        print(
            classification_report(
                y_test, y_pred, target_names=["compose", "compose_generate"]
            )
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)
        print(f"Saved mode selector to {path}")


def load(path: Path) -> ModeSelector:
    with path.open("rb") as f:
        return pickle.load(f)


def train_mode_selector(outcomes_path: str, model_path: str) -> ModeSelector:
    selector = ModeSelector()
    selector.train(Path(outcomes_path))
    selector.save(Path(model_path))
    return selector


if __name__ == "__main__":  # pragma: no cover
    import sys

    if len(sys.argv) != 3:
        print(
            "Usage: python -m knowledge3d.models.mode_selector <outcomes.jsonl> <model.pkl>"
        )
        raise SystemExit(1)
    train_mode_selector(sys.argv[1], sys.argv[2])

