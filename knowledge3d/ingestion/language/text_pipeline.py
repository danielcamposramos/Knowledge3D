"""
Text-language ingestion pipeline.

Transforms raw linguistic artefacts (vocabulary lists, sentences) into
K3D-compatible embeddings and spatial coordinates. The implementation favours
lazy loading of heavyweight models so that callers can decide when to pay the
initialisation cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _require_optional_dependency(module_name: str, install_hint: str) -> None:
    """Raise a helpful error if an optional dependency is missing."""
    try:
        __import__(module_name)
    except ModuleNotFoundError as exc:  # pragma: no cover - executed only when missing
        raise RuntimeError(
            f"Optional dependency '{module_name}' is required for language ingestion.\n"
            f"Install hint: {install_hint}"
        ) from exc


def _pca_reduce(matrix: np.ndarray, components: int) -> np.ndarray:
    """Dimensionality reduction with a PCA fallback that avoids global state."""
    if matrix.ndim != 2:
        raise ValueError("PCA expects a 2D matrix")
    if matrix.shape[0] < components:
        raise ValueError(
            f"PCA requires at least {components} samples, got {matrix.shape[0]}"
        )

    try:
        from sklearn.decomposition import PCA  # type: ignore
    except ModuleNotFoundError:
        # Lightweight fallback using SVD (no whitening, but deterministic)
        matrix_centered = matrix - matrix.mean(axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(matrix_centered, full_matrices=False)
        return (u[:, :components] * s[:components]).astype(np.float32, copy=False)

    pca = PCA(n_components=components)
    return pca.fit_transform(matrix).astype(np.float32, copy=False)


@dataclass
class TextLanguageIngestor:
    """
    Multi-language text ingestion utility.

    Parameters
    ----------
    languages:
        ISO language codes to support.
    sentence_model_name:
        HuggingFace model identifier for sentence embeddings.
    fasttext_dir:
        Directory containing pre-downloaded fastText `.bin` files.
    spacy_model_map:
        Optional explicit map from language code to spaCy model name.
    device:
        Device string forwarded to SentenceTransformer (e.g. 'cpu', 'cuda').
    """

    languages: Sequence[str] = ("en", "pt", "es", "ja", "zh")
    sentence_model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    fasttext_dir: Path = Path("data/fasttext")
    spacy_model_map: Dict[str, str] | None = None
    device: str = "cpu"

    _sentence_model: "SentenceTransformer | None" = field(init=False, default=None)
    _fasttext_models: Dict[str, "fasttext.FastText._FastText"] = field(
        init=False, default_factory=dict
    )
    _spacy_models: Dict[str, "spacy.language.Language"] = field(
        init=False, default_factory=dict
    )

    def _load_sentence_model(self):
        if self._sentence_model is None:
            _require_optional_dependency(
                "sentence_transformers",
                "pip install sentence-transformers",
            )
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._sentence_model = SentenceTransformer(
                self.sentence_model_name, device=self.device
            )
        return self._sentence_model

    def _load_fasttext_model(self, lang: str):
        if lang not in self.languages:
            raise ValueError(f"Language '{lang}' not configured for ingestion")
        model = self._fasttext_models.get(lang)
        if model is None:
            _require_optional_dependency("fasttext", "pip install fasttext")
            import fasttext  # type: ignore

            model_path = self.fasttext_dir / f"cc.{lang}.300.bin"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"fastText model for '{lang}' not found at {model_path}. "
                    "Download from https://fasttext.cc/docs/en/crawl-vectors.html "
                    "and place the .bin file in the configured directory."
                )
            model = fasttext.load_model(str(model_path))
            self._fasttext_models[lang] = model
        return model

    def _load_spacy_model(self, lang: str):
        model = self._spacy_models.get(lang)
        if model is None:
            _require_optional_dependency("spacy", "pip install spacy")
            import spacy  # type: ignore

            model_name = (
                self.spacy_model_map[lang]
                if self.spacy_model_map and lang in self.spacy_model_map
                else self._default_spacy_model(lang)
            )
            try:
                model = spacy.load(model_name)  # type: ignore[call-arg]
            except OSError as exc:
                raise RuntimeError(
                    f"spaCy model '{model_name}' is not installed. "
                    f"Install it via: python -m spacy download {model_name}"
                ) from exc
            self._spacy_models[lang] = model
        return model

    @staticmethod
    def _default_spacy_model(lang: str) -> str:
        mapping = {
            "en": "en_core_web_trf",
            "pt": "pt_core_news_lg",
            "es": "es_core_news_lg",
            "ja": "ja_core_news_lg",
            "zh": "zh_core_web_lg",
        }
        model = mapping.get(lang)
        if model is None:
            raise ValueError(
                f"No default spaCy model configured for language '{lang}'. "
                "Provide an explicit mapping via `spacy_model_map`."
            )
        return model

    # --------------------------------------------------------------------- #
    # Public ingestion API
    # --------------------------------------------------------------------- #
    def ingest_vocabulary(self, lang: str, word_list: Sequence[str]) -> np.ndarray:
        """
        Generate 3D spatial positions for vocabulary tokens.

        Parameters
        ----------
        lang:
            Language code (must match the configured set).
        word_list:
            Iterable of vocabulary tokens.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) representing normalised 3D positions.
        """
        if not word_list:
            raise ValueError("word_list must not be empty")

        fasttext_model = self._load_fasttext_model(lang)
        embeddings = np.vstack(
            [fasttext_model.get_word_vector(token) for token in word_list]
        ).astype(np.float32)

        if embeddings.shape[0] < 3:
            # Degenerate PCA case; fall back to spherical coordinates
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6
            unit = embeddings / norms
            padded = np.pad(unit, ((0, 0), (0, max(0, 3 - unit.shape[1]))))
            return padded[:, :3]

        reduced = _pca_reduce(embeddings, components=3)
        reduced -= reduced.min(axis=0, keepdims=True)
        denom = reduced.max(axis=0, keepdims=True)
        denom[denom == 0.0] = 1.0
        return reduced / denom

    def ingest_grammar_tree(self, lang: str, sentence: str) -> Dict:
        """
        Parse a sentence and produce a structured representation suitable for
        3D placement and swarm processing.
        """
        sentence = sentence.strip()
        if not sentence:
            raise ValueError("Sentence must contain non-whitespace characters")

        nlp = self._load_spacy_model(lang)
        doc = nlp(sentence)

        nodes: List[tuple[str, np.ndarray, int]] = []
        edges: List[tuple[int, int]] = []

        for token in doc:
            depth = sum(1 for _ in token.ancestors)
            position = np.array(
                [
                    self._dep_to_x(token.dep_),
                    min(depth / 10.0, 1.0),
                    token.i / max(len(doc) - 1, 1),
                ],
                dtype=np.float32,
            )
            nodes.append((token.text, position, depth))
            if token.head != token:
                edges.append((token.head.i, token.i))

        sentence_model = self._load_sentence_model()
        sentence_emb = sentence_model.encode(sentence, convert_to_numpy=True)
        embedding = self._resize_embedding(sentence_emb, target_dim=128)

        return {
            "nodes": nodes,
            "edges": edges,
            "embedding": embedding,
            "language": lang,
        }

    # ------------------------------------------------------------------ #
    # Helper transforms
    # ------------------------------------------------------------------ #
    @staticmethod
    def _dep_to_x(dep: str) -> float:
        buckets = [
            "nsubj",
            "obj",
            "iobj",
            "det",
            "amod",
            "advmod",
            "prep",
            "pobj",
        ]
        if dep in buckets:
            return buckets.index(dep) / (len(buckets) - 1 or 1)
        return 0.5

    @staticmethod
    def _resize_embedding(emb: np.ndarray, target_dim: int) -> np.ndarray:
        if emb.ndim != 1:
            raise ValueError("Embedding must be 1D")
        if emb.shape[0] == target_dim:
            return emb.astype(np.float32, copy=False)
        if emb.shape[0] < target_dim:
            return np.pad(emb, (0, target_dim - emb.shape[0])).astype(
                np.float32, copy=False
            )
        # Reduce with PCA on a single sample is undefined; instead chunk.
        segments = np.array_split(emb, target_dim)
        pooled = np.array([segment.mean() for segment in segments], dtype=np.float32)
        if pooled.shape[0] != target_dim:
            pooled = np.resize(pooled, target_dim)
        return pooled


__all__ = ["TextLanguageIngestor"]
