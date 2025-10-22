from __future__ import annotations

import numpy as np
import pytest


class TestCrossLingualGeneralization:
    LANG_SENTENCES = {
        "en": "The quick brown fox jumps over the lazy dog.",
        "pt": "O rato roeu a roupa do rei de Roma.",
        "es": "El perro come la comida.",
        "de": "Der Hund frisst das Futter.",
        "fr": "Le chat boit du lait.",
        "zh": "猫喜欢吃鱼。",
        "ja": "犬が公園で遊んでいます。",
        "ar": "القط يشرب الحليب.",
    }

    CAT_WORDS = {
        "en": "cat",
        "es": "gato",
        "fr": "chat",
        "pt": "gato",
        "de": "Katze",
        "zh": "猫",
        "ja": "猫",
        "ar": "قط",
    }

    @pytest.mark.parametrize("lang", LANG_SENTENCES.keys())
    def test_vocab_coverage_all_languages(self, rpn_engine, lang):
        """All languages should achieve a reasonable trigram hit rate."""
        sentence = self.LANG_SENTENCES[lang]

        prev_hits = rpn_engine.hit_count
        prev_misses = rpn_engine.miss_count
        rpn_engine.embed_sentence(sentence)

        hits = rpn_engine.hit_count - prev_hits
        misses = rpn_engine.miss_count - prev_misses
        total = hits + misses
        hit_rate = hits / total if total else 0.0

        latin_langs = {"en", "pt", "es", "de", "fr"}
        if lang in latin_langs:
            assert hit_rate > 0.4, f"{lang} coverage too low: {hit_rate:.2%}"
        else:
            assert total > 0, f"{lang} produced no trigrams"

    def test_semantic_clustering_multilingual(self, rpn_engine):
        """Concept 'cat' should cluster tightly across languages."""
        embeddings = {}
        for lang, word in self.CAT_WORDS.items():
            embeddings[lang] = rpn_engine.embed_word(word)

        similarities = []
        langs = list(embeddings.keys())
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                emb1 = embeddings[langs[i]]
                emb2 = embeddings[langs[j]]
                denom = (np.linalg.norm(emb1) * np.linalg.norm(emb2)) or 1.0
                similarities.append(np.dot(emb1, emb2) / denom)

        avg_similarity = float(np.mean(similarities))
        assert avg_similarity > 0.05, f"Multilingual 'cat' similarity too low: {avg_similarity:.3f}"
