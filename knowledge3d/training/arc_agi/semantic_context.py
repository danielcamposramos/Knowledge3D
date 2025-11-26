"""
Semantic context using procedural word references (character composition).

Words are stored once as character sequences (symlink pattern) and referenced
by discoveries. This preserves dual-client reality: humans read, AI executes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from knowledge3d.training.arc_agi.semantic_signature import SemanticSignature


class SemanticWord:
    """Semantic word stored as character sequence with meaning/category."""

    def __init__(self, word_id: str, characters: List[int], meaning: str, category: str):
        self.word_id = word_id
        self.characters = characters
        self.meaning = meaning
        self.category = category
        self.references = 0

    def to_dict(self) -> Dict:
        return {
            "word_id": self.word_id,
            "characters": self.characters,
            "meaning": self.meaning,
            "category": self.category,
            "references": self.references,
        }

    @staticmethod
    def from_dict(data: Dict) -> "SemanticWord":
        w = SemanticWord(
            word_id=data["word_id"],
            characters=data.get("characters", []),
            meaning=data.get("meaning", ""),
            category=data.get("category", "unknown"),
        )
        w.references = data.get("references", 0)
        return w


class SemanticVocabulary:
    """Vocabulary of semantic words (deduplicated)."""

    def __init__(self) -> None:
        self.words: Dict[str, SemanticWord] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        for word_id, meaning, category in [
            ("rotation_or_reflection", "Rotation or reflection transformation", "transformation"),
            ("color_transformation", "Color mapping transformation", "transformation"),
            ("pattern_repetition", "Pattern repetition transformation", "transformation"),
            ("geometric_transformation", "Generic geometric transformation", "transformation"),
            ("spatial_rearrangement", "Spatial rearrangement of elements", "transformation"),
            ("asymmetric_input", "Input has no symmetry", "condition"),
            ("symmetric_input", "Input has symmetry", "condition"),
            ("sparse_grid", "Grid with many empty cells", "condition"),
            ("dense_grid", "Grid with few empty cells", "condition"),
            ("multiple_objects", "Input has multiple objects", "condition"),
            ("single_object", "Input has single object", "condition"),
            ("rotation_task", "Task involves rotation", "pattern"),
            ("reflection_task", "Task involves reflection", "pattern"),
            ("color_change_task", "Task involves color changes", "pattern"),
            ("repetition_task", "Task involves repetition", "pattern"),
            ("border_task", "Task involves borders", "pattern"),
            ("has_border", "Grid has border", "property"),
            ("has_repetition", "Grid has repeating pattern", "property"),
            ("connected_components", "Grid has connected components", "property"),
            ("multiple_colors", "Grid has multiple colors", "property"),
        ]:
            self._add_word(word_id, meaning, category)

    def _add_word(self, word_id: str, meaning: str, category: str) -> None:
        characters = [ord(c) for c in word_id]
        self.words[word_id] = SemanticWord(word_id, characters, meaning, category)

    def ref(self, word_id: str) -> Optional[str]:
        if word_id in self.words:
            self.words[word_id].references += 1
            return word_id
        return None

    def resolve(self, word_id: Optional[str]) -> Optional[SemanticWord]:
        if word_id is None:
            return None
        return self.words.get(word_id)

    def get_or_create(self, word_id: str, meaning: str = "", category: str = "unknown") -> str:
        if word_id not in self.words:
            self._add_word(word_id, meaning or f"Semantic concept: {word_id}", category)
        return self.ref(word_id) or word_id

    def to_dict(self) -> Dict:
        return {
            "words": {wid: w.to_dict() for wid, w in self.words.items()},
            "total_words": len(self.words),
            "total_references": sum(w.references for w in self.words.values()),
        }

    @staticmethod
    def from_dict(data: Dict) -> "SemanticVocabulary":
        vocab = SemanticVocabulary()
        vocab.words = {wid: SemanticWord.from_dict(wd) for wid, wd in data.get("words", {}).items()}
        return vocab


class SemanticContext:
    """Record and lookup semantic contexts using vocabulary references."""

    def __init__(self) -> None:
        self.contexts: List[Dict] = []
        self.vocabulary = SemanticVocabulary()

    def record_context(
        self,
        program: str,
        input_grid: Sequence[Sequence[int]],
        output_grid: Optional[Sequence[Sequence[int]]],
        task_id: str,
        score: float,
    ) -> Dict:
        input_sig = SemanticSignature.extract(input_grid)
        output_sig = SemanticSignature.extract(output_grid) if output_grid is not None else {}
        transformation_type = SemanticSignature.compute_transformation_type(input_sig, output_sig)
        when_to_use = self._infer_usage_conditions(input_sig, transformation_type)

        # SOVEREIGN FIX: Store only word refs and lightweight metadata (NO FULL SIGNATURES!)
        # Full signatures contain nested dicts that accumulate memory
        context = {
            "program": program,
            "task_id": task_id,
            "score": score,
            "transformation_type_ref": self.vocabulary.ref(transformation_type),
            "when_to_use_refs": [self.vocabulary.ref(w) for w in when_to_use],
            # Lightweight metadata only (for matching)
            "dimensions": input_sig.get("dimensions", "unknown"),
            "num_colors": input_sig.get("num_colors", 0),
            "sparsity": round(input_sig.get("sparsity", 0.5), 2),
            "sparsity_label": input_sig.get("sparsity_label", "unknown"),
            "symmetry_v": input_sig.get("symmetry_vertical", False),
            "symmetry_h": input_sig.get("symmetry_horizontal", False),
            "has_border": input_sig.get("has_border", False),
            "has_repetition": input_sig.get("has_repetition", False),
            "connected_components": input_sig.get("connected_components", 0),
        }
        self.contexts.append(context)
        return context

    def _infer_usage_conditions(self, input_sig: Dict, transformation_type: str) -> List[str]:
        conditions: List[str] = []
        sparsity = input_sig.get("sparsity", 0.5)
        if sparsity > 0.7:
            conditions.append(self.vocabulary.get_or_create("sparse_grid", category="condition"))
        elif sparsity < 0.3:
            conditions.append(self.vocabulary.get_or_create("dense_grid", category="condition"))

        has_symmetry = any(
            (
                input_sig.get("symmetry_vertical"),
                input_sig.get("symmetry_horizontal"),
                input_sig.get("symmetry_diagonal"),
            )
        )
        if has_symmetry:
            conditions.append(self.vocabulary.get_or_create("symmetric_input", category="condition"))
        else:
            conditions.append(self.vocabulary.get_or_create("asymmetric_input", category="condition"))

        if input_sig.get("has_border"):
            conditions.append(self.vocabulary.get_or_create("has_border", category="property"))
        if input_sig.get("has_repetition"):
            conditions.append(self.vocabulary.get_or_create("has_repetition", category="property"))
        if input_sig.get("num_colors", 0) > 1:
            conditions.append(self.vocabulary.get_or_create("multiple_colors", category="property"))

        if "rotation" in transformation_type:
            conditions.append(self.vocabulary.get_or_create("rotation_task", category="pattern"))
        if "color" in transformation_type:
            conditions.append(self.vocabulary.get_or_create("color_change_task", category="pattern"))
        if "repetition" in transformation_type:
            conditions.append(self.vocabulary.get_or_create("repetition_task", category="pattern"))

        return conditions

    def find_matching_contexts(
        self,
        query_grid: Sequence[Sequence[int]],
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        use_fuzzy_matching: bool = True,
    ) -> List[Dict]:
        query_sig = SemanticSignature.extract(query_grid)
        matches: List[Dict] = []
        for ctx in self.contexts:
            # Use lightweight metadata (no full signatures stored!)
            structural_sim = self._structural_similarity_lite(query_sig, ctx)
            color_sim = self._color_similarity_lite(query_sig, ctx)
            pattern_sim = self._pattern_similarity_lite(query_sig, ctx)
            similarity = 0.4 * structural_sim + 0.3 * pattern_sim + 0.3 * color_sim
            if use_fuzzy_matching and similarity < similarity_threshold:
                if self._fuzzy_match_lite(query_sig, ctx):
                    similarity = max(similarity, similarity_threshold - 0.05)
            if similarity >= similarity_threshold:
                t_ref = ctx.get("transformation_type_ref")
                t_word = self.vocabulary.resolve(t_ref)
                when_refs = ctx.get("when_to_use_refs", [])
                when_words = [self.vocabulary.resolve(r) for r in when_refs]
                matches.append(
                    {
                        "program": ctx["program"],
                        "score": similarity,
                        "transformation_type": t_word.word_id if t_word else "unknown",
                        "when_to_use": [w.word_id for w in when_words if w],
                        "match_components": {
                            "structural": structural_sim,
                            "color": color_sim,
                            "pattern": pattern_sim,
                        },
                    }
                )
        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    # Similarity helpers (lightweight versions using stored metadata)
    def _structural_similarity_lite(self, query_sig: Dict, ctx: Dict) -> float:
        """Compare query signature against lightweight context metadata."""
        score = 0.0
        if query_sig.get("symmetry_vertical") == ctx.get("symmetry_v"):
            score += 0.1
        if query_sig.get("symmetry_horizontal") == ctx.get("symmetry_h"):
            score += 0.1
        sparsity_diff = abs(query_sig.get("sparsity", 0.5) - ctx.get("sparsity", 0.5))
        if sparsity_diff < 0.2:
            score += 0.3
        if query_sig.get("dimensions") == ctx.get("dimensions"):
            score += 0.2
        return min(score, 1.0)

    def _structural_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Legacy function kept for compatibility."""
        score = 0.0
        for key in ("symmetry_vertical", "symmetry_horizontal", "symmetry_diagonal"):
            if sig1.get(key) == sig2.get(key):
                score += 0.1
        sparsity_diff = abs(sig1.get("sparsity", 0.5) - sig2.get("sparsity", 0.5))
        if sparsity_diff < 0.2:
            score += 0.2
        if sig1.get("dimensions") == sig2.get("dimensions"):
            score += 0.2
        return min(score, 1.0)

    def _color_similarity_lite(self, query_sig: Dict, ctx: Dict) -> float:
        """Compare color features using lightweight metadata."""
        score = 0.0
        if query_sig.get("num_colors") == ctx.get("num_colors"):
            score += 1.0
        return min(score, 1.0)

    def _color_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Legacy function kept for compatibility."""
        score = 0.0
        if sig1.get("num_colors") == sig2.get("num_colors"):
            score += 0.5
        colors1 = set(sig1.get("color_distribution", {}).keys())
        colors2 = set(sig2.get("color_distribution", {}).keys())
        if colors1 and colors2:
            jaccard = len(colors1 & colors2) / len(colors1 | colors2)
            score += 0.5 * jaccard
        return min(score, 1.0)

    def _pattern_similarity_lite(self, query_sig: Dict, ctx: Dict) -> float:
        """Compare pattern features using lightweight metadata."""
        score = 0.0
        comp_diff = abs(query_sig.get("connected_components", 0) - ctx.get("connected_components", 0))
        if comp_diff <= 2:
            score += 0.4
        if query_sig.get("has_border") == ctx.get("has_border"):
            score += 0.3
        if query_sig.get("has_repetition") == ctx.get("has_repetition"):
            score += 0.3
        return min(score, 1.0)

    def _pattern_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Legacy function kept for compatibility."""
        score = 0.0
        comp_diff = abs(sig1.get("connected_components", 0) - sig2.get("connected_components", 0))
        if comp_diff <= 2:
            score += 0.4
        if sig1.get("has_border") == sig2.get("has_border"):
            score += 0.3
        if sig1.get("has_repetition") == sig2.get("has_repetition"):
            score += 0.3
        return min(score, 1.0)

    def _fuzzy_match_lite(self, query_sig: Dict, ctx: Dict) -> bool:
        """Fuzzy matching using lightweight metadata."""
        d1 = query_sig.get("dimensions", "0x0")
        d2 = ctx.get("dimensions", "0x0")
        try:
            h1, w1 = map(int, str(d1).split("x"))
            h2, w2 = map(int, str(d2).split("x"))
            if h1 and h2 and w1 and w2:
                if min(h1 / h2, h2 / h1) > 0.5 and min(w1 / w2, w2 / w1) > 0.5:
                    return True
        except Exception:
            pass
        if query_sig.get("has_border") == ctx.get("has_border"):
            return True
        return False

    def _fuzzy_match(self, sig1: Dict, sig2: Dict) -> bool:
        """Legacy function kept for compatibility."""
        d1 = sig1.get("dimensions", "0x0")
        d2 = sig2.get("dimensions", "0x0")
        try:
            h1, w1 = map(int, str(d1).split("x"))
            h2, w2 = map(int, str(d2).split("x"))
            if h1 and h2 and w1 and w2:
                if min(h1 / h2, h2 / h1) > 0.5 and min(w1 / w2, w2 / w1) > 0.5:
                    return True
        except Exception:
            pass
        if sig1.get("has_border") == sig2.get("has_border"):
            return True
        return False

    # Persistence
    def save(self, path: Path) -> None:
        state = {"contexts": self.contexts, "vocabulary": self.vocabulary.to_dict(), "total_contexts": len(self.contexts)}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        total_refs = state["vocabulary"].get("total_references", 0)
        total_words = state["vocabulary"].get("total_words", 0)
        if total_refs:
            savings = (1 - total_words / total_refs) * 100
            print(f"[SemanticContext] Storage savings: {savings:.1f}% ({total_words} words, {total_refs} refs)")

    def load(self, path: Path) -> None:
        if not path.exists():
            print(f"[SemanticContext] No checkpoint at {path}, starting fresh")
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        self.contexts = state.get("contexts", [])
        vocab_state = state.get("vocabulary", {})
        self.vocabulary = SemanticVocabulary.from_dict(vocab_state) if vocab_state else SemanticVocabulary()
        total_refs = vocab_state.get("total_references", 0)
        print(
            f"[SemanticContext] Loaded {len(self.contexts)} contexts, "
            f"{len(self.vocabulary.words)} words, {total_refs} references"
        )


__all__ = ["SemanticContext", "SemanticVocabulary", "SemanticWord"]
