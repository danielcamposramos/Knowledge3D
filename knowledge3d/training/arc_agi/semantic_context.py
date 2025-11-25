"""
Semantic context recorder for discovered programs.

Attaches input/output signatures and inferred usage hints to programs so TRM
can route by context instead of blind trial.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from knowledge3d.training.arc_agi.semantic_signature import SemanticSignature


class SemanticContext:
    """Record and lookup semantic contexts for programs."""

    def __init__(self) -> None:
        self.context_index: Dict[str, List[Dict]] = {}  # input signature hash -> contexts

    def record_context(
        self,
        program: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        task_id: str,
        score: float,
    ) -> Dict:
        input_sig = SemanticSignature.extract(input_grid)
        output_sig = SemanticSignature.extract(output_grid)
        transformation_type = SemanticSignature.compute_transformation_type(input_sig, output_sig)

        context = {
            "program": program,
            "task_id": task_id,
            "score": float(score),
            "input_signature": input_sig,
            "output_signature": output_sig,
            "transformation_type": transformation_type,
            "when_to_use": self._infer_usage_conditions(input_sig, transformation_type),
        }

        sig_hash = input_sig["signature_hash"]
        self.context_index.setdefault(sig_hash, []).append(context)
        return context

    @staticmethod
    def _infer_usage_conditions(input_sig: Dict, transformation_type: str) -> List[str]:
        conditions: List[str] = []
        structural = input_sig["structural"]

        if structural["symmetric_vertical"] or structural["symmetric_horizontal"]:
            conditions.append("symmetric_input")
        else:
            conditions.append("asymmetric_input")

        if structural["sparsity_label"] == "sparse":
            conditions.append("sparse_pattern")
        elif structural["sparsity_label"] == "dense":
            conditions.append("dense_pattern")

        if transformation_type == "rotation_or_reflection":
            conditions.append("rotation_task")
        elif transformation_type == "recoloring":
            conditions.append("color_mapping")
        elif transformation_type == "pattern_completion":
            conditions.append("pattern_completion")

        return conditions

    def find_matching_contexts(
        self,
        query_grid: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.5,
        use_fuzzy_matching: bool = True,
    ) -> List[Dict]:
        """
        Return contexts using multi-component similarity (structural/color/pattern).
        """
        query_sig = SemanticSignature.extract(query_grid)
        matches: List[Dict] = []

        for ctxs in self.context_index.values():
            for ctx in ctxs:
                input_sig = ctx.get("input_signature", {})
                structural_sim = self._structural_similarity(query_sig, input_sig)
                color_sim = self._color_similarity(query_sig, input_sig)
                pattern_sim = self._pattern_similarity(query_sig, input_sig)
                similarity = 0.4 * structural_sim + 0.3 * pattern_sim + 0.3 * color_sim

                if use_fuzzy_matching and similarity < similarity_threshold:
                    if self._fuzzy_match(query_sig, input_sig):
                        similarity = max(similarity, similarity_threshold - 0.05)

                if similarity >= similarity_threshold:
                    matches.append(
                        {
                            "program": ctx.get("program"),
                            "score": similarity,
                            "transformation_type": ctx.get("transformation_type"),
                            "when_to_use": ctx.get("when_to_use", []),
                            "input_signature": input_sig,
                            "match_components": {
                                "structural": structural_sim,
                                "color": color_sim,
                                "pattern": pattern_sim,
                            },
                        }
                    )

        matches.sort(key=lambda x: x["score"], reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------ #
    # Similarity helpers
    # ------------------------------------------------------------------ #
    def _structural_similarity(self, sig1: Dict, sig2: Dict) -> float:
        s1 = sig1.get("structural", {})
        s2 = sig2.get("structural", {})
        score = 0.0
        for key in ("symmetric_vertical", "symmetric_horizontal", "symmetric_diagonal"):
            if s1.get(key) == s2.get(key):
                score += 0.1
        sparsity_diff = abs(s1.get("sparsity", 0.5) - s2.get("sparsity", 0.5))
        if sparsity_diff < 0.2:
            score += 0.2
        if s1.get("dimensions") == s2.get("dimensions"):
            score += 0.2
        return min(score, 1.0)

    def _color_similarity(self, sig1: Dict, sig2: Dict) -> float:
        c1 = sig1.get("color", {})
        c2 = sig2.get("color", {})
        score = 0.0
        if c1.get("num_colors") == c2.get("num_colors"):
            score += 0.5
        colors1 = set(c1.get("color_distribution", {}).keys())
        colors2 = set(c2.get("color_distribution", {}).keys())
        if colors1 and colors2:
            jaccard = len(colors1 & colors2) / len(colors1 | colors2)
            score += 0.5 * jaccard
        return min(score, 1.0)

    def _pattern_similarity(self, sig1: Dict, sig2: Dict) -> float:
        p1 = sig1.get("pattern", {})
        p2 = sig2.get("pattern", {})
        score = 0.0
        comp_diff = abs(p1.get("num_components", 0) - p2.get("num_components", 0))
        if comp_diff <= 2:
            score += 0.4
        if p1.get("has_border") == p2.get("has_border"):
            score += 0.3
        if p1.get("has_repetition") == p2.get("has_repetition"):
            score += 0.3
        return min(score, 1.0)

    def _fuzzy_match(self, sig1: Dict, sig2: Dict) -> bool:
        # Dimensions similarity
        d1 = sig1.get("structural", {}).get("dimensions", "0x0")
        d2 = sig2.get("structural", {}).get("dimensions", "0x0")
        try:
            h1, w1 = map(int, str(d1).split("x"))
            h2, w2 = map(int, str(d2).split("x"))
            ratio_h = min(h1 / h2, h2 / h1) if h1 and h2 else 0
            ratio_w = min(w1 / w2, w2 / w1) if w1 and w2 else 0
            if ratio_h > 0.5 and ratio_w > 0.5:
                return True
        except Exception:
            pass

        # If border presence matches, consider near-match
        if sig1.get("pattern", {}).get("has_border") == sig2.get("pattern", {}).get("has_border"):
            return True

        return False

    def save(self, path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        # Flatten contexts for easier inspection
        flat = []
        for ctxs in self.context_index.values():
            flat.extend(ctxs)
        with path.open("w", encoding="utf-8") as f:
            json.dump({"context_index": self.context_index, "contexts": flat}, f, indent=2)

    def load(self, path) -> None:
        import json
        if not path.exists():
            print(f"[SemanticContext] No checkpoint at {path}, starting fresh")
            return
        with path.open("r", encoding="utf-8") as f:
            state = json.load(f)
        self.context_index = state.get("context_index", {})
        print(f"[SemanticContext] Loaded contexts for {len(self.context_index)} signatures from {path}")


__all__ = ["SemanticContext"]
