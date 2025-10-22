from __future__ import annotations

import numpy as np


class TestDomainTransfer:
    DOMAIN_PROMPTS = {
        "technical": [
            "Explain backpropagation in neural networks.",
            "What causes CUDA warp divergence?",
            "How does a GPU scheduler work?",
        ],
        "natural": [
            "Why do leaves change color in autumn?",
            "How do birds navigate during migration?",
            "What causes ocean tides?",
        ],
        "reasoning": [
            "If A implies B and B implies C, does A imply C?",
            "Three boxes are all mislabeled. Pick one fruit to fix the labels.",
            "Five pirates split 100 coins. How do you maximize your share?",
        ],
    }

    def test_technical_reasoning_similarity(self, rpn_engine):
        """Technical knowledge should transfer to logical reasoning."""
        domain_embeddings = {}
        for domain, prompts in self.DOMAIN_PROMPTS.items():
            embeddings = [rpn_engine.embed_sentence(prompt) for prompt in prompts]
            domain_embeddings[domain] = np.mean(embeddings, axis=0)

        tech_emb = domain_embeddings["technical"]
        reasoning_emb = domain_embeddings["reasoning"]

        denom = (np.linalg.norm(tech_emb) * np.linalg.norm(reasoning_emb)) or 1.0
        similarity = float(np.dot(tech_emb, reasoning_emb) / denom)

        assert similarity > 0.05, f"Tech→Reasoning transfer weak: {similarity:.3f}"
