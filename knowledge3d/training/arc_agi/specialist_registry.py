"""
Specialist registry — stores specialist embeddings in Galaxy.

Each specialist has:
- ID (e.g., "extraction", "rotation", "recolor")
- Embedding (TernaryVector) — what tasks it's good at
- Adapter weights (optional) — LoRA-style specialization
"""

from __future__ import annotations

from typing import Dict, List, Optional

from knowledge3d.cranium.ternary import TernaryGalaxy, TernaryVector

# Bootstrap specialists (learned embeddings will replace these)
BOOTSTRAP_SPECIALISTS = {
    "extraction": {
        "description": "Tasks that extract sub-regions from input",
        "keywords": ["crop", "extract", "select", "subset", "shrink"],
    },
    "expansion": {
        "description": "Tasks that expand or tile the input",
        "keywords": ["tile", "repeat", "expand", "grow", "scale_up"],
    },
    "rotation": {
        "description": "Tasks involving rotation transforms",
        "keywords": ["rotate", "90", "180", "270", "turn", "spin"],
    },
    "reflection": {
        "description": "Tasks involving flip/mirror transforms",
        "keywords": ["flip", "mirror", "reflect", "horizontal", "vertical"],
    },
    "recolor": {
        "description": "Tasks that change colors",
        "keywords": ["recolor", "replace", "swap", "color", "palette"],
    },
    "pattern": {
        "description": "Tasks involving pattern recognition/completion",
        "keywords": ["pattern", "repeat", "sequence", "fill", "complete"],
    },
    "composition": {
        "description": "Tasks combining multiple operations",
        "keywords": ["compose", "chain", "multiple", "combine", "sequence"],
    },
    "spatial": {
        "description": "Tasks with spatial reasoning",
        "keywords": ["move", "translate", "position", "align", "center"],
    },
    "logical": {
        "description": "Tasks with logical operations",
        "keywords": ["and", "or", "xor", "mask", "filter", "condition"],
    },
}


class SpecialistRegistry:
    """
    Manages specialist embeddings and routing.
    """

    def __init__(self):
        self.galaxy = TernaryGalaxy()
        self.specialists: Dict[str, Dict] = {}
        self._bootstrap()

    def _bootstrap(self) -> None:
        """Initialize with bootstrap specialists."""
        for spec_id, spec_data in BOOTSTRAP_SPECIALISTS.items():
            embedding = self._keywords_to_embedding(spec_data["keywords"])
            self.specialists[spec_id] = {
                "id": spec_id,
                "embedding": embedding,
                "description": spec_data["description"],
                "usage_count": 0,
                "success_count": 0,
            }
            self.galaxy.store_frame(f"specialist_{spec_id}", spec_data["description"], embedding)

    def _keywords_to_embedding(self, keywords: List[str]) -> TernaryVector:
        """Convert keywords to embedding via hashing."""
        embedding = [0.0] * 128
        for kw in keywords:
            for i, char in enumerate(kw):
                idx = (ord(char) + i * 7) % 128
                embedding[idx] += 1.0

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in embedding]
        return TernaryVector(ternary)

    def get_specialist_embedding(self, spec_id: str) -> Optional[TernaryVector]:
        """Get embedding for specialist."""
        if spec_id in self.specialists:
            return self.specialists[spec_id]["embedding"]
        return None

    def update_specialist(self, spec_id: str, task_embedding: TernaryVector, success: bool) -> None:
        """
        Update specialist based on task outcome.
        """
        if spec_id not in self.specialists:
            return

        spec = self.specialists[spec_id]
        spec["usage_count"] += 1
        if success:
            spec["success_count"] += 1

        current_emb = spec["embedding"].to_python()
        task_emb = task_embedding.to_python()

        learning_rate = 0.1 if success else -0.05
        new_emb = [c + learning_rate * (t - c) for c, t in zip(current_emb, task_emb)]

        norm = sum(x * x for x in new_emb) ** 0.5
        if norm > 0:
            new_emb = [x / norm for x in new_emb]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in new_emb]
        spec["embedding"] = TernaryVector(ternary)

    def list_specialists(self) -> List[str]:
        """List all specialist IDs."""
        return list(self.specialists.keys())


__all__ = ["SpecialistRegistry", "BOOTSTRAP_SPECIALISTS"]
