# CODEX: Sovereign Swarm Architecture — Grammar Evolution + TRM Router

**Priority:** HIGH — Core architectural upgrade
**Date:** December 12, 2025
**Prerequisite:** Training run completed, metrics collected

---

## Overview

This briefing implements three interconnected sovereign upgrades:

1. **Grammar Galaxy Evolution** — Discovery, validation, promotion of rules (not read-only)
2. **Size Pattern as Embedding** — Model-learned evaluation (no hardcoded ratios)
3. **TRM Swarm Coordinator** — Router-as-model, dynamic worker spawning

**Constraints:**
- NO external libraries (no numpy, no cupy in hot path)
- NO stubs or placeholders — full PTX implementation
- Leverage existing kernels and codecs
- Weights = logic, Galaxy = knowledge

**Codex Contribution Directive:**
You are encouraged to contribute improvements, optimizations, and architectural insights within these constraints. If you see a better approach that maintains sovereignty, implement it and document why. Your implementation experience is valuable — feed back what works and what doesn't.

---

## Part 1: Grammar Galaxy Evolution

### 1.1 New RPN Opcodes

**File:** `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` (APPEND)

```python
# ============================================================
# Grammar Evolution Opcodes
# Cross-modality discovery → validation → promotion
# ============================================================

OP_GRAMMAR_OBSERVE = 0xE5      # visual_emb text_emb → correlation score
OP_GRAMMAR_PROPOSE = 0xE6      # rpn_program context → rule_id (tentative)
OP_GRAMMAR_VALIDATE = 0xE7    # rule_id success → updated quality_score
OP_GRAMMAR_PROMOTE = 0xE8      # rule_id → move to shared if quality >= threshold
OP_GRAMMAR_QUERY = 0xE9        # embedding k → top-k matching rules

# Add to __all__
__all__ += [
    "OP_GRAMMAR_OBSERVE", "OP_GRAMMAR_PROPOSE", "OP_GRAMMAR_VALIDATE",
    "OP_GRAMMAR_PROMOTE", "OP_GRAMMAR_QUERY",
]
```

### 1.2 Grammar Galaxy with Discovery Space

**File:** `knowledge3d/training/arc_agi/grammar_galaxy.py` (MODIFY)

Add two-tier memory model:

```python
from knowledge3d.cranium.ternary import TernaryVector, TernaryGalaxy
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge


class GrammarGalaxy:
    """
    Two-tier grammar storage:
    - Canonical + Promoted: Shared across workers (read via snapshot)
    - Local Discoveries: Worker-private (tentative rules)
    """

    def __init__(
        self,
        rules: Optional[List[GrammarRule]] = None,
        snapshot: Optional[bytes] = None,  # NEW: Initialize from snapshot
        **kwargs,
    ):
        # ... existing __init__ ...

        # NEW: Discovery infrastructure
        self.cosine_bridge = CosineSimilarityBridge()
        self._local_discoveries: Dict[str, Dict] = {}
        self._discovery_threshold = 0.6
        self._promotion_threshold = 0.7
        self._min_usage_for_promotion = 3

        # If snapshot provided, load from it (no file I/O)
        if snapshot is not None:
            self._load_from_snapshot(snapshot)

    # ------------------------------------------------------------------ #
    # Cross-Modality Observation
    # ------------------------------------------------------------------ #
    def observe_pattern(
        self,
        visual_embedding: List[float],
        text_embedding: List[float],
        context: str,
    ) -> Optional[str]:
        """
        Observe cross-modal correlation. If strong, propose tentative rule.

        RPN: visual_emb text_emb GRAMMAR_OBSERVE
        """
        correlation = self.cosine_bridge.compute_similarity(
            [visual_embedding], text_embedding
        )[0]

        if correlation >= self._discovery_threshold:
            # Synthesize rule RPN from correlation pattern
            rule_rpn = self._synthesize_rule_rpn(visual_embedding, text_embedding)
            return self.propose_rule(rule_rpn, context, correlation)
        return None

    def _synthesize_rule_rpn(
        self,
        visual_emb: List[float],
        text_emb: List[float],
    ) -> str:
        """
        Synthesize RPN program from cross-modal embeddings.

        The RPN encodes the TRANSFORMATION that maps visual → text semantics.
        """
        # Find dominant dimensions (top-k by magnitude)
        vis_top = sorted(enumerate(visual_emb), key=lambda x: abs(x[1]), reverse=True)[:8]
        txt_top = sorted(enumerate(text_emb), key=lambda x: abs(x[1]), reverse=True)[:8]

        # Build RPN that captures the mapping
        rpn_parts = []
        for (vi, vv), (ti, tv) in zip(vis_top, txt_top):
            # Encode dimension mapping: visual_dim → text_dim with weight
            weight = tv / (vv + 1e-6)
            rpn_parts.append(f"DIM_{vi} {weight:.4f} MUL DIM_{ti} STORE")

        return " ".join(rpn_parts) + " CROSS_MODAL_RULE"

    # ------------------------------------------------------------------ #
    # Discovery Lifecycle
    # ------------------------------------------------------------------ #
    def propose_rule(self, rpn_program: str, context: str, confidence: float = 0.0) -> str:
        """
        Add tentative rule to local discovery space.

        RPN: rpn_program context GRAMMAR_PROPOSE → rule_id
        """
        rule_id = f"DISC_{hash(rpn_program) & 0xFFFFFF:06x}"

        # Compile to ternary for fast similarity lookup
        rule_embedding = self._compile_rule_to_embedding(rpn_program)

        self._local_discoveries[rule_id] = {
            "rpn_program": rpn_program,
            "embedding": rule_embedding,
            "context": context,
            "usage_count": 0,
            "success_count": 0,
            "quality_score": confidence,
            "created_epoch": getattr(self, "_current_epoch", 0),
        }

        return rule_id

    def validate_usage(self, rule_id: str, success: bool) -> float:
        """
        Update quality score based on usage outcome.

        RPN: rule_id success GRAMMAR_VALIDATE → quality_score
        """
        if rule_id not in self._local_discoveries:
            # Check if it's a promoted/canonical rule (always valid)
            if rule_id in self.rules:
                return 1.0
            return 0.0

        rule = self._local_discoveries[rule_id]
        rule["usage_count"] += 1
        if success:
            rule["success_count"] += 1

        # Bayesian quality estimate
        rule["quality_score"] = rule["success_count"] / rule["usage_count"]

        # Auto-promote if ready
        self._try_promote(rule_id)

        return rule["quality_score"]

    def _try_promote(self, rule_id: str) -> bool:
        """
        Promote to shared galaxy if quality threshold met.

        RPN: rule_id GRAMMAR_PROMOTE → success
        """
        if rule_id not in self._local_discoveries:
            return False

        rule = self._local_discoveries[rule_id]

        if (rule["quality_score"] >= self._promotion_threshold and
            rule["usage_count"] >= self._min_usage_for_promotion):

            # Create GrammarRule from discovery
            new_rule = GrammarRule(
                rule_id=rule_id,
                language="discovered",
                pattern="cross_modal",
                rpn_program=rule["rpn_program"],
                domain="discovered",
                description=f"Discovered from: {rule['context']}",
                is_canonical=False,
            )

            # Add to shared rules
            self.rules[rule_id] = new_rule

            # Remove from local discoveries
            del self._local_discoveries[rule_id]

            print(f"[GRAMMAR PROMOTE] {rule_id} promoted (quality={rule['quality_score']:.2f}, usage={rule['usage_count']})")
            return True

        return False

    def query_similar(self, embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Find top-k rules similar to embedding.

        RPN: embedding k GRAMMAR_QUERY → [(rule_id, score), ...]
        """
        scores = []

        # Search canonical + promoted rules
        for rule_id, rule in self.rules.items():
            rule_emb = self._get_rule_embedding(rule)
            if rule_emb:
                sim = self.cosine_bridge.compute_similarity([embedding], rule_emb)[0]
                scores.append((rule_id, sim))

        # Search local discoveries
        for rule_id, rule_data in self._local_discoveries.items():
            sim = self.cosine_bridge.compute_similarity(
                [embedding], rule_data["embedding"]
            )[0]
            scores.append((rule_id, sim))

        # Return top-k
        scores.sort(key=lambda x: -x[1])
        return scores[:k]

    def _compile_rule_to_embedding(self, rpn_program: str) -> List[float]:
        """Compile RPN to fixed-size embedding for similarity search."""
        # Use token hashing for simple embedding
        tokens = rpn_program.split()
        embedding = [0.0] * 128
        for i, token in enumerate(tokens):
            idx = hash(token) % 128
            embedding[idx] += 1.0 / (i + 1)  # Position-weighted

        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        return embedding

    def _get_rule_embedding(self, rule: GrammarRule) -> Optional[List[float]]:
        """Get or compute embedding for existing rule."""
        # Cache embeddings to avoid recomputation
        cache_key = f"_emb_{rule.rule_id}"
        if hasattr(rule, cache_key):
            return getattr(rule, cache_key)

        emb = self._compile_rule_to_embedding(rule.rpn_program)
        setattr(rule, cache_key, emb)
        return emb

    # ------------------------------------------------------------------ #
    # Snapshot Serialization (for worker transfer)
    # ------------------------------------------------------------------ #
    def to_snapshot(self) -> bytes:
        """
        Serialize galaxy to bytes for worker transfer.
        NO file I/O — pure memory.
        """
        import json

        data = {
            "rules": {
                rule_id: {
                    "rule_id": rule.rule_id,
                    "rpn_program": rule.rpn_program,
                    "pattern": rule.pattern,
                    "language": rule.language,
                    "domain": getattr(rule, "domain", "general"),
                }
                for rule_id, rule in self.rules.items()
            },
            "promoted_count": len([r for r in self.rules.values() if not getattr(r, "is_canonical", True)]),
        }

        return json.dumps(data).encode("utf-8")

    def _load_from_snapshot(self, snapshot: bytes) -> None:
        """Load from snapshot bytes (no file I/O)."""
        import json

        data = json.loads(snapshot.decode("utf-8"))

        for rule_id, rule_data in data.get("rules", {}).items():
            if rule_id not in self.rules:
                self.rules[rule_id] = GrammarRule(
                    rule_id=rule_data["rule_id"],
                    rpn_program=rule_data["rpn_program"],
                    pattern=rule_data.get("pattern", "unknown"),
                    language=rule_data.get("language", "en"),
                    domain=rule_data.get("domain", "general"),
                )

    def merge_discoveries(self, other_discoveries: Dict[str, Dict]) -> int:
        """
        Merge discoveries from worker back to main galaxy.
        Called after worker completes.
        """
        merged = 0
        for rule_id, rule_data in other_discoveries.items():
            if rule_id in self._local_discoveries:
                # Combine usage stats
                existing = self._local_discoveries[rule_id]
                existing["usage_count"] += rule_data["usage_count"]
                existing["success_count"] += rule_data["success_count"]
                existing["quality_score"] = (
                    existing["success_count"] / existing["usage_count"]
                    if existing["usage_count"] > 0 else 0.0
                )
            else:
                self._local_discoveries[rule_id] = rule_data
            merged += 1
            self._try_promote(rule_id)

        return merged
```

---

## Part 2: Size Pattern as Embedding

### 2.1 Size Pattern Encoder

**File:** `knowledge3d/training/arc_agi/size_pattern_encoder.py` (CREATE)

```python
"""
Size pattern encoding — model-learned, not hardcoded ratios.

Encodes input→output size relationships as TernaryVector embeddings
that the model can learn to interpret.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge


class SizePatternEncoder:
    """
    Encode size relationships as learnable embeddings.

    No hardcoded thresholds — the model learns what patterns mean.
    """

    def __init__(self):
        self.cosine_bridge = CosineSimilarityBridge()
        self.embedding_dim = 64

    def encode_task_pattern(self, train_examples: List[Dict]) -> TernaryVector:
        """
        Encode size pattern from training examples.

        Returns embedding that captures:
        - Average size change (shrink/grow/same)
        - Variance in size change
        - Aspect ratio preservation
        - Directional consistency
        """
        if not train_examples:
            return self._neutral_embedding()

        features = []

        for ex in train_examples:
            inp = ex.get("input", [])
            out = ex.get("output", [])
            if not inp or not out:
                continue

            h_in, w_in = len(inp), len(inp[0]) if inp else 0
            h_out, w_out = len(out), len(out[0]) if out else 0

            if h_in == 0 or w_in == 0:
                continue

            # Raw ratios (can be < 1 for shrink, > 1 for grow)
            h_ratio = h_out / h_in
            w_ratio = w_out / w_in

            # Aspect ratio change
            aspect_in = w_in / h_in if h_in > 0 else 1.0
            aspect_out = w_out / h_out if h_out > 0 else 1.0
            aspect_change = aspect_out / aspect_in if aspect_in > 0 else 1.0

            # Size change direction (-1 = shrink, 0 = same, +1 = grow)
            h_dir = -1 if h_ratio < 0.9 else (1 if h_ratio > 1.1 else 0)
            w_dir = -1 if w_ratio < 0.9 else (1 if w_ratio > 1.1 else 0)

            features.append({
                "h_ratio": h_ratio,
                "w_ratio": w_ratio,
                "h_dir": h_dir,
                "w_dir": w_dir,
                "aspect_change": aspect_change,
                "area_ratio": (h_out * w_out) / (h_in * w_in),
            })

        if not features:
            return self._neutral_embedding()

        # Aggregate features
        n = len(features)
        avg_h_ratio = sum(f["h_ratio"] for f in features) / n
        avg_w_ratio = sum(f["w_ratio"] for f in features) / n
        avg_area_ratio = sum(f["area_ratio"] for f in features) / n
        avg_aspect_change = sum(f["aspect_change"] for f in features) / n

        var_h_ratio = sum((f["h_ratio"] - avg_h_ratio)**2 for f in features) / n
        var_w_ratio = sum((f["w_ratio"] - avg_w_ratio)**2 for f in features) / n

        # Direction consistency (how consistent is the direction across examples?)
        h_dirs = [f["h_dir"] for f in features]
        w_dirs = [f["w_dir"] for f in features]
        h_consistency = abs(sum(h_dirs)) / n  # 1.0 = all same direction
        w_consistency = abs(sum(w_dirs)) / n

        # Build embedding vector
        embedding = [0.0] * self.embedding_dim

        # Encode features into embedding dimensions
        # (Spread across dimensions for model to learn)
        embedding[0] = avg_h_ratio
        embedding[1] = avg_w_ratio
        embedding[2] = avg_area_ratio
        embedding[3] = avg_aspect_change
        embedding[4] = var_h_ratio
        embedding[5] = var_w_ratio
        embedding[6] = h_consistency
        embedding[7] = w_consistency

        # Direction indicators (ternary-friendly: -1, 0, +1)
        embedding[8] = sum(h_dirs) / n  # Average direction
        embedding[9] = sum(w_dirs) / n

        # Derived features
        embedding[10] = 1.0 if avg_area_ratio < 0.5 else 0.0  # Strong shrink
        embedding[11] = 1.0 if avg_area_ratio > 2.0 else 0.0  # Strong grow
        embedding[12] = 1.0 if abs(avg_aspect_change - 1.0) < 0.1 else 0.0  # Aspect preserved

        # Fill remaining dimensions with interaction terms
        for i in range(13, self.embedding_dim):
            idx1 = i % 8
            idx2 = (i * 7) % 8
            embedding[i] = embedding[idx1] * embedding[idx2]

        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        # Convert to ternary
        ternary = [
            1 if x > 0.2 else (-1 if x < -0.2 else 0)
            for x in embedding
        ]

        return TernaryVector(ternary)

    def encode_candidate_signature(
        self,
        candidate: Sequence[Sequence[int]],
        expected: Sequence[Sequence[int]],
    ) -> TernaryVector:
        """
        Encode size relationship between candidate and expected output.
        """
        h_cand = len(candidate)
        w_cand = len(candidate[0]) if candidate else 0
        h_exp = len(expected)
        w_exp = len(expected[0]) if expected else 0

        if h_exp == 0 or w_exp == 0:
            return self._neutral_embedding()

        h_ratio = h_cand / h_exp
        w_ratio = w_cand / w_exp
        area_ratio = (h_cand * w_cand) / (h_exp * w_exp)

        embedding = [0.0] * self.embedding_dim
        embedding[0] = h_ratio
        embedding[1] = w_ratio
        embedding[2] = area_ratio
        embedding[3] = w_cand / h_cand if h_cand > 0 else 1.0  # Candidate aspect
        embedding[4] = w_exp / h_exp if h_exp > 0 else 1.0    # Expected aspect

        # Direction
        embedding[8] = -1 if h_ratio < 0.9 else (1 if h_ratio > 1.1 else 0)
        embedding[9] = -1 if w_ratio < 0.9 else (1 if w_ratio > 1.1 else 0)

        # Match indicators
        embedding[10] = 1.0 if abs(h_ratio - 1.0) < 0.01 else 0.0  # Exact height
        embedding[11] = 1.0 if abs(w_ratio - 1.0) < 0.01 else 0.0  # Exact width
        embedding[12] = 1.0 if h_cand == h_exp and w_cand == w_exp else 0.0  # Exact match

        # Normalize and convert to ternary
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [
            1 if x > 0.2 else (-1 if x < -0.2 else 0)
            for x in embedding
        ]

        return TernaryVector(ternary)

    def should_evaluate(
        self,
        candidate: Sequence[Sequence[int]],
        expected: Sequence[Sequence[int]],
        task_pattern: TernaryVector,
    ) -> Tuple[bool, float]:
        """
        Model-based evaluation decision.

        Returns (should_evaluate, confidence).
        No hardcoded ratios — purely similarity-based.
        """
        candidate_sig = self.encode_candidate_signature(candidate, expected)

        # Compute similarity between candidate signature and task pattern
        similarity = self.cosine_bridge.compute_similarity(
            [candidate_sig.to_python()],
            task_pattern.to_python(),
        )[0]

        # Positive similarity = candidate fits task pattern
        # Negative similarity = candidate contradicts pattern
        # Zero = neutral

        return similarity > -0.3, similarity  # Allow neutral/positive

    def _neutral_embedding(self) -> TernaryVector:
        """Neutral embedding for unknown/empty patterns."""
        return TernaryVector([0] * self.embedding_dim)


__all__ = ["SizePatternEncoder"]
```

### 2.2 Integration with Sovereign Pipeline

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py` (MODIFY)

Replace hardcoded size logic:

```python
from knowledge3d.training.arc_agi.size_pattern_encoder import SizePatternEncoder


class SovereignPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.size_encoder = SizePatternEncoder()

    def process_task(self, task: Dict) -> Dict:
        # Encode size pattern ONCE per task
        task_size_pattern = self.size_encoder.encode_task_pattern(
            task.get("train", [])
        )

        # Pass to all evaluation paths
        task["size_pattern_embedding"] = task_size_pattern

        # ... rest of processing ...

    def _should_evaluate_candidate(
        self,
        candidate_output: Sequence[Sequence[int]],
        expected_output: Sequence[Sequence[int]],
        size_pattern_embedding: TernaryVector,  # CHANGED: embedding not string
    ) -> Tuple[bool, float]:
        """
        Model-based evaluation — no hardcoded thresholds.
        """
        return self.size_encoder.should_evaluate(
            candidate_output,
            expected_output,
            size_pattern_embedding,
        )
```

---

## Part 3: TRM Swarm Coordinator

### 3.1 Specialist Registry

**File:** `knowledge3d/training/arc_agi/specialist_registry.py` (CREATE)

```python
"""
Specialist registry — stores specialist embeddings in Galaxy.

Each specialist has:
- ID (e.g., "extraction", "rotation", "recolor")
- Embedding (TernaryVector) — what tasks it's good at
- Adapter weights (optional) — LoRA-style specialization
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from knowledge3d.cranium.ternary import TernaryVector, TernaryGalaxy


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
            # Store in galaxy
            self.galaxy.store_frame(
                f"specialist_{spec_id}",
                spec_data["description"],
                embedding,
            )

    def _keywords_to_embedding(self, keywords: List[str]) -> TernaryVector:
        """Convert keywords to embedding via hashing."""
        embedding = [0.0] * 128
        for kw in keywords:
            for i, char in enumerate(kw):
                idx = (ord(char) + i * 7) % 128
                embedding[idx] += 1.0

        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        # Ternary
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

        Moves specialist embedding toward successful tasks,
        away from failed tasks (online learning).
        """
        if spec_id not in self.specialists:
            return

        spec = self.specialists[spec_id]
        spec["usage_count"] += 1
        if success:
            spec["success_count"] += 1

        # Simple online update: blend embeddings
        current_emb = spec["embedding"].to_python()
        task_emb = task_embedding.to_python()

        learning_rate = 0.1 if success else -0.05

        new_emb = [
            c + learning_rate * (t - c)
            for c, t in zip(current_emb, task_emb)
        ]

        # Re-normalize and convert to ternary
        norm = sum(x*x for x in new_emb) ** 0.5
        if norm > 0:
            new_emb = [x / norm for x in new_emb]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in new_emb]
        spec["embedding"] = TernaryVector(ternary)

    def list_specialists(self) -> List[str]:
        """List all specialist IDs."""
        return list(self.specialists.keys())


__all__ = ["SpecialistRegistry", "BOOTSTRAP_SPECIALISTS"]
```

### 3.2 TRM Swarm Coordinator

**File:** `knowledge3d/training/arc_agi/trm_swarm_coordinator.py` (CREATE)

```python
"""
TRM Swarm Coordinator — router-as-model for dynamic worker spawning.

The router learns:
- Which specialists to invoke for a task
- How many workers to spawn
- How to aggregate results

Weights = logic (routing decisions)
Galaxy = knowledge (specialist embeddings, task patterns)
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, Future
from typing import Any, Callable, Dict, List, Optional, Tuple

from knowledge3d.cranium.ternary import TernaryVector
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.training.arc_agi.specialist_registry import SpecialistRegistry
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy, get_grammar_galaxy


class TRMSwarmCoordinator:
    """
    Master coordinator that routes tasks to specialists dynamically.

    No fixed worker count — spawns based on task complexity.
    No fixed routing rules — uses learned similarity.
    """

    def __init__(
        self,
        max_workers: int = 9,  # Tesla 3-6-9: up to 9 tiers
        spawn_threshold: float = 0.3,
        min_workers: int = 1,
    ):
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.spawn_threshold = spawn_threshold

        self.specialist_registry = SpecialistRegistry()
        self.cosine_bridge = CosineSimilarityBridge()

        # Learned parameters (updated via feedback)
        self._aggregation_weights: Dict[str, float] = {
            spec_id: 1.0 for spec_id in self.specialist_registry.list_specialists()
        }

    def embed_task(self, task: Dict) -> TernaryVector:
        """
        Compute task embedding from examples.

        Combines:
        - Size pattern
        - Color distribution
        - Spatial features
        - Semantic hints (if available)
        """
        embedding = [0.0] * 128

        train_examples = task.get("train", [])

        for ex in train_examples:
            inp = ex.get("input", [])
            out = ex.get("output", [])

            if not inp or not out:
                continue

            # Size features
            h_ratio = len(out) / max(1, len(inp))
            w_ratio = (len(out[0]) if out else 1) / max(1, len(inp[0]) if inp else 1)

            embedding[0] += h_ratio
            embedding[1] += w_ratio

            # Color distribution
            inp_colors = set()
            out_colors = set()
            for row in inp:
                inp_colors.update(row)
            for row in out:
                out_colors.update(row)

            embedding[2] += len(inp_colors)
            embedding[3] += len(out_colors)
            embedding[4] += len(out_colors - inp_colors)  # New colors
            embedding[5] += len(inp_colors - out_colors)  # Removed colors

        n = max(1, len(train_examples))
        embedding = [x / n for x in embedding]

        # Hash semantic hints if available
        hints = task.get("semantic_hints", [])
        for hint in hints:
            for i, char in enumerate(str(hint)):
                idx = (ord(char) + i * 13) % 128
                embedding[idx] += 0.1

        # Normalize
        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in embedding]
        return TernaryVector(ternary)

    def route_task(self, task_embedding: TernaryVector) -> List[Tuple[str, float]]:
        """
        Route task to specialists based on learned similarity.

        Returns: [(specialist_id, confidence), ...] sorted by confidence
        """
        scores = []
        task_emb = task_embedding.to_python()

        for spec_id in self.specialist_registry.list_specialists():
            spec_emb = self.specialist_registry.get_specialist_embedding(spec_id)
            if spec_emb is None:
                continue

            sim = self.cosine_bridge.compute_similarity(
                [task_emb], spec_emb.to_python()
            )[0]

            # Weight by historical success
            weight = self._aggregation_weights.get(spec_id, 1.0)
            scores.append((spec_id, sim * weight))

        # Sort by score descending
        scores.sort(key=lambda x: -x[1])

        # Filter by threshold
        selected = [(s, c) for s, c in scores if c >= self.spawn_threshold]

        # Ensure minimum workers
        if len(selected) < self.min_workers and scores:
            selected = scores[:self.min_workers]

        return selected[:self.max_workers]

    def get_optimal_parallelism(
        self,
        task_embedding: TernaryVector,
        work_items: int,
    ) -> int:
        """
        Determine optimal worker count based on task and workload.
        """
        routing = self.route_task(task_embedding)

        # Number of relevant specialists
        n_specialists = len(routing)

        # Balance with work items
        optimal = min(
            n_specialists,
            work_items,
            self.max_workers,
        )

        return max(self.min_workers, optimal)

    def partition_work_dynamic(
        self,
        items: List[Any],
        task_embedding: TernaryVector,
    ) -> List[List[Any]]:
        """
        Partition work based on task complexity.

        Uses round-robin for balanced load.
        """
        n_workers = self.get_optimal_parallelism(task_embedding, len(items))

        if n_workers <= 0 or not items:
            return [items] if items else []

        # Round-robin partitioning (max 1 item difference between workers)
        return [items[i::n_workers] for i in range(n_workers)]

    def spawn_workers(
        self,
        task: Dict,
        worker_fn: Callable,
        grammar_snapshot: bytes,
    ) -> List[Future]:
        """
        Spawn workers dynamically based on routing decision.
        """
        task_embedding = task.get("task_embedding") or self.embed_task(task)
        routing = self.route_task(task_embedding)

        n_workers = len(routing)
        if n_workers == 0:
            n_workers = self.min_workers
            routing = [(None, 0.0)] * n_workers

        futures = []

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for i, (spec_id, confidence) in enumerate(routing):
                future = executor.submit(
                    worker_fn,
                    task,
                    spec_id,
                    confidence,
                    grammar_snapshot,
                    i,  # worker_id for round-robin assignment
                    n_workers,
                )
                futures.append((spec_id, future))

        return futures

    def aggregate_results(
        self,
        results: List[Tuple[str, Any, float]],  # (spec_id, output, confidence)
        task_embedding: TernaryVector,
    ) -> Tuple[Any, float]:
        """
        Aggregate specialist outputs via learned weighting.

        Returns: (best_result, confidence)
        """
        if not results:
            return None, 0.0

        # Weight by specialist confidence and historical success
        weighted_results = []
        for spec_id, output, confidence in results:
            weight = self._aggregation_weights.get(spec_id, 1.0)
            final_score = confidence * weight
            weighted_results.append((output, final_score, spec_id))

        # Sort by weighted score
        weighted_results.sort(key=lambda x: -x[1])

        best_output, best_score, best_spec = weighted_results[0]
        return best_output, best_score

    def update_from_feedback(
        self,
        task_embedding: TernaryVector,
        specialist_results: List[Tuple[str, bool]],  # (spec_id, success)
    ) -> None:
        """
        Update routing weights based on task outcome.
        """
        for spec_id, success in specialist_results:
            if spec_id is None:
                continue

            # Update specialist embedding
            self.specialist_registry.update_specialist(
                spec_id, task_embedding, success
            )

            # Update aggregation weight
            current = self._aggregation_weights.get(spec_id, 1.0)
            delta = 0.1 if success else -0.05
            self._aggregation_weights[spec_id] = max(0.1, min(2.0, current + delta))


__all__ = ["TRMSwarmCoordinator"]
```

---

## Part 4: Integration

### 4.1 Parallel Candidate Generator Update

**File:** `knowledge3d/training/arc_agi/parallel_candidate_generator.py` (MODIFY)

```python
from knowledge3d.training.arc_agi.trm_swarm_coordinator import TRMSwarmCoordinator
from knowledge3d.training.arc_agi.grammar_galaxy import get_grammar_galaxy


# Module-level for worker reuse
_worker_grammar: Optional[GrammarGalaxy] = None


def _init_worker(grammar_snapshot: bytes) -> None:
    """Worker initializer — runs once per worker process."""
    global _worker_grammar
    if _worker_grammar is None:
        _worker_grammar = GrammarGalaxy(snapshot=grammar_snapshot)


def _worker_generate(
    task: Dict,
    specialist_id: Optional[str],
    confidence: float,
    grammar_snapshot: bytes,
    worker_id: int,
    n_workers: int,
) -> Tuple[str, List[Any], float, Dict]:
    """
    Worker function with specialist context.

    Returns: (specialist_id, candidates, confidence, discoveries)
    """
    global _worker_grammar
    if _worker_grammar is None:
        _init_worker(grammar_snapshot)

    # Get work partition (round-robin)
    hints = task.get("semantic_hints", [])
    my_hints = hints[worker_id::n_workers] if hints else []

    # Generate candidates with specialist bias
    candidates = []
    # ... generation logic using specialist_id to bias search ...

    # Return any grammar discoveries for merging
    discoveries = dict(_worker_grammar._local_discoveries)

    return specialist_id, candidates, confidence, discoveries


class ParallelCandidateGenerator:
    """
    Parallel candidate generation with TRM swarm coordination.
    """

    def __init__(self, ...):
        self.swarm = TRMSwarmCoordinator()
        self.grammar = get_grammar_galaxy()

    def generate(self, task: Dict) -> List[Any]:
        """Generate candidates using dynamic worker spawning."""

        # Embed task once
        task_embedding = self.swarm.embed_task(task)
        task["task_embedding"] = task_embedding

        # Get grammar snapshot (no file I/O in workers)
        grammar_snapshot = self.grammar.to_snapshot()

        # Spawn workers dynamically
        futures = self.swarm.spawn_workers(
            task,
            _worker_generate,
            grammar_snapshot,
        )

        # Collect results
        all_candidates = []
        for spec_id, future in futures:
            try:
                spec_id, candidates, confidence, discoveries = future.result()
                all_candidates.extend(candidates)

                # Merge grammar discoveries
                self.grammar.merge_discoveries(discoveries)

            except Exception as e:
                print(f"[WORKER ERROR] {spec_id}: {e}")

        return all_candidates
```

---

## Part 5: Discovery Preservation & Progressive Scoring

### Philosophy: Near-Misses Are Stepping Stones

The goal is 100% accuracy, but **discoveries must not be lost because they weren't perfect**. An 85% match today can become 100% tomorrow through progressive refinement.

**Core Principle:**
- Minimum threshold: **85%** — anything above this is worth preserving
- Goal threshold: **100%** — perfection is the target, not the gate
- Near-misses (85-99%) are **learning signals**, not failures

### 5.1 Progressive Score System

**File:** `knowledge3d/training/arc_agi/progressive_scorer.py` (CREATE)

```python
"""
Progressive scoring — discoveries improve over iterations.

No fixed thresholds — model learns what "good enough to keep" means.
Near-misses are preserved and refined, not discarded.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from knowledge3d.cranium.ternary import TernaryVector, TernaryGalaxy
from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge


class ProgressiveScorer:
    """
    Scores discoveries with preservation-first philosophy.

    - 85%+ : PRESERVE (add to discovery space)
    - 95%+ : PROMOTE (high confidence, ready for production)
    - 100% : CANONICAL (perfect match, permanent)

    Thresholds are LEARNABLE, not fixed.
    """

    def __init__(self):
        self.cosine_bridge = CosineSimilarityBridge()
        self.galaxy = TernaryGalaxy()

        # Initial thresholds (model will learn to adjust these)
        self._preserve_threshold = 0.85   # Minimum to keep
        self._promote_threshold = 0.95    # Ready for production use
        self._canonical_threshold = 1.0   # Perfect match

        # Threshold learning rate
        self._threshold_lr = 0.01

        # Track score distributions for adaptive thresholds
        self._score_history: List[float] = []
        self._max_history = 1000

    def score_discovery(
        self,
        candidate_output: List[List[int]],
        expected_output: List[List[int]],
        context: str,
    ) -> Tuple[float, str]:
        """
        Score a discovery and determine its fate.

        Returns: (score, fate) where fate is:
        - "discard" : below preserve threshold
        - "preserve" : worth keeping, needs refinement
        - "promote" : high confidence, ready for use
        - "canonical" : perfect match
        """
        # Compute exact match score
        exact_score = self._compute_exact_score(candidate_output, expected_output)

        # Compute fuzzy/semantic score for near-misses
        fuzzy_score = self._compute_fuzzy_score(candidate_output, expected_output)

        # Combined score (weight exact higher but don't ignore fuzzy)
        combined = 0.7 * exact_score + 0.3 * fuzzy_score

        # Track for adaptive thresholds
        self._record_score(combined)

        # Determine fate
        if combined >= self._canonical_threshold:
            fate = "canonical"
        elif combined >= self._promote_threshold:
            fate = "promote"
        elif combined >= self._preserve_threshold:
            fate = "preserve"
        else:
            fate = "discard"

        return combined, fate

    def _compute_exact_score(
        self,
        candidate: List[List[int]],
        expected: List[List[int]],
    ) -> float:
        """Compute pixel-exact match ratio."""
        if not candidate or not expected:
            return 0.0

        h_cand, w_cand = len(candidate), len(candidate[0]) if candidate else 0
        h_exp, w_exp = len(expected), len(expected[0]) if expected else 0

        # Shape must match for exact scoring
        if h_cand != h_exp or w_cand != w_exp:
            # Partial credit for size
            size_match = min(h_cand, h_exp) * min(w_cand, w_exp)
            size_total = max(h_cand, h_exp) * max(w_cand, w_exp)
            return 0.5 * (size_match / size_total) if size_total > 0 else 0.0

        # Count matching pixels
        matches = 0
        total = h_exp * w_exp

        for y in range(h_exp):
            for x in range(w_exp):
                if candidate[y][x] == expected[y][x]:
                    matches += 1

        return matches / total if total > 0 else 0.0

    def _compute_fuzzy_score(
        self,
        candidate: List[List[int]],
        expected: List[List[int]],
    ) -> float:
        """
        Compute fuzzy/semantic similarity.

        Captures structural similarity even when pixels don't match exactly.
        """
        if not candidate or not expected:
            return 0.0

        # Extract features
        cand_features = self._extract_features(candidate)
        exp_features = self._extract_features(expected)

        # Cosine similarity of feature vectors
        similarity = self.cosine_bridge.compute_similarity(
            [cand_features], exp_features
        )[0]

        # Normalize to [0, 1]
        return (similarity + 1.0) / 2.0

    def _extract_features(self, grid: List[List[int]]) -> List[float]:
        """Extract feature vector from grid."""
        if not grid:
            return [0.0] * 64

        h, w = len(grid), len(grid[0]) if grid else 0
        features = [0.0] * 64

        # Size features
        features[0] = h / 30.0  # Normalized height
        features[1] = w / 30.0  # Normalized width
        features[2] = (h * w) / 900.0  # Normalized area
        features[3] = w / h if h > 0 else 1.0  # Aspect ratio

        # Color distribution
        color_counts = [0] * 10
        for row in grid:
            for pixel in row:
                if 0 <= pixel < 10:
                    color_counts[pixel] += 1

        total_pixels = h * w
        for i, count in enumerate(color_counts):
            features[4 + i] = count / total_pixels if total_pixels > 0 else 0.0

        # Spatial features (quadrant distribution)
        mid_h, mid_w = h // 2, w // 2
        quadrants = [0, 0, 0, 0]
        for y, row in enumerate(grid):
            for x, pixel in enumerate(row):
                if pixel != 0:  # Non-background
                    q = (1 if y >= mid_h else 0) + (2 if x >= mid_w else 0)
                    quadrants[q] += 1

        for i, q in enumerate(quadrants):
            features[14 + i] = q / total_pixels if total_pixels > 0 else 0.0

        # Edge density
        edges = 0
        for y in range(h):
            for x in range(w):
                if x > 0 and grid[y][x] != grid[y][x-1]:
                    edges += 1
                if y > 0 and grid[y][x] != grid[y-1][x]:
                    edges += 1
        features[18] = edges / (2 * total_pixels) if total_pixels > 0 else 0.0

        # Normalize
        norm = sum(x*x for x in features) ** 0.5
        if norm > 0:
            features = [x / norm for x in features]

        return features

    def _record_score(self, score: float) -> None:
        """Record score for adaptive threshold learning."""
        self._score_history.append(score)
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)

    def adapt_thresholds(self) -> None:
        """
        Adapt thresholds based on score distribution.

        Goal: preserve ~top 60% of discoveries (learnable retention rate).
        """
        if len(self._score_history) < 100:
            return  # Need enough data

        sorted_scores = sorted(self._score_history)
        n = len(sorted_scores)

        # Target: 85th percentile becomes preserve threshold
        # This makes preserve_threshold adaptive to actual distribution
        target_preserve_idx = int(n * 0.40)  # Keep top 60%
        target_preserve = sorted_scores[target_preserve_idx]

        # Smooth update
        self._preserve_threshold += self._threshold_lr * (
            target_preserve - self._preserve_threshold
        )

        # Clamp to reasonable range
        self._preserve_threshold = max(0.70, min(0.90, self._preserve_threshold))

        # Promote threshold scales with preserve
        self._promote_threshold = min(0.99, self._preserve_threshold + 0.10)

    def get_thresholds(self) -> Dict[str, float]:
        """Get current (possibly adapted) thresholds."""
        return {
            "preserve": self._preserve_threshold,
            "promote": self._promote_threshold,
            "canonical": self._canonical_threshold,
        }


class DiscoveryPreserver:
    """
    Preserves near-miss discoveries for progressive refinement.

    A discovery at 85% today might become 100% after:
    - More training examples seen
    - Related patterns learned
    - Composition with other discoveries
    """

    def __init__(self):
        self.scorer = ProgressiveScorer()
        self.galaxy = TernaryGalaxy()
        self._preserved: Dict[str, Dict] = {}

    def evaluate_and_preserve(
        self,
        discovery_id: str,
        rpn_program: str,
        candidate_output: List[List[int]],
        expected_output: List[List[int]],
        context: str,
    ) -> Tuple[float, str, bool]:
        """
        Evaluate discovery and preserve if worthy.

        Returns: (score, fate, was_preserved)
        """
        score, fate = self.scorer.score_discovery(
            candidate_output, expected_output, context
        )

        was_preserved = False

        if fate != "discard":
            # Preserve the discovery
            self._preserved[discovery_id] = {
                "rpn_program": rpn_program,
                "score": score,
                "fate": fate,
                "context": context,
                "attempts": 1,
                "best_score": score,
                "improvement_history": [score],
            }

            # Store embedding in galaxy for similarity search
            embedding = self._program_to_embedding(rpn_program)
            self.galaxy.store_frame(
                f"discovery_{discovery_id}",
                f"{fate}:{score:.3f}",
                embedding,
            )

            was_preserved = True

            if fate == "preserve":
                print(f"[PRESERVE] {discovery_id}: {score:.2%} — near-miss, kept for refinement")
            elif fate == "promote":
                print(f"[PROMOTE] {discovery_id}: {score:.2%} — high confidence")
            elif fate == "canonical":
                print(f"[CANONICAL] {discovery_id}: {score:.2%} — perfect match!")

        return score, fate, was_preserved

    def attempt_refinement(
        self,
        discovery_id: str,
        new_candidate: List[List[int]],
        expected: List[List[int]],
    ) -> Tuple[float, bool]:
        """
        Attempt to improve a preserved discovery.

        Returns: (new_score, improved)
        """
        if discovery_id not in self._preserved:
            return 0.0, False

        record = self._preserved[discovery_id]
        new_score, _ = self.scorer.score_discovery(
            new_candidate, expected, record["context"]
        )

        record["attempts"] += 1
        record["improvement_history"].append(new_score)

        improved = new_score > record["best_score"]
        if improved:
            old_best = record["best_score"]
            record["best_score"] = new_score
            record["score"] = new_score

            # Update fate if improved enough
            if new_score >= 1.0:
                record["fate"] = "canonical"
            elif new_score >= 0.95:
                record["fate"] = "promote"

            print(f"[IMPROVED] {discovery_id}: {old_best:.2%} → {new_score:.2%}")

        return new_score, improved

    def get_refinement_candidates(self, k: int = 10) -> List[Tuple[str, Dict]]:
        """
        Get top-k discoveries most likely to benefit from refinement.

        Prioritizes:
        - Near-threshold discoveries (close to next tier)
        - Discoveries with improving trend
        - Discoveries with few attempts
        """
        candidates = []

        for disc_id, record in self._preserved.items():
            if record["fate"] == "canonical":
                continue  # Already perfect

            # Score based on potential
            score = record["score"]
            attempts = record["attempts"]

            # Close to next threshold = high priority
            if record["fate"] == "preserve":
                gap_to_promote = 0.95 - score
                priority = 1.0 - gap_to_promote  # Closer = higher priority
            else:
                gap_to_canonical = 1.0 - score
                priority = 1.0 - gap_to_canonical

            # Boost for few attempts (unexplored potential)
            if attempts < 5:
                priority *= 1.5

            # Boost for improving trend
            history = record["improvement_history"]
            if len(history) >= 2 and history[-1] > history[-2]:
                priority *= 1.2

            candidates.append((disc_id, record, priority))

        # Sort by priority descending
        candidates.sort(key=lambda x: -x[2])

        return [(disc_id, record) for disc_id, record, _ in candidates[:k]]

    def _program_to_embedding(self, rpn_program: str) -> TernaryVector:
        """Convert RPN program to embedding."""
        tokens = rpn_program.split()
        embedding = [0.0] * 128

        for i, token in enumerate(tokens):
            idx = hash(token) % 128
            embedding[idx] += 1.0 / (i + 1)

        norm = sum(x*x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]

        ternary = [1 if x > 0.1 else (-1 if x < -0.1 else 0) for x in embedding]
        return TernaryVector(ternary)

    def get_preservation_stats(self) -> Dict:
        """Get statistics about preserved discoveries."""
        stats = {
            "total": len(self._preserved),
            "by_fate": {"preserve": 0, "promote": 0, "canonical": 0},
            "avg_score": 0.0,
            "avg_attempts": 0.0,
        }

        if not self._preserved:
            return stats

        total_score = 0.0
        total_attempts = 0

        for record in self._preserved.values():
            stats["by_fate"][record["fate"]] += 1
            total_score += record["score"]
            total_attempts += record["attempts"]

        n = len(self._preserved)
        stats["avg_score"] = total_score / n
        stats["avg_attempts"] = total_attempts / n

        return stats


__all__ = ["ProgressiveScorer", "DiscoveryPreserver"]
```

### 5.2 Integration with Training Loop

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py` (MODIFY)

Add progressive scoring to evaluation:

```python
from knowledge3d.training.arc_agi.progressive_scorer import DiscoveryPreserver


class SovereignPipeline:
    def __init__(self, ...):
        # ... existing init ...
        self.preserver = DiscoveryPreserver()

    def evaluate_candidate(
        self,
        candidate_id: str,
        rpn_program: str,
        candidate_output: List[List[int]],
        expected_output: List[List[int]],
        context: str,
    ) -> Dict:
        """
        Evaluate with progressive scoring and preservation.
        """
        score, fate, preserved = self.preserver.evaluate_and_preserve(
            candidate_id,
            rpn_program,
            candidate_output,
            expected_output,
            context,
        )

        return {
            "score": score,
            "fate": fate,
            "preserved": preserved,
            "exact_match": score >= 1.0,
            "near_miss": 0.85 <= score < 1.0,
            "high_confidence": score >= 0.95,
        }

    def end_of_epoch_refinement(self) -> None:
        """
        At end of epoch, attempt to refine preserved discoveries.
        """
        # Adapt thresholds based on distribution
        self.preserver.scorer.adapt_thresholds()

        # Get refinement candidates
        candidates = self.preserver.get_refinement_candidates(k=20)

        print(f"[REFINEMENT] Attempting refinement on {len(candidates)} discoveries")

        # Log preservation stats
        stats = self.preserver.get_preservation_stats()
        print(f"[PRESERVATION] Total: {stats['total']}, "
              f"Preserve: {stats['by_fate']['preserve']}, "
              f"Promote: {stats['by_fate']['promote']}, "
              f"Canonical: {stats['by_fate']['canonical']}")
```

### 5.3 Key Points

| Aspect | Value | Learnable? |
|--------|-------|------------|
| Preserve threshold | 85% (initial) | YES — adapts to distribution |
| Promote threshold | 95% (initial) | YES — scales with preserve |
| Canonical threshold | 100% | NO — perfection is fixed |
| Retention rate | ~60% of scored items | YES — implicitly via threshold |

**Near-Miss Philosophy:**
- 85% match = "almost there" → preserve and refine
- 95% match = "production ready" → promote
- 100% match = "perfect" → canonical (permanent)

**Progressive Refinement:**
- Preserved discoveries can be re-evaluated with new context
- Score history tracked for trend analysis
- High-potential discoveries prioritized for refinement

---

## Files Summary

| File | Action | Purpose |
|------|--------|---------|
| `rpn_opcodes.py` | MODIFY | Grammar evolution opcodes |
| `grammar_galaxy.py` | MODIFY | Two-tier memory, discovery lifecycle |
| `size_pattern_encoder.py` | CREATE | Embedding-based size patterns |
| `sovereign_pipeline.py` | MODIFY | Size encoder + progressive scorer + dataset + reporter |
| `specialist_registry.py` | CREATE | Specialist embeddings |
| `trm_swarm_coordinator.py` | CREATE | Router-as-model coordinator |
| `parallel_candidate_generator.py` | MODIFY | Integration |
| `progressive_scorer.py` | CREATE | Discovery preservation, adaptive thresholds |
| `session_reporter.py` | CREATE | Structured JSON training reports |
| `unified_dataset_loader.py` | CREATE | ARC-AGI 1 + 2 interleaved loading |
| `migrate_kernels.sh` | CREATE | Move non-canonical kernels to Old_Attempts |
| `cleanup_arc_agi.sh` | CREATE | Organize arc_agi folder structure |

---

## Verification

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Test 1: Grammar discovery
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.grammar_galaxy import GrammarGalaxy

g = GrammarGalaxy()
print(f'Initial rules: {len(g.rules)}')

# Simulate discovery
rule_id = g.propose_rule('DIM_0 2.0 MUL DIM_1 STORE', 'test_context')
print(f'Proposed: {rule_id}')

# Simulate usage
for _ in range(5):
    score = g.validate_usage(rule_id, success=True)
print(f'Quality after 5 successes: {score:.2f}')

# Check if promoted
print(f'Promoted: {rule_id in g.rules}')
print('=== GRAMMAR DISCOVERY OK ===')
"

# Test 2: Size pattern encoder
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.size_pattern_encoder import SizePatternEncoder

encoder = SizePatternEncoder()

# Extraction pattern (shrink)
examples = [
    {'input': [[0]*10 for _ in range(10)], 'output': [[1]*3 for _ in range(3)]},
    {'input': [[0]*8 for _ in range(8)], 'output': [[1]*2 for _ in range(2)]},
]
pattern = encoder.encode_task_pattern(examples)
print(f'Extraction pattern: {pattern.to_python()[:10]}...')

# Test evaluation
candidate = [[0]*3 for _ in range(3)]
expected = [[1]*3 for _ in range(3)]
should_eval, conf = encoder.should_evaluate(candidate, expected, pattern)
print(f'Should evaluate 3x3 candidate: {should_eval}, confidence: {conf:.2f}')
print('=== SIZE PATTERN ENCODER OK ===')
"

# Test 3: TRM Swarm routing
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.trm_swarm_coordinator import TRMSwarmCoordinator

swarm = TRMSwarmCoordinator()

# Create test task
task = {
    'train': [
        {'input': [[0]*10]*10, 'output': [[1]*3]*3},  # Extraction-like
    ],
    'semantic_hints': ['extract', 'crop', 'subset'],
}

embedding = swarm.embed_task(task)
routing = swarm.route_task(embedding)

print(f'Task embedding: {embedding.to_python()[:5]}...')
print(f'Routing decision:')
for spec_id, score in routing[:3]:
    print(f'  {spec_id}: {score:.3f}')

n_workers = swarm.get_optimal_parallelism(embedding, 20)
print(f'Optimal workers: {n_workers}')
print('=== TRM SWARM COORDINATOR OK ===')
"

# Test 4: Progressive scoring
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.progressive_scorer import ProgressiveScorer, DiscoveryPreserver

scorer = ProgressiveScorer()

# Perfect match
perfect = [[1, 2], [3, 4]]
score, fate = scorer.score_discovery(perfect, perfect, 'test')
print(f'Perfect match: {score:.2%} -> {fate}')
assert fate == 'canonical', 'Perfect should be canonical'

# Near-miss (1 pixel off)
near_miss = [[1, 2], [3, 5]]  # 4 -> 5
expected = [[1, 2], [3, 4]]
score, fate = scorer.score_discovery(near_miss, expected, 'test')
print(f'Near-miss (1 pixel): {score:.2%} -> {fate}')

# Test preservation
preserver = DiscoveryPreserver()
score, fate, preserved = preserver.evaluate_and_preserve(
    'test_disc_001',
    'GRID 2 2 FILL',
    [[1, 1], [1, 1]],
    [[1, 1], [1, 2]],  # 87.5% match
    'test_context',
)
print(f'Preservation: score={score:.2%}, fate={fate}, preserved={preserved}')

# Test refinement
new_score, improved = preserver.attempt_refinement(
    'test_disc_001',
    [[1, 1], [1, 2]],  # Now perfect
    [[1, 1], [1, 2]],
)
print(f'Refinement: new_score={new_score:.2%}, improved={improved}')

stats = preserver.get_preservation_stats()
print(f'Stats: {stats}')
print('=== PROGRESSIVE SCORER OK ===')
"
```

---

## Part 6: Organization, Logging & Dual-Dataset Training

### 6.1 Kernel Organization

**All CUDA kernels must live in the kernels folder.** Move deprecated/experimental versions to `Old_Attempts`.

**Directory Structure:**
```
knowledge3d/cranium/
├── kernels/                          # Active kernels ONLY
│   ├── codec_ops.cu                  # Audio/Video/Image codecs
│   ├── drawing_transform_ops.cu      # Drawing Galaxy transforms
│   ├── color_convert.cu              # Color space conversions
│   ├── filter_convolution.cu         # Image filters
│   ├── gradient_rasterizer.cu        # Gradient rendering
│   ├── vectordotmap_encoder.cu       # VectorDotMap codec
│   └── ... (other active kernels)
│
├── kernels/Old_Attempts/             # Deprecated/experimental versions
│   ├── codec_ops_v1.cu               # Version history
│   ├── drawing_ops_draft.cu
│   └── ...
│
├── ptx/                              # Compiled PTX (auto-generated)
│   └── ...
│
└── ptx_runtime/                      # Python wrappers
    └── ...
```

**Migration Script:**
```bash
#!/bin/bash
# migrate_kernels.sh — Move non-canonical kernels to Old_Attempts

KERNELS_DIR="knowledge3d/cranium/kernels"
OLD_DIR="$KERNELS_DIR/Old_Attempts"

mkdir -p "$OLD_DIR"

# Files to keep in kernels/ (canonical list)
CANONICAL=(
    "codec_ops.cu"
    "drawing_transform_ops.cu"
    "color_convert.cu"
    "filter_convolution.cu"
    "gradient_rasterizer.cu"
    "vectordotmap_encoder.cu"
    "trm_ops.cu"
    "ternary_ops.cu"
)

# Move non-canonical .cu files
for f in "$KERNELS_DIR"/*.cu; do
    basename=$(basename "$f")
    keep=false
    for canonical in "${CANONICAL[@]}"; do
        if [ "$basename" = "$canonical" ]; then
            keep=true
            break
        fi
    done
    if [ "$keep" = false ]; then
        echo "Moving $basename to Old_Attempts/"
        mv "$f" "$OLD_DIR/"
    fi
done

echo "Kernel migration complete."
```

### 6.2 ARC-AGI Folder Cleanup

**The `arc_agi/` folder should contain ONLY ARC-AGI specific files.**

**Clean Structure:**
```
knowledge3d/training/arc_agi/
├── __init__.py
├── candidate_generator.py            # Candidate generation
├── drawing_galaxy.py                 # Drawing primitives registry
├── grammar_galaxy.py                 # Grammar rules + discovery
├── multimodal_parser.py              # Task parsing
├── parallel_candidate_generator.py   # Parallel execution
├── progressive_scorer.py             # Discovery preservation
├── rpn_executor.py                   # RPN execution
├── size_pattern_encoder.py           # Size embeddings
├── sovereign_pipeline.py             # Main pipeline
├── specialist_registry.py            # Specialist embeddings
├── trm_swarm_coordinator.py          # Router-as-model
├── unified_dataset_loader.py         # NEW: Dual-dataset loader
│
├── logs/                             # Training session reports
│   └── session_YYYYMMDD_HHMMSS.json
│
└── Old_Attempts/                     # Deprecated implementations
    └── ...
```

**Cleanup Script:**
```bash
#!/bin/bash
# cleanup_arc_agi.sh — Organize ARC-AGI folder

ARC_DIR="knowledge3d/training/arc_agi"
OLD_DIR="$ARC_DIR/Old_Attempts"
LOGS_DIR="$ARC_DIR/logs"

mkdir -p "$OLD_DIR" "$LOGS_DIR"

# Canonical files (keep in main folder)
CANONICAL=(
    "__init__.py"
    "candidate_generator.py"
    "drawing_galaxy.py"
    "grammar_galaxy.py"
    "multimodal_parser.py"
    "parallel_candidate_generator.py"
    "progressive_scorer.py"
    "rpn_executor.py"
    "size_pattern_encoder.py"
    "sovereign_pipeline.py"
    "specialist_registry.py"
    "trm_swarm_coordinator.py"
    "unified_dataset_loader.py"
)

# Move non-canonical .py files
for f in "$ARC_DIR"/*.py; do
    basename=$(basename "$f")
    keep=false
    for canonical in "${CANONICAL[@]}"; do
        if [ "$basename" = "$canonical" ]; then
            keep=true
            break
        fi
    done
    if [ "$keep" = false ]; then
        echo "Moving $basename to Old_Attempts/"
        mv "$f" "$OLD_DIR/"
    fi
done

echo "ARC-AGI folder cleanup complete."
```

### 6.3 Training Session Reports

**Every training run produces a structured JSON report.**

**File:** `knowledge3d/training/arc_agi/session_reporter.py` (CREATE)

```python
"""
Training session reporter — structured logs for every run.

Outputs to: knowledge3d/training/arc_agi/logs/session_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionReporter:
    """
    Generates structured training session reports.
    """

    def __init__(self, session_name: Optional[str] = None):
        self.start_time = datetime.now()
        self.session_id = session_name or self.start_time.strftime("%Y%m%d_%H%M%S")

        # Report data
        self.report: Dict[str, Any] = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "config": {},
            "datasets": {
                "arc_agi_1": {"path": None, "task_count": 0},
                "arc_agi_2": {"path": None, "task_count": 0},
                "total_tasks": 0,
            },
            "epochs": [],
            "summary": {
                "total_epochs": 0,
                "best_epoch": None,
                "best_accuracy": 0.0,
                "final_accuracy": 0.0,
                "discoveries": {
                    "proposed": 0,
                    "preserved": 0,
                    "promoted": 0,
                    "canonical": 0,
                },
                "grammar_rules": {
                    "initial": 0,
                    "final": 0,
                    "discovered": 0,
                },
            },
            "errors": [],
            "warnings": [],
        }

        # Ensure logs directory exists
        self.logs_dir = Path(__file__).parent / "logs"
        self.logs_dir.mkdir(exist_ok=True)

    def set_config(self, config: Dict[str, Any]) -> None:
        """Record training configuration."""
        self.report["config"] = config

    def set_datasets(
        self,
        arc1_path: str,
        arc1_count: int,
        arc2_path: str,
        arc2_count: int,
    ) -> None:
        """Record dataset information."""
        self.report["datasets"] = {
            "arc_agi_1": {"path": arc1_path, "task_count": arc1_count},
            "arc_agi_2": {"path": arc2_path, "task_count": arc2_count},
            "total_tasks": arc1_count + arc2_count,
        }

    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, Any],
    ) -> None:
        """Log metrics for an epoch."""
        epoch_data = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            **metrics,
        }
        self.report["epochs"].append(epoch_data)

        # Update best
        accuracy = metrics.get("accuracy", 0.0)
        if accuracy > self.report["summary"]["best_accuracy"]:
            self.report["summary"]["best_accuracy"] = accuracy
            self.report["summary"]["best_epoch"] = epoch

        self.report["summary"]["total_epochs"] = epoch
        self.report["summary"]["final_accuracy"] = accuracy

    def log_discovery_stats(
        self,
        proposed: int,
        preserved: int,
        promoted: int,
        canonical: int,
    ) -> None:
        """Log discovery preservation statistics."""
        self.report["summary"]["discoveries"] = {
            "proposed": proposed,
            "preserved": preserved,
            "promoted": promoted,
            "canonical": canonical,
        }

    def log_grammar_stats(
        self,
        initial: int,
        final: int,
        discovered: int,
    ) -> None:
        """Log grammar evolution statistics."""
        self.report["summary"]["grammar_rules"] = {
            "initial": initial,
            "final": final,
            "discovered": discovered,
        }

    def log_error(self, error: str, context: Optional[str] = None) -> None:
        """Log an error."""
        self.report["errors"].append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "context": context,
        })

    def log_warning(self, warning: str) -> None:
        """Log a warning."""
        self.report["warnings"].append({
            "timestamp": datetime.now().isoformat(),
            "warning": warning,
        })

    def finalize(self) -> str:
        """Finalize and save the report. Returns the file path."""
        end_time = datetime.now()
        self.report["end_time"] = end_time.isoformat()
        self.report["duration_seconds"] = (end_time - self.start_time).total_seconds()

        # Save to file
        report_path = self.logs_dir / f"session_{self.session_id}.json"
        with open(report_path, "w") as f:
            json.dump(self.report, f, indent=2)

        print(f"[SESSION REPORT] Saved to {report_path}")
        return str(report_path)

    def print_summary(self) -> None:
        """Print human-readable summary."""
        s = self.report["summary"]
        d = s["discoveries"]
        g = s["grammar_rules"]

        print("\n" + "=" * 60)
        print(f"SESSION REPORT: {self.session_id}")
        print("=" * 60)
        print(f"Duration: {self.report['duration_seconds']:.1f}s")
        print(f"Total epochs: {s['total_epochs']}")
        print(f"Best accuracy: {s['best_accuracy']:.2%} (epoch {s['best_epoch']})")
        print(f"Final accuracy: {s['final_accuracy']:.2%}")
        print(f"\nDiscoveries:")
        print(f"  Proposed: {d['proposed']}")
        print(f"  Preserved: {d['preserved']} (85%+ matches)")
        print(f"  Promoted: {d['promoted']} (95%+ matches)")
        print(f"  Canonical: {d['canonical']} (100% matches)")
        print(f"\nGrammar Rules:")
        print(f"  Initial: {g['initial']}")
        print(f"  Final: {g['final']} (+{g['discovered']} discovered)")
        print(f"\nErrors: {len(self.report['errors'])}")
        print(f"Warnings: {len(self.report['warnings'])}")
        print("=" * 60 + "\n")


__all__ = ["SessionReporter"]
```

### 6.4 Unified Dataset Loader (ARC-AGI 1 + 2)

**Double the training data by including ARC-AGI 1 with proper alternation.**

**Paths:**
- **ARC-AGI 1:** `/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/training/`
- **ARC-AGI 2:** `/K3D/Knowledge3D.local/datasets/exams/arc-src/data/training/`

**Alternation Pattern:** 1 task from ARC-1, then 1 task from ARC-2 (same difficulty band)

**File:** `knowledge3d/training/arc_agi/unified_dataset_loader.py` (CREATE)

```python
"""
Unified ARC-AGI dataset loader — combines ARC-AGI 1 and ARC-AGI 2.

Alternates between datasets for balanced exposure:
- Task N from ARC-1 (difficulty level D)
- Task N from ARC-2 (difficulty level D)

This doubles training data while maintaining difficulty progression.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


# Dataset paths
ARC_AGI_1_PATH = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data")
ARC_AGI_2_PATH = Path("/K3D/Knowledge3D.local/datasets/exams/arc-src/data")


class UnifiedDatasetLoader:
    """
    Loads and alternates between ARC-AGI 1 and ARC-AGI 2 datasets.
    """

    def __init__(
        self,
        arc1_path: Optional[Path] = None,
        arc2_path: Optional[Path] = None,
        split: str = "training",
    ):
        self.arc1_path = arc1_path or ARC_AGI_1_PATH
        self.arc2_path = arc2_path or ARC_AGI_2_PATH
        self.split = split

        # Load task lists
        self.arc1_tasks = self._load_task_list(self.arc1_path / split)
        self.arc2_tasks = self._load_task_list(self.arc2_path / split)

        # Sort by estimated difficulty (task complexity heuristic)
        self.arc1_tasks = self._sort_by_difficulty(self.arc1_tasks)
        self.arc2_tasks = self._sort_by_difficulty(self.arc2_tasks)

        print(f"[DATASET] ARC-AGI 1: {len(self.arc1_tasks)} tasks from {self.arc1_path}")
        print(f"[DATASET] ARC-AGI 2: {len(self.arc2_tasks)} tasks from {self.arc2_path}")
        print(f"[DATASET] Total: {len(self.arc1_tasks) + len(self.arc2_tasks)} tasks")

    def _load_task_list(self, path: Path) -> List[Tuple[str, Path]]:
        """Load list of (task_id, file_path) from directory."""
        if not path.exists():
            print(f"[WARNING] Dataset path not found: {path}")
            return []

        tasks = []
        for f in sorted(path.glob("*.json")):
            task_id = f.stem
            tasks.append((task_id, f))

        return tasks

    def _sort_by_difficulty(
        self,
        tasks: List[Tuple[str, Path]],
    ) -> List[Tuple[str, Path]]:
        """
        Sort tasks by estimated difficulty.

        Difficulty heuristic based on:
        - Number of training examples
        - Grid sizes
        - Color count
        """
        scored_tasks = []

        for task_id, path in tasks:
            try:
                with open(path) as f:
                    data = json.load(f)

                train = data.get("train", [])
                test = data.get("test", [])

                # Difficulty factors
                n_examples = len(train)
                avg_input_size = 0
                avg_output_size = 0
                total_colors = set()

                for ex in train:
                    inp = ex.get("input", [[]])
                    out = ex.get("output", [[]])
                    avg_input_size += len(inp) * len(inp[0]) if inp else 0
                    avg_output_size += len(out) * len(out[0]) if out else 0
                    for row in inp:
                        total_colors.update(row)
                    for row in out:
                        total_colors.update(row)

                if n_examples > 0:
                    avg_input_size /= n_examples
                    avg_output_size /= n_examples

                # Difficulty score (higher = harder)
                difficulty = (
                    (1 / (n_examples + 1)) * 10 +  # Fewer examples = harder
                    avg_input_size / 100 +
                    avg_output_size / 100 +
                    len(total_colors) * 0.5
                )

                scored_tasks.append((difficulty, task_id, path))

            except Exception:
                # Default difficulty for problematic tasks
                scored_tasks.append((50.0, task_id, path))

        # Sort by difficulty (easiest first)
        scored_tasks.sort(key=lambda x: x[0])

        return [(task_id, path) for _, task_id, path in scored_tasks]

    def __len__(self) -> int:
        """Total number of tasks across both datasets."""
        return len(self.arc1_tasks) + len(self.arc2_tasks)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate tasks with alternation: ARC-1, ARC-2, ARC-1, ARC-2, ...

        Pairs tasks of similar difficulty.
        """
        max_len = max(len(self.arc1_tasks), len(self.arc2_tasks))

        for i in range(max_len):
            # ARC-AGI 1 task
            if i < len(self.arc1_tasks):
                task_id, path = self.arc1_tasks[i]
                yield self._load_task(task_id, path, source="arc_agi_1")

            # ARC-AGI 2 task (same difficulty band)
            if i < len(self.arc2_tasks):
                task_id, path = self.arc2_tasks[i]
                yield self._load_task(task_id, path, source="arc_agi_2")

    def _load_task(
        self,
        task_id: str,
        path: Path,
        source: str,
    ) -> Dict[str, Any]:
        """Load a single task with metadata."""
        with open(path) as f:
            data = json.load(f)

        return {
            "task_id": task_id,
            "source": source,
            "path": str(path),
            "train": data.get("train", []),
            "test": data.get("test", []),
        }

    def get_arc1_tasks(self) -> List[Dict[str, Any]]:
        """Get all ARC-AGI 1 tasks."""
        return [
            self._load_task(task_id, path, "arc_agi_1")
            for task_id, path in self.arc1_tasks
        ]

    def get_arc2_tasks(self) -> List[Dict[str, Any]]:
        """Get all ARC-AGI 2 tasks."""
        return [
            self._load_task(task_id, path, "arc_agi_2")
            for task_id, path in self.arc2_tasks
        ]

    def get_interleaved_batches(
        self,
        batch_size: int = 2,
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Get batches with 1 ARC-1 + 1 ARC-2 task per batch.

        batch_size must be even for proper alternation.
        """
        if batch_size % 2 != 0:
            batch_size += 1
            print(f"[WARNING] batch_size adjusted to {batch_size} for even alternation")

        half = batch_size // 2
        max_pairs = min(len(self.arc1_tasks), len(self.arc2_tasks))

        for i in range(0, max_pairs, half):
            batch = []

            # Add ARC-1 tasks
            for j in range(half):
                if i + j < len(self.arc1_tasks):
                    task_id, path = self.arc1_tasks[i + j]
                    batch.append(self._load_task(task_id, path, "arc_agi_1"))

            # Add ARC-2 tasks
            for j in range(half):
                if i + j < len(self.arc2_tasks):
                    task_id, path = self.arc2_tasks[i + j]
                    batch.append(self._load_task(task_id, path, "arc_agi_2"))

            if batch:
                yield batch

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            "arc_agi_1": {
                "path": str(self.arc1_path),
                "task_count": len(self.arc1_tasks),
            },
            "arc_agi_2": {
                "path": str(self.arc2_path),
                "task_count": len(self.arc2_tasks),
            },
            "total_tasks": len(self.arc1_tasks) + len(self.arc2_tasks),
            "split": self.split,
        }


__all__ = ["UnifiedDatasetLoader", "ARC_AGI_1_PATH", "ARC_AGI_2_PATH"]
```

### 6.5 Integration with Training Pipeline

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py` (MODIFY)

```python
from knowledge3d.training.arc_agi.unified_dataset_loader import UnifiedDatasetLoader
from knowledge3d.training.arc_agi.session_reporter import SessionReporter


class SovereignPipeline:
    def __init__(self, ...):
        # ... existing init ...

        # Unified dataset (ARC-1 + ARC-2)
        self.dataset = UnifiedDatasetLoader(split="training")

        # Session reporter
        self.reporter = SessionReporter()
        self.reporter.set_datasets(
            arc1_path=str(self.dataset.arc1_path),
            arc1_count=len(self.dataset.arc1_tasks),
            arc2_path=str(self.dataset.arc2_path),
            arc2_count=len(self.dataset.arc2_tasks),
        )

    def train(self, epochs: int, **config) -> str:
        """
        Run training with session reporting.

        Returns: Path to session report.
        """
        self.reporter.set_config(config)
        initial_grammar_count = len(self.grammar.rules)

        for epoch in range(epochs):
            epoch_metrics = self._run_epoch(epoch)
            self.reporter.log_epoch(epoch, epoch_metrics)

            # End-of-epoch refinement
            self.end_of_epoch_refinement()

        # Final stats
        self.reporter.log_grammar_stats(
            initial=initial_grammar_count,
            final=len(self.grammar.rules),
            discovered=len(self.grammar.rules) - initial_grammar_count,
        )

        # Preservation stats
        stats = self.preserver.get_preservation_stats()
        self.reporter.log_discovery_stats(
            proposed=stats["total"],
            preserved=stats["by_fate"]["preserve"],
            promoted=stats["by_fate"]["promote"],
            canonical=stats["by_fate"]["canonical"],
        )

        # Finalize and print
        report_path = self.reporter.finalize()
        self.reporter.print_summary()

        return report_path

    def _run_epoch(self, epoch: int) -> Dict[str, Any]:
        """Run single epoch over interleaved dataset."""
        correct = 0
        total = 0
        arc1_correct = 0
        arc1_total = 0
        arc2_correct = 0
        arc2_total = 0

        for task in self.dataset:
            result = self.process_task(task)
            total += 1

            if result.get("exact_match"):
                correct += 1

            # Track per-dataset metrics
            if task["source"] == "arc_agi_1":
                arc1_total += 1
                if result.get("exact_match"):
                    arc1_correct += 1
            else:
                arc2_total += 1
                if result.get("exact_match"):
                    arc2_correct += 1

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total_tasks": total,
            "correct": correct,
            "arc1_accuracy": arc1_correct / arc1_total if arc1_total > 0 else 0.0,
            "arc2_accuracy": arc2_correct / arc2_total if arc2_total > 0 else 0.0,
        }
```

### 6.6 Verification

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Test 1: Unified dataset loader
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.unified_dataset_loader import UnifiedDatasetLoader

loader = UnifiedDatasetLoader(split='training')
stats = loader.get_stats()

print(f'ARC-AGI 1 tasks: {stats[\"arc_agi_1\"][\"task_count\"]}')
print(f'ARC-AGI 2 tasks: {stats[\"arc_agi_2\"][\"task_count\"]}')
print(f'Total tasks: {stats[\"total_tasks\"]}')

# Test interleaving
count = 0
for task in loader:
    count += 1
    if count <= 4:
        print(f'  Task {count}: {task[\"task_id\"]} ({task[\"source\"]})')
    if count > 4:
        break

print(f'Interleaving works: {count >= 4}')
print('=== UNIFIED DATASET LOADER OK ===')
"

# Test 2: Session reporter
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.session_reporter import SessionReporter

reporter = SessionReporter('test_session')
reporter.set_config({'epochs': 10, 'batch_size': 2})
reporter.set_datasets('/path/arc1', 400, '/path/arc2', 400)

# Simulate epochs
for epoch in range(3):
    reporter.log_epoch(epoch, {
        'accuracy': 0.5 + epoch * 0.1,
        'total_tasks': 800,
        'correct': 400 + epoch * 80,
    })

reporter.log_discovery_stats(100, 60, 25, 10)
reporter.log_grammar_stats(747, 800, 53)

# Don't save (test only)
reporter.print_summary()
print('=== SESSION REPORTER OK ===')
"

# Test 3: Check dataset paths exist
ls -la /K3D/K3D_llama_cpp/datasets/ARC-AGI-master/data/training/*.json | head -5
ls -la /K3D/Knowledge3D.local/datasets/exams/arc-src/data/training/*.json 2>/dev/null | head -5 || echo "ARC-2 path may differ"
```

---

## Success Criteria

1. **Grammar evolves** — discoveries → validation → promotion
2. **Size patterns learned** — no hardcoded ratios
3. **Dynamic routing** — workers spawned based on task similarity
4. **Round-robin load** — balanced partitioning
5. **No external libs** — pure PTX + RPN
6. **Snapshot transfer** — no file I/O in workers
7. **Near-misses preserved** — 85%+ discoveries kept for refinement
8. **Progressive improvement** — preserved discoveries can improve over epochs
9. **Adaptive thresholds** — preserve/promote thresholds learn from distribution
10. **Kernel organization** — all active kernels in `kernels/`, old in `Old_Attempts/`
11. **ARC-AGI cleanup** — folder contains only canonical files
12. **Session reports** — JSON reports in `arc_agi/logs/` for every run
13. **Dual-dataset training** — ARC-AGI 1 + 2 interleaved (800 tasks total)

---

## Design Notes for Codex

**Your contribution is valued.** If during implementation you discover:
- A cleaner architectural pattern
- An optimization that maintains sovereignty
- A bug in the spec
- A better way to achieve the same goal

...implement it and document the change. The goal is a working, elegant system — not blind adherence to spec.

**Key questions to consider:**
- How should preserved discoveries persist across training runs?
- Should threshold adaptation be per-task-type or global?
- What's the optimal balance between exact vs fuzzy scoring?
- How do we prevent discovery space explosion?

Feed back your insights. Architecture is iterative.

---

## Part 7: ARC-AGI 3 Integration & Standard Output Format

**Phase 2 — After Parts 1-6 are complete**

### 7.1 Overview

ARC-AGI 3 introduces a new paradigm: **abundant examples** for efficient skill acquisition (unlike the few-shot setup of ARC-AGI 1/2). We need:

1. **Ingest ARC-AGI 3 dataset** into K3D's sovereign format
2. **Adapt evaluation tools** to work with our hot path
3. **Standard output format** — output results in the same format other models use (for benchmarking)

**Key Principle:** Sovereignty stays internal. The "print layer" at the end converts RPN results to standard JSON format for external evaluation.

```
┌─────────────────────────────────────────────────────────────┐
│                    K3D SOVEREIGN HOT PATH                   │
│  (PTX + RPN + TernaryGalaxy — NO external libs)            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STANDARD OUTPUT LAYER                    │
│  (Converts RPN grid → JSON format for benchmarking)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    [ARC Prize Evaluation API]
```

### 7.2 Repository Setup

**Clone ARC-AGI 3 tools:**
```bash
cd /K3D/K3D_llama_cpp/datasets/
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
git clone https://github.com/arcprize/arc-agi-benchmarking.git
```

**Dataset paths:**
```python
ARC_AGI_3_PATH = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/data")
ARC_BENCHMARKING_PATH = Path("/K3D/K3D_llama_cpp/datasets/arc-agi-benchmarking")
```

### 7.3 Standard Output Adapter

**File:** `knowledge3d/training/arc_agi/standard_output_adapter.py` (CREATE)

```python
"""
Standard Output Adapter — converts K3D sovereign results to ARC Prize format.

Internal: RPN programs, TernaryVector embeddings, procedural grids
External: JSON with 2D grid arrays (standard benchmark format)

This is the "print layer" — sovereignty stays internal.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class StandardOutputAdapter:
    """
    Converts sovereign K3D results to standard ARC evaluation format.

    ARC Prize format:
    {
        "task_id": {
            "attempt_1": [[int, ...], ...],
            "attempt_2": [[int, ...], ...],
        },
        ...
    }
    """

    def __init__(self, max_attempts: int = 2):
        self.max_attempts = max_attempts
        self.results: Dict[str, Dict[str, List[List[int]]]] = {}

    def record_attempt(
        self,
        task_id: str,
        attempt_number: int,
        grid: Sequence[Sequence[int]],
    ) -> None:
        """
        Record an attempt for a task.

        Args:
            task_id: ARC task identifier (e.g., "007bbfb7")
            attempt_number: 1 or 2 (ARC allows 2 attempts per task)
            grid: 2D grid as nested lists of integers (0-9)
        """
        if task_id not in self.results:
            self.results[task_id] = {}

        attempt_key = f"attempt_{attempt_number}"

        # Convert to standard format (list of lists of ints)
        standard_grid = [
            [int(cell) for cell in row]
            for row in grid
        ]

        self.results[task_id][attempt_key] = standard_grid

    def record_from_rpn_result(
        self,
        task_id: str,
        attempt_number: int,
        rpn_grid: Any,  # Could be numpy array, list, or TernaryVector
    ) -> None:
        """
        Record attempt from RPN execution result.

        Handles various internal representations.
        """
        # Convert from various internal formats
        if hasattr(rpn_grid, 'tolist'):
            # numpy array or similar
            grid = rpn_grid.tolist()
        elif hasattr(rpn_grid, 'to_python'):
            # TernaryVector
            grid = rpn_grid.to_python()
        else:
            # Already a list
            grid = rpn_grid

        self.record_attempt(task_id, attempt_number, grid)

    def to_submission_json(self) -> str:
        """
        Convert all results to ARC Prize submission JSON format.
        """
        return json.dumps(self.results, indent=2)

    def save_submission(
        self,
        output_path: Optional[Path] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Save submission file for ARC Prize evaluation.

        Returns path to saved file.
        """
        if output_path is None:
            timestamp = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"submissions/arc_submission_{timestamp}.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(self.to_submission_json())

        print(f"[SUBMISSION] Saved to {output_path}")
        return str(output_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get submission statistics."""
        total_tasks = len(self.results)
        tasks_with_2_attempts = sum(
            1 for r in self.results.values()
            if len(r) >= 2
        )

        return {
            "total_tasks": total_tasks,
            "tasks_with_2_attempts": tasks_with_2_attempts,
            "coverage": tasks_with_2_attempts / total_tasks if total_tasks > 0 else 0.0,
        }


class ARCEvaluationBridge:
    """
    Bridge to ARC Prize evaluation tools.

    Wraps arc-agi-benchmarking utilities for sovereign integration.
    """

    def __init__(
        self,
        benchmarking_path: Optional[Path] = None,
    ):
        self.benchmarking_path = benchmarking_path or Path(
            "/K3D/K3D_llama_cpp/datasets/arc-agi-benchmarking"
        )

    def evaluate_submission(
        self,
        submission_path: Path,
        ground_truth_path: Path,
    ) -> Dict[str, Any]:
        """
        Evaluate submission against ground truth.

        Returns metrics compatible with ARC Prize leaderboard.
        """
        # Load submission
        with open(submission_path) as f:
            submission = json.load(f)

        # Load ground truth
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Evaluate
        correct = 0
        total = 0
        results_by_task = {}

        for task_id, expected in ground_truth.items():
            if task_id not in submission:
                results_by_task[task_id] = {
                    "status": "missing",
                    "correct": False,
                }
                total += 1
                continue

            attempts = submission[task_id]
            expected_output = expected.get("test", [{}])[0].get("output", [])

            task_correct = False

            # Check each attempt
            for attempt_key in ["attempt_1", "attempt_2"]:
                if attempt_key not in attempts:
                    continue

                attempt_output = attempts[attempt_key]

                if self._grids_match(attempt_output, expected_output):
                    task_correct = True
                    break

            results_by_task[task_id] = {
                "status": "correct" if task_correct else "incorrect",
                "correct": task_correct,
            }

            if task_correct:
                correct += 1
            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "results_by_task": results_by_task,
        }

    def _grids_match(
        self,
        grid1: List[List[int]],
        grid2: List[List[int]],
    ) -> bool:
        """Check if two grids are identical."""
        if len(grid1) != len(grid2):
            return False

        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
            if row1 != row2:
                return False

        return True


__all__ = ["StandardOutputAdapter", "ARCEvaluationBridge"]
```

### 7.4 Unified Dataset Loader v2 (ARC 1 + 2 + 3)

**File:** `knowledge3d/training/arc_agi/unified_dataset_loader.py` (MODIFY)

Add ARC-AGI 3 support:

```python
# Add to existing file

ARC_AGI_3_PATH = Path("/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/data")


class UnifiedDatasetLoaderV2(UnifiedDatasetLoader):
    """
    Extended loader supporting ARC-AGI 1, 2, AND 3.

    ARC-AGI 3 has abundant examples — different training strategy.
    """

    def __init__(
        self,
        arc1_path: Optional[Path] = None,
        arc2_path: Optional[Path] = None,
        arc3_path: Optional[Path] = None,
        split: str = "training",
        include_arc3: bool = True,
    ):
        # Initialize base (ARC 1 + 2)
        super().__init__(arc1_path, arc2_path, split)

        self.include_arc3 = include_arc3
        self.arc3_path = arc3_path or ARC_AGI_3_PATH
        self.arc3_tasks = []

        if include_arc3 and self.arc3_path.exists():
            self.arc3_tasks = self._load_task_list(self.arc3_path / split)
            self.arc3_tasks = self._sort_by_difficulty(self.arc3_tasks)
            print(f"[DATASET] ARC-AGI 3: {len(self.arc3_tasks)} tasks from {self.arc3_path}")

        print(f"[DATASET] Grand total: {len(self)} tasks")

    def __len__(self) -> int:
        """Total tasks across all three datasets."""
        return (
            len(self.arc1_tasks) +
            len(self.arc2_tasks) +
            len(self.arc3_tasks)
        )

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate with 3-way alternation: ARC-1, ARC-2, ARC-3, ...

        Groups tasks of similar difficulty from each dataset.
        """
        max_len = max(
            len(self.arc1_tasks),
            len(self.arc2_tasks),
            len(self.arc3_tasks),
        )

        for i in range(max_len):
            # ARC-AGI 1 task
            if i < len(self.arc1_tasks):
                task_id, path = self.arc1_tasks[i]
                yield self._load_task(task_id, path, source="arc_agi_1")

            # ARC-AGI 2 task
            if i < len(self.arc2_tasks):
                task_id, path = self.arc2_tasks[i]
                yield self._load_task(task_id, path, source="arc_agi_2")

            # ARC-AGI 3 task
            if i < len(self.arc3_tasks):
                task_id, path = self.arc3_tasks[i]
                yield self._load_task(task_id, path, source="arc_agi_3")

    def get_arc3_tasks(self) -> List[Dict[str, Any]]:
        """Get all ARC-AGI 3 tasks."""
        return [
            self._load_task(task_id, path, "arc_agi_3")
            for task_id, path in self.arc3_tasks
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics for all three datasets."""
        stats = super().get_stats()
        stats["arc_agi_3"] = {
            "path": str(self.arc3_path),
            "task_count": len(self.arc3_tasks),
        }
        stats["total_tasks"] = len(self)
        return stats


__all__ += ["UnifiedDatasetLoaderV2", "ARC_AGI_3_PATH"]
```

### 7.5 Sovereign Pipeline with Standard Output

**File:** `knowledge3d/training/arc_agi/sovereign_pipeline.py` (MODIFY)

Add standard output integration:

```python
from knowledge3d.training.arc_agi.standard_output_adapter import (
    StandardOutputAdapter,
    ARCEvaluationBridge,
)


class SovereignPipeline:
    def __init__(self, ...):
        # ... existing init ...

        # Standard output adapter (for benchmarking)
        self.output_adapter = StandardOutputAdapter(max_attempts=2)
        self.evaluation_bridge = ARCEvaluationBridge()

    def process_task_with_output(
        self,
        task: Dict,
        record_submission: bool = True,
    ) -> Dict:
        """
        Process task and optionally record for submission.

        The hot path remains sovereign — only the final output
        is converted to standard format.
        """
        # === SOVEREIGN HOT PATH ===
        # (PTX + RPN + TernaryGalaxy)
        result = self.process_task(task)

        # === STANDARD OUTPUT LAYER ===
        # (Convert to benchmark format)
        if record_submission and result.get("output_grid"):
            task_id = task["task_id"]

            # Record best attempt
            self.output_adapter.record_from_rpn_result(
                task_id=task_id,
                attempt_number=1,
                rpn_grid=result["output_grid"],
            )

            # If we have a second-best candidate, record it too
            if result.get("second_best_grid"):
                self.output_adapter.record_from_rpn_result(
                    task_id=task_id,
                    attempt_number=2,
                    rpn_grid=result["second_best_grid"],
                )

        return result

    def generate_submission(
        self,
        session_id: Optional[str] = None,
    ) -> str:
        """
        Generate submission file for ARC Prize evaluation.

        Returns path to submission JSON.
        """
        return self.output_adapter.save_submission(session_id=session_id)

    def evaluate_against_ground_truth(
        self,
        submission_path: Path,
        ground_truth_path: Path,
    ) -> Dict:
        """
        Evaluate submission using ARC Prize format.
        """
        return self.evaluation_bridge.evaluate_submission(
            submission_path,
            ground_truth_path,
        )

    def run_official_evaluation(
        self,
        dataset_version: str = "arc_agi_3",
    ) -> Dict:
        """
        Run evaluation in official ARC Prize format.

        1. Process all tasks (sovereign hot path)
        2. Generate submission file (standard output)
        3. Evaluate against ground truth
        4. Return leaderboard-compatible metrics
        """
        print(f"[EVAL] Running official evaluation for {dataset_version}")

        # Select dataset
        if dataset_version == "arc_agi_3":
            dataset = self.dataset.get_arc3_tasks()
            gt_path = ARC_AGI_3_PATH / "solutions.json"
        elif dataset_version == "arc_agi_2":
            dataset = self.dataset.get_arc2_tasks()
            gt_path = ARC_AGI_2_PATH / "solutions.json"
        else:
            dataset = self.dataset.get_arc1_tasks()
            gt_path = ARC_AGI_1_PATH / "solutions.json"

        # Process tasks
        for task in dataset:
            self.process_task_with_output(task, record_submission=True)

        # Generate submission
        submission_path = Path(self.generate_submission())

        # Evaluate
        results = self.evaluate_against_ground_truth(
            submission_path,
            gt_path,
        )

        print(f"[EVAL] Accuracy: {results['accuracy']:.2%}")
        print(f"[EVAL] Correct: {results['correct']}/{results['total']}")

        return results
```

### 7.6 Session Reporter Update

**File:** `knowledge3d/training/arc_agi/session_reporter.py` (MODIFY)

Add ARC-AGI 3 tracking:

```python
# Update __init__ datasets structure
self.report["datasets"] = {
    "arc_agi_1": {"path": None, "task_count": 0},
    "arc_agi_2": {"path": None, "task_count": 0},
    "arc_agi_3": {"path": None, "task_count": 0},  # NEW
    "total_tasks": 0,
}

# Update set_datasets method
def set_datasets(
    self,
    arc1_path: str,
    arc1_count: int,
    arc2_path: str,
    arc2_count: int,
    arc3_path: Optional[str] = None,  # NEW
    arc3_count: int = 0,              # NEW
) -> None:
    """Record dataset information."""
    self.report["datasets"] = {
        "arc_agi_1": {"path": arc1_path, "task_count": arc1_count},
        "arc_agi_2": {"path": arc2_path, "task_count": arc2_count},
        "arc_agi_3": {"path": arc3_path, "task_count": arc3_count},
        "total_tasks": arc1_count + arc2_count + arc3_count,
    }

# Update _run_epoch to track ARC-3 metrics
def _run_epoch(self, epoch: int) -> Dict[str, Any]:
    # ... existing code ...
    arc3_correct = 0
    arc3_total = 0

    for task in self.dataset:
        # ... existing processing ...

        if task["source"] == "arc_agi_3":
            arc3_total += 1
            if result.get("exact_match"):
                arc3_correct += 1

    return {
        # ... existing metrics ...
        "arc3_accuracy": arc3_correct / arc3_total if arc3_total > 0 else 0.0,
    }
```

### 7.7 Verification

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Test 1: Clone ARC-AGI 3 repos (if not already present)
if [ ! -d "/K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents" ]; then
    cd /K3D/K3D_llama_cpp/datasets/
    git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
    git clone https://github.com/arcprize/arc-agi-benchmarking.git
fi

# Test 2: Standard output adapter
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.standard_output_adapter import StandardOutputAdapter

adapter = StandardOutputAdapter()

# Simulate recording results
adapter.record_attempt('007bbfb7', 1, [[1, 2], [3, 4]])
adapter.record_attempt('007bbfb7', 2, [[1, 2], [3, 5]])
adapter.record_attempt('00d62c1b', 1, [[0, 0, 1], [1, 0, 0]])

# Check format
submission = adapter.to_submission_json()
print('Submission format:')
print(submission[:200])

stats = adapter.get_stats()
print(f'Stats: {stats}')
print('=== STANDARD OUTPUT ADAPTER OK ===')
"

# Test 3: Check ARC-AGI 3 dataset structure
ls -la /K3D/K3D_llama_cpp/datasets/ARC-AGI-3-Agents/ 2>/dev/null || echo "ARC-AGI 3 not yet cloned"

# Test 4: Unified loader v2
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.arc_agi.unified_dataset_loader import UnifiedDatasetLoaderV2

loader = UnifiedDatasetLoaderV2(split='training', include_arc3=True)
stats = loader.get_stats()

print(f'ARC-AGI 1: {stats.get(\"arc_agi_1\", {}).get(\"task_count\", 0)} tasks')
print(f'ARC-AGI 2: {stats.get(\"arc_agi_2\", {}).get(\"task_count\", 0)} tasks')
print(f'ARC-AGI 3: {stats.get(\"arc_agi_3\", {}).get(\"task_count\", 0)} tasks')
print(f'Total: {stats[\"total_tasks\"]} tasks')
print('=== UNIFIED LOADER V2 OK ===')
"
```

### 7.8 Files Summary (Part 7)

| File | Action | Purpose |
|------|--------|---------|
| `standard_output_adapter.py` | CREATE | Convert RPN → standard JSON for benchmarking |
| `unified_dataset_loader.py` | MODIFY | Add ARC-AGI 3 support (UnifiedDatasetLoaderV2) |
| `sovereign_pipeline.py` | MODIFY | Add standard output integration |
| `session_reporter.py` | MODIFY | Track ARC-AGI 3 metrics |

### 7.9 Success Criteria (Part 7)

14. **ARC-AGI 3 integrated** — dataset loaded and alternated with ARC 1/2
15. **Standard output format** — submission JSON compatible with ARC Prize
16. **Evaluation bridge** — can evaluate against ground truth
17. **Hot path sovereignty preserved** — standard output is a "print layer" only

---

## Part 8: Math Benchmark Extensions (Multi-Modal Reasoning)

### 8.1 Overview

K3D sovereign architecture must demonstrate **multi-modal mathematical reasoning** across diverse benchmark types. This extends beyond visual reasoning (ARC-AGI) to:

- **Grade School Math** — step-by-step arithmetic reasoning
- **Competition Math** — complex problem-solving (AMC, AIME, Olympiad)
- **Abstract Math** — proof-based and symbolic reasoning

**Sovereignty Constraint**: All benchmarks use the same hot path (PTX + RPN + TernaryGalaxy). Only the **input proceduralization** and **output print layer** adapt per benchmark.

```
┌─────────────────────────────────────────────────────────────────┐
│                    BENCHMARK INGESTION LAYER                    │
│  (Parse problem text → RPN programs + metadata)                │
│  [GSM8K | MATH | MMLU | Omni-MATH | AMC-AIME]                  │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    K3D SOVEREIGN HOT PATH                       │
│  (PTX + RPN + TernaryGalaxy — NO external libs)                │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STANDARD OUTPUT LAYER                        │
│  (Convert RPN result → benchmark-specific answer format)       │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 Available Datasets (Cloned)

| Dataset | Path | Description | Task Count |
|---------|------|-------------|------------|
| **GSM8K** | `/K3D/K3D_llama_cpp/datasets/GSM8K/` | Grade school math word problems | ~8,500 |
| **MATH** | `/K3D/K3D_llama_cpp/datasets/math/` | Competition-level (AMC, AIME, Olympiad) | ~12,500 |
| **MMLU** | `/K3D/K3D_llama_cpp/datasets/MMLU/` | Multi-task (math subset) | ~14,000 (total) |
| **Omni-MATH** | `/K3D/K3D_llama_cpp/datasets/Omni-MATH/` | Olympiad-level problems | 4,428 |
| **AMC-AIME** | `/K3D/K3D_llama_cpp/datasets/AMC-AIME/` | American Math Competition archive | ~2,000+ |

**Note**: FrontierMath is a private benchmark (no public dataset). We can prepare the adapter but cannot train on it directly.

### 8.3 Math Problem Proceduralizer

```python
# File: knowledge3d/training/math_benchmarks/math_proceduralizer.py
"""
Convert text-based math problems into RPN programs for sovereign execution.
Uses Character Galaxy for text → procedural conversion.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Iterator
import json
import re

# Dataset paths
MATH_DATASETS = {
    "gsm8k": Path("/K3D/K3D_llama_cpp/datasets/GSM8K/grade_school_math/data"),
    "math": Path("/K3D/K3D_llama_cpp/datasets/math"),
    "mmlu": Path("/K3D/K3D_llama_cpp/datasets/MMLU"),
    "omni_math": Path("/K3D/K3D_llama_cpp/datasets/Omni-MATH"),
    "amc_aime": Path("/K3D/K3D_llama_cpp/datasets/AMC-AIME"),
}


class MathProceduralizer:
    """
    Transform text math problems into procedural RPN representation.

    The key insight: math expressions are already procedural!
    "2 + 3 * 4" → RPN: [2, 3, 4, MUL, ADD]

    Word problems need:
    1. Extract quantities (numbers)
    2. Extract operations (implied by words)
    3. Build solution RPN
    """

    def __init__(self):
        # Math operation keywords → RPN opcodes
        self._operation_patterns = {
            r"\b(add|plus|sum|more|together)\b": "ADD",
            r"\b(subtract|minus|less|fewer|difference)\b": "SUB",
            r"\b(multiply|times|product|of)\b": "MUL",
            r"\b(divide|split|share|per|ratio)\b": "DIV",
            r"\b(square|squared)\b": "SQR",
            r"\b(root|sqrt)\b": "SQRT",
            r"\b(power|exponent)\b": "POW",
            r"\b(percent|%)\b": "PCT",
            r"\b(total|altogether|result)\b": "EQ",  # signals answer
        }

        # Number extraction pattern
        self._number_pattern = re.compile(
            r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|\b\d+/\d+\b"
        )

    def proceduralize_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a math problem to procedural form.

        Returns:
            {
                "problem_id": str,
                "source": str,  # dataset name
                "difficulty": int,  # 1-10
                "problem_rpn": List[str],  # problem as RPN tokens
                "solution_rpn": List[str],  # solution steps as RPN
                "answer": Any,  # final answer (for verification)
                "metadata": Dict,  # original problem data
            }
        """
        text = problem.get("question", problem.get("problem", ""))
        solution = problem.get("answer", problem.get("solution", ""))

        # Extract quantities
        quantities = self._extract_quantities(text)

        # Extract operation hints
        operations = self._extract_operations(text)

        # Build problem RPN (procedural representation)
        problem_rpn = self._build_problem_rpn(quantities, operations)

        # Parse solution if available
        solution_rpn = self._parse_solution(solution) if solution else []

        # Extract final answer
        answer = self._extract_answer(solution)

        return {
            "problem_id": problem.get("id", problem.get("problem_id", hash(text))),
            "source": problem.get("source", "unknown"),
            "difficulty": problem.get("level", problem.get("difficulty", 5)),
            "problem_rpn": problem_rpn,
            "solution_rpn": solution_rpn,
            "answer": answer,
            "metadata": {
                "original_text": text,
                "original_solution": solution,
                "quantities_found": quantities,
                "operations_implied": operations,
            },
        }

    def _extract_quantities(self, text: str) -> List[float]:
        """Extract all numbers from problem text."""
        matches = self._number_pattern.findall(text)
        quantities = []
        for m in matches:
            try:
                if "/" in m:  # fraction
                    num, den = m.split("/")
                    quantities.append(float(num) / float(den))
                else:
                    quantities.append(float(m))
            except ValueError:
                continue
        return quantities

    def _extract_operations(self, text: str) -> List[str]:
        """Extract implied operations from word problem."""
        text_lower = text.lower()
        operations = []
        for pattern, opcode in self._operation_patterns.items():
            if re.search(pattern, text_lower):
                operations.append(opcode)
        return operations

    def _build_problem_rpn(
        self, quantities: List[float], operations: List[str]
    ) -> List[str]:
        """Build RPN representation of the problem."""
        rpn = []

        # Push quantities
        for q in quantities:
            rpn.append(f"PUSH {q}")

        # Add operation hints
        for op in operations:
            rpn.append(f"HINT {op}")

        # Problem marker
        rpn.append("SOLVE")

        return rpn

    def _parse_solution(self, solution: str) -> List[str]:
        """Parse solution text into RPN steps."""
        rpn = []

        # Look for step-by-step calculations
        # Pattern: "... = number" or "Answer: number"
        steps = re.findall(r"=\s*([-+]?\d*\.?\d+)", solution)

        for i, step in enumerate(steps):
            rpn.append(f"STEP_{i}: {step}")

        # Final answer marker
        answer_match = re.search(
            r"(?:answer|final|result)[:\s]*\$?([-+]?\d*\.?\d+)",
            solution,
            re.IGNORECASE,
        )
        if answer_match:
            rpn.append(f"ANSWER: {answer_match.group(1)}")

        return rpn

    def _extract_answer(self, solution: str) -> Optional[float]:
        """Extract the final numerical answer."""
        if not solution:
            return None

        # GSM8K format: "#### answer"
        gsm8k_match = re.search(r"####\s*([-+]?\d*\.?\d+)", solution)
        if gsm8k_match:
            return float(gsm8k_match.group(1))

        # MATH format: "\\boxed{answer}"
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", solution)
        if boxed_match:
            try:
                return float(boxed_match.group(1))
            except ValueError:
                return boxed_match.group(1)  # symbolic answer

        # Fallback: last number in solution
        numbers = self._number_pattern.findall(solution)
        if numbers:
            try:
                return float(numbers[-1])
            except ValueError:
                return numbers[-1]

        return None


class MathDatasetLoader:
    """
    Unified loader for all math benchmark datasets.
    """

    def __init__(
        self,
        datasets: List[str] = None,
        difficulty_filter: Optional[range] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            datasets: List of datasets to load (default: all available)
            difficulty_filter: Only include problems in this difficulty range
            shuffle: Whether to shuffle problems
        """
        self._datasets = datasets or list(MATH_DATASETS.keys())
        self._difficulty_filter = difficulty_filter
        self._shuffle = shuffle
        self._proceduralizer = MathProceduralizer()
        self._problems: List[Dict[str, Any]] = []

        self._load_datasets()

    def _load_datasets(self) -> None:
        """Load all configured datasets."""
        for ds_name in self._datasets:
            path = MATH_DATASETS.get(ds_name)
            if path and path.exists():
                self._load_dataset(ds_name, path)

    def _load_dataset(self, name: str, path: Path) -> None:
        """Load a specific dataset."""
        if name == "gsm8k":
            self._load_gsm8k(path)
        elif name == "math":
            self._load_math(path)
        elif name == "mmlu":
            self._load_mmlu(path)
        elif name == "omni_math":
            self._load_omni_math(path)
        elif name == "amc_aime":
            self._load_amc_aime(path)

    def _load_gsm8k(self, path: Path) -> None:
        """Load GSM8K dataset."""
        for split in ["train", "test"]:
            split_file = path / f"{split}.jsonl"
            if split_file.exists():
                with open(split_file) as f:
                    for line in f:
                        problem = json.loads(line)
                        problem["source"] = "gsm8k"
                        problem["split"] = split
                        proc = self._proceduralizer.proceduralize_problem(problem)
                        self._add_if_passes_filter(proc)

    def _load_math(self, path: Path) -> None:
        """Load MATH dataset (competition problems)."""
        for difficulty_dir in path.iterdir():
            if difficulty_dir.is_dir() and difficulty_dir.name.startswith("Level"):
                level = int(difficulty_dir.name.replace("Level", "").strip())
                for problem_file in difficulty_dir.glob("*.json"):
                    with open(problem_file) as f:
                        problem = json.load(f)
                        problem["source"] = "math"
                        problem["level"] = level * 2  # Scale 1-5 → 2-10
                        proc = self._proceduralizer.proceduralize_problem(problem)
                        self._add_if_passes_filter(proc)

    def _load_mmlu(self, path: Path) -> None:
        """Load MMLU math subset."""
        math_subjects = [
            "abstract_algebra",
            "college_mathematics",
            "elementary_mathematics",
            "high_school_mathematics",
            "high_school_statistics",
        ]

        for subject in math_subjects:
            subject_file = path / "data" / "test" / f"{subject}_test.csv"
            if subject_file.exists():
                import csv
                with open(subject_file) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 5:
                            problem = {
                                "question": row[0],
                                "choices": row[1:5],
                                "answer": row[5] if len(row) > 5 else None,
                                "source": "mmlu",
                                "subject": subject,
                            }
                            proc = self._proceduralizer.proceduralize_problem(problem)
                            self._add_if_passes_filter(proc)

    def _load_omni_math(self, path: Path) -> None:
        """Load Omni-MATH Olympiad problems."""
        jsonl_file = path / "Omni-Math.jsonl"
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                for line in f:
                    problem = json.loads(line)
                    problem["source"] = "omni_math"
                    # Omni-MATH has difficulty 1-10
                    problem["difficulty"] = problem.get("difficulty", 5)
                    proc = self._proceduralizer.proceduralize_problem(problem)
                    self._add_if_passes_filter(proc)

    def _load_amc_aime(self, path: Path) -> None:
        """Load AMC/AIME competition problems."""
        # AMC dataset comes as ZIP, need to extract and parse
        zip_file = path / "AMC.zip"
        if zip_file.exists():
            import zipfile
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(tmpdir)

                # Parse extracted problems
                for json_file in Path(tmpdir).rglob("*.json"):
                    try:
                        with open(json_file) as f:
                            problem = json.load(f)
                            problem["source"] = "amc_aime"
                            proc = self._proceduralizer.proceduralize_problem(problem)
                            self._add_if_passes_filter(proc)
                    except json.JSONDecodeError:
                        continue

    def _add_if_passes_filter(self, problem: Dict[str, Any]) -> None:
        """Add problem if it passes difficulty filter."""
        if self._difficulty_filter is None:
            self._problems.append(problem)
        elif problem.get("difficulty", 5) in self._difficulty_filter:
            self._problems.append(problem)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over problems."""
        problems = self._problems.copy()
        if self._shuffle:
            import random
            random.shuffle(problems)
        yield from problems

    def __len__(self) -> int:
        return len(self._problems)

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        from collections import Counter

        sources = Counter(p["source"] for p in self._problems)
        difficulties = Counter(p.get("difficulty", 5) for p in self._problems)

        return {
            "total_problems": len(self._problems),
            "by_source": dict(sources),
            "by_difficulty": dict(difficulties),
            "datasets_loaded": self._datasets,
        }
```

### 8.4 Math Output Adapter

```python
# File: knowledge3d/training/math_benchmarks/math_output_adapter.py
"""
Convert sovereign RPN results to standard benchmark output formats.
"""

from typing import Dict, Any, List, Optional
import json
import re


class MathOutputAdapter:
    """
    Converts K3D RPN execution results to benchmark-specific formats.

    Each benchmark has different answer format expectations:
    - GSM8K: single number
    - MATH: LaTeX expression (\\boxed{...})
    - MMLU: letter choice (A/B/C/D)
    - Omni-MATH: number or symbolic expression
    """

    def __init__(self):
        self._results: Dict[str, Dict[str, Any]] = {}

    def record_result(
        self,
        problem_id: str,
        rpn_stack: List[Any],
        source: str,
    ) -> None:
        """
        Record a result from sovereign execution.

        Args:
            problem_id: Unique problem identifier
            rpn_stack: Final RPN stack after execution
            source: Dataset source (for format selection)
        """
        # Extract answer from stack
        raw_answer = rpn_stack[-1] if rpn_stack else None

        # Format for benchmark
        formatted = self._format_for_benchmark(raw_answer, source)

        self._results[problem_id] = {
            "raw_answer": raw_answer,
            "formatted_answer": formatted,
            "source": source,
        }

    def _format_for_benchmark(self, answer: Any, source: str) -> str:
        """Format answer for specific benchmark."""
        if answer is None:
            return ""

        if source == "gsm8k":
            # GSM8K expects plain number
            return str(self._to_number(answer))

        elif source == "math":
            # MATH expects LaTeX boxed
            return f"\\boxed{{{answer}}}"

        elif source == "mmlu":
            # MMLU expects letter choice
            return self._to_letter_choice(answer)

        elif source == "omni_math":
            # Omni-MATH can be number or expression
            return str(answer)

        elif source == "amc_aime":
            # AMC/AIME expects integer 0-999
            num = self._to_number(answer)
            if num is not None:
                return str(int(num) % 1000)
            return str(answer)

        return str(answer)

    def _to_number(self, value: Any) -> Optional[float]:
        """Convert value to number."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.replace(",", ""))
            except ValueError:
                return None
        return None

    def _to_letter_choice(self, value: Any) -> str:
        """Convert value to MMLU letter choice."""
        if isinstance(value, str) and value.upper() in "ABCD":
            return value.upper()
        if isinstance(value, int) and 0 <= value <= 3:
            return "ABCD"[value]
        return "A"  # default

    def to_submission_format(self, benchmark: str) -> str:
        """Export results in benchmark's submission format."""
        results = {
            pid: data["formatted_answer"]
            for pid, data in self._results.items()
            if data["source"] == benchmark or benchmark == "all"
        }
        return json.dumps(results, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        from collections import Counter

        sources = Counter(d["source"] for d in self._results.values())

        return {
            "total_recorded": len(self._results),
            "by_source": dict(sources),
        }
```

### 8.5 Benchmark Evaluator

```python
# File: knowledge3d/training/math_benchmarks/benchmark_evaluator.py
"""
Evaluate K3D sovereign results against benchmark ground truth.
"""

from typing import Dict, Any, List, Optional
import re


class MathBenchmarkEvaluator:
    """
    Evaluate sovereign math reasoning against benchmarks.

    Scoring is benchmark-specific:
    - GSM8K: exact numerical match
    - MATH: symbolic equivalence (tricky!)
    - MMLU: exact letter match
    - Omni-MATH: numerical or symbolic match
    - AMC-AIME: integer match (mod 1000 for AIME)
    """

    def __init__(self, tolerance: float = 1e-6):
        self._tolerance = tolerance
        self._results: List[Dict[str, Any]] = []

    def evaluate(
        self,
        problem_id: str,
        predicted: Any,
        ground_truth: Any,
        source: str,
    ) -> Dict[str, Any]:
        """
        Evaluate a single prediction.

        Returns:
            {
                "problem_id": str,
                "correct": bool,
                "predicted": Any,
                "ground_truth": Any,
                "source": str,
                "match_type": str,  # "exact", "numerical", "symbolic", "partial"
            }
        """
        correct = False
        match_type = "none"

        if source == "gsm8k":
            correct, match_type = self._evaluate_gsm8k(predicted, ground_truth)
        elif source == "math":
            correct, match_type = self._evaluate_math(predicted, ground_truth)
        elif source == "mmlu":
            correct, match_type = self._evaluate_mmlu(predicted, ground_truth)
        elif source == "omni_math":
            correct, match_type = self._evaluate_omni(predicted, ground_truth)
        elif source == "amc_aime":
            correct, match_type = self._evaluate_amc(predicted, ground_truth)

        result = {
            "problem_id": problem_id,
            "correct": correct,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "source": source,
            "match_type": match_type,
        }

        self._results.append(result)
        return result

    def _evaluate_gsm8k(self, predicted: Any, truth: Any) -> tuple:
        """GSM8K: exact numerical match."""
        try:
            pred_num = float(str(predicted).replace(",", ""))
            truth_num = float(str(truth).replace(",", ""))

            if abs(pred_num - truth_num) < self._tolerance:
                return True, "exact"

            # Check relative tolerance for large numbers
            if truth_num != 0:
                rel_diff = abs(pred_num - truth_num) / abs(truth_num)
                if rel_diff < 1e-4:
                    return True, "numerical"

            return False, "none"
        except (ValueError, TypeError):
            return str(predicted) == str(truth), "string"

    def _evaluate_math(self, predicted: Any, truth: Any) -> tuple:
        """MATH: symbolic equivalence (simplified)."""
        # Extract from \\boxed{} if present
        pred_str = str(predicted)
        truth_str = str(truth)

        boxed_pred = re.search(r"\\boxed\{([^}]+)\}", pred_str)
        boxed_truth = re.search(r"\\boxed\{([^}]+)\}", truth_str)

        if boxed_pred:
            pred_str = boxed_pred.group(1)
        if boxed_truth:
            truth_str = boxed_truth.group(1)

        # Normalize LaTeX
        pred_norm = self._normalize_latex(pred_str)
        truth_norm = self._normalize_latex(truth_str)

        if pred_norm == truth_norm:
            return True, "exact"

        # Try numerical comparison
        try:
            pred_num = float(pred_norm)
            truth_num = float(truth_norm)
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "numerical"
        except ValueError:
            pass

        return False, "none"

    def _normalize_latex(self, latex: str) -> str:
        """Normalize LaTeX expression for comparison."""
        # Remove whitespace
        s = re.sub(r"\s+", "", latex)
        # Remove common LaTeX commands
        s = re.sub(r"\\(left|right|big|Big)", "", s)
        # Normalize fractions
        s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        return s.lower()

    def _evaluate_mmlu(self, predicted: Any, truth: Any) -> tuple:
        """MMLU: exact letter match."""
        pred_letter = str(predicted).strip().upper()[:1]
        truth_letter = str(truth).strip().upper()[:1]

        if pred_letter == truth_letter:
            return True, "exact"
        return False, "none"

    def _evaluate_omni(self, predicted: Any, truth: Any) -> tuple:
        """Omni-MATH: numerical or symbolic match."""
        # Try numerical first
        try:
            pred_num = float(str(predicted))
            truth_num = float(str(truth))
            if abs(pred_num - truth_num) < self._tolerance:
                return True, "numerical"
        except ValueError:
            pass

        # Fall back to string comparison
        if str(predicted).strip() == str(truth).strip():
            return True, "exact"

        return False, "none"

    def _evaluate_amc(self, predicted: Any, truth: Any) -> tuple:
        """AMC/AIME: integer match (mod 1000 for AIME)."""
        try:
            pred_int = int(float(str(predicted)))
            truth_int = int(float(str(truth)))

            # AIME answers are 0-999
            if pred_int % 1000 == truth_int % 1000:
                return True, "exact"

            return False, "none"
        except (ValueError, TypeError):
            return False, "none"

    def get_metrics(self) -> Dict[str, Any]:
        """Calculate aggregate metrics."""
        from collections import defaultdict

        by_source = defaultdict(lambda: {"correct": 0, "total": 0})

        for r in self._results:
            source = r["source"]
            by_source[source]["total"] += 1
            if r["correct"]:
                by_source[source]["correct"] += 1

        metrics = {
            "overall": {
                "correct": sum(1 for r in self._results if r["correct"]),
                "total": len(self._results),
                "accuracy": (
                    sum(1 for r in self._results if r["correct"]) / len(self._results)
                    if self._results else 0.0
                ),
            },
            "by_source": {},
        }

        for source, data in by_source.items():
            acc = data["correct"] / data["total"] if data["total"] > 0 else 0.0
            metrics["by_source"][source] = {
                "correct": data["correct"],
                "total": data["total"],
                "accuracy": acc,
            }

        return metrics
```

### 8.6 Integration with Sovereign Pipeline

```python
# File: knowledge3d/training/math_benchmarks/sovereign_math_pipeline.py
"""
Sovereign pipeline extension for math benchmarks.
"""

from typing import Dict, Any, List, Optional, Iterator
from pathlib import Path

from knowledge3d.training.math_benchmarks.math_proceduralizer import (
    MathDatasetLoader,
    MathProceduralizer,
)
from knowledge3d.training.math_benchmarks.math_output_adapter import MathOutputAdapter
from knowledge3d.training.math_benchmarks.benchmark_evaluator import MathBenchmarkEvaluator
from knowledge3d.training.arc_agi.sovereign_pipeline import SovereignPipeline


class SovereignMathPipeline(SovereignPipeline):
    """
    Extends SovereignPipeline for math benchmark training.

    The hot path remains sovereign (PTX + RPN).
    Only I/O layers adapt for math problems.
    """

    def __init__(
        self,
        datasets: List[str] = None,
        difficulty_range: Optional[range] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._math_loader = MathDatasetLoader(
            datasets=datasets or ["gsm8k", "math"],
            difficulty_filter=difficulty_range,
        )
        self._math_adapter = MathOutputAdapter()
        self._math_evaluator = MathBenchmarkEvaluator()
        self._proceduralizer = MathProceduralizer()

    def train_on_math(
        self,
        epochs: int = 10,
        log_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Train on math benchmarks.

        Returns:
            Training metrics including per-benchmark accuracy.
        """
        all_metrics = []

        for epoch in range(epochs):
            epoch_correct = 0
            epoch_total = 0

            for i, problem in enumerate(self._math_loader):
                # Execute through sovereign hot path
                result = self._execute_sovereign(problem)

                # Record result
                self._math_adapter.record_result(
                    problem["problem_id"],
                    result.get("stack", []),
                    problem["source"],
                )

                # Evaluate
                eval_result = self._math_evaluator.evaluate(
                    problem["problem_id"],
                    result.get("answer"),
                    problem["answer"],
                    problem["source"],
                )

                if eval_result["correct"]:
                    epoch_correct += 1
                epoch_total += 1

                if (i + 1) % log_interval == 0:
                    acc = epoch_correct / epoch_total
                    print(f"[Epoch {epoch+1}] Problem {i+1}: {acc:.2%} accuracy")

            metrics = self._math_evaluator.get_metrics()
            metrics["epoch"] = epoch + 1
            all_metrics.append(metrics)

            print(f"[Epoch {epoch+1}] Overall: {metrics['overall']['accuracy']:.2%}")
            for source, data in metrics["by_source"].items():
                print(f"  {source}: {data['accuracy']:.2%}")

        return {
            "epochs": epochs,
            "final_metrics": all_metrics[-1] if all_metrics else {},
            "history": all_metrics,
        }

    def _execute_sovereign(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute problem through sovereign hot path.

        This uses the same PTX + RPN execution as ARC-AGI.
        """
        # Use parent class execution
        return self.execute_task({
            "input": problem["problem_rpn"],
            "metadata": problem["metadata"],
        })


# Unified training entry point
def train_multimodal(
    arc_epochs: int = 10,
    math_epochs: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train on both ARC-AGI and math benchmarks.

    This demonstrates K3D's multi-modal reasoning capability
    using the same sovereign hot path.
    """
    output_dir = output_dir or Path("/K3D/Knowledge3D.local/logs/multimodal/")
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = SovereignMathPipeline(
        datasets=["gsm8k", "math", "omni_math"],
        difficulty_range=range(1, 8),  # Exclude hardest problems initially
    )

    results = {
        "arc_agi": None,
        "math": None,
    }

    # Phase 1: ARC-AGI training
    print("=" * 60)
    print("PHASE 1: ARC-AGI Visual Reasoning")
    print("=" * 60)
    # Use parent class for ARC training
    # results["arc_agi"] = pipeline.train(epochs=arc_epochs)

    # Phase 2: Math benchmark training
    print("=" * 60)
    print("PHASE 2: Math Reasoning Benchmarks")
    print("=" * 60)
    results["math"] = pipeline.train_on_math(epochs=math_epochs)

    # Save combined report
    import json
    report_path = output_dir / "multimodal_training_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nReport saved to: {report_path}")

    return results
```

### 8.7 Session Reporter Update (Math Benchmarks)

```python
# Extend SessionReporter to track math benchmark metrics

# In session_reporter.py, add to __init__:
self.report["math_benchmarks"] = {
    "datasets": [],
    "total_problems": 0,
    "by_source": {},
}

# Add method:
def set_math_benchmarks(
    self,
    datasets: List[str],
    total_problems: int,
    by_source: Dict[str, int],
) -> None:
    """Record math benchmark configuration."""
    self.report["math_benchmarks"] = {
        "datasets": datasets,
        "total_problems": total_problems,
        "by_source": by_source,
    }

# Update record_epoch to include math metrics:
def record_epoch(
    self,
    epoch: int,
    metrics: Dict[str, Any],
    math_metrics: Optional[Dict[str, Any]] = None,
) -> None:
    """Record epoch metrics including math benchmarks."""
    epoch_data = {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        **metrics,
    }

    if math_metrics:
        epoch_data["math_accuracy"] = math_metrics.get("overall", {}).get("accuracy", 0.0)
        epoch_data["math_by_source"] = math_metrics.get("by_source", {})

    self.report["epoch_history"].append(epoch_data)
```

### 8.8 Verification

```bash
cd "/mnt/arquivos/EchoSystems AI Studios/Knowledge 3D Standard/GitHub/Knowledge3D"

# Test 1: Verify datasets cloned
echo "=== Dataset Verification ==="
ls -la /K3D/K3D_llama_cpp/datasets/GSM8K/
ls -la /K3D/K3D_llama_cpp/datasets/math/
ls -la /K3D/K3D_llama_cpp/datasets/MMLU/
ls -la /K3D/K3D_llama_cpp/datasets/Omni-MATH/
ls -la /K3D/K3D_llama_cpp/datasets/AMC-AIME/

# Test 2: Math proceduralizer
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.math_benchmarks.math_proceduralizer import (
    MathProceduralizer,
    MathDatasetLoader,
)

proc = MathProceduralizer()

# Test problem
problem = {
    'question': 'Alice has 5 apples. Bob gives her 3 more. How many apples does Alice have?',
    'answer': '#### 8',
    'source': 'gsm8k',
}

result = proc.proceduralize_problem(problem)
print('Problem RPN:', result['problem_rpn'])
print('Answer:', result['answer'])
print('=== PROCEDURALIZER OK ===')
"

# Test 3: Dataset loader
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.math_benchmarks.math_proceduralizer import MathDatasetLoader

loader = MathDatasetLoader(datasets=['gsm8k'])
stats = loader.get_stats()

print(f'Total problems: {stats[\"total_problems\"]}')
print(f'By source: {stats[\"by_source\"]}')
print('=== DATASET LOADER OK ===')
"

# Test 4: Math evaluator
PYTHONPATH=. /K3D/Knowledge3D.local/envs/k3d-cranium/bin/python -c "
from knowledge3d.training.math_benchmarks.benchmark_evaluator import MathBenchmarkEvaluator

evaluator = MathBenchmarkEvaluator()

# Test GSM8K
evaluator.evaluate('p1', 8, 8, 'gsm8k')
evaluator.evaluate('p2', 42.0, '42', 'gsm8k')
evaluator.evaluate('p3', 10, 11, 'gsm8k')

# Test MATH
evaluator.evaluate('p4', '\\\\boxed{5}', '\\\\boxed{5}', 'math')

# Test MMLU
evaluator.evaluate('p5', 'B', 'B', 'mmlu')
evaluator.evaluate('p6', 'A', 'C', 'mmlu')

metrics = evaluator.get_metrics()
print(f'Overall accuracy: {metrics[\"overall\"][\"accuracy\"]:.2%}')
print(f'GSM8K: {metrics[\"by_source\"][\"gsm8k\"][\"accuracy\"]:.2%}')
print('=== EVALUATOR OK ===')
"
```

### 8.9 Files Summary (Part 8)

| File | Action | Purpose |
|------|--------|---------|
| `math_benchmarks/math_proceduralizer.py` | CREATE | Convert text problems → RPN |
| `math_benchmarks/math_output_adapter.py` | CREATE | Format answers for benchmarks |
| `math_benchmarks/benchmark_evaluator.py` | CREATE | Evaluate against ground truth |
| `math_benchmarks/sovereign_math_pipeline.py` | CREATE | Unified math training pipeline |
| `session_reporter.py` | MODIFY | Add math benchmark tracking |

### 8.10 Success Criteria (Part 8)

18. **Math datasets loaded** — GSM8K, MATH, MMLU, Omni-MATH, AMC-AIME parsed and ready
19. **Proceduralizer working** — word problems → RPN representation
20. **Per-benchmark evaluation** — correct scoring for each benchmark format
21. **Unified training** — single pipeline handles ARC-AGI + math benchmarks
22. **Session reports include math** — accuracy tracked per-benchmark

---

## Implementation Order

**Phase 1 (Parts 1-6):** Core sovereign swarm architecture
- Grammar evolution
- Size pattern encoding
- TRM Swarm Coordinator
- Progressive scoring
- Organization & dual-dataset

**Phase 2 (Part 7):** ARC-AGI 3 & benchmarking
- Clone ARC-AGI 3 repos
- Standard output adapter
- Unified loader v2
- Official evaluation pipeline

**Phase 3 (Part 8):** Math benchmark extensions
- Math proceduralizer
- Benchmark-specific adapters
- Unified multimodal training
- Comprehensive evaluation metrics

---

**END OF SPECIFICATION**

Claude (Architecture Partner)
December 12, 2025

Sources consulted for math benchmarks:
- [GSM8K](https://github.com/openai/grade-school-math) - OpenAI grade school math
- [MATH](https://github.com/hendrycks/math) - Hendrycks competition math
- [MMLU](https://github.com/hendrycks/test) - Multi-task benchmark
- [Omni-MATH](https://github.com/KbsdJames/Omni-MATH) - Olympiad-level problems
- [AMC-AIME](https://github.com/ryanrudes/amc) - American Math Competition archive
- [FrontierMath](https://epoch.ai/frontiermath) - Private benchmark (no public dataset)
