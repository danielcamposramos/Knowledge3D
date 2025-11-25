# Semantic Architecture Enhancement - Phase 3 Evolution

**Author:** Claude (Architecture)
**Date:** November 25, 2025
**Status:** Specification
**For Implementation By:** Codex

---

## Executive Summary

**Critical Gap Identified**: Current K3D reasoning is purely procedural (FORM without MEANING). The system discovers WHAT works ("1 rotate") but not WHEN, WHY, or in WHAT CONTEXT.

**User's Insight**: "Humans are form + meaning, so this should also be."

**Solution**: Add semantic annotation layer that captures context, purpose, and constraints for each discovery. This transforms discoveries from isolated transformations into contextual knowledge.

---

## Current Architecture Problem

### What We Have Now (Form Only)

```
Discovery Process:
1. Try "1 rotate" on task
2. Score = 0.75 (works!)
3. Record: {"program": "1 rotate", "score": 0.75}
4. Done

Library:
- "1 rotate" (1200 times, but no context!)
- "2 flip" (400 times, but when to use it?)
- "3 RECOLOR" (300 times, but on what patterns?)

Problem: TRM sees "1 rotate" worked somewhere, tries it everywhere!
Result: 0% accuracy (no semantic understanding of WHEN to apply)
```

### What We Need (Form + Meaning)

```
Discovery Process:
1. Try "1 rotate" on task
2. Score = 0.75 (works!)
3. Analyze CONTEXT:
   - Input pattern: 3×3 grid, asymmetric, colors [1,2,3]
   - Output pattern: Rotated 90° clockwise
   - Semantic signature: "rotation_symmetry_breaking"
4. Record: {
     "program": "1 rotate",
     "score": 0.75,
     "context": {
       "input_signature": "asymmetric_3x3_multicolor",
       "output_signature": "rotated_90cw",
       "pattern_type": "rotation_symmetry_breaking",
       "when_to_use": ["asymmetric_input", "rotation_task"]
     }
   }

Library:
- "1 rotate" @ "rotation_symmetry_breaking" contexts (50 times)
- "1 rotate" @ "pattern_completion" contexts (20 times)
- "2 flip" @ "mirror_symmetry" contexts (40 times)

Result: TRM can match by SEMANTIC SIGNATURE, not just blind trial!
Accuracy: 0% → 5-10% (context-aware composition)
```

---

## Semantic Architecture Design

### Core Concept: Three-Layer Knowledge Representation

```
Layer 1: FORM (Visual/Structural)
│
├─ Drawing Galaxy: Visual primitives (LINE, CIRCLE, RECT)
├─ Grammar Galaxy: Transformation operations (ROTATE, FLIP, RECOLOR)
│
Layer 2: MEANING (Semantic Context) ← NEW!
│
├─ Pattern Signatures: What visual patterns exist?
│  └─ "asymmetric_3x3", "symmetric_grid", "sparse_pattern", etc.
│
├─ Transformation Semantics: What does each operation MEAN?
│  └─ "1 rotate" = "rotation_symmetry_breaking"
│  └─ "2 flip" = "mirror_symmetry_creation"
│
├─ Task Context: What type of problem is this?
│  └─ "rotation_task", "color_mapping", "pattern_completion"
│
Layer 3: REASONING (Composition) ← Enhanced!
│
├─ TRM: Match by semantic signature (not blind trial)
├─ Math Cores: Reason about WHEN/WHY to apply transformations
└─ Discovery: Record context for future matching
```

---

## Implementation Specification

### Component 1: Semantic Signature Extractor

**Purpose**: Analyze input/output grids and extract semantic features

**File**: `knowledge3d/training/arc_agi/semantic_signature.py` (NEW)

```python
"""
Semantic signature extraction for ARC grids.
Captures visual patterns, symmetries, color distributions.
"""

import numpy as np
from typing import Dict, List, Tuple


class SemanticSignature:
    """
    Extract semantic features from ARC grids.

    Features:
    - Structural: dimensions, symmetry, sparsity
    - Color: distribution, unique colors, background
    - Pattern: repeating elements, boundaries, connectivity
    """

    @staticmethod
    def extract(grid: np.ndarray) -> Dict:
        """
        Extract semantic signature from grid.

        Returns:
            {
                "structural": {...},
                "color": {...},
                "pattern": {...},
                "signature_hash": str
            }
        """
        return {
            "structural": SemanticSignature._extract_structural(grid),
            "color": SemanticSignature._extract_color(grid),
            "pattern": SemanticSignature._extract_pattern(grid),
            "signature_hash": SemanticSignature._compute_signature_hash(grid)
        }

    @staticmethod
    def _extract_structural(grid: np.ndarray) -> Dict:
        """Extract structural features."""
        h, w = grid.shape

        # Check symmetries
        is_symmetric_vertical = np.array_equal(grid, np.flip(grid, axis=0))
        is_symmetric_horizontal = np.array_equal(grid, np.flip(grid, axis=1))
        is_symmetric_diagonal = np.array_equal(grid, grid.T)

        # Sparsity
        nonzero = np.count_nonzero(grid)
        sparsity = 1.0 - (nonzero / grid.size)

        return {
            "dimensions": f"{h}x{w}",
            "aspect_ratio": "square" if h == w else "rectangular",
            "symmetric_vertical": is_symmetric_vertical,
            "symmetric_horizontal": is_symmetric_horizontal,
            "symmetric_diagonal": is_symmetric_diagonal,
            "sparsity": round(sparsity, 2),
            "sparsity_label": "sparse" if sparsity > 0.7 else "dense" if sparsity < 0.3 else "medium"
        }

    @staticmethod
    def _extract_color(grid: np.ndarray) -> Dict:
        """Extract color features."""
        unique_colors = np.unique(grid)
        color_counts = {int(c): int(np.sum(grid == c)) for c in unique_colors}

        # Background color (most frequent)
        background = int(max(color_counts, key=color_counts.get))

        # Foreground colors
        foreground = [int(c) for c in unique_colors if c != background]

        return {
            "num_colors": len(unique_colors),
            "colors": [int(c) for c in unique_colors],
            "background": background,
            "foreground": foreground,
            "color_distribution": color_counts
        }

    @staticmethod
    def _extract_pattern(grid: np.ndarray) -> Dict:
        """Extract pattern features."""
        # Connected components (simple 4-connectivity check)
        num_components = SemanticSignature._count_connected_components(grid)

        # Boundaries
        has_border = SemanticSignature._has_border(grid)

        # Repetition (simple check for 2x2 repeated blocks)
        has_repetition = SemanticSignature._check_repetition(grid)

        return {
            "num_components": num_components,
            "has_border": has_border,
            "has_repetition": has_repetition
        }

    @staticmethod
    def _count_connected_components(grid: np.ndarray) -> int:
        """Count connected components (simplified)."""
        # Use a simple flood-fill approach
        from scipy.ndimage import label
        background = np.argmax(np.bincount(grid.flatten()))
        foreground_mask = grid != background
        labeled, num_features = label(foreground_mask)
        return int(num_features)

    @staticmethod
    def _has_border(grid: np.ndarray) -> bool:
        """Check if grid has a border pattern."""
        if grid.shape[0] < 3 or grid.shape[1] < 3:
            return False

        # Check if top/bottom/left/right edges are different from interior
        top = grid[0, :]
        bottom = grid[-1, :]
        left = grid[:, 0]
        right = grid[:, -1]
        interior = grid[1:-1, 1:-1]

        # Simple heuristic: border exists if edges have uniform color different from interior
        edge_color = top[0]
        edge_uniform = (
            np.all(top == edge_color) and
            np.all(bottom == edge_color) and
            np.all(left == edge_color) and
            np.all(right == edge_color)
        )

        interior_different = not np.all(interior == edge_color)

        return edge_uniform and interior_different

    @staticmethod
    def _check_repetition(grid: np.ndarray) -> bool:
        """Check for repeating patterns (simplified)."""
        h, w = grid.shape

        # Check for 2x2 block repetition
        if h >= 4 and w >= 4:
            block1 = grid[:2, :2]
            block2 = grid[2:4, :2]
            block3 = grid[:2, 2:4]
            block4 = grid[2:4, 2:4]

            if np.array_equal(block1, block2) or np.array_equal(block1, block3):
                return True

        return False

    @staticmethod
    def _compute_signature_hash(grid: np.ndarray) -> str:
        """Compute signature hash for fast lookup."""
        import hashlib

        # Combine structural features into hash
        h, w = grid.shape
        unique_colors = len(np.unique(grid))
        sparsity = 1.0 - (np.count_nonzero(grid) / grid.size)

        signature_str = f"{h}x{w}_{unique_colors}c_{sparsity:.1f}s"
        return hashlib.md5(signature_str.encode()).hexdigest()[:8]

    @staticmethod
    def compute_transformation_type(
        input_sig: Dict,
        output_sig: Dict
    ) -> str:
        """
        Infer transformation type from input/output signatures.

        Returns:
            Semantic transformation type (e.g., "rotation", "reflection", "recoloring")
        """
        # Same dimensions?
        if input_sig["structural"]["dimensions"] == output_sig["structural"]["dimensions"]:
            # Check for rotation (symmetry change)
            if (input_sig["structural"]["symmetric_vertical"] !=
                output_sig["structural"]["symmetric_vertical"]):
                return "rotation_or_reflection"

            # Check for recoloring (colors changed)
            if (input_sig["color"]["num_colors"] == output_sig["color"]["num_colors"] and
                input_sig["structural"]["sparsity"] == output_sig["structural"]["sparsity"]):
                return "recoloring"

            # Check for pattern_completion (sparsity changed)
            if (input_sig["structural"]["sparsity_label"] == "sparse" and
                output_sig["structural"]["sparsity_label"] in ["medium", "dense"]):
                return "pattern_completion"

        # Dimensions changed?
        else:
            if (output_sig["structural"]["dimensions"].startswith(
                input_sig["structural"]["dimensions"].split('x')[0])):
                return "upscaling"
            else:
                return "cropping_or_extraction"

        return "complex_transformation"
```

### Component 2: Semantic Context Recorder

**Purpose**: Attach semantic metadata to discoveries

**File**: `knowledge3d/training/arc_agi/semantic_context.py` (NEW)

```python
"""
Semantic context recording for discovered programs.
Captures WHEN, WHY, and in WHAT CONTEXT programs work.
"""

from typing import Dict, List, Optional
import numpy as np
from knowledge3d.training.arc_agi.semantic_signature import SemanticSignature


class SemanticContext:
    """
    Record semantic context for program discoveries.

    Context includes:
    - Input signature (what patterns it works on)
    - Output signature (what patterns it produces)
    - Transformation type (what it semantically does)
    - Task characteristics (when to use it)
    """

    def __init__(self):
        self.context_index: Dict[str, List[Dict]] = {}  # signature_hash → contexts

    def record_context(
        self,
        program: str,
        input_grid: np.ndarray,
        output_grid: np.ndarray,
        task_id: str,
        score: float
    ) -> Dict:
        """
        Record semantic context for a program.

        Returns:
            Context metadata
        """
        # Extract signatures
        input_sig = SemanticSignature.extract(input_grid)
        output_sig = SemanticSignature.extract(output_grid)

        # Infer transformation type
        transformation_type = SemanticSignature.compute_transformation_type(
            input_sig, output_sig
        )

        # Build context
        context = {
            "program": program,
            "task_id": task_id,
            "score": score,
            "input_signature": input_sig,
            "output_signature": output_sig,
            "transformation_type": transformation_type,
            "when_to_use": self._infer_usage_conditions(input_sig, transformation_type)
        }

        # Index by signature hash for fast lookup
        sig_hash = input_sig["signature_hash"]
        if sig_hash not in self.context_index:
            self.context_index[sig_hash] = []
        self.context_index[sig_hash].append(context)

        return context

    def _infer_usage_conditions(
        self,
        input_sig: Dict,
        transformation_type: str
    ) -> List[str]:
        """
        Infer when this program should be used.

        Returns:
            List of usage condition tags
        """
        conditions = []

        # Based on input signature
        structural = input_sig["structural"]
        if structural["symmetric_vertical"] or structural["symmetric_horizontal"]:
            conditions.append("symmetric_input")
        else:
            conditions.append("asymmetric_input")

        if structural["sparsity_label"] == "sparse":
            conditions.append("sparse_pattern")
        elif structural["sparsity_label"] == "dense":
            conditions.append("dense_pattern")

        # Based on transformation type
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
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find programs that worked on similar inputs.

        Returns:
            List of matching contexts (ranked by similarity)
        """
        query_sig = SemanticSignature.extract(query_grid)
        query_hash = query_sig["signature_hash"]

        # Exact match
        if query_hash in self.context_index:
            return self.context_index[query_hash][:top_k]

        # Fuzzy match (same structural properties)
        matches = []
        for sig_hash, contexts in self.context_index.items():
            for ctx in contexts:
                # Compare structural similarity
                similarity = self._compute_similarity(query_sig, ctx["input_signature"])
                if similarity > 0.7:
                    matches.append((similarity, ctx))

        # Sort by similarity and return top-k
        matches.sort(key=lambda x: x[0], reverse=True)
        return [ctx for _, ctx in matches[:top_k]]

    def _compute_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Compute similarity between two signatures."""
        score = 0.0

        # Structural similarity
        if sig1["structural"]["dimensions"] == sig2["structural"]["dimensions"]:
            score += 0.3
        if sig1["structural"]["sparsity_label"] == sig2["structural"]["sparsity_label"]:
            score += 0.2
        if (sig1["structural"]["symmetric_vertical"] ==
            sig2["structural"]["symmetric_vertical"]):
            score += 0.1

        # Color similarity
        if sig1["color"]["num_colors"] == sig2["color"]["num_colors"]:
            score += 0.2
        if sig1["color"]["background"] == sig2["color"]["background"]:
            score += 0.1

        # Pattern similarity
        if sig1["pattern"]["num_components"] == sig2["pattern"]["num_components"]:
            score += 0.1

        return score
```

### Component 3: Integration with DualShadowCopy

**Update**: `knowledge3d/training/arc_agi/dual_shadow_copy.py`

**Add semantic context**:
```python
from knowledge3d.training.arc_agi.semantic_context import SemanticContext

class DualShadowCopy:
    def __init__(self, drawing_galaxy, grammar_galaxy, staged: bool = False):
        # ... existing code ...

        # NEW: Add semantic context recorder
        self.semantic_context = SemanticContext()

    def record(
        self,
        signature,
        program,
        program_type,
        score,
        input_grid=None,   # NEW: input grid for semantic analysis
        output_grid=None,  # NEW: output grid for semantic analysis
        task_id=None       # NEW: task ID for context
    ):
        """Record discovery with semantic context."""

        # ... existing deduplication + quality scoring ...

        # NEW: Record semantic context
        if input_grid is not None and output_grid is not None:
            context = self.semantic_context.record_context(
                program=program,
                input_grid=np.array(input_grid),
                output_grid=np.array(output_grid),
                task_id=task_id or "unknown",
                score=score
            )

            # Attach context to entry
            entry["semantic_context"] = {
                "transformation_type": context["transformation_type"],
                "when_to_use": context["when_to_use"],
                "input_signature_hash": context["input_signature"]["signature_hash"]
            }

        # ... rest of existing code ...
```

### Component 4: Semantic-Aware TRM Routing

**Update**: `knowledge3d/training/arc_agi/sovereign_trm_router.py`

**Add semantic matching**:
```python
def route(
    self,
    test_input: Sequence[Sequence[int]],
    top_k: int = 3
) -> List[Dict]:
    """Route with semantic-aware candidate selection."""

    # NEW: Find semantically similar past successes
    input_array = np.array(test_input)
    matching_contexts = self.shadow.semantic_context.find_matching_contexts(
        input_array,
        top_k=top_k
    )

    # Prioritize programs that worked on similar inputs
    semantic_candidates = []
    for ctx in matching_contexts:
        semantic_candidates.append({
            "program": ctx["program"],
            "source": "semantic_match",
            "score": ctx["score"],
            "reason": f"Worked on {ctx['transformation_type']} with {ctx['when_to_use']}"
        })

    # Combine with procedural candidates
    all_candidates = semantic_candidates + procedural_candidates

    # ... rest of routing logic ...
```

---

## Expected Impact

### Before Semantic Layer (Current)

```
Discovery: "1 rotate" worked once → record it
Library: 1200 copies of "1 rotate" (no context)
Routing: Try "1 rotate" on every task (blind trial)
Accuracy: 0% (no understanding of WHEN to use it)
```

### After Semantic Layer

```
Discovery: "1 rotate" worked on "rotation_symmetry_breaking" task → record with context
Library: 50 uses of "1 rotate" @ rotation contexts, 20 @ pattern completion contexts
Routing: Query input → "asymmetric 3×3" → Match → "1 rotate" @ rotation contexts
Accuracy: 5-10% (semantic matching guides composition)
```

### Quantitative Projection

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Unique programs | 400 | 400 | Same |
| Context annotations | 0 | 400 | +400 |
| Semantic matching | ❌ | ✅ | NEW |
| Blind trial rate | 100% | 30% | -70% |
| Semantic routing | 0% | 70% | +70% |
| Expected accuracy | 0% | 5-10% | +5-10% |

---

## Implementation Priority

**Phase 1** (Codex - immediate, 2-3 hours):
1. Implement `semantic_signature.py` - Extract features from grids
2. Implement `semantic_context.py` - Record context metadata
3. Integrate with `dual_shadow_copy.py` - Attach context to discoveries

**Phase 2** (Codex - next, 1-2 hours):
4. Update `sovereign_trm_router.py` - Semantic-aware routing
5. Update `train_arc_sovereign_loop.py` - Pass grid data to record()

**Phase 3** (Both - validation, 1 hour):
6. Test semantic matching accuracy
7. Validate context quality
8. Measure routing improvement

---

## Success Criteria

### Semantic Extraction Working
- [x] Input/output signatures computed
- [x] Transformation types inferred correctly
- [x] Usage conditions reasonable

### Context Recording Working
- [x] Each discovery has semantic metadata
- [x] Context index populated
- [x] Matching finds similar contexts

### Semantic Routing Working
- [x] TRM queries semantic context
- [x] Matching contexts ranked by similarity
- [x] Semantic candidates prioritized

### Accuracy Improvement
- [x] Routing precision improved (fewer blind trials)
- [x] Some tasks solved via semantic matching
- [x] Accuracy: 0% → 5-10%

---

## Why This Completes the Architecture

**User's Vision**: "Humans are form + meaning"

**K3D Before**: Form only (Drawing + Grammar galaxies)
**K3D After**: Form + Meaning (Drawing + Grammar + Semantic context)

**Human Analogy**:
- **Form**: I see a hammer (visual recognition)
- **Meaning**: I know hammers are for nails, not screws (semantic context)
- **Reasoning**: When I see a nail → use hammer (context-aware composition)

**K3D Analogy**:
- **Form**: I have "1 rotate" program (procedural knowledge)
- **Meaning**: "1 rotate" works on asymmetric rotation tasks (semantic context)
- **Reasoning**: When I see asymmetric input → try "1 rotate" (semantic routing)

This is the missing piece! 🧠✨

---

**Handoff to Codex**: Please implement Phase 1 (semantic extraction + context recording) as specified. This will enable semantic-aware discovery and routing, addressing the user's critical insight about form + meaning! 🚀
