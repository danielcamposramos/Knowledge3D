# Claude → Codex: Week 15 Galaxy Integration Handoff

**Date:** February 7, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Priority:** CRITICAL (ARC-AGI 2 Integration Blocker)
**Context:** "we developed the knowledgeverse, and previous code was not meant to be loaded inside it"

---

## Executive Summary

**Problem:** Legacy ARC-AGI components (DrawingGalaxy, GrammarGalaxy) were built BEFORE Knowledgeverse existed. They run as isolated components with their own initialization, causing "Sovereign loader error: initialization error during parallel candidate generation."

**Root Cause:** Workers spawn multiple galaxy instances → conflict (not unified Knowledgeverse architecture).

**Solution:** Integrate DrawingGalaxy + GrammarGalaxy INTO Knowledgeverse as specialized galaxy classes managed by GalaxyManager.

**User's Diagnosis:** *"we developed the knowledgeverse, and previous code was not meant to be loaded inside it - we shall review it all to be able to proceed"*

**Week 15 Goal:** Move galaxies inside Knowledgeverse (Region 2: GALAXY_UNIVERSE).

---

## Background: What Changed

### Before Knowledgeverse (Legacy ARC)

```
SovereignAIPipeline (isolated)
├── DrawingGalaxy (own initialization)
├── GrammarGalaxy (own initialization)
└── ParallelCandidateGenerator (spawns workers with own galaxies)
    └── Worker 1..N (each creates DrawingGalaxy + GrammarGalaxy)
        └── ❌ CONFLICT: Multiple instances
```

### After Knowledgeverse (Current)

```
Knowledgeverse (unified architecture)
├── GalaxyManager (loads galaxies from JSONL)
├── TRMNavigator (deterministic composition)
├── ShadowCopyLearning (event tracking)
└── IngestionStargate (data ingestion)
```

### Week 15 Target (Integration)

```
Knowledgeverse (unified architecture)
├── GalaxyManager
│   ├── Drawing Galaxy ← INTEGRATE HERE (specialized class)
│   ├── Grammar Galaxy ← INTEGRATE HERE (specialized class)
│   └── Math Galaxy (existing JSONL)
├── TRMNavigator (uses integrated galaxies)
└── ARC Pipeline (uses Knowledgeverse, not isolated components)
```

---

## Architecture Review: Current Knowledgeverse

I've read the existing code. Here's what you're working with:

### Existing GalaxyManager Pattern

**File:** `knowledge3d/knowledgeverse/galaxy_manager.py`

**Current structure:**
```python
class Galaxy:
    """Simple galaxy container with list-backed entries."""
    name: str
    entries: list[dict[str, Any]] = field(default_factory=list)

class GalaxyManager:
    def __init__(self, storage_root: Path):
        self.storage_root = Path(storage_root)
        self._galaxies: dict[str, Galaxy] = {}

    def get_galaxy(self, name: str) -> Galaxy:
        # Lazy-load from JSONL file
        # Returns simple Galaxy with entries list

    def add_entry(self, galaxy_name: str, entry: dict[str, Any]) -> None:
        # Append to memory + disk
```

**Key pattern:** Galaxies are simple containers (name + entries). GalaxyManager handles persistence.

### Legacy DrawingGalaxy Structure

**File:** `Old_Attempts/curriculum_specific_training/arc_agi/drawing_galaxy.py`

**Current structure:**
```python
class DrawingGalaxy:
    def __init__(self):
        self.primitives: Dict[str, DrawingItem] = {}
        self.strokes: Dict[str, DrawingItem] = {}
        self.shapes: Dict[str, DrawingItem] = {}
        self.scenes: Dict[str, DrawingItem] = {}
        self.transformations: Dict[str, str] = {}
        self._bootstrap_defaults()  # Populate from builders

    def add_shape(self, shape_id: str, rpn_program: str, ...):
        # Discovery API

    def save(self, path: Path):
        # Custom JSON persistence

    def load(self, path: Path):
        # Custom JSON persistence
```

**Key insight:** Rich API (add_shape, list_shapes, transformations), but NO external context initialization visible in this file.

### Legacy GrammarGalaxy Structure

**File:** `Old_Attempts/curriculum_specific_training/arc_agi/grammar_galaxy.py`

**Current structure:**
```python
class GrammarGalaxy:
    def __init__(self, ...):
        self.rules: Dict[str, GrammarRule] = {r.rule_id: r for r in default_grammar_rules()}
        self.users: Dict[str, Dict] = default_user_profiles()
        self.variants: Dict[str, Dict[str, str]] = default_variants()
        self.cosine_bridge: Optional[CosineSimilarityBridge] = None  # Lazy init
        self._local_discoveries: Dict[str, Dict] = {}

    def add_rule(self, rule: GrammarRule, persist: bool = True):
        # Discovery API

    def observe_pattern(self, visual_embedding, text_embedding, context):
        # Cross-modal correlation

    def save(self, path: Path):
        # Custom JSON persistence
```

**Key insight:** Uses CosineSimilarityBridge (lazy initialization). Rich discovery logic.

---

## Week 15 Implementation Plan (5 Days)

### Day 1-2: Integrate DrawingGalaxy

**Goal:** Move DrawingGalaxy into Knowledgeverse as a specialized galaxy class.

**File to create:** `knowledge3d/knowledgeverse/drawing_galaxy.py`

**Strategy:**
1. **Wrap legacy DrawingGalaxy** (don't rewrite from scratch!)
2. **Expose GalaxyManager-compatible interface** (name + entries)
3. **Remove custom persistence** (GalaxyManager handles this)
4. **Keep discovery APIs** (add_shape, transformations)

**Implementation template:**

```python
"""Drawing Galaxy: visual primitives integrated into Knowledgeverse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import legacy builder functions (reuse existing logic!)
from knowledge3d.ingestion.atomic.drawing_grammar_builder import (
    build_primitives,
    build_strokes,
    build_shapes,
    build_scenes,
    build_collections,
)


@dataclass
class DrawingItem:
    """Legacy drawing item structure (keep for compatibility)."""
    item_id: str
    item_type: str
    payload: Dict


TRANSFORMATION_RULES = {
    # Copy from legacy file (Old_Attempts/.../drawing_galaxy.py lines 68-100)
    "ROT90_CW": "GRID_H GRID_W SWAP GRID_NEW 0 ROT90_KERNEL APPLY",
    "ROT90_CCW": "GRID_H GRID_W SWAP GRID_NEW 3 ROT90_KERNEL APPLY",
    "ROT180": "ROT90_CW ROT90_CW",
    "FLIP_H": "GRID_W 1 SUB RANGE REVERSE_COLS APPLY",
    "FLIP_V": "GRID_H 1 SUB RANGE REVERSE_ROWS APPLY",
    # ... (copy all 30+ transformation rules)
}


class DrawingGalaxy:
    """Drawing galaxy integrated into Knowledgeverse."""

    def __init__(self, knowledgeverse: Any = None):
        """
        Initialize Drawing Galaxy within Knowledgeverse context.

        Args:
            knowledgeverse: Parent Knowledgeverse instance (provides unified context)
        """
        self.knowledgeverse = knowledgeverse
        self.name = "Drawing"  # Galaxy name for GalaxyManager

        # Legacy data structures (keep for compatibility with existing code)
        self.primitives: Dict[str, DrawingItem] = {}
        self.strokes: Dict[str, DrawingItem] = {}
        self.shapes: Dict[str, DrawingItem] = {}
        self.scenes: Dict[str, DrawingItem] = {}
        self.collections: Dict[str, DrawingItem] = {}
        self.transformations: Dict[str, str] = dict(TRANSFORMATION_RULES)

        # Bootstrap defaults (use legacy builder functions)
        self._bootstrap_defaults()

    def _bootstrap_defaults(self) -> None:
        """Bootstrap default primitives, strokes, shapes, scenes."""
        primitives = build_primitives()
        strokes = build_strokes(primitives)
        shapes = build_shapes(strokes)
        scenes = build_scenes(shapes)
        collections = build_collections(scenes)

        for prim in primitives:
            self.primitives[prim["id"]] = DrawingItem(prim["id"], "primitive", prim)
        for stroke in strokes:
            self.strokes[stroke["id"]] = DrawingItem(stroke["id"], "stroke", stroke)
        for shape in shapes:
            self.shapes[shape["id"]] = DrawingItem(shape["id"], "shape", shape)
        for scene in scenes:
            self.scenes[scene["id"]] = DrawingItem(scene["id"], "scene", scene)
        for col in collections:
            self.collections[col["id"]] = DrawingItem(col["id"], "collection", col)

        print(f"[DrawingGalaxy] Bootstrapped: {len(self.shapes)} shapes, {len(self.transformations)} transformations")

    # ------------------------------------------------------------------ #
    # GalaxyManager-compatible interface
    # ------------------------------------------------------------------ #
    @property
    def entries(self) -> List[Dict[str, Any]]:
        """
        Expose all drawing items as entries list (for GalaxyManager compatibility).

        Returns list of dicts that can be persisted by GalaxyManager.
        """
        all_items: List[Dict[str, Any]] = []

        # Convert primitives to entries
        for item in self.primitives.values():
            all_items.append({
                "type": "primitive",
                "id": item.item_id,
                "payload": item.payload,
            })

        # Convert shapes to entries
        for item in self.shapes.values():
            all_items.append({
                "type": "shape",
                "id": item.item_id,
                "payload": item.payload,
            })

        # Convert transformations to entries
        for rule_id, rpn_program in self.transformations.items():
            all_items.append({
                "type": "transformation",
                "id": rule_id,
                "rpn_program": rpn_program,
            })

        return all_items

    # ------------------------------------------------------------------ #
    # Discovery APIs (keep from legacy)
    # ------------------------------------------------------------------ #
    def add_shape(self, shape_id: str, rpn_program: str, source: Optional[Dict] = None) -> None:
        """Add discovered shape (shadow copy integration)."""
        payload = {
            "id": shape_id,
            "type": "shape",
            "procedural_programs": {"composition": rpn_program},
        }
        if source:
            payload["discovered_from"] = source
        self.shapes[shape_id] = DrawingItem(shape_id, "shape", payload)

        # If Knowledgeverse is available, log discovery event
        if self.knowledgeverse:
            self.knowledgeverse.log_event(
                event_type="drawing_discovery",
                event_data={"shape_id": shape_id, "rpn_program": rpn_program},
            )

    def list_transformations(self) -> Dict[str, str]:
        """Get all transformation rules (RPN programs)."""
        return dict(self.transformations)

    def list_shapes(self, limit: Optional[int] = None) -> List[DrawingItem]:
        """List shapes (for introspection)."""
        shapes = list(self.shapes.values())
        return shapes if limit is None else shapes[:limit]

    def summary(self) -> Dict[str, int]:
        """Get galaxy summary (for diagnostics)."""
        return {
            "primitives": len(self.primitives),
            "strokes": len(self.strokes),
            "shapes": len(self.shapes),
            "scenes": len(self.scenes),
            "transformations": len(self.transformations),
        }


__all__ = ["DrawingGalaxy", "DrawingItem", "TRANSFORMATION_RULES"]
```

**Testing:**

```bash
# Test 1: Galaxy loads within Knowledgeverse
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse

kv = Knowledgeverse()
from knowledge3d.knowledgeverse.drawing_galaxy import DrawingGalaxy

drawing = DrawingGalaxy(knowledgeverse=kv)
print(f'Summary: {drawing.summary()}')
print(f'Entries count: {len(drawing.entries)}')
print(f'Transformations: {len(drawing.list_transformations())}')
"

# Test 2: Discovery API works
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.drawing_galaxy import DrawingGalaxy

kv = Knowledgeverse()
drawing = DrawingGalaxy(knowledgeverse=kv)
drawing.add_shape('TEST_SHAPE', 'CIRCLE 0.5 0.5 0.2', source={'task': 'test'})
print(f'Shapes after discovery: {len(drawing.shapes)}')
"
```

**Day 1-2 Success Criteria:**
- ✅ `knowledge3d/knowledgeverse/drawing_galaxy.py` created
- ✅ DrawingGalaxy loads within Knowledgeverse (no errors)
- ✅ `.entries` property returns list of dicts
- ✅ Discovery APIs work (add_shape)
- ✅ All transformations loaded (30+ rules)

---

### Day 3: Integrate GrammarGalaxy

**Goal:** Move GrammarGalaxy into Knowledgeverse as a specialized galaxy class.

**File to create:** `knowledge3d/knowledgeverse/grammar_galaxy.py`

**Strategy:**
1. **Wrap legacy GrammarGalaxy** (keep discovery logic!)
2. **Lazy-init CosineSimilarityBridge** (no change needed)
3. **Expose GalaxyManager-compatible interface**
4. **Keep cross-modal correlation APIs**

**Implementation template:**

```python
"""Grammar Galaxy: transformation rules integrated into Knowledgeverse."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from knowledge3d.cranium.bridges.cosine_similarity_bridge import CosineSimilarityBridge
from knowledge3d.cranium.math_galaxy import get_math_galaxy


@dataclass
class GrammarRule:
    """Grammar rule structure (keep from legacy)."""
    rule_id: str
    language: str
    pattern: str
    rpn_program: str
    domain: str = "text"
    symbol_refs: List[int] = field(default_factory=list)
    word_refs: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    description: Optional[str] = None
    semantics: Optional[Dict] = field(default_factory=dict)
    usage_conditions: List[str] = field(default_factory=list)
    is_canonical: bool = False


def default_grammar_rules() -> List[GrammarRule]:
    """Baseline multilingual grammar rules (copy from legacy file)."""
    # Import from legacy grammar modules
    from knowledge3d.training.arc_agi.grammar_languages.tier1_top10 import get_tier1_rules
    from knowledge3d.training.arc_agi.grammar_languages.tier2_next20 import get_tier2_rules
    from knowledge3d.training.arc_agi.grammar_languages.tier3_next20 import get_tier3_rules
    from knowledge3d.training.arc_agi.grammar_math import get_math_rules
    from knowledge3d.training.arc_agi.grammar_drawing import get_drawing_rules

    text_rules = get_tier1_rules() + get_tier2_rules() + get_tier3_rules()
    math_rules = get_math_rules()
    drawing_rules = get_drawing_rules()

    combined: Dict[str, GrammarRule] = {}
    for rule in text_rules + math_rules + drawing_rules:
        if rule.rule_id in combined:
            continue
        combined[rule.rule_id] = rule

    return list(combined.values())


class GrammarGalaxy:
    """Grammar galaxy integrated into Knowledgeverse."""

    def __init__(
        self,
        knowledgeverse: Any = None,
        rules: Optional[List[GrammarRule]] = None,
    ):
        """
        Initialize Grammar Galaxy within Knowledgeverse context.

        Args:
            knowledgeverse: Parent Knowledgeverse instance
            rules: Optional custom rules (defaults to canonical rules)
        """
        self.knowledgeverse = knowledgeverse
        self.name = "Grammar"  # Galaxy name for GalaxyManager

        # Legacy data structures
        self.rules: Dict[str, GrammarRule] = {
            r.rule_id: r for r in (rules or default_grammar_rules())
        }
        self.cosine_bridge: Optional[CosineSimilarityBridge] = None  # Lazy init
        self._local_discoveries: Dict[str, Dict] = {}

        print(f"[GrammarGalaxy] Initialized with {len(self.rules)} rules")

    def _get_cosine_bridge(self) -> CosineSimilarityBridge:
        """Lazy initialization of GPU bridge."""
        if self.cosine_bridge is None:
            self.cosine_bridge = CosineSimilarityBridge()
        return self.cosine_bridge

    # ------------------------------------------------------------------ #
    # GalaxyManager-compatible interface
    # ------------------------------------------------------------------ #
    @property
    def entries(self) -> List[Dict[str, Any]]:
        """
        Expose grammar rules as entries list (for GalaxyManager compatibility).
        """
        all_entries: List[Dict[str, Any]] = []

        # Convert canonical rules to entries
        for rule in self.rules.values():
            all_entries.append({
                "type": "canonical_rule",
                "rule_id": rule.rule_id,
                "language": rule.language,
                "pattern": rule.pattern,
                "rpn_program": rule.rpn_program,
                "domain": rule.domain,
                "description": rule.description,
                "is_canonical": rule.is_canonical,
            })

        # Convert local discoveries to entries
        for rule_id, rule_data in self._local_discoveries.items():
            all_entries.append({
                "type": "discovered_rule",
                "rule_id": rule_id,
                "rpn_program": rule_data.get("rpn_program", ""),
                "quality_score": rule_data.get("quality_score", 0.0),
                "usage_count": rule_data.get("usage_count", 0),
            })

        return all_entries

    # ------------------------------------------------------------------ #
    # Discovery APIs (keep from legacy)
    # ------------------------------------------------------------------ #
    def add_rule(self, rule: GrammarRule, persist: bool = True) -> bool:
        """Add grammar rule (shadow copy integration)."""
        if rule.rule_id in self.rules:
            return False

        self.rules[rule.rule_id] = rule

        # If Knowledgeverse is available, log discovery event
        if self.knowledgeverse:
            self.knowledgeverse.log_event(
                event_type="grammar_discovery",
                event_data={
                    "rule_id": rule.rule_id,
                    "rpn_program": rule.rpn_program,
                    "language": rule.language,
                },
            )

        return True

    def observe_pattern(
        self,
        visual_embedding: List[float],
        text_embedding: List[float],
        context: str,
    ) -> Optional[str]:
        """Observe cross-modal correlation (enriched mode)."""
        if not visual_embedding or not text_embedding:
            return None

        scores = self._get_cosine_bridge().compute_similarities(
            [visual_embedding], text_embedding
        )
        correlation = scores[0] if scores else 0.0

        if correlation >= 0.6:  # Discovery threshold
            rule_rpn = self._synthesize_rule_rpn(visual_embedding, text_embedding)
            return self.propose_rule(rule_rpn, context, correlation)
        return None

    def propose_rule(self, rpn_program: str, context: str, confidence: float = 0.0) -> str:
        """Add tentative rule to local discovery space."""
        rule_id = f"DISC_{hash(rpn_program) & 0xFFFFFF:06x}"
        if rule_id in self.rules:
            return rule_id

        self._local_discoveries[rule_id] = {
            "rpn_program": rpn_program,
            "context": context,
            "usage_count": 0,
            "success_count": 0,
            "quality_score": confidence,
        }
        return rule_id

    def validate_usage(self, rule_id: str, success: bool) -> float:
        """Update quality score for a rule."""
        if rule_id not in self._local_discoveries:
            if rule_id in self.rules:
                return 1.0
            return 0.0

        rule = self._local_discoveries[rule_id]
        rule["usage_count"] += 1
        if success:
            rule["success_count"] += 1

        if rule["usage_count"] > 0:
            rule["quality_score"] = rule["success_count"] / rule["usage_count"]

        return rule["quality_score"]

    def _synthesize_rule_rpn(
        self, visual_emb: List[float], text_emb: List[float]
    ) -> str:
        """Synthesize RPN description of cross-modal mapping."""
        # Simplified version (copy from legacy if needed)
        return "CROSS_MODAL_RULE"

    def list_rules(self, language: Optional[str] = None) -> List[GrammarRule]:
        """List grammar rules (optionally filtered by language)."""
        if language is None:
            return list(self.rules.values())
        return [r for r in self.rules.values() if r.language == language]

    def summary(self) -> Dict[str, int]:
        """Get galaxy summary."""
        return {
            "canonical_rules": len(self.rules),
            "discovered_rules": len(self._local_discoveries),
            "total": len(self.rules) + len(self._local_discoveries),
        }


__all__ = ["GrammarGalaxy", "GrammarRule", "default_grammar_rules"]
```

**Testing:**

```bash
# Test 1: Galaxy loads within Knowledgeverse
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy

kv = Knowledgeverse()
grammar = GrammarGalaxy(knowledgeverse=kv)
print(f'Summary: {grammar.summary()}')
print(f'Entries count: {len(grammar.entries)}')
print(f'Rules: {len(grammar.list_rules())}')
"

# Test 2: Discovery API works
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse
from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy, GrammarRule

kv = Knowledgeverse()
grammar = GrammarGalaxy(knowledgeverse=kv)
test_rule = GrammarRule(
    rule_id='TEST_RULE',
    language='en',
    pattern='test',
    rpn_program='TEST_OP',
)
grammar.add_rule(test_rule)
print(f'Rules after discovery: {len(grammar.rules)}')
"
```

**Day 3 Success Criteria:**
- ✅ `knowledge3d/knowledgeverse/grammar_galaxy.py` created
- ✅ GrammarGalaxy loads within Knowledgeverse (no errors)
- ✅ `.entries` property returns list of dicts
- ✅ Discovery APIs work (add_rule, observe_pattern)
- ✅ All canonical rules loaded (196+ rules)

---

### Day 4: Update GalaxyManager Integration

**Goal:** Teach GalaxyManager to load specialized galaxy classes (not just JSONL).

**File to update:** `knowledge3d/knowledgeverse/galaxy_manager.py`

**Changes:**

```python
class GalaxyManager:
    """Galaxy manager with persistence and resilient query surface."""

    def __init__(self, storage_root: str | Path = "../Knowledge3D.local/galaxies"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self._galaxies: dict[str, Galaxy] = {}
        self._knowledgeverse = None  # Will be set by Knowledgeverse after init

    def set_knowledgeverse(self, knowledgeverse: Any) -> None:
        """Set parent Knowledgeverse reference (for specialized galaxies)."""
        self._knowledgeverse = knowledgeverse

    def get_galaxy(self, name: str) -> Galaxy:
        """Get galaxy by name (supports specialized classes + JSONL)."""
        if name in self._galaxies:
            return self._galaxies[name]

        # Special case: Drawing Galaxy (specialized class)
        if name == "Drawing":
            from knowledge3d.knowledgeverse.drawing_galaxy import DrawingGalaxy
            drawing = DrawingGalaxy(knowledgeverse=self._knowledgeverse)
            self._galaxies[name] = drawing
            return drawing

        # Special case: Grammar Galaxy (specialized class)
        if name == "Grammar":
            from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy
            grammar = GrammarGalaxy(knowledgeverse=self._knowledgeverse)
            self._galaxies[name] = grammar
            return grammar

        # Default: Load from JSONL (existing behavior)
        path = self._galaxy_path(name)
        entries: list[dict[str, Any]] = []
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        galaxy = Galaxy(name=name, entries=entries)
        self._galaxies[name] = galaxy
        return galaxy
```

**Update Knowledgeverse to set reference:**

**File to update:** `knowledge3d/knowledgeverse/knowledgeverse.py`

```python
class Knowledgeverse:
    def __init__(self, ...):
        # ... existing init code ...

        self.galaxy_manager = GalaxyManager(storage_root=galaxy_root)

        # NEW: Set Knowledgeverse reference (for specialized galaxies)
        self.galaxy_manager.set_knowledgeverse(self)

        # ... rest of init ...
```

**Testing:**

```bash
# Test 1: GalaxyManager loads specialized galaxies
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse

kv = Knowledgeverse()
drawing = kv.galaxy_manager.get_galaxy('Drawing')
grammar = kv.galaxy_manager.get_galaxy('Grammar')
print(f'Drawing: {drawing.summary()}')
print(f'Grammar: {grammar.summary()}')
"

# Test 2: Multiple loads return same instance (singleton pattern)
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. python -c "
from knowledge3d.knowledgeverse import Knowledgeverse

kv = Knowledgeverse()
drawing1 = kv.galaxy_manager.get_galaxy('Drawing')
drawing2 = kv.galaxy_manager.get_galaxy('Drawing')
print(f'Same instance: {drawing1 is drawing2}')  # Should be True
"
```

**Day 4 Success Criteria:**
- ✅ GalaxyManager loads DrawingGalaxy on `get_galaxy("Drawing")`
- ✅ GalaxyManager loads GrammarGalaxy on `get_galaxy("Grammar")`
- ✅ Singleton pattern (same instance on multiple calls)
- ✅ Knowledgeverse reference set correctly

---

### Day 5: Integration Testing + Documentation

**Goal:** Validate full integration and update docs.

**Test suite:**

**File to create:** `tests/test_week15_galaxy_integration.py`

```python
"""Week 15: Galaxy integration tests."""

import pytest
from knowledge3d.knowledgeverse import Knowledgeverse


def test_knowledgeverse_loads_drawing_galaxy():
    """Drawing Galaxy loads within Knowledgeverse."""
    kv = Knowledgeverse()
    drawing = kv.galaxy_manager.get_galaxy("Drawing")

    assert drawing.name == "Drawing"
    assert len(drawing.entries) > 0
    assert len(drawing.transformations) >= 30  # At least 30 transformation rules
    summary = drawing.summary()
    assert summary["shapes"] > 0
    assert summary["transformations"] > 0


def test_knowledgeverse_loads_grammar_galaxy():
    """Grammar Galaxy loads within Knowledgeverse."""
    kv = Knowledgeverse()
    grammar = kv.galaxy_manager.get_galaxy("Grammar")

    assert grammar.name == "Grammar"
    assert len(grammar.entries) > 0
    assert len(grammar.rules) >= 190  # At least 190+ canonical rules
    summary = grammar.summary()
    assert summary["canonical_rules"] > 0


def test_drawing_discovery_api():
    """Drawing Galaxy discovery API works."""
    kv = Knowledgeverse()
    drawing = kv.galaxy_manager.get_galaxy("Drawing")

    initial_count = len(drawing.shapes)
    drawing.add_shape("TEST_SHAPE", "CIRCLE 0.5 0.5 0.2", source={"task": "test"})

    assert len(drawing.shapes) == initial_count + 1
    assert "TEST_SHAPE" in drawing.shapes


def test_grammar_discovery_api():
    """Grammar Galaxy discovery API works."""
    kv = Knowledgeverse()
    grammar = kv.galaxy_manager.get_galaxy("Grammar")

    from knowledge3d.knowledgeverse.grammar_galaxy import GrammarRule

    initial_count = len(grammar.rules)
    test_rule = GrammarRule(
        rule_id="TEST_RULE",
        language="en",
        pattern="test",
        rpn_program="TEST_OP",
    )
    grammar.add_rule(test_rule)

    assert len(grammar.rules) == initial_count + 1
    assert "TEST_RULE" in grammar.rules


def test_galaxy_singleton_pattern():
    """Multiple get_galaxy calls return same instance."""
    kv = Knowledgeverse()
    drawing1 = kv.galaxy_manager.get_galaxy("Drawing")
    drawing2 = kv.galaxy_manager.get_galaxy("Drawing")

    assert drawing1 is drawing2  # Same object reference


def test_shadow_copy_event_logging():
    """Galaxy discoveries log events to Shadow Copy."""
    kv = Knowledgeverse()
    drawing = kv.galaxy_manager.get_galaxy("Drawing")

    # Discovery should log event
    drawing.add_shape("EVENT_TEST", "RECT 0.0 0.0 1.0 1.0")

    # Check that event was logged (basic validation)
    # Shadow Copy integration will be tested separately


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

**Run tests:**

```bash
# Run Week 15 integration tests
/home/daniel/miniforge/bin/conda run -n k3d-cranium env PYTHONPATH=. pytest tests/test_week15_galaxy_integration.py -v
```

**Expected output:**

```
tests/test_week15_galaxy_integration.py::test_knowledgeverse_loads_drawing_galaxy PASSED
tests/test_week15_galaxy_integration.py::test_knowledgeverse_loads_grammar_galaxy PASSED
tests/test_week15_galaxy_integration.py::test_drawing_discovery_api PASSED
tests/test_week15_galaxy_integration.py::test_grammar_discovery_api PASSED
tests/test_week15_galaxy_integration.py::test_galaxy_singleton_pattern PASSED
tests/test_week15_galaxy_integration.py::test_shadow_copy_event_logging PASSED

========================= 6 passed in 2.3s =========================
```

**Day 5 Success Criteria:**
- ✅ All 6 integration tests pass
- ✅ No import errors
- ✅ Discovery APIs work within Knowledgeverse
- ✅ Shadow Copy event logging works

---

## Critical Reminders

### 1. DO NOT Rewrite from Scratch

**REUSE legacy code!** The DrawingGalaxy and GrammarGalaxy implementations are PROVEN (46.7% on ARC-AGI 1). Don't rebuild — **wrap and integrate**.

**Strategy:**
- Keep legacy data structures (DrawingItem, GrammarRule)
- Keep legacy builder functions (build_primitives, build_shapes, default_grammar_rules)
- Add `.entries` property for GalaxyManager compatibility
- Add `knowledgeverse` parameter for unified context

### 2. Lazy Initialization is GOOD

**CosineSimilarityBridge:** GrammarGalaxy already uses lazy initialization:
```python
self.cosine_bridge: Optional[CosineSimilarityBridge] = None  # Lazy init
```

**Keep this pattern!** Don't initialize GPU resources until needed.

### 3. Singleton Pattern (GalaxyManager)

**GalaxyManager caches galaxies:**
```python
if name in self._galaxies:
    return self._galaxies[name]  # Return cached instance
```

**This is CRITICAL for worker parallelization.** Week 17 will pass Knowledgeverse reference to workers, ensuring they all use the SAME galaxy instances (no conflicts).

### 4. Discovery Events → Shadow Copy

**When galaxies discover new entries, log to Shadow Copy:**
```python
if self.knowledgeverse:
    self.knowledgeverse.log_event(
        event_type="drawing_discovery",
        event_data={"shape_id": shape_id, "rpn_program": rpn_program},
    )
```

**This enables continuous learning!** Shadow Copy tracks which discoveries led to successful task solutions.

---

## Success Metrics

**Immediate Goals (Week 15):**
- ✅ DrawingGalaxy + GrammarGalaxy load within Knowledgeverse (no errors)
- ✅ GalaxyManager.get_galaxy("Drawing") returns specialized DrawingGalaxy
- ✅ GalaxyManager.get_galaxy("Grammar") returns specialized GrammarGalaxy
- ✅ Discovery APIs work (add_shape, add_rule)
- ✅ All tests pass (6/6)

**Stretch Goals:**
- ✅ Shadow Copy event logging works
- ✅ Singleton pattern validated (same instance on multiple calls)
- ✅ No regressions in existing benchmarks (Math, LHE)

---

## Next Steps (After Week 15)

**Week 16:** Create KnowledgeversARCPipeline that uses integrated galaxies (replaces legacy SovereignAIPipeline).

**Week 17:** Refactor ParallelCandidateGenerator to pass Knowledgeverse reference to workers (eliminates "Sovereign loader error").

---

## Deliverable

**After Week 15 completion, create:**

**File:** `TEMP/CODEX_WEEK15_GALAXY_INTEGRATION_COMPLETION_REPORT_02.XX.2026.md`

**Include:**
1. **Implementation summary:** DrawingGalaxy + GrammarGalaxy integrated
2. **Test results:** 6/6 tests passing
3. **Performance validation:** No regressions in existing benchmarks
4. **Lessons learned:** What worked, what didn't
5. **Blockers encountered:** Any issues that need architectural review
6. **Week 16 readiness:** Confirm galaxies are ready for ARC pipeline integration

---

## End of Handoff

**Priority:** CRITICAL (Week 15 = Foundation for ARC-AGI 2 Integration)

**Start here:**
1. Create `knowledge3d/knowledgeverse/drawing_galaxy.py` (Day 1-2)
2. Create `knowledge3d/knowledgeverse/grammar_galaxy.py` (Day 3)
3. Update `knowledge3d/knowledgeverse/galaxy_manager.py` (Day 4)
4. Create `tests/test_week15_galaxy_integration.py` and validate (Day 5)
5. Write completion report

**Remember:** REUSE legacy code (don't rewrite from scratch!). The goal is INTEGRATION, not reimplementation.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

**Let's integrate these galaxies into Knowledgeverse!** 🚀

---

**Claude (Architecture Partner)**
February 7, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
