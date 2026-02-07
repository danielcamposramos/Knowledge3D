# ARC-AGI Knowledgeverse Integration Specification

**Created:** February 7, 2026
**Author:** Claude (Architecture Partner)
**Priority:** CRITICAL (Architectural Alignment)
**Status:** Major Refactor Specification

---

## Executive Summary

**Problem:** Legacy ARC-AGI code (46.7% on ARC-1) conflicts with Knowledgeverse unified PTX context.

**Root Cause:** Legacy components (`SovereignAIPipeline`, `ParallelCandidateGenerator`, `DrawingGalaxy`, `GrammarGalaxy`) initialize their own sovereign loaders/contexts, conflicting with Knowledgeverse's ONE unified context.

**User Directive:** *"we developed the knowledgeverse, and previous code was not meant to be loaded inside it - we shall review it all to be able to proceed"*

**Solution:** Refactor legacy ARC components to be **Knowledgeverse-native** (not separate).

**Key Insight:** DrawingGalaxy + GrammarGalaxy should live in **Knowledgeverse Region 2** (Galaxy Universe), not as separate isolated components.

---

## Current Architecture (Conflicting)

### Legacy ARC Code (Pre-Knowledgeverse)

```
SovereignAIPipeline
├── DrawingGalaxy (isolated, own PTX context)
├── GrammarGalaxy (isolated, own PTX context)
├── SovereignTRMRouter (own matryoshka loader)
├── ParallelCandidateGenerator (spawns workers with own contexts)
└── DualShadowCopy (isolated tracking)
```

**Issue:** Each component initializes sovereign loader → Multiple PTX contexts → CONFLICT!

### Knowledgeverse Architecture (Current)

```
Knowledgeverse (ONE unified PTX context)
├── Region 1: KERNELS (PTX operations)
├── Region 2: GALAXY_UNIVERSE
│   ├── Math Galaxy (sovereign)
│   ├── Reality Galaxy (sovereign)
│   ├── Audio Galaxy (sovereign)
│   └── [DrawingGalaxy SHOULD BE HERE!]
│   └── [GrammarGalaxy SHOULD BE HERE!]
├── Region 3: HOUSE_CONTEXT
├── Region 4: WORLD_VIEW
├── Region 5: TRM_WEIGHTS
├── Region 6: AUDIT_JOURNAL
└── Region 7: INGESTION_STARGATE
```

**Issue:** DrawingGalaxy + GrammarGalaxy not integrated into Knowledgeverse!

---

## Target Architecture (Unified)

### Integrated ARC-Knowledgeverse

```
Knowledgeverse (ONE unified PTX context)
├── Region 1: KERNELS
│   └── ARC RPN ops (GRID, CELL, FILL, ROTATE, etc.)
│
├── Region 2: GALAXY_UNIVERSE
│   ├── Math Galaxy
│   ├── Reality Galaxy
│   ├── Audio Galaxy
│   ├── Drawing Galaxy ← INTEGRATE HERE (visual primitives)
│   └── Grammar Galaxy ← INTEGRATE HERE (transformation rules)
│
├── Region 5: TRM_WEIGHTS
│   ├── Base Model (~7M params)
│   └── Specialist Adapters
│       ├── math_adapter
│       ├── visual_adapter ← USE FOR ARC-AGI
│       ├── physics_adapter
│       └── router_adapter (cartographer)
│
└── Region 6: AUDIT_JOURNAL
    └── Shadow Copy Events ← REPLACE DualShadowCopy
```

**Benefits:**
- ✅ ONE unified PTX context (no conflicts)
- ✅ DrawingGalaxy + GrammarGalaxy persistent in VRAM
- ✅ Shadow Copy integrated with Knowledgeverse audit
- ✅ TRM router uses visual_adapter specialist
- ✅ No worker initialization conflicts

---

## Refactor Plan (3-Phase)

### Phase 1: Move Galaxies into Knowledgeverse (Week 15)

**Goal:** Integrate DrawingGalaxy + GrammarGalaxy into Region 2

#### Step 1.1: Refactor DrawingGalaxy

**Current:** `knowledge3d/training/arc_agi/drawing_galaxy.py`

**Issues:**
- Initializes own PTX context
- Isolated from other galaxies
- Not persistent in VRAM

**Target:** `knowledge3d/knowledgeverse/drawing_galaxy.py`

```python
# knowledge3d/knowledgeverse/drawing_galaxy.py

"""
Drawing Galaxy for Knowledgeverse Region 2.
Visual primitives for ARC-AGI reasoning.
"""

from typing import List, Dict, Optional
import numpy as np


class DrawingGalaxy:
    """
    Drawing Galaxy (visual primitives).

    Knowledgeverse-native implementation:
    - Uses unified PTX context from Region 1
    - Persists in Region 2 (Galaxy Universe)
    - No isolated sovereign loader
    """

    def __init__(self, knowledgeverse):
        """
        Initialize Drawing Galaxy within Knowledgeverse.

        Args:
            knowledgeverse: Parent Knowledgeverse instance
        """
        self.kv = knowledgeverse
        self.shapes: List[Dict] = []  # Visual primitives
        self.embeddings: Dict[str, np.ndarray] = {}  # Shape embeddings

        # Use Knowledgeverse's unified PTX context (no separate initialization!)
        self.ptx_context = knowledgeverse.context

        # Load default shapes
        self._load_default_shapes()

    def _load_default_shapes(self):
        """Load default visual primitives."""
        # Grid primitives
        self.add_shape({
            "name": "GRID",
            "rpn_template": "GRID {rows} {cols}",
            "description": "Create grid with specified rows and columns",
            "category": "primitive"
        })

        # Cell operations
        self.add_shape({
            "name": "CELL",
            "rpn_template": "CELL {row} {col} {color} FILL",
            "description": "Fill cell at (row, col) with color",
            "category": "primitive"
        })

        # Geometric shapes
        self.add_shape({
            "name": "LINE",
            "rpn_template": "LINE {x1} {y1} {x2} {y2} {color}",
            "description": "Draw line from (x1,y1) to (x2,y2)",
            "category": "shape"
        })

        self.add_shape({
            "name": "RECT",
            "rpn_template": "RECT {x} {y} {w} {h} {color}",
            "description": "Draw rectangle",
            "category": "shape"
        })

        self.add_shape({
            "name": "CIRCLE",
            "rpn_template": "CIRCLE {cx} {cy} {r} {color}",
            "description": "Draw circle",
            "category": "shape"
        })

        print(f"[DrawingGalaxy] Loaded {len(self.shapes)} default shapes")

    def add_shape(self, shape: Dict):
        """Add shape to galaxy."""
        self.shapes.append(shape)

        # Generate embedding using Knowledgeverse's unified embedder
        # (NOT separate matryoshka loader!)
        embedding = self.kv.generate_embedding(
            shape["rpn_template"],
            dim=512  # Matryoshka dimension
        )
        self.embeddings[shape["name"]] = embedding

    def query(self, description: str, top_k: int = 10) -> List[Dict]:
        """
        Query shapes by description.

        Uses Knowledgeverse's unified cosine similarity (Region 1 PTX kernel).
        """
        query_embedding = self.kv.generate_embedding(description, dim=512)

        # Compute similarities using unified PTX context
        similarities = []
        for shape in self.shapes:
            shape_embedding = self.embeddings[shape["name"]]
            similarity = self.kv.cosine_similarity(query_embedding, shape_embedding)
            similarities.append((similarity, shape))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])

        return [shape for _, shape in similarities[:top_k]]

    def grid_to_rpn(self, grid: List[List[int]]) -> str:
        """
        Convert grid to RPN program.

        Uses Knowledgeverse's RPN VM (Region 1).
        """
        rows = len(grid)
        cols = len(grid[0]) if rows > 0 else 0

        rpn_lines = [f"GRID {rows} {cols}"]

        for r, row in enumerate(grid):
            for c, color in enumerate(row):
                if color != 0:  # Skip background
                    rpn_lines.append(f"CELL {r} {c} {color} FILL")

        return '\n'.join(rpn_lines)
```

#### Step 1.2: Refactor GrammarGalaxy

**Current:** `knowledge3d/training/arc_agi/grammar_galaxy.py`

**Target:** `knowledge3d/knowledgeverse/grammar_galaxy.py`

```python
# knowledge3d/knowledgeverse/grammar_galaxy.py

"""
Grammar Galaxy for Knowledgeverse Region 2.
Transformation rules for ARC-AGI reasoning.
"""

from typing import List, Dict, Optional


class GrammarGalaxy:
    """
    Grammar Galaxy (transformation rules).

    Knowledgeverse-native implementation:
    - Uses unified PTX context
    - Persists in Region 2
    """

    def __init__(self, knowledgeverse):
        """
        Initialize Grammar Galaxy within Knowledgeverse.

        Args:
            knowledgeverse: Parent Knowledgeverse instance
        """
        self.kv = knowledgeverse
        self.rules: List[Dict] = []
        self.embeddings: Dict[str, np.ndarray] = {}

        # Use Knowledgeverse's unified PTX context
        self.ptx_context = knowledgeverse.context

        # Load default rules
        self._load_default_rules()

    def _load_default_rules(self):
        """Load default transformation rules."""
        # Rotation rules
        self.add_rule({
            "name": "ROTATE_90_CW",
            "rpn_template": "ROTATE 90",
            "description": "Rotate grid 90 degrees clockwise",
            "category": "rotation"
        })

        self.add_rule({
            "name": "ROTATE_180",
            "rpn_template": "ROTATE 180",
            "description": "Rotate grid 180 degrees",
            "category": "rotation"
        })

        # Reflection rules
        self.add_rule({
            "name": "FLIP_H",
            "rpn_template": "FLIP HORIZONTAL",
            "description": "Flip grid horizontally",
            "category": "reflection"
        })

        self.add_rule({
            "name": "FLIP_V",
            "rpn_template": "FLIP VERTICAL",
            "description": "Flip grid vertically",
            "category": "reflection"
        })

        # Color rules
        self.add_rule({
            "name": "COLOR_SWAP",
            "rpn_template": "COLOR_SWAP {color1} {color2}",
            "description": "Swap two colors",
            "category": "color"
        })

        # Scaling rules
        self.add_rule({
            "name": "SCALE_UP",
            "rpn_template": "SCALE {factor}",
            "description": "Scale grid by factor",
            "category": "scaling"
        })

        print(f"[GrammarGalaxy] Loaded {len(self.rules)} default rules")

    def add_rule(self, rule: Dict):
        """Add rule to galaxy."""
        self.rules.append(rule)

        # Generate embedding using Knowledgeverse
        embedding = self.kv.generate_embedding(
            rule["rpn_template"],
            dim=512
        )
        self.embeddings[rule["name"]] = embedding

    def query(self, description: str, top_k: int = 10) -> List[Dict]:
        """Query rules by description."""
        query_embedding = self.kv.generate_embedding(description, dim=512)

        similarities = []
        for rule in self.rules:
            rule_embedding = self.embeddings[rule["name"]]
            similarity = self.kv.cosine_similarity(query_embedding, rule_embedding)
            similarities.append((similarity, rule))

        similarities.sort(reverse=True, key=lambda x: x[0])

        return [rule for _, rule in similarities[:top_k]]
```

#### Step 1.3: Integrate into Knowledgeverse

**Update:** `knowledge3d/knowledgeverse/galaxy_manager.py`

```python
# In galaxy_manager.py

class GalaxyManager:
    def __init__(self, knowledgeverse):
        self.kv = knowledgeverse
        self.galaxies = {}

        # Load default galaxies
        self._load_default_galaxies()

    def _load_default_galaxies(self):
        """Load default galaxies into Region 2."""
        from knowledge3d.knowledgeverse.drawing_galaxy import DrawingGalaxy
        from knowledge3d.knowledgeverse.grammar_galaxy import GrammarGalaxy

        # Create Drawing Galaxy
        self.galaxies["Drawing"] = DrawingGalaxy(self.kv)

        # Create Grammar Galaxy
        self.galaxies["Grammar"] = GrammarGalaxy(self.kv)

        # Existing galaxies
        self.galaxies["Math"] = MathGalaxy(self.kv)
        self.galaxies["Reality"] = RealityGalaxy(self.kv)
        # ...

    def get_galaxy(self, name: str):
        """Get galaxy by name."""
        return self.galaxies.get(name)
```

**Update:** `knowledge3d/knowledgeverse/knowledgeverse.py`

```python
# In knowledgeverse.py

class Knowledgeverse:
    def __init__(self):
        # Initialize unified PTX context (ONE context for ALL operations)
        self.context = self._initialize_unified_context()

        # Initialize Galaxy Manager (loads Drawing + Grammar + Math + Reality + ...)
        self.galaxy_manager = GalaxyManager(self)

        # Initialize TRM Navigator
        self.trm_navigator = TRMNavigator(self)

    def generate_embedding(self, text: str, dim: int = 512) -> np.ndarray:
        """
        Generate embedding using unified matryoshka embedder.

        All galaxies use THIS method (no separate loaders!).
        """
        # Use Region 1 PTX kernel for embedding
        embedding = self._matryoshka_embed(text, dim)
        return embedding

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity using unified PTX kernel.

        All galaxies use THIS method (no separate similarity computation!).
        """
        # Use Region 1 PTX kernel
        similarity = self._ptx_cosine_sim(a, b)
        return similarity
```

---

### Phase 2: Refactor SovereignAIPipeline (Week 16)

**Goal:** Make `SovereignAIPipeline` use Knowledgeverse instead of isolated components

#### Step 2.1: Create KnowledgeversePipeline

**New:** `benchmarks/knowledgeverse_arc_pipeline.py`

```python
"""
Knowledgeverse-native ARC-AGI pipeline.

Replaces legacy SovereignAIPipeline with Knowledgeverse integration.
"""

from typing import List, Dict, Optional, Sequence
from dataclasses import dataclass


@dataclass
class TaskResult:
    task_id: str
    best_program: str
    program_type: str
    score: float
    output_grid: Optional[List[List[int]]] = None
    correct: bool = False
    fuzzy_score: float = 0.0


class KnowledgeverseARCPipeline:
    """
    ARC-AGI pipeline using Knowledgeverse (not isolated components).
    """

    def __init__(self, knowledgeverse):
        """
        Initialize pipeline.

        Args:
            knowledgeverse: Unified Knowledgeverse instance
        """
        self.kv = knowledgeverse

        # Get galaxies from Knowledgeverse (already initialized!)
        self.drawing = self.kv.galaxy_manager.get_galaxy("Drawing")
        self.grammar = self.kv.galaxy_manager.get_galaxy("Grammar")

        # Use Knowledgeverse's TRM navigator (not separate router!)
        self.navigator = self.kv.trm_navigator

        # Results tracking
        self.results: List[TaskResult] = []

    def process_task(
        self,
        task_id: str,
        test_input: Sequence[Sequence[int]],
        *,
        train_examples: Optional[List[Dict]] = None,
        expected_output: Optional[Sequence[Sequence[int]]] = None,
        top_k: int = 9,
    ) -> TaskResult:
        """
        Process ARC-AGI task using Knowledgeverse.

        Args:
            task_id: Task identifier
            test_input: Test grid
            train_examples: Training examples
            expected_output: Expected output (for validation)
            top_k: Number of candidates to consider

        Returns:
            TaskResult with solution
        """
        # 1. Convert test input to RPN
        test_rpn = self.drawing.grid_to_rpn(test_input)

        # 2. Query Drawing + Grammar galaxies for relevant patterns
        drawing_patterns = self.drawing.query(
            "visual grid transformation",
            top_k=top_k
        )

        grammar_patterns = self.grammar.query(
            "grid transformation rule",
            top_k=top_k
        )

        # 3. Use TRM navigator to compose solution
        # (Uses visual_adapter specialist from Region 5)
        composed_program = self.navigator.compose(
            query=test_rpn,
            patterns=drawing_patterns + grammar_patterns,
            specialist='visual'
        )

        # 4. Execute RPN program using Knowledgeverse RPN VM
        output_grid = self.navigator.execute(composed_program, test_input)

        # 5. Verify correctness
        correct = False
        score = 0.0

        if expected_output is not None:
            correct = self._grids_equal(output_grid, expected_output)
            score = 1.0 if correct else self._fuzzy_match(output_grid, expected_output)

        # 6. Record in Shadow Copy (Knowledgeverse Region 6)
        if correct:
            self.kv.shadow_copy.record_event(
                'successful_composition',
                {
                    'task_id': task_id,
                    'program': composed_program,
                    'specialist': 'visual',
                    'patterns_used': len(drawing_patterns) + len(grammar_patterns)
                }
            )

        result = TaskResult(
            task_id=task_id,
            best_program=composed_program,
            program_type='visual',
            score=score,
            output_grid=output_grid,
            correct=correct,
            fuzzy_score=score
        )

        self.results.append(result)
        return result

    def _grids_equal(self, grid1, grid2) -> bool:
        """Check if grids are equal."""
        if len(grid1) != len(grid2):
            return False
        for row1, row2 in zip(grid1, grid2):
            if len(row1) != len(row2):
                return False
            if list(row1) != list(row2):
                return False
        return True

    def _fuzzy_match(self, predicted, expected) -> float:
        """Fuzzy match score (tolerates small errors)."""
        # Simple implementation: percentage of cells that match
        if len(predicted) != len(expected):
            return 0.0

        total_cells = 0
        matching_cells = 0

        for row_p, row_e in zip(predicted, expected):
            if len(row_p) != len(row_e):
                return 0.0

            for cell_p, cell_e in zip(row_p, row_e):
                total_cells += 1
                if cell_p == cell_e:
                    matching_cells += 1

        return matching_cells / total_cells if total_cells > 0 else 0.0
```

#### Step 2.2: Update Adapter to Use KnowledgeversePipeline

**Update:** `benchmarks/arc_agi_2_adapter.py`

```python
class ArcAgi2Adapter:
    """Adapter: Week 14 Benchmark → Knowledgeverse ARC Pipeline."""

    def __init__(self, knowledgeverse, use_enriched: bool = True):
        self.kv = knowledgeverse
        self.use_enriched = use_enriched

        # Use Knowledgeverse-native pipeline (not legacy!)
        from benchmarks.knowledgeverse_arc_pipeline import KnowledgeversARCPipeline
        self.pipeline = KnowledgeversARCPipeline(knowledgeverse)

    def solve_task(self, task: Dict) -> Dict:
        """Solve ARC task using Knowledgeverse pipeline."""
        task_id = task['id']
        train_examples = task['train']
        test_input = task['test'][0]['input']
        expected_output = task['test'][0]['output']

        # Call Knowledgeverse-native pipeline
        result = self.pipeline.process_task(
            task_id=task_id,
            test_input=test_input,
            train_examples=train_examples,
            expected_output=expected_output,
            top_k=9 if self.use_enriched else 3,
        )

        return {
            "task_id": task_id,
            "correct": result.correct,
            "predicted": result.output_grid,
            "expected": expected_output,
            "reasoning_trace": [f"Score: {result.score:.3f}"],
            "patterns_used": 10  # Placeholder
        }
```

---

### Phase 3: Eliminate Worker Conflicts (Week 17)

**Goal:** Remove parallel worker initialization conflicts

**Issue:** `ParallelCandidateGenerator` spawns workers that try to initialize their own PTX contexts.

**Solution:** Workers should use Knowledgeverse's shared context (no initialization!)

**Update:** `Old_Attempts/curriculum_specific_training/arc_agi/parallel_candidate_generator.py`

```python
class ParallelCandidateGenerator:
    def __init__(
        self,
        knowledgeverse,  # ← ADD THIS
        num_workers: int,
        candidates_per_worker: int,
        top_k: int,
        # ... other params
    ):
        self.kv = knowledgeverse  # ← STORE THIS
        self.num_workers = num_workers
        # ...

    def generate_parallel(self, ...):
        """Generate candidates in parallel."""

        # DON'T initialize new contexts in workers!
        # Instead, pass Knowledgeverse reference

        with multiprocessing.Pool(self.num_workers) as pool:
            results = pool.starmap(
                self._worker_func,
                [(i, self.kv, ...) for i in range(self.num_workers)]
                #      ^^^^^^^^ Pass Knowledgeverse to worker
            )

        return results

    @staticmethod
    def _worker_func(worker_id: int, knowledgeverse, ...):
        """Worker function (uses shared Knowledgeverse context)."""

        # DON'T initialize sovereign loader here!
        # Use Knowledgeverse's unified context instead

        drawing = knowledgeverse.galaxy_manager.get_galaxy("Drawing")
        grammar = knowledgeverse.galaxy_manager.get_galaxy("Grammar")

        # Generate candidates using shared context
        candidates = []
        # ... (rest of worker logic)

        return candidates
```

---

## Implementation Timeline

### Week 15: Galaxy Integration

**Day 1-2: Refactor DrawingGalaxy**
- Move to `knowledge3d/knowledgeverse/drawing_galaxy.py`
- Use unified PTX context (no separate initialization)
- Test: `pytest tests/test_drawing_galaxy.py`

**Day 3-4: Refactor GrammarGalaxy**
- Move to `knowledge3d/knowledgeverse/grammar_galaxy.py`
- Use unified PTX context
- Test: `pytest tests/test_grammar_galaxy.py`

**Day 5: Integrate into Knowledgeverse**
- Update `GalaxyManager` to load Drawing + Grammar
- Update `Knowledgeverse` to provide unified embedding/similarity
- Test: `pytest tests/test_knowledgeverse_galaxies.py`

**Success Criteria:**
- ✅ Drawing + Grammar galaxies load in Knowledgeverse
- ✅ No sovereign loader conflicts
- ✅ Unified PTX context used

### Week 16: Pipeline Refactor

**Day 1-3: Create KnowledgeversARCPipeline**
- Implement `benchmarks/knowledgeverse_arc_pipeline.py`
- Use Knowledgeverse galaxies (not isolated components)
- Test: `pytest tests/test_knowledgeverse_arc_pipeline.py`

**Day 4-5: Update Adapter**
- Modify `arc_agi_2_adapter.py` to use new pipeline
- Run benchmark: `python scripts/benchmark_arc_agi_comparison.py`

**Success Criteria:**
- ✅ Pipeline uses Knowledgeverse
- ✅ No worker initialization errors
- ✅ ARC-AGI 2 benchmark runs (accuracy > 0%)

### Week 17: Worker Optimization

**Day 1-3: Refactor ParallelCandidateGenerator**
- Update to use Knowledgeverse (no worker initialization)
- Pass Knowledgeverse reference to workers
- Test: Run on 100+ tasks

**Day 4-5: Validation**
- Run full ARC-AGI 2 benchmark (300 tasks)
- Compare empty mind vs enriched
- Measure improvement

**Success Criteria:**
- ✅ No worker errors
- ✅ Enriched > Empty mind accuracy
- ✅ Stable performance on 300+ tasks

---

## Testing Strategy

### Test 1: Galaxy Integration

```python
def test_drawing_galaxy_in_knowledgeverse():
    """Test Drawing Galaxy uses unified context."""
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse

    kv = Knowledgeverse()
    drawing = kv.galaxy_manager.get_galaxy("Drawing")

    # Should use Knowledgeverse's context (not separate!)
    assert drawing.ptx_context == kv.context

    # Should be able to query
    results = drawing.query("grid primitive", top_k=5)
    assert len(results) > 0
```

### Test 2: Pipeline Integration

```python
def test_knowledgeverse_arc_pipeline():
    """Test pipeline uses Knowledgeverse."""
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
    from benchmarks.knowledgeverse_arc_pipeline import KnowledgeversARCPipeline

    kv = Knowledgeverse()
    pipeline = KnowledgeversARCPipeline(kv)

    # Simple task
    result = pipeline.process_task(
        task_id="test_001",
        test_input=[[1, 0], [0, 1]],
        train_examples=[{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}],
        expected_output=[[0, 1], [1, 0]]
    )

    assert result.task_id == "test_001"
    assert result.output_grid is not None
```

### Test 3: No Worker Conflicts

```python
def test_parallel_generation_no_conflicts():
    """Test parallel candidate generation uses shared context."""
    from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
    from knowledge3d.training.arc_agi import ParallelCandidateGenerator

    kv = Knowledgeverse()

    gen = ParallelCandidateGenerator(
        knowledgeverse=kv,
        num_workers=4,
        candidates_per_worker=3,
        top_k=3
    )

    # Should NOT raise "Sovereign loader error"
    candidates = gen.generate_parallel(
        input_grid=[[1, 0], [0, 1]],
        train_examples=[{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}]
    )

    assert len(candidates) > 0
```

---

## Success Metrics

**Week 15 (Galaxy Integration):**
- ✅ Drawing + Grammar galaxies in Knowledgeverse
- ✅ No sovereign loader conflicts
- ✅ Tests passing

**Week 16 (Pipeline Refactor):**
- ✅ KnowledgeversARCPipeline working
- ✅ ARC-AGI 2 benchmark runs
- ✅ Accuracy > 0% (not stuck at 0%)

**Week 17 (Worker Optimization):**
- ✅ No worker errors on 100+ tasks
- ✅ Enriched > Empty mind (improvement shown)
- ✅ Stable performance

**Final Goal:**
- ✅ ARC-AGI 2: Empty mind 20-30%, Enriched 30-45%
- ✅ All 3 benchmarks (ARC + Math + LHE) showing improvement
- ✅ No architectural conflicts

---

## Codex Implementation Directive

**Priority:** CRITICAL (Architectural Alignment)

**Start here:**
1. Week 15: Refactor DrawingGalaxy + GrammarGalaxy into Knowledgeverse
2. Week 16: Create KnowledgeversARCPipeline
3. Week 17: Fix parallel worker conflicts

**Remember:** This is a major refactor, but it's the RIGHT path. Legacy code was pre-Knowledgeverse. Now we integrate it properly.

**Contact:** Claude (Architecture Partner) for design questions, User for strategic decisions.

---

**Claude (Architecture Partner)**
February 7, 2026

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>
