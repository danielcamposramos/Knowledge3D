# ARC-AGI Training Optimization Specification

**Date**: November 27, 2025
**Author**: Claude (Architecture Partner)
**Target**: Codex (Implementation Lead)
**Priority**: CRITICAL - Training stalled, immediate optimization required
**Status**: Ready for implementation

---

## Executive Summary

**Problem**: Library growth stalled at 52 programs for 5 consecutive runs (Runs 006-010)
**Root Cause**: System only generates primitive single-step programs, no compositional discovery
**Evidence**: GPU underutilized (1.12% avg), accuracy plateau (0-3.33%), zero library growth
**Solution**: Implement N-step compositional discovery + 9-core parallel generation

**Expected Impact**:
- Library: 52 → 100+ programs (compositions unlock complexity)
- Accuracy: 3.33% → 10%+ (complex patterns become solvable)
- Runtime: 16-24 min → 2-3 min per run (9× parallel speedup)
- GPU utilization: 1.12% → 10-15% (better resource usage)

---

## Optimization 1: N-Step Compositional Discovery (CRITICAL)

### Problem Statement

Current system generates only **single-step primitives**:
- rotate_90
- flip_horizontal
- recolor_to_5
- translate_1_0

**Missing**: Ability to **chain** discovered programs into multi-step compositions:
- rotate_90 → flip_horizontal
- recolor_to_5 → rotate_90 → flip_horizontal
- translate_1_0 → recolor_to_3 → rotate_180

**Evidence of stagnation**:
- Library stuck at 52 programs (all single-step primitives discovered)
- Accuracy cannot improve beyond 3.33% (complex tasks require multi-step reasoning)

### Architecture Design

#### Core Principle: Open-Ended Composition (N Steps)

**IMPORTANT**: Do NOT hardcode composition depth. System must explore N-step chains dynamically.

**Composition Strategy**:
1. Start with depth=2 (compose pairs of primitives)
2. If depth=2 compositions solve new tasks, continue to depth=3
3. If depth=3 compositions solve new tasks, continue to depth=4
4. Continue until diminishing returns (no new solutions for 2 consecutive depths)

**Why N-step matters**:
- Simple tasks: 2-step compositions (rotate → recolor)
- Medium tasks: 3-4 step compositions (flip → rotate → recolor → translate)
- Hard tasks: 5+ step compositions (complex object manipulations)

#### Implementation Structure

**File**: `knowledge3d/training/arc_agi/compositional_generator.py` (NEW)

**Class**: `CompositionalCandidateGenerator`

**Key Methods**:

```python
class CompositionalCandidateGenerator:
    """
    Generate N-step compositions of discovered programs.

    Composition strategy:
    1. Enumerate all valid K-step chains (K=2,3,4,...)
    2. Execute chains on input grids
    3. Score outputs against expected outputs
    4. Keep high-quality compositions (score >0.45)
    5. Add to candidate pool
    """

    def __init__(self, library, max_depth=6, beam_width=10):
        """
        Args:
            library: DualShadowCopy instance (discovered programs)
            max_depth: Maximum composition depth (default 6)
            beam_width: How many programs to consider at each depth (default 10)
        """
        self.library = library
        self.max_depth = max_depth
        self.beam_width = beam_width

    def generate_compositions(self, input_grid, expected_output, current_depth=2):
        """
        Generate N-step compositions using beam search.

        Algorithm:
        1. Start at depth=2 (pairs)
        2. Enumerate all pairs of top-quality programs
        3. Execute and score compositions
        4. If score >0.45, add to library and try depth+1
        5. If no high-quality compositions found, stop

        Returns:
            List of (composition, score, depth) tuples
        """
        compositions = []

        # Get top programs by quality score
        top_programs = self.library.get_top_k_programs(self.beam_width)

        # Iteratively deepen: try depth=2, 3, 4, ... up to max_depth
        for depth in range(2, self.max_depth + 1):
            print(f"[COMPOSITIONAL] Exploring depth={depth} compositions...")

            # Generate all K-step chains
            depth_compositions = self._enumerate_chains(
                programs=top_programs,
                depth=depth,
                input_grid=input_grid,
                expected_output=expected_output
            )

            # Filter by quality threshold
            high_quality = [c for c in depth_compositions if c['score'] > 0.45]

            if high_quality:
                print(f"[COMPOSITIONAL] Found {len(high_quality)} high-quality "
                      f"depth={depth} compositions!")
                compositions.extend(high_quality)

                # Add best compositions to library
                for comp in sorted(high_quality, key=lambda c: c['score'], reverse=True)[:3]:
                    self._add_to_library(comp)
            else:
                # No high-quality compositions at this depth, stop deepening
                print(f"[COMPOSITIONAL] No high-quality depth={depth} compositions found, "
                      f"stopping at depth={depth-1}")
                break

        return compositions

    def _enumerate_chains(self, programs, depth, input_grid, expected_output):
        """
        Enumerate all depth-K chains of programs.

        Uses iterative deepening:
        - Depth 2: all pairs (P1 → P2)
        - Depth 3: all triplets (P1 → P2 → P3)
        - Depth K: all K-tuples

        Returns:
            List of {chain, score, output, description} dicts
        """
        import itertools

        chains = []

        # Generate all K-permutations (order matters: rotate→flip != flip→rotate)
        for chain_tuple in itertools.permutations(programs, depth):
            # Execute chain: apply each program sequentially
            output_grid = input_grid.copy()
            chain_description = []

            for program in chain_tuple:
                try:
                    # Apply program to current grid state
                    output_grid = self._execute_program(program, output_grid)
                    chain_description.append(program.get('name', 'unknown'))
                except Exception as e:
                    # Invalid chain (e.g., program failed), skip
                    break
            else:
                # All programs executed successfully, score the result
                score = self._score_output(output_grid, expected_output)

                chains.append({
                    'chain': chain_tuple,
                    'score': score,
                    'output': output_grid,
                    'description': ' → '.join(chain_description),
                    'depth': depth
                })

        return chains

    def _execute_program(self, program, input_grid):
        """Execute a single program on input grid."""
        # Use RPN interpreter to execute program code
        from knowledge3d.cranium.rpn_interp import execute_rpn

        program_code = program.get('program', '')
        output_grid = execute_rpn(program_code, input_grid)

        return output_grid

    def _score_output(self, output_grid, expected_output):
        """Score how well output matches expected output (0.0-1.0)."""
        from knowledge3d.training.arc_agi.sovereign_pipeline import score_candidate

        return score_candidate(output_grid, expected_output)

    def _add_to_library(self, composition):
        """Add high-quality composition to library as a new discovered program."""
        # Create program entry
        program_entry = {
            'program': self._chain_to_rpn(composition['chain']),
            'quality_score': composition['score'],
            'description': composition['description'],
            'depth': composition['depth'],
            'source': 'compositional_discovery'
        }

        # Add to DualShadowCopy library
        self.library.add_program(program_entry)

        print(f"[COMPOSITIONAL] Added depth={composition['depth']} composition: "
              f"{composition['description']} (score={composition['score']:.2f})")

    def _chain_to_rpn(self, chain):
        """Convert a chain of programs into a single RPN program string."""
        # Concatenate RPN programs with appropriate sequencing
        rpn_programs = [prog.get('program', '') for prog in chain]
        return ' '.join(rpn_programs)
```

#### Integration Points

**File**: `knowledge3d/training/arc_agi/candidate_generator.py`

**Modification**: Add compositional generation alongside procedural generation

```python
def generate_candidates(self, input_examples, expected_output, semantic_hints=None):
    """
    Generate candidates using THREE strategies:
    1. Procedural generation (existing)
    2. Semantic-guided generation (existing)
    3. **Compositional generation (NEW)**
    """
    candidates = []

    # Strategy 1: Procedural (existing)
    procedural_candidates = self._generate_procedural_candidates(input_examples)
    candidates.extend(procedural_candidates)

    # Strategy 2: Semantic-guided (existing)
    if semantic_hints:
        semantic_candidates = self._generate_semantic_guided_candidates(
            input_examples, semantic_hints
        )
        candidates.extend(semantic_candidates)

    # Strategy 3: Compositional (NEW!)
    compositional_candidates = self._generate_compositional_candidates(
        input_examples[0], expected_output
    )
    candidates.extend(compositional_candidates)

    return candidates

def _generate_compositional_candidates(self, input_grid, expected_output):
    """
    Generate N-step compositions of discovered programs.

    NEW METHOD - implements compositional discovery!
    """
    from knowledge3d.training.arc_agi.compositional_generator import (
        CompositionalCandidateGenerator
    )

    # Get current library state
    library = self.shadow_copy  # DualShadowCopy instance

    # Create compositional generator
    comp_gen = CompositionalCandidateGenerator(
        library=library,
        max_depth=6,      # Explore up to 6-step compositions
        beam_width=10     # Consider top 10 programs at each step
    )

    # Generate compositions
    compositions = comp_gen.generate_compositions(
        input_grid=input_grid,
        expected_output=expected_output,
        current_depth=2   # Start at depth=2
    )

    print(f"[COMPOSITIONAL GEN] Generated {len(compositions)} compositional candidates")

    return [c['output'] for c in compositions]
```

#### Composition Pruning Strategy

**Why pruning matters**: N-step composition explodes combinatorially:
- Depth 2 with 10 programs: 10×9 = 90 chains
- Depth 3 with 10 programs: 10×9×8 = 720 chains
- Depth 4 with 10 programs: 10×9×8×7 = 5,040 chains
- Depth 6 with 10 programs: 151,200 chains!

**Solution: Beam Search with Quality Pruning**

```python
def _enumerate_chains_with_pruning(self, programs, depth, input_grid, expected_output):
    """
    Enumerate chains using beam search to avoid combinatorial explosion.

    Algorithm:
    1. Start with all single programs (depth=1)
    2. For each depth increment:
       a. Extend top-K chains from previous depth
       b. Score extended chains
       c. Keep top-K by score (beam search)
    3. Continue until max_depth or no high-quality extensions
    """
    # Initialize beam with single programs
    beam = [
        {
            'chain': [prog],
            'output': self._execute_program(prog, input_grid),
            'score': self._score_output(
                self._execute_program(prog, input_grid),
                expected_output
            ),
            'depth': 1
        }
        for prog in programs
    ]

    # Sort by score, keep top-K
    beam = sorted(beam, key=lambda c: c['score'], reverse=True)[:self.beam_width]

    all_chains = beam.copy()

    # Iteratively deepen
    for d in range(2, depth + 1):
        new_beam = []

        # Extend each chain in beam by one program
        for chain_entry in beam:
            for prog in programs:
                # Don't apply same program consecutively (usually redundant)
                if chain_entry['chain'][-1] == prog:
                    continue

                # Extend chain
                extended_chain = chain_entry['chain'] + [prog]

                # Execute new program on previous output
                try:
                    extended_output = self._execute_program(prog, chain_entry['output'])
                    extended_score = self._score_output(extended_output, expected_output)

                    new_beam.append({
                        'chain': extended_chain,
                        'output': extended_output,
                        'score': extended_score,
                        'depth': d
                    })
                except:
                    # Execution failed, skip this extension
                    continue

        # Sort by score, keep top-K
        new_beam = sorted(new_beam, key=lambda c: c['score'], reverse=True)[:self.beam_width]

        # If no high-quality extensions, stop
        if not any(c['score'] > 0.45 for c in new_beam):
            print(f"[COMPOSITIONAL] No high-quality depth={d} chains, stopping")
            break

        beam = new_beam
        all_chains.extend(new_beam)

    return all_chains
```

---

## Optimization 2: Multi-Core Parallel Generation (Tesla 3-6-9)

### Problem Statement

Current system generates candidates **sequentially** on a single thread:
- GPU utilization: 1.12% avg (98.88% idle!)
- Runtime: 16-24 minutes per run
- Bottleneck: CPU-bound candidate generation

**Evidence**: GPU metrics show massive headroom (94.2% idle even at peak 5.8% utilization)

### Architecture Design: Tesla 3-6-9 Pattern

**Principle**: Spawn 9 parallel cores, each generates 6 candidates, select top 3

**Why this pattern**:
- **9 cores**: Maximizes parallelism without overwhelming GPU (~10-15% total utilization)
- **6 candidates each**: Focused generation (quality over quantity)
- **Top 3 selection**: Best candidates across all cores (diversity + quality)

**Expected speedup**: 9× parallel execution + better candidate quality

#### Implementation Structure

**File**: `knowledge3d/training/arc_agi/parallel_generator.py` (NEW)

```python
import multiprocessing as mp
from knowledge3d.cranium.math_core_pool import MathCorePool

class ParallelCandidateGenerator:
    """
    Generate candidates in parallel using 9 math cores (Tesla 9).

    Each core generates 6 candidates (Tesla 6).
    Top 3 candidates selected across all cores (Tesla 3).
    """

    def __init__(self, num_cores=9, candidates_per_core=6, top_k=3):
        """
        Args:
            num_cores: Number of parallel cores (default 9, Tesla pattern)
            candidates_per_core: Candidates per core (default 6, Tesla pattern)
            top_k: Top candidates to select (default 3, Tesla pattern)
        """
        self.num_cores = num_cores
        self.candidates_per_core = candidates_per_core
        self.top_k = top_k

        # Initialize math core pool
        self.pool = MathCorePool(num_cores=num_cores)

    def generate_parallel(self, input_grid, expected_output, library, semantic_hints=None):
        """
        Generate candidates in parallel across 9 cores.

        Algorithm:
        1. Spawn 9 cores
        2. Each core generates 6 candidates (compositional + procedural)
        3. Unite all candidates (54 total)
        4. Select top 3 by quality score

        Returns:
            List of top-3 candidates
        """
        print(f"[PARALLEL GEN] Spawning {self.num_cores} cores...")

        # Create generation tasks (one per core)
        tasks = [
            {
                'core_id': i,
                'input_grid': input_grid,
                'expected_output': expected_output,
                'library': library,
                'semantic_hints': semantic_hints,
                'max_candidates': self.candidates_per_core
            }
            for i in range(self.num_cores)
        ]

        # Execute in parallel
        results = self.pool.map(
            func=self._generate_on_core,
            tasks=tasks
        )

        # Flatten results (list of lists → single list)
        all_candidates = []
        for core_result in results:
            all_candidates.extend(core_result['candidates'])

        print(f"[PARALLEL GEN] Generated {len(all_candidates)} candidates "
              f"({self.num_cores} cores × {self.candidates_per_core} each)")

        # Select top-3 by quality score
        top_candidates = self._select_top_k(all_candidates, self.top_k)

        print(f"[PARALLEL GEN] Selected top {self.top_k} candidates "
              f"(scores: {[c['score'] for c in top_candidates]})")

        return top_candidates

    def _generate_on_core(self, task):
        """
        Generate candidates on a single core.

        Called in parallel by pool.map().

        Each core generates:
        - 3 compositional candidates (depth 2-4)
        - 3 procedural candidates (primitives)

        Total: 6 candidates per core
        """
        core_id = task['core_id']
        input_grid = task['input_grid']
        expected_output = task['expected_output']
        library = task['library']
        max_candidates = task['max_candidates']

        print(f"[CORE {core_id}] Generating {max_candidates} candidates...")

        candidates = []

        # Generate compositional candidates (depth 2-4)
        from knowledge3d.training.arc_agi.compositional_generator import (
            CompositionalCandidateGenerator
        )
        comp_gen = CompositionalCandidateGenerator(library, max_depth=4, beam_width=5)
        comp_candidates = comp_gen.generate_compositions(input_grid, expected_output)
        candidates.extend(comp_candidates[:3])  # Take top 3 compositional

        # Generate procedural candidates (primitives)
        from knowledge3d.training.arc_agi.candidate_generator import generate_primitive
        proc_candidates = [
            generate_primitive(input_grid)
            for _ in range(3)
        ]
        candidates.extend(proc_candidates)

        print(f"[CORE {core_id}] Generated {len(candidates)} candidates")

        return {'core_id': core_id, 'candidates': candidates}

    def _select_top_k(self, candidates, k):
        """
        Select top-K candidates by quality score.

        Applies diversity filter: if top candidates are too similar,
        replace with next-best diverse candidate.
        """
        # Sort by score
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c['score'],
            reverse=True
        )

        # Apply diversity filter
        top_k = []
        for cand in sorted_candidates:
            if len(top_k) >= k:
                break

            # Check if candidate is diverse from already-selected candidates
            if self._is_diverse(cand, top_k):
                top_k.append(cand)

        return top_k

    def _is_diverse(self, candidate, existing_candidates, threshold=0.8):
        """
        Check if candidate is diverse (not too similar to existing).

        Uses grid similarity: if grids are >80% identical, reject.
        """
        if not existing_candidates:
            return True

        for existing in existing_candidates:
            similarity = self._compute_similarity(
                candidate['output'],
                existing['output']
            )
            if similarity > threshold:
                return False  # Too similar, not diverse

        return True

    def _compute_similarity(self, grid1, grid2):
        """Compute similarity between two grids (0.0 = different, 1.0 = identical)."""
        import numpy as np

        if grid1.shape != grid2.shape:
            return 0.0

        matches = np.sum(grid1 == grid2)
        total = grid1.size

        return matches / total
```

#### Integration Points

**File**: `scripts/train_arc_sovereign_loop.py`

**Modification**: Replace sequential generation with parallel generation

```python
# BEFORE (sequential):
candidates = candidate_generator.generate_candidates(
    input_examples=task['train'],
    expected_output=task['train'][0]['output'],
    semantic_hints=semantic_hints
)

# AFTER (parallel):
from knowledge3d.training.arc_agi.parallel_generator import ParallelCandidateGenerator

parallel_gen = ParallelCandidateGenerator(
    num_cores=9,           # Tesla 9
    candidates_per_core=6, # Tesla 6
    top_k=3                # Tesla 3
)

candidates = parallel_gen.generate_parallel(
    input_grid=task['train'][0]['input'],
    expected_output=task['train'][0]['output'],
    library=shadow_copy,
    semantic_hints=semantic_hints
)
```

---

## Optimization 3: Semantic Cross-Pattern Composition

### Problem Statement

Current semantic layer is **too conservative**:
- Detects pattern types: rotation, flip, color, sparse
- Generates variants **within** pattern type (rotate_90, rotate_180, rotate_270)
- **Missing**: Cross-pattern compositions (rotation + color, flip + sparse)

**Evidence**: Pattern types stuck at 4 since Run 003

### Architecture Design

**Principle**: When semantic layer detects multiple patterns, generate compositions across pattern types

**Example**:
```
Semantic hints: ["rotation", "color_change", "translation"]

Current (conservative):
- Generate rotation variants (90°, 180°, 270°)
- Generate color variants (recolor_3, recolor_5)
- Generate translation variants (1,0), (0,1)

New (cross-pattern):
- Generate rotation+color compositions (rotate_90 → recolor_5)
- Generate flip+translation compositions (flip_h → translate_1_0)
- Generate rotation+color+translation (rotate_90 → recolor_5 → translate_1_0)
```

#### Implementation

**File**: `knowledge3d/training/arc_agi/sovereign_pipeline.py`

**Modification**: Enhance semantic hint extraction to support cross-pattern composition

```python
def extract_semantic_hints_with_cross_patterns(self, task_description, semantic_contexts):
    """
    Extract semantic hints including cross-pattern compositions.

    NEW: When multiple pattern types detected, generate cross-pattern hints.
    """
    hints = []
    pattern_types = set()

    # Extract individual pattern hints
    for ctx in semantic_contexts:
        if "transformation_type" in ctx:
            word = ctx["transformation_type"]
            hints.append(word)

            # Track pattern type
            if word in ["rotate_90", "rotate_180", "rotate_270"]:
                pattern_types.add("rotation")
            elif word in ["flip_horizontal", "flip_vertical"]:
                pattern_types.add("flip")
            elif "recolor" in word:
                pattern_types.add("color")
            elif "translate" in word:
                pattern_types.add("translation")

    # NEW: Cross-pattern hints
    if len(pattern_types) >= 2:
        # Multiple patterns detected → generate cross-pattern hints
        cross_patterns = list(itertools.combinations(pattern_types, 2))
        for p1, p2 in cross_patterns:
            hints.append(f"cross_pattern:{p1}+{p2}")
            print(f"[SEMANTIC CROSS] Detected {p1}+{p2} cross-pattern")

    return hints, pattern_types
```

**File**: `knowledge3d/training/arc_agi/candidate_generator.py`

**Modification**: Generate compositions when cross-pattern hints present

```python
def _generate_semantic_guided_candidates(self, input_examples, semantic_hints):
    """
    Generate candidates guided by semantic hints.

    NEW: Support cross-pattern hints (e.g., "cross_pattern:rotation+color")
    """
    candidates = []

    # Separate single-pattern and cross-pattern hints
    single_pattern_hints = [h for h in semantic_hints if not h.startswith("cross_pattern:")]
    cross_pattern_hints = [h for h in semantic_hints if h.startswith("cross_pattern:")]

    # Generate single-pattern candidates (existing)
    for hint in single_pattern_hints:
        pattern_candidates = self._generate_pattern_candidates(hint, input_examples)
        candidates.extend(pattern_candidates)

    # Generate cross-pattern compositions (NEW!)
    for hint in cross_pattern_hints:
        # Parse hint: "cross_pattern:rotation+color" → ["rotation", "color"]
        patterns = hint.replace("cross_pattern:", "").split("+")

        # Get programs for each pattern
        pattern_programs = {}
        for pattern in patterns:
            pattern_programs[pattern] = self._get_programs_for_pattern(pattern)

        # Generate compositions across patterns
        cross_compositions = self._compose_across_patterns(
            pattern_programs,
            input_examples[0]['input']
        )
        candidates.extend(cross_compositions)

        print(f"[SEMANTIC CROSS] Generated {len(cross_compositions)} "
              f"cross-pattern compositions for {hint}")

    return candidates

def _compose_across_patterns(self, pattern_programs, input_grid):
    """
    Compose programs across different pattern types.

    Example:
    - pattern_programs = {
        "rotation": [rotate_90, rotate_180],
        "color": [recolor_3, recolor_5]
      }
    - Compositions: rotate_90 → recolor_3, rotate_90 → recolor_5,
                     rotate_180 → recolor_3, rotate_180 → recolor_5
    """
    import itertools

    compositions = []

    # Get all cross-pattern pairs
    pattern_names = list(pattern_programs.keys())
    for p1, p2 in itertools.combinations(pattern_names, 2):
        programs_p1 = pattern_programs[p1]
        programs_p2 = pattern_programs[p2]

        # Generate all pairs (p1_prog → p2_prog)
        for prog1 in programs_p1:
            for prog2 in programs_p2:
                # Execute composition
                try:
                    intermediate = self._execute_program(prog1, input_grid)
                    output = self._execute_program(prog2, intermediate)

                    compositions.append({
                        'output': output,
                        'chain': [prog1, prog2],
                        'description': f"{prog1['name']} → {prog2['name']}",
                        'source': 'cross_pattern_composition'
                    })
                except:
                    continue

    return compositions
```

---

## Testing & Validation Strategy

### Unit Tests

**File**: `knowledge3d/cranium/tests/test_compositional_generator.py` (NEW)

```python
def test_composition_depth_2():
    """Test 2-step composition generation."""
    # Setup: library with 2 programs (rotate_90, flip_h)
    library = create_test_library([
        {'program': 'rotate_90', 'quality_score': 0.8},
        {'program': 'flip_horizontal', 'quality_score': 0.7}
    ])

    # Generate compositions
    comp_gen = CompositionalCandidateGenerator(library, max_depth=2)
    compositions = comp_gen.generate_compositions(
        input_grid=test_grid,
        expected_output=test_expected_output
    )

    # Validate: should generate rotate_90→flip_h and flip_h→rotate_90
    assert len(compositions) >= 2
    assert any('rotate_90 → flip_horizontal' in c['description'] for c in compositions)

def test_composition_depth_n():
    """Test N-step composition (depth=4)."""
    library = create_test_library([
        {'program': 'rotate_90', 'quality_score': 0.8},
        {'program': 'flip_horizontal', 'quality_score': 0.7},
        {'program': 'recolor_5', 'quality_score': 0.6}
    ])

    comp_gen = CompositionalCandidateGenerator(library, max_depth=4)
    compositions = comp_gen.generate_compositions(test_grid, test_expected_output)

    # Should explore depths 2, 3, 4
    depths = [c['depth'] for c in compositions]
    assert max(depths) >= 3  # At least depth 3 explored

def test_parallel_generation_tesla_3_6_9():
    """Test parallel generation (9 cores × 6 candidates → top 3)."""
    parallel_gen = ParallelCandidateGenerator(
        num_cores=9,
        candidates_per_core=6,
        top_k=3
    )

    top_candidates = parallel_gen.generate_parallel(
        input_grid=test_grid,
        expected_output=test_expected_output,
        library=test_library
    )

    # Should return exactly 3 candidates
    assert len(top_candidates) == 3

    # Candidates should be sorted by score (descending)
    scores = [c['score'] for c in top_candidates]
    assert scores == sorted(scores, reverse=True)
```

### Integration Tests

**File**: `scripts/test_compositional_training.py` (NEW)

```python
"""
Test compositional training on a small subset of ARC tasks.

Expected behavior:
- Library should grow beyond 52 programs (compositions discovered)
- Accuracy should improve as compositions solve more complex tasks
- Runtime should decrease (parallel generation speedup)
"""

def test_compositional_training_short_run():
    """Run 3 epochs on 5 tasks with compositional generation enabled."""
    result = run_arc_training(
        num_tasks=5,
        num_epochs=3,
        enable_compositional=True,
        enable_parallel=True
    )

    # Validate library growth
    assert result['final_library_size'] > result['initial_library_size'], \
        "Library should grow with compositional discovery"

    # Validate runtime improvement
    assert result['avg_epoch_time'] < 2.0, \
        "Parallel generation should reduce epoch time to <2 min"

    # Validate accuracy improvement
    assert result['final_accuracy'] >= result['initial_accuracy'], \
        "Accuracy should not decrease"
```

### Validation Metrics

**Track these metrics across Runs 011-015**:

1. **Library Growth**:
   - Programs: Should grow 52 → 100+ (compositions)
   - Composition depth distribution (depth 2/3/4/5/6 counts)

2. **Accuracy**:
   - Peak accuracy: Should exceed 3.33%
   - Final accuracy: Should show upward trend
   - Tasks solved: Should diversify (not just same 1-2 tasks)

3. **Performance**:
   - Runtime per epoch: Should decrease to 2-3 min (from 16-24 min)
   - GPU utilization: Should increase to 10-15% (from 1.12%)

4. **Compositional Discovery**:
   - Depth distribution: Track how many depth-2, depth-3, etc. compositions discovered
   - Source breakdown: % compositional vs. procedural vs. semantic

---

## Implementation Checklist for Codex

### Phase 1: Compositional Discovery (Priority 1 - CRITICAL)

- [ ] Create `knowledge3d/training/arc_agi/compositional_generator.py`
- [ ] Implement `CompositionalCandidateGenerator` class
  - [ ] `generate_compositions()` method (N-step beam search)
  - [ ] `_enumerate_chains_with_pruning()` method (beam search pruning)
  - [ ] `_execute_program()` method (RPN execution)
  - [ ] `_score_output()` method (grid comparison)
  - [ ] `_add_to_library()` method (save to DualShadowCopy)
  - [ ] `_chain_to_rpn()` method (serialize composition)
- [ ] Integrate with `candidate_generator.py`:
  - [ ] Add `_generate_compositional_candidates()` method
  - [ ] Call from `generate_candidates()` alongside procedural/semantic
- [ ] Write unit tests (`test_compositional_generator.py`)
- [ ] Validate on small task subset (3 tasks × 3 epochs)

**Success Criteria**:
- Library grows beyond 52 programs
- Compositions of depth 2-4 discovered
- At least one new task solved (accuracy >3.33%)

### Phase 2: Parallel Generation (Priority 2 - HIGH)

- [ ] Create `knowledge3d/training/arc_agi/parallel_generator.py`
- [ ] Implement `ParallelCandidateGenerator` class
  - [ ] `generate_parallel()` method (spawn 9 cores)
  - [ ] `_generate_on_core()` method (per-core generation)
  - [ ] `_select_top_k()` method (top-3 selection with diversity)
  - [ ] `_is_diverse()` method (diversity filter)
  - [ ] `_compute_similarity()` method (grid similarity)
- [ ] Integrate with `train_arc_sovereign_loop.py`:
  - [ ] Replace sequential generation with parallel generation
  - [ ] Configure Tesla 3-6-9 pattern (9 cores × 6 candidates → top 3)
- [ ] Measure speedup (compare runtime before/after)
- [ ] Write unit tests (`test_parallel_generator.py`)

**Success Criteria**:
- Runtime per epoch: <3 minutes (down from 16-24 min)
- GPU utilization: 10-15% (up from 1.12%)
- Top-3 selection working (diverse, high-quality candidates)

### Phase 3: Semantic Cross-Pattern (Priority 3 - MEDIUM)

- [ ] Modify `sovereign_pipeline.py`:
  - [ ] Add `extract_semantic_hints_with_cross_patterns()` method
  - [ ] Detect cross-pattern hints (rotation+color, flip+translation, etc.)
- [ ] Modify `candidate_generator.py`:
  - [ ] Add `_compose_across_patterns()` method
  - [ ] Generate cross-pattern compositions from hints
- [ ] Validate on tasks with multiple pattern types
- [ ] Write unit tests (`test_cross_pattern_composition.py`)

**Success Criteria**:
- Pattern types grow beyond 4
- Cross-pattern compositions discovered
- Accuracy improvement on multi-pattern tasks

### Phase 4: Integration & Validation (Priority 4 - FINAL)

- [ ] Run full training (Run 011: 60 tasks × 27 epochs × 6 cycles)
- [ ] Capture metrics:
  - [ ] Library size, composition depth distribution
  - [ ] Accuracy peak/final
  - [ ] Runtime, GPU utilization
- [ ] Update `TEMP/ARC_TRAINING_LOG.md` with Run 011 results
- [ ] Compare Runs 011-015 vs. Runs 006-010 (stall vs. growth)
- [ ] Document results in `TEMP/COMPOSITIONAL_OPTIMIZATION_RESULTS_11.27.2025.md`

**Success Criteria**:
- Library: 52 → 100+ programs
- Accuracy: 3.33% → 10%+
- Runtime: 16-24 min → 2-3 min per run
- Sustained growth over 5 runs (no stall)

---

## Expected Timeline

**Phase 1 (Compositional Discovery)**: 2-4 hours coding + testing
**Phase 2 (Parallel Generation)**: 1-2 hours coding + testing
**Phase 3 (Semantic Cross-Pattern)**: 1-2 hours coding + testing
**Phase 4 (Integration & Validation)**: 1 hour + ~3 hours GPU time (Run 011)

**Total**: ~6-10 hours implementation + ~3 hours validation runtime

---

## Communication Protocol

**After Phase 1 complete**:
- Report library growth (52 → X programs)
- Report composition depth distribution
- Report any new tasks solved

**After Phase 2 complete**:
- Report runtime improvement (before/after)
- Report GPU utilization increase
- Report candidate quality metrics

**After Phase 4 complete**:
- Update `TEMP/ARC_TRAINING_LOG.md` with Run 011-015
- Create `TEMP/COMPOSITIONAL_OPTIMIZATION_RESULTS_11.27.2025.md`
- Summarize impact (library, accuracy, runtime)

---

## Risk Mitigation

**Risk 1**: Compositional explosion (too many chains)
**Mitigation**: Beam search pruning (beam_width=10, quality threshold=0.45)

**Risk 2**: Parallel overhead exceeds speedup
**Mitigation**: Profile first, ensure GPU I/O not bottleneck

**Risk 3**: Compositions too complex, don't generalize
**Mitigation**: Track composition success rate, prune ineffective depths

**Risk 4**: Integration breaks existing system
**Mitigation**: Feature flags to enable/disable compositional/parallel generation independently

---

## Success Definition

**Optimization successful if Runs 011-015 show**:

1. ✅ Library growth: 52 → 100+ programs (compositions discovered)
2. ✅ Accuracy improvement: 3.33% → 10%+ (new tasks solvable)
3. ✅ Runtime reduction: 16-24 min → 2-3 min per run (parallel speedup)
4. ✅ Sustained growth: No stall over 5 runs (library continues expanding)

**If successful, continue training to 99% with compositional system**
**If stalled again, escalate to Claude for architecture review**

---

## References

- [ARC_TRAINING_LOG.md](ARC_TRAINING_LOG.md) - Runs 001-010 documentation
- [HANDOFF_SUMMARY_FOR_DANIEL_11.26.2025.md](HANDOFF_SUMMARY_FOR_DANIEL_11.26.2025.md) - Training handoff
- [PHASE2_INTENSIVE_TRAINING_COMPLETE_11.26.2025.md](PHASE2_INTENSIVE_TRAINING_COMPLETE_11.26.2025.md) - Architecture validation
- [CODEX_ARC_TRAINING_HANDOFF_11.26.2025.md](CODEX_ARC_TRAINING_HANDOFF_11.26.2025.md) - Training environment setup

---

**END OF SPECIFICATION**

**Status**: Ready for Codex implementation
**Next Action**: Codex implements Phase 1 (Compositional Discovery)
**Target**: Run 011 with compositional+parallel generation by end of November 27, 2025
