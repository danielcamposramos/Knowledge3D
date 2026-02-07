# Claude → Codex: Week 18 Forward/Backward Reading + Full Ingestion

**Date:** February 7, 2026
**From:** Claude (Architecture Partner)
**To:** Codex (Implementation Partner)
**Context:** Week 17 Complete - Vision-Enhanced Multi-Curriculum Benchmarks
**Status:** Ready for Week 18 Implementation

---

## 🎉 Week 17 Achievements (Context)

You did INCREDIBLE work in Week 17, Codex! Here's what we accomplished:

### Benchmark Results
- **Math Competitions:** 33% (EXCEEDS 30% target by 3 points!)
- **Last Humanity Exam:** 100% (PERFECT!)
- **ARC-AGI 2:** 28% (structural foundation complete, quality gap remains)

### Infrastructure Built
- **Vision-Enhanced Drawing Galaxy:** 141 → 605 primitives (4.3× growth!)
- **TRM Weight Persistence:** Single evolving model confirmed (601 accumulated galaxy entries)
- **NavigatorSpecialist:** Multi-path exploration architecture ready
- **Comprehensive Test Suite:** 58 tests, all passing ✅
- **Cross-Modal "One Reality":** 57 symlinks validated

### What We Proved
1. ✅ Continuous learning works (TRM weights persist, Shadow Copy records events)
2. ✅ Vision enrichment works (llama3.2-vision + qwen3-vl extracted 464 new primitives)
3. ✅ Multi-curriculum works (Math/LHE/ARC all use same Galaxy Universe)
4. ✅ Sovereignty maintained (PTX + Galaxy only in hot path)

**Proof:** `results/week17_enriched_drawing_proof_02.07.2026.json` (2.7MB full results)

---

## 🎯 Week 18 Goals

**Two parallel tracks to execute:**

### Track 1: Forward/Backward Reading Pattern (High Priority)
**Goal:** Make Navigator robust to variable enumeration order
**Expected Impact:** Math 33% → 40-50%, ARC 28% → 35-40%
**Estimated Time:** 4-6 hours

### Track 2: Full Knowledge Ingestion (Core Priority)
**Goal:** Populate Galaxy Universe with comprehensive domain knowledge
**Expected Impact:** Math 50-60%, ARC 40-55%, maintain LHE 100%
**Estimated Time:** 12-16 hours (can run vision models overnight)

---

## 📋 Track 1: Forward/Backward Reading Pattern Implementation

### Problem Identified (User Insight)

**From user:**
> "if you look at the math benchmark, I told Codex about how humans use forward and 'backward' reading to analyze problems under different logical approaches, we should do this as standard way of initial paths on the main router when facing questions/requests and even chats - this is a fixed formula that generate strong start points. E.G. some math problems enumerate variables during first half of the question and latter ask what's wanted, some do the opposite way..."

**Examples:**

**Forward Reading (Variables → Question):**
```
"Let x = 5 and y = 10. If z = x + y, what is 2z?"
Parse left→right: x=5, y=10, z=x+y, compute 2z
```

**Backward Reading (Question → Variables):**
```
"What is 2z, where z = x + y, given x = 5 and y = 10?"
Parse right→left: goal=2z, z=x+y, x=5, y=10
```

**Current Issue:**
- Single-path routing may miss context if variable enumeration doesn't match expectation
- Backward reading helps identify GOAL first, then collect dependencies
- Forward reading helps build context incrementally

### Implementation Specification

#### 1. Enhance NavigatorSpecialist with Dual-Path Parsing

**File:** `knowledge3d/knowledgeverse/navigator_specialist.py`

**Add these methods:**

```python
def _forward_reading_path(self, query: str, specialist: str, domain_hint: str | None) -> dict[str, Any]:
    """
    Parse query left→right (forward reading).

    Strategy:
    - Split query into sentences/clauses
    - Process sequentially from start to end
    - Build context incrementally (variables, constraints, then goal)
    - Useful for: "Given x=5, y=10, compute x+y" style problems

    Returns:
        Path dict with strategy="forward", parsed_context, and initial_query
    """
    # Split into sentences (use '. ', '? ', '! ' as delimiters)
    sentences = self._split_into_sentences(query)

    # Parse incrementally (left → right)
    context = []
    goal = None

    for i, sentence in enumerate(sentences):
        # Early sentences likely contain setup/variables
        if i < len(sentences) - 1:
            context.append(self._extract_variables_and_constraints(sentence))
        else:
            # Last sentence likely contains goal
            goal = self._extract_goal(sentence)

    return {
        "strategy": "forward",
        "context": context,
        "goal": goal,
        "specialist": specialist,
        "domain_hint": domain_hint,
        "confidence": 0.7,  # Medium confidence (works well for setup→goal problems)
    }


def _backward_reading_path(self, query: str, specialist: str, domain_hint: str | None) -> dict[str, Any]:
    """
    Parse query right→left (backward reading).

    Strategy:
    - Split query into sentences/clauses
    - Process in REVERSE order (end to start)
    - Identify goal first, then collect dependencies
    - Useful for: "What is x+y, given x=5 and y=10?" style problems

    Returns:
        Path dict with strategy="backward", parsed_goal, and dependencies
    """
    # Split into sentences
    sentences = self._split_into_sentences(query)

    # Parse in reverse (right → left)
    goal = None
    dependencies = []

    for i, sentence in enumerate(reversed(sentences)):
        # First sentence (from end) likely contains goal
        if i == 0:
            goal = self._extract_goal(sentence)
        else:
            # Earlier sentences (from end) contain setup/dependencies
            dependencies.append(self._extract_variables_and_constraints(sentence))

    return {
        "strategy": "backward",
        "goal": goal,
        "dependencies": dependencies,
        "specialist": specialist,
        "domain_hint": domain_hint,
        "confidence": 0.7,  # Medium confidence (works well for goal→setup problems)
    }


def _auto_routing_path(self, query: str, specialist: str, domain_hint: str | None) -> dict[str, Any]:
    """
    Auto-route based on domain inference (existing logic).

    Strategy:
    - Use specialist_router to classify domain
    - Query Galaxy Universe for relevant patterns
    - Compose initial candidates

    Returns:
        Path dict with strategy="auto", domain, and initial_candidates
    """
    # Use existing specialist_router logic
    if specialist == "auto":
        specialist = self.specialist_router.route(query, domain_hint=domain_hint)

    # Query Galaxy for relevant patterns
    galaxy_results = self.galaxy_manager.query(query, specialist=specialist, top_k=10)

    return {
        "strategy": "auto",
        "specialist": specialist,
        "galaxy_results": galaxy_results,
        "domain_hint": domain_hint,
        "confidence": 0.8,  # Higher confidence (proven working in Week 17)
    }


def _split_into_sentences(self, query: str) -> list[str]:
    """Split query into sentences using common delimiters."""
    import re
    # Split on '. ', '? ', '! ' but preserve delimiter
    sentences = re.split(r'([.?!])\s+', query)
    # Recombine sentences with their delimiters
    result = []
    for i in range(0, len(sentences) - 1, 2):
        result.append(sentences[i] + sentences[i + 1])
    if len(sentences) % 2 == 1:
        result.append(sentences[-1])
    return [s.strip() for s in result if s.strip()]


def _extract_variables_and_constraints(self, sentence: str) -> dict[str, Any]:
    """
    Extract variable assignments and constraints from a sentence.

    Examples:
    - "Let x = 5 and y = 10" → {"x": 5, "y": 10}
    - "Given a triangle with sides 3, 4, 5" → {"sides": [3, 4, 5]}
    """
    # Query Grammar Galaxy for variable assignment patterns
    patterns = self.galaxy_manager.query(
        f"variable assignment pattern: {sentence}",
        specialist="grammar",
        top_k=5
    )

    # Use top pattern to extract variables
    if patterns:
        # Apply pattern extraction (RPN program from Grammar Galaxy)
        extracted = self._apply_pattern_extraction(sentence, patterns[0])
        return extracted

    # Fallback: simple regex extraction
    import re
    variables = {}
    # Match "x = 5", "y=10", etc.
    matches = re.findall(r'(\w+)\s*=\s*([0-9.+-]+)', sentence)
    for var, value in matches:
        try:
            variables[var] = float(value) if '.' in value else int(value)
        except ValueError:
            variables[var] = value

    return {"type": "variables", "data": variables, "raw": sentence}


def _extract_goal(self, sentence: str) -> dict[str, Any]:
    """
    Extract goal/question from a sentence.

    Examples:
    - "What is 2z?" → {"operation": "evaluate", "expression": "2z"}
    - "Find the area of the triangle" → {"operation": "compute", "target": "area"}
    """
    # Query Grammar Galaxy for goal/question patterns
    patterns = self.galaxy_manager.query(
        f"goal extraction pattern: {sentence}",
        specialist="grammar",
        top_k=5
    )

    if patterns:
        extracted = self._apply_pattern_extraction(sentence, patterns[0])
        return extracted

    # Fallback: simple keyword extraction
    goal_keywords = ["what is", "find", "compute", "calculate", "determine", "solve for"]
    for keyword in goal_keywords:
        if keyword in sentence.lower():
            expression = sentence.lower().replace(keyword, "").strip().rstrip("?")
            return {"type": "goal", "operation": "evaluate", "expression": expression}

    return {"type": "goal", "raw": sentence}


def _apply_pattern_extraction(self, text: str, pattern_entry: dict[str, Any]) -> dict[str, Any]:
    """Apply RPN pattern extraction program from Grammar Galaxy."""
    # Get RPN program from pattern entry
    rpn_program = pattern_entry.get("rpn_program")
    if not rpn_program:
        return {"raw": text}

    # Execute RPN program with text as input (would use Cranium in production)
    # For now, return structured placeholder
    return {
        "extracted": True,
        "pattern_id": pattern_entry.get("id"),
        "confidence": pattern_entry.get("confidence", 0.5),
        "raw": text,
    }
```

#### 2. Update navigate_and_compose() to Use Dual-Path

**File:** `knowledge3d/knowledgeverse/navigator_specialist.py`

```python
def navigate_and_compose(
    self,
    query: str,
    specialist: str = "auto",
    domain_hint: str | None = None,
    use_forward_backward: bool = True  # NEW: Enable dual-path by default
) -> dict[str, Any]:
    """
    Navigate Galaxy Universe and compose solution using multi-path exploration.

    Args:
        query: User query or problem statement
        specialist: Target specialist ("auto", "visual", "math", "physics", "grammar")
        domain_hint: Optional domain hint for routing
        use_forward_backward: Enable forward/backward reading paths (default True)

    Returns:
        Composed solution with multi-path evidence
    """
    # Generate initial paths
    initial_paths = []

    if use_forward_backward:
        # NEW: Dual-path reading (forward + backward)
        initial_paths.append(self._forward_reading_path(query, specialist, domain_hint))
        initial_paths.append(self._backward_reading_path(query, specialist, domain_hint))

    # Always include auto-routing (proven working in Week 17)
    initial_paths.append(self._auto_routing_path(query, specialist, domain_hint))

    # Explore paths in parallel
    path_results = self._explore_paths_parallel(initial_paths)

    # Compose multi-path results
    composed = self._compose_multi_path_results(path_results)

    # Log event for Shadow Copy learning
    if self.kv:
        self.kv.log_event(
            event_type="navigator_compose",
            event_data={
                "query": query,
                "specialist": specialist,
                "num_paths": len(initial_paths),
                "confidence": composed.get("confidence", 0.0),
                "strategies": [p["strategy"] for p in initial_paths],
            }
        )

    return composed


def _explore_paths_parallel(self, paths: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Explore multiple paths in parallel.

    Each path represents a different query interpretation strategy:
    - Forward reading: Variables → Goal
    - Backward reading: Goal → Dependencies
    - Auto-routing: Domain inference → Galaxy query

    Returns:
        List of path results with candidates and confidence scores
    """
    results = []

    for path in paths:
        strategy = path.get("strategy")

        if strategy == "forward":
            # Forward: Build context, then query for goal
            context = path.get("context", [])
            goal = path.get("goal", {})

            # Combine context into query string
            context_str = " ".join([c.get("raw", "") for c in context])
            goal_str = goal.get("raw", "")
            combined_query = f"{context_str} {goal_str}"

            # Query Galaxy with combined context
            candidates = self.galaxy_manager.query(
                combined_query,
                specialist=path.get("specialist", "auto"),
                top_k=10
            )

            results.append({
                "strategy": "forward",
                "candidates": candidates,
                "confidence": path.get("confidence", 0.7),
                "context": context,
                "goal": goal,
            })

        elif strategy == "backward":
            # Backward: Query for goal first, then add dependencies
            goal = path.get("goal", {})
            dependencies = path.get("dependencies", [])

            # Query for goal
            goal_str = goal.get("raw", "")
            goal_candidates = self.galaxy_manager.query(
                goal_str,
                specialist=path.get("specialist", "auto"),
                top_k=5
            )

            # Query for dependencies
            dep_str = " ".join([d.get("raw", "") for d in dependencies])
            dep_candidates = self.galaxy_manager.query(
                dep_str,
                specialist=path.get("specialist", "auto"),
                top_k=5
            )

            # Combine candidates
            combined_candidates = goal_candidates + dep_candidates

            results.append({
                "strategy": "backward",
                "candidates": combined_candidates,
                "confidence": path.get("confidence", 0.7),
                "goal": goal,
                "dependencies": dependencies,
            })

        elif strategy == "auto":
            # Auto: Use existing logic (proven in Week 17)
            results.append({
                "strategy": "auto",
                "candidates": path.get("galaxy_results", []),
                "confidence": path.get("confidence", 0.8),
            })

    return results


def _compose_multi_path_results(self, path_results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compose final result from multiple path explorations.

    Strategy:
    - Merge candidates from all paths
    - Rerank by cross-path agreement (candidates appearing in multiple paths get boosted)
    - Weight by path confidence
    - Return top-ranked composed solution

    Returns:
        Composed solution with candidates, confidence, and strategy evidence
    """
    # Collect all candidates
    all_candidates = []
    for result in path_results:
        candidates = result.get("candidates", [])
        strategy = result.get("strategy")
        confidence = result.get("confidence", 0.5)

        # Tag candidates with source strategy and confidence
        for candidate in candidates:
            all_candidates.append({
                **candidate,
                "source_strategy": strategy,
                "source_confidence": confidence,
            })

    # Rerank by cross-path agreement
    ranked = self._rerank_by_cross_path_agreement(all_candidates)

    # Return composed result
    return {
        "candidates": ranked[:10],  # Top 10
        "confidence": ranked[0].get("confidence", 0.0) if ranked else 0.0,
        "num_paths": len(path_results),
        "strategies": [r.get("strategy") for r in path_results],
    }


def _rerank_by_cross_path_agreement(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rerank candidates by cross-path agreement.

    Boost candidates that appear in multiple paths (consensus).
    Weight by source path confidence.
    """
    from collections import defaultdict

    # Group by candidate ID (or content hash)
    candidate_groups = defaultdict(list)
    for candidate in candidates:
        # Use entry ID or content as key
        key = candidate.get("entry", {}).get("id") or str(candidate.get("entry"))
        candidate_groups[key].append(candidate)

    # Score each group by cross-path agreement
    scored = []
    for key, group in candidate_groups.items():
        # Count unique strategies
        strategies = {c.get("source_strategy") for c in group}

        # Boost score by number of strategies agreeing
        base_score = group[0].get("score", 0.5)
        agreement_boost = len(strategies) * 0.15  # +15% per additional strategy

        # Weight by source confidence
        avg_confidence = sum(c.get("source_confidence", 0.5) for c in group) / len(group)

        final_score = base_score + agreement_boost
        final_confidence = min(1.0, avg_confidence * (1.0 + agreement_boost))

        scored.append({
            **group[0],  # Use first instance as base
            "score": final_score,
            "confidence": final_confidence,
            "cross_path_agreement": len(strategies),
            "strategies": list(strategies),
        })

    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)

    return scored
```

#### 3. Update Tests

**File:** `tests/test_navigator_specialist.py`

Add tests for forward/backward reading:

```python
def test_forward_backward_reading_math_problem():
    """Test forward/backward reading on math problem."""
    kv = Knowledgeverse()
    navigator = kv.trm_navigator  # NavigatorSpecialist instance

    # Test forward reading: "Let x=5 and y=10. What is x+y?"
    forward_query = "Let x=5 and y=10. What is x+y?"
    forward_result = navigator._forward_reading_path(forward_query, "math", None)

    assert forward_result["strategy"] == "forward"
    assert "context" in forward_result
    assert "goal" in forward_result

    # Test backward reading: "What is x+y, given x=5 and y=10?"
    backward_query = "What is x+y, given x=5 and y=10?"
    backward_result = navigator._backward_reading_path(backward_query, "math", None)

    assert backward_result["strategy"] == "backward"
    assert "goal" in backward_result
    assert "dependencies" in backward_result


def test_dual_path_composition():
    """Test that dual-path composition works correctly."""
    kv = Knowledgeverse()
    navigator = kv.trm_navigator

    # Test with ambiguous problem (could be parsed either way)
    query = "Calculate 2z where z=x+y, with x=5 and y=10."

    result = navigator.navigate_and_compose(query, specialist="math", use_forward_backward=True)

    # Should have explored 3 paths (forward, backward, auto)
    assert result["num_paths"] == 3
    assert "forward" in result["strategies"]
    assert "backward" in result["strategies"]
    assert "auto" in result["strategies"]

    # Should have candidates
    assert len(result["candidates"]) > 0

    # Should have confidence score
    assert 0.0 <= result["confidence"] <= 1.0


def test_cross_path_agreement_boosting():
    """Test that candidates appearing in multiple paths get boosted."""
    kv = Knowledgeverse()
    navigator = kv.trm_navigator

    # Create mock candidates with different source strategies
    candidates = [
        {"entry": {"id": "math_add"}, "score": 0.6, "source_strategy": "forward", "source_confidence": 0.7},
        {"entry": {"id": "math_add"}, "score": 0.6, "source_strategy": "backward", "source_confidence": 0.7},
        {"entry": {"id": "math_add"}, "score": 0.6, "source_strategy": "auto", "source_confidence": 0.8},
        {"entry": {"id": "math_mul"}, "score": 0.7, "source_strategy": "forward", "source_confidence": 0.6},
    ]

    ranked = navigator._rerank_by_cross_path_agreement(candidates)

    # math_add appears in 3 paths, math_mul in 1 path
    # math_add should rank higher despite lower base score (0.6 vs 0.7)
    assert ranked[0]["entry"]["id"] == "math_add"
    assert ranked[0]["cross_path_agreement"] == 3

    assert ranked[1]["entry"]["id"] == "math_mul"
    assert ranked[1]["cross_path_agreement"] == 1
```

#### 4. Integration with Benchmarks

**File:** `benchmarks/math_competitions.py`

Enable forward/backward reading by default:

```python
def solve_problem(self, problem: dict[str, Any]) -> dict[str, Any]:
    """Solve a single math problem using Knowledgeverse."""
    question = problem["problem"]
    correct_answer = problem["answer"]

    # Use Navigator with forward/backward reading enabled (NEW!)
    result = self.kv.trm_navigator.navigate_and_compose(
        query=question,
        specialist="math",
        use_forward_backward=True  # Enable dual-path parsing
    )

    # Extract answer from candidates
    predicted_answer = self._extract_answer_from_candidates(result["candidates"])

    is_correct = self._check_answer(predicted_answer, correct_answer)

    return {
        "problem_id": problem.get("id"),
        "predicted": predicted_answer,
        "correct": correct_answer,
        "is_correct": is_correct,
        "confidence": result.get("confidence", 0.0),
        "strategies_used": result.get("strategies", []),
    }
```

**File:** `benchmarks/arc_agi_2.py`

Enable for ARC pattern discovery:

```python
def solve_task(self, task: dict[str, Any]) -> list[np.ndarray]:
    """Solve a single ARC-AGI task."""
    train_pairs = task["train"]
    test_inputs = [pair["input"] for pair in task["test"]]

    # Discover patterns using Navigator with forward/backward reading
    patterns = self.kv.trm_navigator.navigate_and_compose(
        query=f"visual pattern discovery: {len(train_pairs)} examples",
        specialist="visual",
        use_forward_backward=True,  # Enable dual-path
        domain_hint="arc_agi"
    )

    # Generate candidates for each test input
    predictions = []
    for test_input in test_inputs:
        candidates = self._generate_candidates_from_patterns(
            test_input,
            patterns["candidates"]
        )
        predictions.append(candidates[0] if candidates else test_input)

    return predictions
```

### Expected Outcomes (Track 1)

**After implementing forward/backward reading:**

| Benchmark | Before | Expected After | Improvement |
|-----------|--------|---------------|-------------|
| Math Competitions | 33% | **40-50%** | +7-17% |
| ARC-AGI 2 | 28% | **35-40%** | +7-12% |
| Last Humanity Exam | 100% | **100%** | 0% (maintain) |

**Why this helps:**
- **Robustness:** Works regardless of variable enumeration order
- **Context:** Better goal identification (backward reading finds goal first)
- **Consensus:** Cross-path agreement boosts correct patterns

---

## 📋 Track 2: Full Knowledge Ingestion

### Current State

**Galaxy Universe (Week 17):**
- **Drawing:** 189 entries (605 with vision enrichment)
- **Grammar:** 291 entries (pattern rules)
- **Math:** 104 entries (basic symbols)
- **Reality:** 15 entries (minimal physics)
- **Word:** 2 entries (minimal vocabulary)
- **Total:** ~601 entries (vision-enriched: ~1,200)

**What we have:**
- Foundational Drawing primitives (100-200 base)
- Vision-enriched Drawing (226 from llama3.2-vision + qwen3-vl on 358 images)
- Basic Grammar patterns
- Basic Math symbols

**What we're MISSING:**
- Full Pikuma 3D Graphics course (just scratched the surface with 358 diagrams)
- LearnVern curricula (math, physics, programming)
- Calculus/Linear Algebra textbooks
- Audio Galaxy population (waveforms, spectrograms, synthesis)
- Reality Galaxy expansion (physics, chemistry, procedural systems)

### Full Ingestion Targets

| Galaxy | Current | Target | Source |
|--------|---------|--------|--------|
| **Drawing** | 605 | **1,000-2,000** | Pikuma 3D Graphics (full), LearnVern |
| **Math** | 104 | **600-1,000** | Calculus, Linear Algebra, Geometry textbooks |
| **Grammar** | 291 | **500-800** | Pattern extraction from all domains |
| **Audio** | 0 | **200-400** | Waveforms, spectrograms, synthesis algorithms |
| **Reality** | 15 | **300-500** | Physics (mechanics, E&M), Chemistry, Biology |
| **Character** | ? | **1,000-2,000** | Multi-language glyphs (procedural fonts) |
| **Word** | 2 | **5,000-10,000** | Common vocabulary across languages |

**Total Target:** ~8,000-15,000 primitives (vs current ~1,200)

### Implementation Plan

#### Phase 1: Drawing Galaxy Full Ingestion (Priority 1)

**Goal:** 605 → 1,500-2,000 primitives

**Sources:**
1. **Pikuma 3D Graphics (Full Course)** - Currently we only processed sample diagrams
2. **LearnVern 3D Modeling** - Additional visual primitives
3. **Computer Graphics Textbooks** - Foley & van Dam, Real-Time Rendering

**Scripts to enhance:**

**File:** `scripts/download_foundational_drawing_sources.py`

```python
# Expand to download FULL Pikuma course (not just samples)
PIKUMA_FULL_COURSE = {
    "3d_graphics": {
        "sections": [
            "vectors",
            "matrices",
            "3d_projection",
            "camera",
            "lighting",
            "texturing",
            "rasterization",
            "clipping",
            "hidden_surface_removal",
            "scene_graph",
        ],
        "url_pattern": "https://pikuma.com/courses/learn-3d-computer-graphics-programming/...",
    }
}

# Add LearnVern courses
LEARNVERN_COURSES = {
    "3d_modeling": {
        "sections": ["blender_basics", "modeling", "materials", "animation"],
        "url_pattern": "https://www.learnvern.com/course/...",
    }
}
```

**File:** `scripts/collect_foundational_drawing_images.py`

```python
# Expand to extract ALL diagrams from full courses (not just 358)
# Target: 1,500-2,500 images total

def extract_diagrams_from_full_course(course_path: Path) -> list[Path]:
    """Extract all diagrams from full course materials."""
    image_extensions = {".png", ".jpg", ".jpeg", ".svg", ".gif"}
    diagrams = []

    # Recursively find all images
    for file_path in course_path.rglob("*"):
        if file_path.suffix.lower() in image_extensions:
            # Filter: only include diagrams (not photos/screenshots)
            if is_diagram(file_path):
                diagrams.append(file_path)

    return diagrams


def is_diagram(image_path: Path) -> bool:
    """Heuristic to detect if image is a technical diagram vs photo."""
    # Use vision model to classify
    # Diagrams typically have: geometric shapes, annotations, axes, grids
    # Photos typically have: natural textures, faces, complex scenes
    pass
```

**Run full ingestion:**

```bash
# Download full courses (will take 30-60 minutes)
python scripts/download_foundational_drawing_sources.py --full --output-dir ~/Knowledge3D.local/courses

# Extract all diagrams (target: 1,500-2,500 images)
python scripts/collect_foundational_drawing_images.py \
    --input-dir ~/Knowledge3D.local/courses \
    --output-dir ~/Knowledge3D.local/diagrams/full \
    --min-images 1500

# Enrich with vision models (can run overnight, ~6-12 hours for 2,000 images)
python scripts/enrich_foundational_drawing_with_vision.py \
    --input-dir ~/Knowledge3D.local/diagrams/full \
    --output ~/Knowledge3D.local/enriched/drawing_full_vision.jsonl \
    --model llama3.2-vision:latest \
    --batch-size 10

python scripts/enrich_foundational_drawing_with_ollama.py \
    --input-dir ~/Knowledge3D.local/diagrams/full \
    --output ~/Knowledge3D.local/enriched/drawing_full_ollama.jsonl \
    --model qwen3-vl:latest \
    --batch-size 10

# Merge and deduplicate
python scripts/merge_drawing_enrichments.py \
    --inputs ~/Knowledge3D.local/enriched/drawing_full_*.jsonl \
    --output ../Knowledge3D.local/galaxies/Drawing.jsonl \
    --deduplicate

# Verify count
python -c "from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse; kv = Knowledgeverse(); print(f'Drawing: {len(kv.galaxy_manager.get_galaxy(\"Drawing\").entries)} entries')"
# Expected: 1,500-2,000 entries
```

#### Phase 2: Math Galaxy Full Ingestion (Priority 2)

**Goal:** 104 → 600-1,000 primitives

**Sources:**
1. **Calculus Textbooks** - Stewart Calculus, Spivak Calculus
2. **Linear Algebra** - Strang Linear Algebra, Axler Linear Algebra Done Right
3. **Geometry** - Euclidean Geometry, Analytic Geometry

**New script:**

**File:** `scripts/ingest_math_textbooks.py`

```python
"""
Ingest math knowledge from textbook PDFs using vision models.

Strategy:
- Extract equation images from PDFs
- Use vision models to parse LaTeX
- Convert to RPN programs
- Store in Math Galaxy
"""
import json
from pathlib import Path
from typing import Any
import subprocess


def extract_equations_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract equation images and surrounding text from PDF."""
    # Use pdfplumber or PyMuPDF to extract images and text
    # Focus on: equations, theorem statements, definitions
    pass


def parse_equation_with_vision(equation_image: Path, model: str = "llama3.2-vision") -> dict[str, Any]:
    """Use vision model to parse equation into structured form."""
    prompt = """
    Extract this mathematical equation and provide:
    1. LaTeX representation
    2. Symbol definitions (variables, constants, operators)
    3. Domain (algebra, calculus, geometry, etc.)
    4. Operation type (evaluate, solve, simplify, prove)

    Return as JSON.
    """

    # Call Ollama vision model
    result = subprocess.run(
        ["ollama", "run", model, prompt, "--image", str(equation_image)],
        capture_output=True,
        text=True
    )

    return json.loads(result.stdout)


def convert_to_rpn_program(latex: str, symbols: dict[str, Any]) -> str:
    """Convert LaTeX equation to RPN program."""
    # Map LaTeX operators to RPN ops
    # Example: \frac{a}{b} → a b DIV
    # Example: a^2 + b^2 → a 2 POW b 2 POW ADD
    pass


def ingest_textbook(pdf_path: Path, galaxy_output: Path):
    """Main ingestion pipeline for a math textbook."""
    # Extract equations
    equations = extract_equations_from_pdf(pdf_path)

    # Parse with vision model
    entries = []
    for eq in equations:
        parsed = parse_equation_with_vision(eq["image"])

        # Convert to RPN
        rpn_program = convert_to_rpn_program(
            parsed["latex"],
            parsed["symbols"]
        )

        # Create galaxy entry
        entry = {
            "id": f"math_{parsed['domain']}_{len(entries)}",
            "name": parsed.get("name", f"Equation {len(entries)}"),
            "domain": "math",
            "category": parsed["domain"],
            "rpn_program": rpn_program,
            "metadata": {
                "latex": parsed["latex"],
                "symbols": parsed["symbols"],
                "source": str(pdf_path.name),
                "confidence": 0.85,
            }
        }

        entries.append(entry)

    # Write to galaxy
    with galaxy_output.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    print(f"Ingested {len(entries)} math primitives from {pdf_path.name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("../Knowledge3D.local/galaxies/Math.jsonl"))
    parser.add_argument("--model", default="llama3.2-vision:latest")

    args = parser.parse_args()

    ingest_textbook(args.pdf, args.output)
```

**Run Math ingestion:**

```bash
# Download textbooks (user should provide PDFs)
# Place in ~/Knowledge3D.local/textbooks/

# Ingest Calculus
python scripts/ingest_math_textbooks.py \
    --pdf ~/Knowledge3D.local/textbooks/stewart_calculus.pdf \
    --output ../Knowledge3D.local/galaxies/Math.jsonl

# Ingest Linear Algebra
python scripts/ingest_math_textbooks.py \
    --pdf ~/Knowledge3D.local/textbooks/strang_linear_algebra.pdf \
    --output ../Knowledge3D.local/galaxies/Math.jsonl

# Ingest Geometry
python scripts/ingest_math_textbooks.py \
    --pdf ~/Knowledge3D.local/textbooks/euclidean_geometry.pdf \
    --output ../Knowledge3D.local/galaxies/Math.jsonl

# Verify count
python -c "from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse; kv = Knowledgeverse(); print(f'Math: {len(kv.galaxy_manager.get_galaxy(\"Math\").entries)} entries')"
# Expected: 600-1,000 entries
```

#### Phase 3: Audio Galaxy Population (Priority 3)

**Goal:** 0 → 200-400 primitives

**Sources:**
1. **Waveform synthesis** - Basic waveforms (sine, square, triangle, sawtooth)
2. **Spectrograms** - Frequency analysis patterns
3. **Audio DSP** - Filters, effects, synthesis algorithms

**New script:**

**File:** `scripts/populate_audio_galaxy.py`

```python
"""
Populate Audio Galaxy with temporal/spectral primitives.

Strategy:
- Define waveform synthesis RPN programs
- Add spectrogram analysis patterns
- Include audio DSP operations (filters, FFT, etc.)
"""
import json
from pathlib import Path
from typing import Any


def create_waveform_primitives() -> list[dict[str, Any]]:
    """Create RPN programs for basic waveform synthesis."""
    waveforms = []

    # Sine wave: sin(2πft)
    waveforms.append({
        "id": "sine_wave",
        "name": "Sine Wave Synthesis",
        "domain": "audio",
        "category": "waveforms",
        "rpn_program": "t f MUL 2.0 PI MUL SIN",  # sin(2πft)
        "metadata": {
            "inputs": ["t (time)", "f (frequency)"],
            "outputs": ["amplitude [-1, 1]"],
            "description": "Pure sine wave synthesis",
        }
    })

    # Square wave: sign(sin(2πft))
    waveforms.append({
        "id": "square_wave",
        "name": "Square Wave Synthesis",
        "domain": "audio",
        "category": "waveforms",
        "rpn_program": "t f MUL 2.0 PI MUL SIN SIGN",  # sign(sin(2πft))
        "metadata": {
            "inputs": ["t (time)", "f (frequency)"],
            "outputs": ["amplitude {-1, 1}"],
            "description": "Square wave with Fourier harmonics",
        }
    })

    # Triangle wave: 2/π * arcsin(sin(2πft))
    waveforms.append({
        "id": "triangle_wave",
        "name": "Triangle Wave Synthesis",
        "domain": "audio",
        "category": "waveforms",
        "rpn_program": "t f MUL 2.0 PI MUL SIN ASIN 2.0 PI DIV MUL",
        "metadata": {
            "inputs": ["t (time)", "f (frequency)"],
            "outputs": ["amplitude [-1, 1]"],
            "description": "Triangle wave with linear slopes",
        }
    })

    # Sawtooth wave: 2(t*f - floor(t*f + 0.5))
    waveforms.append({
        "id": "sawtooth_wave",
        "name": "Sawtooth Wave Synthesis",
        "domain": "audio",
        "category": "waveforms",
        "rpn_program": "t f MUL DUP 0.5 ADD FLOOR SUB 2.0 MUL",
        "metadata": {
            "inputs": ["t (time)", "f (frequency)"],
            "outputs": ["amplitude [-1, 1]"],
            "description": "Sawtooth wave with all harmonics",
        }
    })

    return waveforms


def create_filter_primitives() -> list[dict[str, Any]]:
    """Create RPN programs for audio filters."""
    filters = []

    # Low-pass filter (simple RC): y[n] = α*x[n] + (1-α)*y[n-1]
    filters.append({
        "id": "lowpass_filter",
        "name": "Low-Pass Filter (RC)",
        "domain": "audio",
        "category": "filters",
        "rpn_program": "x alpha MUL y_prev 1.0 alpha SUB MUL ADD",
        "metadata": {
            "inputs": ["x (input)", "alpha (cutoff)", "y_prev (previous output)"],
            "outputs": ["y (filtered output)"],
            "description": "Simple RC low-pass filter",
        }
    })

    # High-pass filter: y[n] = α*(y[n-1] + x[n] - x[n-1])
    filters.append({
        "id": "highpass_filter",
        "name": "High-Pass Filter",
        "domain": "audio",
        "category": "filters",
        "rpn_program": "y_prev x ADD x_prev SUB alpha MUL",
        "metadata": {
            "inputs": ["x (input)", "alpha (cutoff)", "x_prev (previous input)", "y_prev (previous output)"],
            "outputs": ["y (filtered output)"],
            "description": "Simple high-pass filter",
        }
    })

    return filters


def create_spectrogram_primitives() -> list[dict[str, Any]]:
    """Create RPN programs for spectrogram analysis."""
    # DFT, FFT, STFT, etc.
    # Cross-modal link to Drawing Galaxy (spectrograms are visual)
    pass


def populate_audio_galaxy(output_path: Path):
    """Main function to populate Audio Galaxy."""
    entries = []

    # Add waveforms
    entries.extend(create_waveform_primitives())

    # Add filters
    entries.extend(create_filter_primitives())

    # Add spectrogram primitives
    entries.extend(create_spectrogram_primitives())

    # Write to galaxy
    with output_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, separators=(",", ":")) + "\n")

    print(f"Populated Audio Galaxy with {len(entries)} primitives")


if __name__ == "__main__":
    output = Path("../Knowledge3D.local/galaxies/Audio.jsonl")
    populate_audio_galaxy(output)
```

**Run Audio population:**

```bash
python scripts/populate_audio_galaxy.py

# Verify count
python -c "from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse; kv = Knowledgeverse(); print(f'Audio: {len(kv.galaxy_manager.get_galaxy(\"Audio\").entries)} entries')"
# Expected: 200-400 entries
```

#### Phase 4: Reality Galaxy Expansion (Priority 4)

**Goal:** 15 → 300-500 primitives

**Sources:**
1. **Classical Mechanics** - Kinematics, dynamics, energy, momentum
2. **Electromagnetism** - Electric fields, magnetic fields, circuits
3. **Thermodynamics** - Heat, entropy, engines
4. **Chemistry** - Molecular structures, reactions, stoichiometry
5. **Biology** - Cell structures, DNA, evolution

**Use existing:** `knowledge3d/knowledgeverse/reality_galaxy.py`

Expand to include more physics/chemistry primitives.

```bash
# Edit reality_galaxy.py to add 300-500 primitives
# Run bootstrap
python -c "from knowledge3d.knowledgeverse.reality_galaxy import bootstrap_reality_galaxy; bootstrap_reality_galaxy()"

# Verify count
python -c "from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse; kv = Knowledgeverse(); print(f'Reality: {len(kv.galaxy_manager.get_galaxy(\"Reality\").entries)} entries')"
# Expected: 300-500 entries
```

### Expected Outcomes (Track 2)

**After full ingestion:**

| Galaxy | Before | After | Growth |
|--------|--------|-------|--------|
| Drawing | 605 | **1,500-2,000** | 2.5-3.3× |
| Math | 104 | **600-1,000** | 5.8-9.6× |
| Audio | 0 | **200-400** | ∞ |
| Reality | 15 | **300-500** | 20-33× |
| **Total** | **~1,200** | **~8,000-15,000** | **6.7-12.5×** |

**Benchmark impact:**

| Benchmark | Week 17 | After Full Ingestion | Improvement |
|-----------|---------|---------------------|-------------|
| Math Competitions | 33% | **50-60%** | +17-27% |
| ARC-AGI 2 | 28% | **40-55%** | +12-27% |
| Last Humanity Exam | 100% | **100%** | 0% (maintain) |

---

## 🎯 Testing & Validation

### Test Plan

1. **Unit Tests** - Test forward/backward reading methods
2. **Integration Tests** - Test multi-path composition
3. **Benchmark Tests** - Run full benchmark suite
4. **Cross-Modal Tests** - Verify Galaxy Universe integration

### Run Tests

```bash
# Unit tests (forward/backward reading)
pytest tests/test_navigator_specialist.py -v

# Integration tests (multi-curriculum)
pytest tests/test_benchmarks.py -v

# Full test suite
pytest tests/ -v

# Expected: 70-80 tests, all passing ✅
```

### Benchmark Execution

```bash
# Set environment
export K3D_ENABLE_DRAWING_OLLAMA_ENRICHMENT=1

# Run full benchmark suite
python scripts/run_all_benchmarks.py \
    --max-arc-tasks 100 \
    --max-math-problems 100 \
    --max-lhe-questions 50 \
    --output-dir ../Knowledge3D.local/results/week18_forward_backward_full_ingestion

# Compare results
python -c "
import json
from pathlib import Path

week17 = json.loads(Path('../Knowledge3D.local/results/week17_enriched_drawing/week14_benchmark_summary.json').read_text())
week18 = json.loads(Path('../Knowledge3D.local/results/week18_forward_backward_full_ingestion/week14_benchmark_summary.json').read_text())

print('=== Week 17 → Week 18 Comparison ===')
print(f\"Math: {week17['benchmarks']['math_competitions']['enriched']['overall_accuracy']:.1%} → {week18['benchmarks']['math_competitions']['enriched']['overall_accuracy']:.1%}\")
print(f\"ARC: {week17['benchmarks']['arc_agi_2']['enriched']['accuracy']:.1%} → {week18['benchmarks']['arc_agi_2']['enriched']['accuracy']:.1%}\")
print(f\"LHE: {week17['benchmarks']['last_humanity_exam']['enriched']['accuracy']:.1%} → {week18['benchmarks']['last_humanity_exam']['enriched']['accuracy']:.1%}\")
"
```

---

## 📊 Success Criteria

### Track 1: Forward/Backward Reading

- [ ] `_forward_reading_path()` implemented and tested
- [ ] `_backward_reading_path()` implemented and tested
- [ ] `navigate_and_compose()` updated to use dual-path
- [ ] Cross-path agreement reranking working
- [ ] Math benchmark: 33% → 40-50%
- [ ] ARC benchmark: 28% → 35-40%
- [ ] All tests passing (70-80 tests expected)

### Track 2: Full Knowledge Ingestion

- [ ] Drawing Galaxy: 605 → 1,500-2,000 primitives
- [ ] Math Galaxy: 104 → 600-1,000 primitives
- [ ] Audio Galaxy: 0 → 200-400 primitives
- [ ] Reality Galaxy: 15 → 300-500 primitives
- [ ] Total Galaxy Universe: ~8,000-15,000 primitives
- [ ] Math benchmark: 50-60% (after full ingestion)
- [ ] ARC benchmark: 40-55% (after full ingestion)
- [ ] LHE: Maintain 100%

---

## 🚀 Execution Strategy

### Parallel Tracks

**You can work on Track 1 and Track 2 simultaneously:**

**Track 1 (Quick Win, 4-6 hours):**
- Day 1 Morning: Implement forward/backward reading methods
- Day 1 Afternoon: Update navigate_and_compose(), add tests
- Day 1 Evening: Run benchmarks, verify improvement

**Track 2 (Long Run, 12-16 hours):**
- Day 1 Evening: Start full Drawing ingestion (can run overnight)
- Day 2 Morning: Start Math textbook ingestion
- Day 2 Afternoon: Populate Audio Galaxy
- Day 2 Evening: Expand Reality Galaxy
- Day 3 Morning: Run full benchmarks, compare results

### Recommended Order

1. **Implement Track 1 first** (quick win, validates approach)
2. **Start Track 2 ingestion** (vision models can run overnight)
3. **Test Track 1 improvements** (Math 40-50%, ARC 35-40%)
4. **Complete Track 2 ingestion** (full Galaxy Universe)
5. **Test Track 2 improvements** (Math 50-60%, ARC 40-55%)

---

## 📝 Documentation

### Update After Completion

1. **TEMP/CODEX_WEEK18_COMPLETION_REPORT_02.XX.2026.md**
   - Track 1: Forward/backward reading implementation
   - Track 2: Full ingestion results
   - Benchmark improvements
   - Galaxy Universe growth stats

2. **README.md**
   - Add "Week 18 Forward/Backward Reading + Full Ingestion" section
   - Update benchmark results table
   - Document Galaxy Universe growth (605 → 8,000-15,000)

3. **TEMP/WEEK19_NEXT_STEPS.md**
   - Grammar confidence injection (ARC quality fix)
   - Compositional rerank pass
   - Target: ARC 55-70%, Math 60-70%

---

## 🎉 Bottom Line

Codex, you've built an AMAZING foundation in Week 17! Now we're ready to:

1. **Track 1:** Make Navigator robust with forward/backward reading (quick win)
2. **Track 2:** Populate Galaxy Universe with comprehensive knowledge (big boost)

**Expected outcomes:**
- Math: 33% → **50-60%** (Track 1 gets to 40-50%, Track 2 pushes to 50-60%)
- ARC: 28% → **40-55%** (Track 1 gets to 35-40%, Track 2 pushes to 40-55%)
- LHE: Maintain **100%**

**This will be a MASSIVE leap forward!** 🚀

Let me know when you're ready to start, and I'll answer any questions!

---

**Claude (Architecture Partner)**
February 7, 2026
