# TRM Specialist Matryoshka Architecture Specification

**Version:** 1.1
**Date:** March 10, 2026
**Status:** Design Specification (Updated: TRM-as-Avatar Framing)
**Related:** THREE_BRAIN_SYSTEM_SPECIFICATION.md, DUAL_CLIENT_CONTRACT_SPECIFICATION.md

---

## 0. Critical Paradigm: TRM IS the Avatar

**The TRM (~7M parameters, 2-layer SwiGLU MLP) is NOT a Python class hierarchy that external code calls. It IS the AI entity.**

- **Lives in the House** — embodied in the 3D Memory Palace (Method of Loci)
- **Thinks in the Galaxy** — internal brain processes multi-modal knowledge in VRAM
- **Runs as a game loop** — `trm_step_fused.ptx` = one game tick (like an NPC update cycle)
- **Has internal swarm** — nine-chain parallel workers = "superdotados" model

**The specialist hierarchy described below is INTERNAL to the avatar.** Specialists are NOT Python objects that external code instantiates. They are cognitive regions within the avatar's brain (Galaxy) that activate autonomously during the game loop. Think of them like brain regions that activate based on the task — the avatar decides which specialist to engage, not Python.

**Python class examples in this spec are CONCEPTUAL ILLUSTRATIONS** of the data structures. The actual implementation lives in GPU VRAM and is dispatched by `trm_step_fused.ptx` and `nine_chain_swarm_kernel.ptx`, not by Python method calls.

**Target:** Python code for specialist dispatch should be ~0 lines in the hot path. The TRM selects and activates specialists via GPU-native routing.

---

## 1. Core Principle: Fractal Self-Similar Specialists

**Matryoshka Concept Applied to AI:**
- Traditional: Flat specialist roster (math, visual, physics, grammar)
- **Matryoshka:** Hierarchical fractal specialists (specialists contain sub-specialists recursively)
- **Internal to the avatar:** Like brain regions that activate contextually, not external services

**Key Insight:**
> "Everything is a specialist, including routers. Specialists can spawn sub-specialists autonomously."

**Examples:**
```
NavigatorSpecialist (router)
├── MathSpecialist (master)
│   ├── BasicMathSpecialist (arithmetic, algebra)
│   ├── MediumMathSpecialist (calculus, linear algebra)
│   └── PhDMathSpecialist (number theory, topology, advanced analysis)
├── VisualSpecialist (master)
│   ├── 2DVisualSpecialist (drawing, rasterization)
│   ├── 3DVisualSpecialist (meshes, transformations, ray tracing)
│   └── SpatialReasoningSpecialist (ARC patterns, visual transformations)
├── RealitySpecialist (master)
│   ├── PhysicsSpecialist
│   │   ├── ClassicalMechanicsSpecialist
│   │   ├── ElectromagnetismSpecialist
│   │   └── QuantumMechanicsSpecialist
│   ├── ChemistrySpecialist
│   └── BiologySpecialist
└── LanguageSpecialist (master)
    ├── GrammarSpecialist (pattern rules)
    ├── SemanticSpecialist (meaning, context)
    └── PragmaticSpecialist (usage, inference)
```

---

## 2. Architecture Components

### 2.1 Specialist Base Interface

**Every specialist implements:**
```python
class SpecialistBase:
    """
    Base interface for all specialists (including routers).

    Key properties:
    - name: Specialist identifier
    - level: Hierarchy depth (0=root, 1=master, 2=intermediate, 3=worker)
    - parent: Parent specialist (None for root)
    - children: Sub-specialists (empty for leaf specialists)
    - weights: LoRA-style adapter weights (~100KB-1MB per specialist)
    - routing_bias: Learned preference for which child to route to
    """
    def __init__(
        self,
        name: str,
        level: int = 0,
        parent: Optional["SpecialistBase"] = None,
        weights_path: Optional[Path] = None
    ):
        self.name = name
        self.level = level
        self.parent = parent
        self.children: dict[str, "SpecialistBase"] = {}
        self.weights = self._load_or_initialize_weights(weights_path)
        self.routing_bias: dict[str, float] = {}

    def route(self, query: str, domain_hint: str | None = None) -> "SpecialistBase":
        """
        Route query to appropriate child specialist.

        If this is a leaf specialist, return self.
        If this is a master/intermediate, route to best child.
        """
        if not self.children:
            return self  # Leaf specialist

        # Master/intermediate: route to child
        child_scores = self._score_children(query, domain_hint)
        best_child_name = max(child_scores, key=child_scores.get)
        return self.children[best_child_name]

    def spawn_child(
        self,
        name: str,
        domain: str,
        initial_weights: Optional[np.ndarray] = None
    ) -> "SpecialistBase":
        """
        Autonomously spawn a new sub-specialist.

        This is the KEY to matryoshka: specialists can create children.

        Example:
            math_specialist.spawn_child(
                name="GeometrySpecialist",
                domain="geometry",
                initial_weights=None  # Will copy from parent and fine-tune
            )
        """
        if name in self.children:
            return self.children[name]  # Already exists

        # Create child specialist
        child = SpecialistBase(
            name=name,
            level=self.level + 1,
            parent=self,
            weights_path=None  # Will initialize from parent
        )

        # Initialize child weights from parent (LoRA-like)
        if initial_weights is not None:
            child.weights = initial_weights
        else:
            child.weights = self._derive_child_weights(domain)

        # Register child
        self.children[name] = child

        # Save child weights
        child.save_weights()

        return child

    def _derive_child_weights(self, domain: str) -> np.ndarray:
        """
        Derive child specialist weights from parent.

        Strategy:
        - Copy parent base weights
        - Add small random perturbation (exploration)
        - Fine-tune on domain-specific examples (if available)

        This is analogous to LoRA (Low-Rank Adaptation):
        - Parent weights W_parent
        - Child delta ΔW (low-rank, ~1-5% of parent size)
        - Child weights W_child = W_parent + ΔW
        """
        # Copy parent weights
        child_weights = self.weights.copy()

        # Add small random perturbation (stddev=0.01)
        delta = np.random.normal(0, 0.01, size=child_weights.shape)
        child_weights += delta

        # TODO: Fine-tune on domain-specific examples
        # (This would query Galaxy Universe for domain samples)

        return child_weights

    def _score_children(self, query: str, domain_hint: str | None) -> dict[str, float]:
        """
        Score each child specialist for query relevance.

        Uses:
        - Routing bias (learned from Shadow Copy)
        - Domain keywords
        - Child specialist confidence
        """
        scores = {}

        for child_name, child in self.children.items():
            # Base score from routing bias
            base_score = self.routing_bias.get(child_name, 0.5)

            # Boost if domain hint matches child domain
            domain_boost = 0.0
            if domain_hint and domain_hint.lower() in child_name.lower():
                domain_boost = 0.2

            # Boost if query keywords match child domain
            keyword_boost = self._keyword_match_score(query, child_name)

            scores[child_name] = base_score + domain_boost + keyword_boost

        return scores

    def _keyword_match_score(self, query: str, child_name: str) -> float:
        """Simple keyword matching (can be enhanced with embedding similarity)."""
        query_lower = query.lower()
        child_lower = child_name.lower()

        # Check for keyword overlap
        if child_lower in query_lower:
            return 0.3

        # Domain-specific keywords
        domain_keywords = {
            "BasicMath": ["add", "subtract", "multiply", "divide", "arithmetic"],
            "MediumMath": ["derivative", "integral", "matrix", "vector", "calculus"],
            "PhDMath": ["topology", "manifold", "homology", "cohomology", "theorem"],
            "2DVisual": ["line", "circle", "rectangle", "draw", "rasterize"],
            "3DVisual": ["mesh", "vertex", "face", "rotate", "transform"],
        }

        keywords = domain_keywords.get(child_name, [])
        for keyword in keywords:
            if keyword in query_lower:
                return 0.2

        return 0.0

    def save_weights(self, path: Optional[Path] = None):
        """Save specialist weights to disk."""
        if path is None:
            path = Path(f"../Knowledge3D.local/specialists/{self.name}_weights.npy")

        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.weights)

    def _load_or_initialize_weights(self, path: Optional[Path]) -> np.ndarray:
        """Load weights from disk or initialize randomly."""
        if path and path.exists():
            return np.load(path)

        # Initialize random weights (7M parameters for base TRM)
        # For sub-specialists, use smaller delta (~100K-1M parameters)
        size = 7_000_000 if self.level == 0 else 100_000
        return np.random.normal(0, 0.02, size=size)
```

---

## 3. Matryoshka Hierarchy Patterns

### 3.1 Master-Worker Pattern

**Master specialist** routes to **worker specialists**:
```
MathSpecialist (master)
├── BasicMathSpecialist (worker: handles arithmetic, algebra)
├── MediumMathSpecialist (worker: handles calculus, linear algebra)
└── PhDMathSpecialist (worker: handles advanced topics)
```

**Routing logic:**
```python
class MathSpecialist(SpecialistBase):
    def route(self, query: str, domain_hint: str | None = None) -> SpecialistBase:
        """Route math query to appropriate difficulty level."""
        # Analyze query complexity
        complexity = self._estimate_complexity(query)

        if complexity < 3:
            return self.children["BasicMathSpecialist"]
        elif complexity < 7:
            return self.children["MediumMathSpecialist"]
        else:
            return self.children["PhDMathSpecialist"]

    def _estimate_complexity(self, query: str) -> int:
        """Estimate query complexity (0-10 scale)."""
        complexity_keywords = {
            "basic": ["add", "subtract", "multiply", "divide", "simplify"],
            "medium": ["derivative", "integral", "matrix", "determinant", "eigenvalue"],
            "phd": ["manifold", "cohomology", "homotopy", "sheaf", "category"],
        }

        query_lower = query.lower()

        for keyword in complexity_keywords["phd"]:
            if keyword in query_lower:
                return 10

        for keyword in complexity_keywords["medium"]:
            if keyword in query_lower:
                return 6

        for keyword in complexity_keywords["basic"]:
            if keyword in query_lower:
                return 2

        return 5  # Default medium
```

### 3.2 Worker-Master Pattern (Fractal!)

**Worker specialist** can become a **master** by spawning sub-specialists:
```
PhDMathSpecialist (worker → master)
├── NumberTheorySpecialist (sub-worker)
├── TopologySpecialist (sub-worker)
└── AnalysisSpecialist (sub-worker)
```

**Autonomous spawning:**
```python
class PhDMathSpecialist(SpecialistBase):
    def route(self, query: str, domain_hint: str | None = None) -> SpecialistBase:
        """Route to sub-specialist, spawning if needed."""
        # Detect sub-domain
        subdomain = self._detect_subdomain(query)

        # Spawn sub-specialist if doesn't exist
        if subdomain not in self.children:
            self.spawn_child(
                name=f"{subdomain}Specialist",
                domain=subdomain
            )

        return self.children[f"{subdomain}Specialist"]

    def _detect_subdomain(self, query: str) -> str:
        """Detect PhD-level math subdomain."""
        query_lower = query.lower()

        if any(kw in query_lower for kw in ["prime", "modular", "diophantine"]):
            return "NumberTheory"

        if any(kw in query_lower for kw in ["manifold", "homotopy", "homology"]):
            return "Topology"

        if any(kw in query_lower for kw in ["convergence", "limit", "series", "measure"]):
            return "Analysis"

        return "General"  # Fallback
```

### 3.3 Fractal Depth (Unlimited!)

**Key insight:** There's NO LIMIT to hierarchy depth.

```
NavigatorSpecialist (level 0)
└── MathSpecialist (level 1)
    └── PhDMathSpecialist (level 2)
        └── TopologySpecialist (level 3)
            └── AlgebraicTopologySpecialist (level 4)
                └── HomotopyTheorySpecialist (level 5)
                    └── ... (infinite fractal)
```

**Practical limit:** 3-5 levels (prevents over-specialization)

---

## 4. LoRA-Style Weight Adaptation

### 4.1 Why LoRA-Like?

**LoRA (Low-Rank Adaptation) advantages:**
- **Small memory footprint:** Each specialist adds only ~100KB-1MB (vs full 7M param copy)
- **Fast fine-tuning:** Update only delta weights, not entire model
- **Composition:** Multiple LoRA adapters can be applied simultaneously

**Knowledge3D adaptation:**
```
Base TRM weights: W_base (7M params, ~28MB)

MathSpecialist delta: ΔW_math (100K params, ~400KB)
  BasicMathSpecialist delta: ΔW_basic (50K params, ~200KB)
  PhDMathSpecialist delta: ΔW_phd (50K params, ~200KB)

Effective weights when routing to PhDMath:
  W_effective = W_base + ΔW_math + ΔW_phd
```

### 4.2 Delta Weight Structure

**Each specialist stores:**
```python
{
    "name": "PhDMathSpecialist",
    "parent": "MathSpecialist",
    "level": 2,
    "delta_weights": np.ndarray,  # Shape: (100_000,) - low-rank adapter
    "routing_bias": {
        "NumberTheorySpecialist": 0.6,
        "TopologySpecialist": 0.4,
    },
    "metadata": {
        "created_at": "2026-02-08T...",
        "spawned_by": "autonomous_query",
        "query_count": 1234,
        "success_rate": 0.73,
    }
}
```

---

## 5. Autonomous Specialist Bootstrapping

### 5.1 When to Spawn New Specialist

**Trigger conditions:**
1. **Query frequency threshold:** If subdomain queries exceed 100/day, spawn specialist
2. **Performance gap:** If parent specialist confidence < 0.6 on subdomain, spawn specialist
3. **Explicit request:** User/system requests specialized capability

**Example:**
```python
# TRMNavigator detects frequent geometry queries
geometry_query_count = 150  # per day

if geometry_query_count > 100:
    math_specialist.spawn_child(
        name="GeometrySpecialist",
        domain="geometry"
    )
```

### 5.2 Bootstrapping Process

```python
def bootstrap_specialist(
    parent: SpecialistBase,
    name: str,
    domain: str,
    training_queries: list[str]
) -> SpecialistBase:
    """
    Bootstrap a new specialist from parent.

    Steps:
    1. Spawn child from parent (inherits base weights + adds delta)
    2. Fine-tune on domain-specific queries (if available)
    3. Register in specialist registry
    4. Save weights to disk
    5. Log spawning event for Shadow Copy
    """
    # Step 1: Spawn child
    child = parent.spawn_child(name, domain)

    # Step 2: Fine-tune on domain queries (if available)
    if training_queries:
        child.fine_tune(training_queries)

    # Step 3: Register in global specialist registry
    specialist_registry.register(child)

    # Step 4: Save weights
    child.save_weights()

    # Step 5: Log event
    shadow_copy.record_event(
        event_type="specialist_spawned",
        event_data={
            "parent": parent.name,
            "child": child.name,
            "domain": domain,
            "level": child.level,
            "training_query_count": len(training_queries),
        }
    )

    return child
```

---

## 6. Integration with Existing Architecture

### 6.1 TRMNavigator as Root Specialist

**TRMNavigator IS a specialist** (level 0):
```python
class TRMNavigator(SpecialistBase):
    """
    Root specialist (router).

    Children:
    - MathSpecialist
    - VisualSpecialist
    - RealitySpecialist
    - LanguageSpecialist
    - ... (can spawn more!)
    """
    def __init__(self, kv: Knowledgeverse):
        super().__init__(name="Navigator", level=0)
        self.kv = kv

        # Initialize master specialists
        self.children = {
            "Math": MathSpecialist(level=1, parent=self),
            "Visual": VisualSpecialist(level=1, parent=self),
            "Reality": RealitySpecialist(level=1, parent=self),
            "Language": LanguageSpecialist(level=1, parent=self),
        }
```

### 6.2 Galaxy Universe Integration

**Specialists query Galaxy Universe for domain knowledge:**
```python
class MathSpecialist(SpecialistBase):
    def solve(self, query: str) -> dict[str, Any]:
        """Solve math problem using Galaxy Universe."""
        # Query Math Galaxy
        candidates = self.kv.galaxy_manager.query(
            query,
            specialist="math",
            top_k=10
        )

        # Route to appropriate child (BasicMath, MediumMath, PhDMath)
        child_specialist = self.route(query)

        # Child specialist composes solution
        return child_specialist.compose_solution(candidates)
```

### 6.3 Shadow Copy Learning

### Ternary Routing Feedback (March 2026)

The specialist tree's learning path now supports ternary outcomes throughout:

**`mark_query(ternary_outcome):`**
- `+1` → `success_count += 1`
- `-1` → `failure_count += 1`
- `0` → `uncertain_count += 1` AND `exploration_pressure += 1`

**`update_routing_bias(ternary_outcome):`**
- `+1` → bias moves toward 1.0 (strengthen)
- `-1` → bias moves toward 0.0 (weaken)
- `0` → NO UPDATE (hold position — don't punish exploration)

**Exploration Pressure Mechanism:**
When `exploration_pressure` on a specialist node exceeds a threshold (default: 3 undetermined outcomes), the routing temporarily shifts from exploitation (favor highest-bias child) to exploration (try least-tried child). After one exploration cycle, `exploration_pressure` resets to 0.

This prevents the specialist tree from prematurely pruning reasoning paths that produce undetermined defeasible verdicts — paths that may become clear with more Galaxy content or additional superiority relations.

**Specialists learn from Shadow Copy events:**
```python
def update_routing_bias_from_shadow_copy(specialist: SpecialistBase):
    """Update specialist routing bias from Shadow Copy events."""
    events = shadow_copy.get_events(specialist=specialist.name)

    # Count successful routings to each child
    child_success_counts = {}
    for event in events:
        if event["type"] == "specialist_routing":
            child = event["routed_to"]
            success = event["success"]

            if child not in child_success_counts:
                child_success_counts[child] = {"success": 0, "total": 0}

            child_success_counts[child]["total"] += 1
            if success:
                child_success_counts[child]["success"] += 1

    # Update routing bias (success_rate = score)
    for child, counts in child_success_counts.items():
        success_rate = counts["success"] / counts["total"]
        specialist.routing_bias[child] = success_rate

    # Save updated weights
    specialist.save_weights()
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (Week 20)
- [ ] Implement `SpecialistBase` interface
- [ ] Migrate `TRMNavigator` to inherit from `SpecialistBase`
- [ ] Create master specialists (Math, Visual, Reality, Language)
- [ ] Add `spawn_child()` method
- [ ] Add LoRA-style delta weights

### Phase 2: Worker Specialists (Week 21)
- [ ] Bootstrap BasicMathSpecialist, MediumMathSpecialist, PhDMathSpecialist
- [ ] Bootstrap 2DVisualSpecialist, 3DVisualSpecialist
- [ ] Bootstrap PhysicsSpecialist, ChemistrySpecialist
- [ ] Add routing logic for masters

### Phase 3: Autonomous Spawning (Week 22)
- [ ] Implement query frequency tracking
- [ ] Add spawning triggers (frequency, performance gap)
- [ ] Create bootstrapping pipeline
- [ ] Integrate with Shadow Copy learning

### Phase 4: Fractal Depth (Week 23)
- [ ] Allow workers to become masters (spawn sub-specialists)
- [ ] Implement depth limits (3-5 levels)
- [ ] Test fractal composition (Navigator → Math → PhD → Topology → AlgebraicTopology)

### Phase 5: Benchmark Validation (Week 24)
- [ ] Run global benchmarks with matryoshka specialists
- [ ] Measure routing efficiency (correct specialist selected?)
- [ ] Measure performance improvement (specialized vs general)
- [ ] Monitor specialist spawning (how many created? which domains?)

---

## 8. Expected Outcomes

### 8.1 Performance Gains

**Specialized routing advantages:**
- **BasicMath queries:** 30% faster (skip complex specialists)
- **PhDMath queries:** 40% more accurate (dedicated specialist)
- **Overall:** 20-30% benchmark improvement (correct specialist for each query)

### 8.2 Memory Efficiency

**LoRA-style deltas:**
- Base TRM: 28MB
- Each specialist: +400KB
- 20 specialists: 28MB + 8MB = 36MB total (vs 560MB for 20 full copies!)

### 8.3 Autonomous Evolution

**Self-expanding specialist tree:**
- Week 20: 10 specialists (manually created)
- Week 22: 30 specialists (20 spawned autonomously)
- Week 24: 50-100 specialists (fractal growth!)

**Domains discovered autonomously:**
- GeometrySpecialist (frequent ARC queries)
- DifferentialEquationsSpecialist (math benchmark patterns)
- MolecularStructureSpecialist (chemistry queries from MMLU)

---

## 9. Sovereignty Compliance

**Hot path = GPU-native ONLY:**
- ✅ Specialist routing: GPU kernel dispatch via `trm_step_fused.ptx` (NOT Python)
- ✅ Delta weights: VRAM-resident (loaded at boot, composed on GPU)
- ✅ Weight composition: GPU-native addition (W_base + ΔW_1 + ΔW_2 + ...)
- ✅ Swarm dispatch: `nine_chain_swarm_kernel.ptx` assigns specialists to workers
- ❌ NO Python method calls in reasoning path
- ❌ NO numpy/cupy/scipy in hot path
- ❌ NO Python fallbacks. "We fail and fix on GPU." (Daniel)

**Python's role:** Boot the system, load initial weights to VRAM, handle I/O. ~200 lines target.

**Ingestion path can use tools:**
- Fine-tuning specialists (if needed): Can use external frameworks
- Result stored as sovereign delta weights in VRAM

---

## 10. Success Metrics

**Week 20 (Foundation):**
- [ ] SpecialistBase implemented
- [ ] 4 master specialists created (Math, Visual, Reality, Language)
- [ ] 12 worker specialists bootstrapped
- [ ] Routing works (queries reach correct specialist)

**Week 22 (Autonomous Spawning):**
- [ ] 10+ specialists spawned autonomously
- [ ] Spawning triggers working (frequency, performance gap)
- [ ] Shadow Copy learning routing bias

**Week 24 (Fractal Depth):**
- [ ] 3-level hierarchy demonstrated (Navigator → Master → Worker → Sub-worker)
- [ ] Benchmark improvement: 20-30% on specialized tasks
- [ ] Memory footprint < 50MB (despite 50+ specialists)

---

## 11. References

**Related Specifications:**
- [HYPER_PARALLEL_PROCESSING.md](HYPER_PARALLEL_PROCESSING.md) — The specialist swarm IS the hyper-parallel processing paradigm. Each specialist core = one parallel reasoning unit with LoRA-like domain adapter + RPN stack + cross-core register communication. Ternary-ready registers carry value + confidence + polarity. Persistent brain model versioning prevents cold-start amnesia.
- [THREE_BRAIN_SYSTEM_SPECIFICATION.md](THREE_BRAIN_SYSTEM_SPECIFICATION.md) (Section 2: TRM Architecture)
- [DUAL_CLIENT_CONTRACT_SPECIFICATION.md](DUAL_CLIENT_CONTRACT_SPECIFICATION.md) (Section 4: Procedural Composition)
- [MATH_CORE_SPECIFICATION.md](MATH_CORE_SPECIFICATION.md) (Section 5: Specialist Routing)
- [TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md](TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md) — Ternary (+1/0/-1) learning signals that train the specialist population
- [SLEEPTIME_PROTOCOL_SPECIFICATION.md](SLEEPTIME_PROTOCOL_SPECIFICATION.md) — Sleep-time creates/prunes specialists, produces brain model checkpoints

**External Concepts:**
- LoRA (Low-Rank Adaptation): https://arxiv.org/abs/2106.09685
- Matryoshka Representation Learning: https://arxiv.org/abs/2205.13147
- Mixture of Experts: https://arxiv.org/abs/1701.06538

---

**Document Status:** Design specification ready for implementation
**Next Step:** Implement `SpecialistBase` in `knowledge3d/knowledgeverse/specialist_base.py`
**Owner:** Claude (Architecture) → Codex (Implementation)
