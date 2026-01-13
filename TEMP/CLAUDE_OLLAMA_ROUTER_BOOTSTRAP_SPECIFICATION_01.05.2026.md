# Ollama-Bootstrapped Router Specialist Specification

**From**: Claude (Architecture Partner)
**To**: Codex (Implementation Lead)
**Date**: January 5, 2026
**Context**: Bootstrap router specialist using local Ollama LLM as data generator

---

## Problem Statement

**Current State**:
- Router specialist is integrated but **untrained** (uniform/random weights)
- Benchmark shows router matching patterns but **defaulting to same rule** (apply_constant_multiple_rule)
- **0% accuracy** - selected rules aren't executing or aren't the right rules
- Need training data: `(semantic_tags → grammar_rule, outcome_performance)`

**Traditional Approach** (slow):
1. Run benchmark → collect routing decisions
2. Filter successful decisions
3. Train router specialist
4. Re-run benchmark → repeat

**Ollama Approach** (fast):
1. Use local LLM to **generate synthetic training data**
2. LLM reasons: "product_rule pattern → use apply_product_rule grammar"
3. Bootstrap router weights from synthetic data
4. Validate on real benchmark

---

## Architecture: Ollama as Data Generator

### What Ollama Provides

**Reasoning capability**: Local LLM (deepseek-r1, qwen2.5, etc.) can reason about calculus concepts:
- "Which calculus rule applies to derivative of product?"
- "Answer: Product rule (f'g + fg')"
- "Grammar rule: apply_product_rule"

**Synthetic training examples**: Generate routing decisions WITHOUT running benchmarks:
```python
RoutingDecision(
    input_data=embed(["derivative", "product_rule"]),
    task_description="Find derivative of (x²)(x³)",
    specialist_weights={"apply_product_rule": 1.0},  # From Ollama reasoning
    outcome_performance=1.0,  # Assume correct
    timestamp="2026-01-05T..."
)
```

**Coverage**: Generate examples for ALL 9 theorem patterns (even rare ones with no benchmark matches)

---

## Implementation Strategy

### Phase 1: Ollama Data Generation

**File**: `scripts/generate_router_training_data_ollama.py`

**Goal**: Use Ollama to generate synthetic routing decisions for all theorem patterns

**Prompt Template**:
```python
ROUTING_PROMPT_TEMPLATE = """You are a calculus expert helping train a routing model.

Given a theorem pattern with semantic tags, identify which grammar rule should be used.

Theorem Pattern: {pattern_id}
Semantic Tags: {semantic_tags}
Domain: {domain}
Description: {description}

Available Grammar Rules:
1. apply_power_rule - derivative of x^n
2. apply_product_rule - derivative of f*g
3. apply_quotient_rule - derivative of f/g
4. apply_chain_rule - derivative of f(g(x))
5. apply_sum_rule - derivative of f+g
6. apply_constant_multiple_rule - derivative of c*f
7. apply_integration_by_parts - integral using integration by parts
8. apply_fundamental_theorem_calculus - definite integral evaluation
9. apply_pythagorean_identity - sin²θ + cos²θ = 1

Question: Which grammar rule should be used for this theorem pattern?

Answer with ONLY the grammar rule name (e.g., "apply_product_rule").
"""
```

**Example Generation Flow**:
```python
import ollama

def generate_routing_decision_ollama(
    pattern: Dict[str, Any],
    ollama_model: str = "deepseek-r1:7b"
) -> RoutingDecision:
    """Generate synthetic routing decision using Ollama."""

    # Build prompt
    prompt = ROUTING_PROMPT_TEMPLATE.format(
        pattern_id=pattern["pattern_id"],
        semantic_tags=", ".join(pattern["semantic_tags"]),
        domain=pattern["domain"],
        description=_describe_pattern(pattern)
    )

    # Query Ollama
    response = ollama.generate(
        model=ollama_model,
        prompt=prompt,
        options={"temperature": 0.0}  # Deterministic
    )

    # Parse grammar rule from response
    grammar_rule = _extract_grammar_rule(response["response"])

    # Create synthetic routing decision
    semantic_embedding = _embed_semantic_tags(pattern["semantic_tags"])

    return RoutingDecision(
        input_data=semantic_embedding,
        task_description=f"{pattern['pattern_id']} pattern",
        specialist_weights={grammar_rule: 1.0},  # Ollama's choice
        outcome_performance=1.0,  # Assume correct (will validate on benchmark)
        timestamp=datetime.now().isoformat()
    )
```

**Generate multiple examples per pattern** (with variations):
```python
def generate_training_dataset_ollama(
    theorem_patterns: List[Dict[str, Any]],
    examples_per_pattern: int = 10,
    ollama_model: str = "deepseek-r1:7b"
) -> List[RoutingDecision]:
    """Generate synthetic training dataset for all patterns."""

    routing_decisions = []

    for pattern in theorem_patterns:
        # Generate multiple examples with slight variations
        for i in range(examples_per_pattern):
            # Add noise to semantic tags (simulates real-world variation)
            noisy_pattern = _add_semantic_noise(pattern, noise_level=0.1)

            decision = generate_routing_decision_ollama(noisy_pattern, ollama_model)
            routing_decisions.append(decision)

            print(f"  Generated {pattern['pattern_id']} example {i+1}/{examples_per_pattern}")

    print(f"\n[Ollama] Generated {len(routing_decisions)} synthetic routing decisions")
    return routing_decisions
```

---

### Phase 2: Semantic Tag Embedding

**Challenge**: Router specialist expects `np.ndarray` input, not raw semantic tags

**Solution**: Embed semantic tags into fixed-size vectors

**Approach A: Simple Hashing** (fast, deterministic):
```python
def embed_semantic_tags_hash(tags: List[str], dim: int = 256) -> np.ndarray:
    """Embed semantic tags using hash-based projection."""
    embedding = np.zeros(dim, dtype=np.float32)

    for tag in tags:
        # Hash tag to multiple indices
        hash_val = hash(tag)
        for i in range(3):  # 3 positions per tag
            idx = (hash_val + i * 31) % dim
            embedding[idx] += 1.0

    # Normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm

    return embedding
```

**Approach B: TF-IDF** (better, uses corpus statistics):
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def embed_semantic_tags_tfidf(
    tags_list: List[List[str]],
    dim: int = 256
) -> List[np.ndarray]:
    """Embed semantic tags using TF-IDF."""

    # Convert tag lists to strings
    tag_strings = [" ".join(tags) for tags in tags_list]

    # Fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=dim)
    embeddings = vectorizer.fit_transform(tag_strings).toarray()

    return [emb.astype(np.float32) for emb in embeddings]
```

**Recommended**: Start with **Approach A** (hash-based) for simplicity, upgrade to TF-IDF later if needed.

---

### Phase 3: Bootstrap Router Specialist

**File**: `scripts/train_router_from_ollama_data.py`

**Goal**: Train router specialist using Ollama-generated synthetic data

**Workflow**:
```python
from knowledge3d.cranium.router_specialist import RouterSpecialistTrainer
from knowledge3d.cranium.adaptive_swarm import AdaptiveSwarmTRM

# Step 1: Generate synthetic training data
print("[OllamaBootstrap] Generating synthetic routing decisions...")
routing_decisions = generate_training_dataset_ollama(
    theorem_patterns=THEOREM_PATTERNS,
    examples_per_pattern=20,  # 20 examples × 9 patterns = 180 decisions
    ollama_model="deepseek-r1:7b"
)

# Step 2: Initialize swarm + router specialist
swarm = AdaptiveSwarmTRM(base_dims=256)
trainer = RouterSpecialistTrainer(swarm)

# Register router specialist
trainer.register_router_specialist(
    num_specialists=len(CALCULUS_RULES),  # 9 grammar rules
    router_dims=256,
    router_rank=16
)

# Step 3: Train router specialist from synthetic data
print("\n[OllamaBootstrap] Training router specialist...")
stats = trainer.train_from_history(
    routing_history=routing_decisions,
    epochs=10,
    filter_threshold=0.8,  # Only use high-confidence Ollama decisions
    learning_rate=0.001
)

# Step 4: Save trained router weights
swarm.save_specialists("ollama_bootstrapped_router.npz")

print(f"\n[OllamaBootstrap] ✓ Router specialist trained")
print(f"  Training samples: {stats['train_samples']}")
print(f"  Validation accuracy: {stats['candidate_metrics']['accuracy']:.2%}")
print(f"  Saved to: ollama_bootstrapped_router.npz")
```

---

### Phase 4: Validate on Real Benchmark

**After bootstrapping**, run MATH benchmark with trained router:

```bash
K3D_TRM_ENABLE_MULTISTEP=1 PYTHONPATH=. python \
  scripts/run_sovereign_math_benchmarks.py \
  --datasets math \
  --max-problems 100 \
  --shuffle --shuffle-seed 123 \
  --use-trm-navigator \
  --load-all-galaxies \
  --enable-book-galaxies \
  --book-galaxy-root /K3D/Knowledge3D.local/galaxies/books_v5_clean2 \
  --router-weights ollama_bootstrapped_router.npz \
  --verbose
```

**Expected improvements**:
- Router selects **diverse rules** (not just apply_constant_multiple_rule)
- Grammar rules **actually execute** (RPN engine invoked)
- Accuracy **improves from 0%** (target: ≥2%)

---

### Phase 5: Iterative Refinement (Optional)

**After first benchmark run**, collect real routing decisions:
1. Real problems → real routing outcomes
2. Mix Ollama synthetic data + real benchmark data
3. Re-train router specialist (continual learning)
4. Router improves from both synthetic reasoning + real-world feedback

---

## Ollama Model Selection

**Recommended models** (local, fast inference):

1. **deepseek-r1:7b** (BEST for reasoning)
   - Excellent at mathematical reasoning
   - Fast inference (~2 tokens/sec on CPU)
   - Understands calculus concepts well

2. **qwen2.5:7b** (Good alternative)
   - Strong math capabilities
   - Slightly faster than deepseek-r1

3. **llama3.2:3b** (Fastest, lower quality)
   - Very fast inference
   - Adequate for simple routing decisions

**Installation** (if not already installed):
```bash
# Pull deepseek-r1:7b (recommended)
ollama pull deepseek-r1:7b

# Or qwen2.5:7b
ollama pull qwen2.5:7b
```

---

## Success Criteria

### Ollama Data Generation
- [ ] Generate ≥180 synthetic routing decisions (20 per pattern × 9 patterns)
- [ ] Ollama selects **diverse rules** (not all defaulting to one rule)
- [ ] Semantic tag embeddings have dim=256, normalized

### Router Specialist Training
- [ ] Router specialist registered in swarm
- [ ] Training converges (loss decreases over epochs)
- [ ] Validation accuracy ≥60% on synthetic data
- [ ] Weights saved to ollama_bootstrapped_router.npz

### Benchmark Validation
- [ ] Router selects diverse rules on real problems
- [ ] Grammar rules execute (logs show RPN engine invocation)
- [ ] MATH accuracy ≥2% (improvement from 0% baseline)
- [ ] Real routing decisions collected for future refinement

---

## Architecture Benefits

**Ollama as Data Generator**:
- ✅ **Fast bootstrap** - don't need benchmark runs to collect training data
- ✅ **Complete coverage** - generate examples for ALL patterns (even rare ones)
- ✅ **Reasoning-based** - Ollama understands calculus concepts, not random guessing
- ✅ **Sovereign-friendly** - Ollama runs locally (no external API calls)
- ✅ **Iterative** - can mix synthetic + real data for continual improvement

**Router Specialist**:
- ✅ **Learned navigation** - weights learn pattern→rule mappings
- ✅ **Continual learning** - improves from real benchmark feedback
- ✅ **No hardcoded logic** - just trained LoRA adapter
- ✅ **Shadow copy compatible** - router self-updates from success

---

## Implementation Checklist

### Immediate Tasks (Codex)
1. Create `scripts/generate_router_training_data_ollama.py`
   - Implement `generate_routing_decision_ollama()`
   - Implement `embed_semantic_tags_hash()`
   - Implement `generate_training_dataset_ollama()`

2. Create `scripts/train_router_from_ollama_data.py`
   - Load Ollama-generated synthetic data
   - Initialize swarm + router specialist
   - Train router from synthetic routing decisions
   - Save trained weights

3. Update `run_sovereign_math_benchmarks.py`
   - Add `--router-weights` flag to load trained router
   - Wire trained router into TRM navigation

4. Add execution logging in `trm_galaxy_reader.py`
   - Log: `[TRM] Executing grammar rule: {rule_name}`
   - Log: `[RPN Engine] Result: {result}`
   - Confirm rules are actually being invoked

### Testing Workflow
1. Generate synthetic data: `python scripts/generate_router_training_data_ollama.py`
2. Train router: `python scripts/train_router_from_ollama_data.py`
3. Run benchmark: `python scripts/run_sovereign_math_benchmarks.py --router-weights ollama_bootstrapped_router.npz`
4. Review logs: Check for diverse rule selection + execution

---

## Expected Timeline

**Ollama bootstrap** (3-4 hours):
1. Data generation script (1 hour)
2. Semantic embedding implementation (30 min)
3. Router training script (1 hour)
4. Execution logging (30 min)
5. Benchmark integration (30 min)

**Validation** (1 hour):
- Generate synthetic data (~10 min)
- Train router (~20 min)
- Run benchmark (~20 min)
- Analyze results (~10 min)

**Total**: ~5 hours from start to validated trained router

---

## Key Architectural Principle

**"Use LLMs where they're good (reasoning), avoid where they're bad (execution)"**

- ✅ Ollama generates **training labels** (semantic reasoning)
- ✅ Router specialist **learns navigation** (LoRA weights)
- ✅ Grammar rules **execute sovereign RPN** (PTX kernels)
- ❌ NO LLM in hot path (inference is learned weights + PTX)

**This is the K3D way**: LLMs as **data generators** for sovereign systems, not as runtime dependencies.

---

**Proceed with Ollama bootstrap, Codex!** This will train the router specialist fast. 🚀
