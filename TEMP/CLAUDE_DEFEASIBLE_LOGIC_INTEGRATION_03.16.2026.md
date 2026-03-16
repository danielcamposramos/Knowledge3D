# Defeasible Logic Integration — Architecture Plan

**Date:** 2026-03-16
**Author:** Claude (Architecture Partner) + Christoph (SPINdle reference)
**Status:** Architecture Spec — Ready for Codex Implementation

---

## 1. What Is Defeasible Logic (and Why It Matters for K3D)

Defeasible logic is **non-monotonic reasoning**: conclusions can be withdrawn when stronger evidence appears. Unlike classical logic (once proven, always proven), defeasible reasoning says "birds typically fly, but penguins don't — and the penguin rule wins because it's more specific."

**SPINdle** (from Christoph's reference) implements this with:
- **Facts** (`>>`): unconditional truths
- **Strict rules** (`->`): always hold (classical implication)
- **Defeasible rules** (`=>`): normally apply, can be overridden
- **Defeaters** (`~>`): block conclusions without proving alternatives
- **Superiority relations**: explicit priority when rules conflict
- **Trust weights**: source attribution with confidence decay

**Conclusion types:**
| SPINdle | Trit Value | Meaning |
|---------|-----------|---------|
| +D (definitely provable) | +1 | Strict chain, cannot be defeated |
| +d (defeasibly provable) | +1 (with tag) | Survives all conflicts |
| -d (defeasibly refuted) | -1 | Defeated by superior rule |
| -D (definitely refuted) | -1 (with tag) | Strict chain to negation |
| undetermined | 0 | No chain, or balanced conflict |

---

## 2. Why This Maps Perfectly to K3D

### 2.1 We Already Have the Primitives

**RPN Standard Tier — Ternary Opcodes (0x70-0x76):**

| Opcode | Name | SPINdle Operation |
|--------|------|-------------------|
| 0x70 TADD | Trit addition | Rule combination (accumulate evidence) |
| 0x71 TMUL | Trit multiplication | Rule conjunction (AND gate) |
| 0x72 TNOT | Trit negation | Negation-as-failure (~p) |
| 0x73 TCOMP | Trit comparison | Superiority check (sign(rule_a - rule_b)) |
| 0x74 TQUANT | Float→trit quantize | Quantize confidence to verdict {+1, 0, -1} |
| 0x75 TPACK | Pack two trits | Compact storage of (definite, defeasible) pair |
| 0x76 TUNPACK | Unpack two trits | Extract (definite, defeasible) pair |

**The key insight:** TPACK/TUNPACK already encode two trits — we can pack `(+D verdict, +d verdict)` into a single scalar. This is EXACTLY the SPINdle output format: every literal gets both a definite and a defeasible proof tag.

### 2.2 Grammar Galaxy Rules Already Have Ternary Masks

From `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` (line 884):
```python
class GrammarRule:
    rule_id: str = "arc_rotation_rule"
    rpn_program: str = "GRID ROTATE_90_CW"
    ternary_mask: np.ndarray = np.array([+1, -1], dtype=np.int8)
```

These ternary relevance masks already express "this rule says YES to ∑, NO to ∫, UNKNOWN to the rest." That IS defeasible rule metadata.

### 2.3 The Swarm IS the Forward Chainer

SPINdle's forward chaining processes rules to derive conclusions. Our Nine-Chain Swarm already:
1. Takes candidates (= facts/literals)
2. Applies Grammar Galaxy rules (= defeasible rules)
3. Scores each path (= proof strength)
4. Halting gate checks convergence (= proof standard)

**What's missing:** principled conflict resolution. Currently the halting gate just counts agreement (majority vote). With defeasible logic, conflicts resolve via **superiority relations** — a bird-flies rule and a penguin-doesn't-fly rule don't just vote; the more specific one DEFEATS the other.

### 2.4 Trust Weights = Existing Confidence System

SPINdle's trust-weighted reasoning assigns weights (0-1) to information sources. We already have:
- `confidence` on Galaxy entries
- `specialist_resonance` / `specialist_coherence` per candidate
- `galaxy_weight` per galaxy contribution

These ARE trust weights. Defeasible logic formalizes what we're already doing intuitively.

---

## 3. Architecture Design

### 3.1 Rule Strength Metadata (Grammar Galaxy Extension)

Extend `GrammarRule` with defeasible metadata:

```
rule_strength: trit
  +1 = strict (->)  : always holds, cannot be defeated
   0 = defeasible (=>) : default, can be overridden (MOST rules)
  -1 = defeater (~>)  : blocks conclusions, doesn't prove anything

superior_to: list[rule_id]
  Explicit superiority relations.
  "This rule defeats those rules when they conflict."

trust_source: str
  Origin attribution (galaxy name, curriculum, bootstrap, discovery).

trust_weight: float [0, 1]
  Source confidence. Decays during sleep-time consolidation if unused.
```

**Key principle:** ALL existing Grammar rules default to `rule_strength = 0` (defeasible). This preserves backward compatibility. Only rules that are mathematically certain (e.g., `2 + 3 = 5`) or axioms get `rule_strength = +1`.

### 3.2 New Kernel: `gre_defeasible_resolver.cu`

**Purpose:** Takes swarm worker conclusions + rule metadata → produces defeasible verdicts.

**Input:**
```c
const float* __restrict__ conclusions,     // [num_workers * num_candidates] raw scores
const int8_t* __restrict__ rule_strengths, // [num_workers] trit per worker's rule
const uint32_t* __restrict__ superiority,  // [num_rules * max_superiors] packed superiority graph
float* __restrict__ verdicts,              // [num_candidates] defeasible verdict scores
uint32_t* __restrict__ proof_tags,         // [num_candidates] packed (D, d) trit pairs
int num_workers,
int num_candidates,
int max_superiors
```

**Algorithm (per candidate):**

```
Phase 1 — Collect support chains:
  For each worker w that scored this candidate:
    support[w] = TQUANT(score[w])  // quantize to {+1, 0, -1}
    strength[w] = rule_strengths[w]

Phase 2 — Apply superiority:
  For each pair (w_a, w_b) where w_a has superiority over w_b:
    If support[w_a] and support[w_b] conflict (TMUL < 0):
      support[w_b] = 0  // defeated

Phase 3 — Aggregate:
  strict_chain = TMUL of all strict (strength=+1) supports
  defeasible_chain = TADD of all surviving defeasible supports

  D_verdict = strict_chain    // definitely provable only from strict rules
  d_verdict = TADD(strict_chain, TQUANT(defeasible_chain))

  proof_tag = TPACK(D_verdict, d_verdict)
  verdict = trit_to_float(d_verdict) * trust_weighted_confidence

Phase 4 — Defeater check:
  For each defeater rule (strength=-1):
    If defeater fires AND no superior rule supports the conclusion:
      d_verdict = 0  // blocked, not defeated to -1
```

**This replaces nothing** — it ENRICHES the existing pipeline. The halting gate still runs after, but now it has principled verdicts instead of raw scores.

### 3.3 Pipeline Integration Point

```
Morton → LED-A* → Frustum → LOD →
  Nine-Chain Swarm (each worker applies Grammar rules) →
    *** DEFEASIBLE RESOLVER (new) *** →
      Halting Gate (now checks verdicts, not just agreement count)
```

The resolver sits BETWEEN swarm output and halting gate. It's a single kernel launch — no Python, fully sovereign.

### 3.4 Halting Gate Enhancement

Current halting gate checks: `minimum_threshold ∧ gap_threshold ∧ agreement_threshold`

Enhanced halting gate adds:
- **proof_strength flag**: Is the winner's verdict +D (strict) or +d (defeasible)?
- **defeat_margin**: How many competing rules were defeated vs. just outscored?
- **unresolved_count**: How many candidates have verdict = 0 (undetermined)?

If `proof_strength = +D` (strict chain), halt immediately — the answer is certain.
If `proof_strength = +d` and `defeat_margin > 0`, halt — the answer survived challenge.
If many candidates undetermined, DON'T halt — need more swarm iterations.

### 3.5 RPN Program Encoding for Defeasible Rules

New RPN programs in Grammar Galaxy can use existing ternary opcodes for defeasible reasoning:

```
// "Penguins don't fly" as RPN (defeasible, superior to bird-flies):
// Input: candidate embedding on stack
GALAXY_LOOKUP "penguin"    // 0xE0: query Galaxy for penguin match
TQUANT                     // 0x74: quantize similarity to trit
DUP                        // duplicate for dual chain
GALAXY_LOOKUP "flies"      // query "flies" concept
TNOT                       // 0x72: negate (penguin → NOT flies)
TMUL                       // 0x71: conjunction (penguin AND NOT flies)
// Result: +1 if penguin confirmed and flies negated, 0 if uncertain, -1 if contradicted
```

**Note:** Galaxy opcodes (0xE0-E2) are in Standard tier but NOT in Extended tier. This is the asymmetry we documented. For defeasible logic, Standard tier is sufficient.

---

## 4. What Christoph's SPINdle Gives Us (and What We Don't Take)

### 4.1 We ABSORB (patterns)

| SPINdle Concept | K3D Implementation |
|-----------------|-------------------|
| Forward chaining | Nine-Chain Swarm (already exists) |
| Strict/defeasible/defeater | `rule_strength` trit on GrammarRule |
| Superiority relations | `superior_to` list on GrammarRule |
| Trust weights | `trust_weight` + `trust_source` on GrammarRule |
| +D/+d/-D/-d conclusions | TPACK(D_verdict, d_verdict) per candidate |
| What-if queries | Swarm worker hypothesis testing (already exists) |
| Why-not explanations | Selection steps trace (already exists) |

### 4.2 We DON'T TAKE (architecture)

- No Rust/Racket runtime dependency
- No SPL parser (our rules are RPN programs in Galaxy, not text)
- No external reasoner process (reasoning is PTX on GPU)
- No Datalog grounding (our grounding is Galaxy spatial navigation)
- No Allen temporal algebra (our temporal reasoning is `gre_temporal_reasoning.cu`)

**We follow K3D's absorption pattern:** SPINdle's CONCEPTS enter the Galaxy as principled metadata. The IMPLEMENTATION stays sovereign PTX.

---

## 5. Implementation Roadmap (for Codex)

### Step 1: GrammarRule Extension (ingestion path, Python OK)

Add three fields to `GrammarRule`:
```python
rule_strength: int = 0       # trit: +1=strict, 0=defeasible, -1=defeater
superior_to: list[str] = []  # rule_ids this rule defeats
trust_weight: float = 1.0    # source confidence [0, 1]
```

Tag existing rules:
- Math axioms (`2+3=5`, power rule) → `rule_strength = +1` (strict)
- All current Grammar rules → `rule_strength = 0` (defeasible, default)
- No defeaters yet (Phase E: sleep-time discovers them)

### Step 2: `gre_defeasible_resolver.cu` (new kernel)

Single-block kernel. Inputs: worker scores, rule strengths, superiority adjacency.
Outputs: per-candidate verdicts + proof tags (TPACK'd trit pairs).
Algorithm as described in §3.2.

**Compile to PTX, wire bridge in `sovereign_bridges.py`.**

### Step 3: Wire into `_apply_specialist_swarm_features()`

After swarm workers score candidates, before halting gate:
1. Collect worker rule_strengths from Grammar Galaxy metadata
2. Build superiority adjacency from `superior_to` references
3. Launch `gre_defeasible_resolver`
4. Thread `specialist_defeasible_verdict` and `specialist_proof_tag` into candidates

### Step 4: Halting Gate Enhancement

Add proof_strength check:
- If winner has `+D` tag → `proof_strength_flag = 1` (certain)
- If winner has `+d` tag → `proof_strength_flag = 1` (survived challenge)
- If winner has `0` tag → `proof_strength_flag = 0` (undetermined — don't trust)

### Step 5: Scoring Integration

Add `specialist_defeasible_verdict` to the RPN scoring expression (like geometry/temporal at 0.03 weight initially). Increase weight as we validate.

---

## 6. Expected Impact

### Immediate (after Step 5):

- **Math benchmarks**: Strict rules for axioms mean `2+3=5` gets +D verdict — the halting gate can trust it immediately instead of waiting for agreement
- **GSM8K**: Competing decomposition strategies resolve via superiority, not random swarm majority
- **LHE**: Multi-hop chains where step 2 defeats step 1's conclusion — defeasible logic handles this natively

### Phase E (future):

- **Sleep-time discovers defeaters**: When a rule leads to wrong answers, sleep-time consolidation creates a defeater rule that blocks it in similar contexts
- **Trust decay**: Unused rules' trust_weight decays toward 0, eventually pruned
- **Abduction**: "Why didn't this work?" → find missing rules that would have proven the goal (SPINdle's why-not query)

### Long-term:

- **The TRM learns defeasible reasoning**: As the game loop runs, the TRM's navigation weights learn WHICH superiority paths lead to correct answers. The explicit rule structure makes this learnable rather than opaque.

---

## 7. Christoph's SPINdle as Reference Implementation

**For validation (NOT runtime dependency):**

The Rust and Racket SPINdle implementations can serve as oracle/reference:
- Given the same rules and facts, does K3D's GPU defeasible resolver reach the same conclusions?
- SPINdle has 1,500+ test cases — we can port key test vectors as benchmark validation
- The WASM build could run in-browser for interactive demos (House visualization of rule conflicts)

**Reference repos:**
- Rust: https://codeberg.org/anuna/spindle-rust
- Racket: https://codeberg.org/anuna/spindle-racket
- Demo: https://spindle-rust.anuna.io/

---

## 8. Sovereignty Compliance

| Component | Path | Sovereign? |
|-----------|------|-----------|
| GrammarRule fields | Ingestion (Python) | N/A (ingestion is flexible) |
| `gre_defeasible_resolver.cu` | Hot path (PTX) | YES |
| Superiority adjacency | Galaxy metadata (VRAM) | YES |
| Verdict computation | RPN ternary opcodes | YES |
| Proof tags | TPACK/TUNPACK (0x75/0x76) | YES |
| Trust weights | Galaxy entry metadata | YES |
| SPINdle reference | Validation only | N/A (not in hot path) |

**Zero new Python in hot path. Zero external dependencies. Pure PTX + Galaxy + RPN.**

---

## Appendix A: SPINdle Quick Reference

```
Facts:      >> bird              (unconditional truth)
Strict:     r1: bird -> animal   (always holds)
Defeasible: r2: bird => flies    (normally holds, can be overridden)
Defeater:   r3: penguin ~> flies (blocks "flies" without proving "not flies")
Superior:   r3 > r2             (penguin rule defeats bird rule)

Conclusions:
  +D bird     (definitely: it's a fact)
  +d animal   (defeasibly: strict rule from fact)
  -d flies    (defeasibly refuted: r3 defeats r2 via superiority)
```

## Appendix B: Ternary Opcode Cheat Sheet (RPN Standard 0x70-0x76)

```
TADD   (0x70): a + b clamped to {-1, 0, +1}  — accumulate evidence
TMUL   (0x71): a * b clamped to {-1, 0, +1}  — conjunction/AND
TNOT   (0x72): -a                              — negation
TCOMP  (0x73): sign(a - b)                    — superiority comparison
TQUANT (0x74): quantize float → trit (±0.33 threshold)
TPACK  (0x75): pack (a, b) → single scalar    — store (D, d) pair
TUNPACK(0x76): unpack scalar → (a, b)         — extract (D, d) pair
```
