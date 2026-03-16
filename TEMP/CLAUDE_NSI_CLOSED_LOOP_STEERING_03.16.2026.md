# NSI Closed Loop — Ternary Contrastive Learning from Defeasible Verdicts

**Date:** 2026-03-16
**Author:** Claude (Architecture Partner), insight from Gemini (NotebookLM research)
**Status:** Architecture Spec — closes the neurosymbolic integration loop

---

## The Gap

We have defeasible logic flowing FORWARD (rules → verdicts → scoring). We have TRM learning flowing from execution events (shadow copy → sleep-time → weight updates). But these two paths aren't connected — AND the learning path itself is crippled by **binary chokepoints** that discard the 0-signal our ternary architecture already produces.

**Current flow (broken loop + binary chokepoints):**
```
Grammar Rules ──→ Defeasible Resolver ──→ Scoring ──→ Answer
                                                        │
Execution Events ──→ Shadow Copy ──→ SleepTime ──→ TRM Weight Update
                                          (disconnected)
                                          (binary: success=True/False)
```

**Closed loop (ternary contrastive throughout):**
```
Grammar Rules ──→ Defeasible Resolver ──→ Scoring ──→ Answer
       ↑                   │                            │
       │                   ▼                            ▼
       │       Defeasible Verdict Events      Execution Events
       │          (+1 / 0 / -1)                 (+1 / 0 / -1)
       │                   │                            │
       │                   ▼                            ▼
       │              Shadow Copy ◄─────────────────────┘
       │                   │
       │              Ternary Contrastive Consolidation
       │                   │
       │         ┌─────────┼─────────┐
       │         ▼         ▼         ▼
       │      +1 path   0 path    -1 path
       │     (reinforce) (explore) (anti-pattern)
       │         │         │         │
       │         ▼         ▼         ▼
       └──── TRM Weight Update ◄─── Auto-Generated Defeater Rules
```

---

## Binary Chokepoints Audit — The Five Wounds

The contrastive learning infrastructure (`execution_quality_tracker.py`, `execution_grammar_detector.py`) already tracks ternary outcomes and generates anti-patterns. But the TRM learning path collapses everything to binary. Here are the exact locations:

### Chokepoint 1: `specialist_base.py:122` — `update_routing_bias()`

```python
def update_routing_bias(self, child_name: str, success: bool, *, alpha: float = 0.1) -> None:
    current = float(self.routing_bias.get(child_name, 0.5))
    target = 1.0 if success else 0.0  # ← BINARY: 1.0 or 0.0, nothing in between
    updated = (float(alpha) * target) + ((1.0 - float(alpha)) * current)
    self.routing_bias[child_name] = max(0.0, min(updated, 1.0))
```

**Problem:** `target = 1.0 if success else 0.0`. The 0-signal (undetermined) doesn't exist. An undetermined outcome gets forced to `success=False → target=0.0`, PUNISHING exploration the same as failure.

**Fix:** Accept ternary outcome. Target: `+1 → 1.0`, `-1 → 0.0`, `0 → current` (no update — hold position).

### Chokepoint 2: `specialist_base.py:128` — `mark_query()`

```python
def mark_query(self, success: bool) -> None:
    self.query_count += 1
    if success:
        self.success_count += 1
    else:
        self.failure_count += 1  # ← BINARY: 0-signal counted as failure
```

**Problem:** No `uncertain_count`. The 0-signal inflates `failure_count`, making the Bayesian quality estimate pessimistic about exploration.

**Fix:** Add `uncertain_count`. Three branches: `+1 → success_count`, `-1 → failure_count`, `0 → uncertain_count`.

### Chokepoint 3: `navigator_specialist.py:337` — `learn_routing_topology()`

```python
bucket = specialist_stats.setdefault(specialist, {"success": 0, "failure": 0})
if success:
    bucket["success"] += 1
    self.router.adjust_specialist_bias(specialist, +0.02)
else:
    bucket["failure"] += 1
    self.router.adjust_specialist_bias(specialist, -0.01)  # ← 0-signal penalizes
```

**Problem:** Binary routing topology. The asymmetric bias (+0.02 vs -0.01) tries to be cautious, but 0-signals still push toward -0.01 which is incorrect.

**Fix:** Three branches. `+1 → +0.02`, `-1 → -0.01`, `0 → 0.0` (no bias change, but increment an `uncertain` counter that drives exploration pressure).

### Chokepoint 4: `trm_navigator.py:1280` — `consolidate_weights_from_events()`

```python
success = ("success" in event_type) or (
    "fail" not in event_type and confidence >= 0.65
)  # ← BINARY: string-match heuristic, 0-signal → success if confidence high enough
```

**Problem:** Events are classified as success/failure by string matching and a confidence threshold. DefeasibleVerdictEvents (which carry explicit trit values) don't exist yet, so the system guesses.

**Fix:** Check for `verdict_trit` field first. If present, use it directly. Fall back to string-match only for legacy execution events.

### Chokepoint 5: `execution_grammar_detector.py:571` — `observe_event()`

```python
outcome = int(event.get("outcome", 0) or 0)
if outcome == 0:
    return {}  # ← DROPPED: 0-signals produce no grammar patterns at all
```

**Problem:** The grammar detector SKIPS 0-outcomes entirely. But 0-outcomes from defeasible resolution mean "conflicting rules, no winner" — that's a DISCOVERY opportunity, not nothing. The pattern "these two rules conflict in this context" is itself valuable Grammar metadata.

**Fix:** 0-outcomes should accumulate as `"exploratory"` polarity patterns. After enough recurrences, they generate Grammar rules with `rule_strength = 0` (defeasible) and metadata tagging them as conflict zones where the TRM should try alternative strategies.

---

## The Contrastive Learning Model We Already Have (But Don't Use Fully)

The `execution_quality_tracker.py` ALREADY tracks ternary state properly:

```python
# Line 400-402: THREE counters exist
success_count = int(record.get("success_count", 0)) + (1 if outcome > 0 else 0)
failure_count = int(record.get("failure_count", 0)) + (1 if outcome < 0 else 0)
uncertain_count = int(record.get("uncertain_count", 0)) + (1 if outcome == 0 else 0)

# Line 411: Ternary trend computation
ternary_trend = _ternary(current_mean - previous_mean)  # returns +1, 0, -1
```

The `execution_grammar_detector.py` ALREADY generates contrastive anti-patterns:

```python
# Line 309: Source attribution for anti-patterns
source = "auto_detected" if is_positive else "auto_detected_contrastive"

# Line 338-344: Ternary confidence + contrastive recommendation
"ternary_confidence": int(ternary_quantize_quality(bayesian_quality)),
"contrastive_recommendation": (
    "reuse_and_promote"
    if is_positive else
    "avoid_or_invert"
),
```

**The infrastructure is there.** The quality tracker counts uncertain outcomes. The grammar detector generates contrastive anti-patterns. But the TRM learning path (`specialist_base`, `navigator_specialist`, `trm_navigator`) collapses everything to binary before it reaches the weight update. The contrastive learning signals are PRODUCED but never CONSUMED.

---

## Ternary Contrastive Learning — Full Design

### Signal Flow Per Verdict Trit

**+1 (Proven / Reinforced):**
```
Defeasible verdict +1
  → Shadow Copy: DefeasibleVerdictEvent(verdict_trit=+1, program_id, confidence)
  → SleepTime consolidation:
      → learn_from_feedback(ternary_outcome=+1)
          → specialist_base.mark_query(outcome=+1) → success_count += 1
          → specialist_base.update_routing_bias(outcome=+1) → target=1.0
          → navigator_specialist.learn_routing_topology(outcome=+1) → bias += 0.02
      → execution_grammar_detector.observe_event(outcome=+1) → positive pattern
          → If recurrent: promote to Grammar Galaxy as canonical rule
          → trust_weight of contributing rule increases
```

**-1 (Defeated / Contrastive Anti-Pattern):**
```
Defeasible verdict -1, was_defeated_by=superior_rule_id
  → Shadow Copy: DefeasibleVerdictEvent(verdict_trit=-1, was_defeated_by, program_id)
  → SleepTime consolidation:
      → learn_from_feedback(ternary_outcome=-1, defeat_source=superior_rule_id)
          → specialist_base.mark_query(outcome=-1) → failure_count += 1
          → specialist_base.update_routing_bias(outcome=-1) → target=0.0
          → navigator_specialist.learn_routing_topology(outcome=-1) → bias -= 0.01
      → execution_grammar_detector.observe_event(outcome=-1) → negative/contrastive pattern
          → If recurrent: promote to Grammar Galaxy as DEFEATER rule
              → rule_strength = -1
              → superior_to = [defeated_rule_id]
              → semantics.source = "defeasible_auto_detected_contrastive"
              → semantics.contrastive_recommendation = "block_in_context"
          → trust_weight of defeated rule decreases
          → trust_weight of superior rule increases
```

**0 (Undetermined / Exploration Signal):**
```
Defeasible verdict 0, proof_tag = (0, 0)
  → Shadow Copy: DefeasibleVerdictEvent(verdict_trit=0, program_id)
  → SleepTime consolidation:
      → learn_from_feedback(ternary_outcome=0)
          → specialist_base.mark_query(outcome=0) → uncertain_count += 1
          → specialist_base.update_routing_bias(outcome=0) → NO UPDATE (hold current)
          → specialist_base.exploration_pressure += 1  ← NEW
          → navigator_specialist.learn_routing_topology(outcome=0) → bias unchanged, uncertain += 1
      → execution_grammar_detector.observe_event(outcome=0) → exploratory pattern ← NEW POLARITY
          → If recurrent: promote to Grammar Galaxy as EXPLORATORY rule
              → rule_strength = 0 (defeasible)
              → semantics.source = "defeasible_auto_detected_exploratory"
              → semantics.contrastive_recommendation = "explore_alternatives"
              → semantics.conflict_zone = [conflicting_rule_ids]
          → Flag this context as a "conflict zone" for future routing
```

### Exploration Pressure Mechanism

When `exploration_pressure` on a specialist node exceeds a threshold (e.g., 3 undetermined outcomes), the TRM routing changes behavior:

1. **Normal routing:** Use `routing_bias` to pick the most successful child specialist
2. **Exploration routing:** When `exploration_pressure >= threshold`, TEMPORARILY ignore `routing_bias` and try the LEAST-TRIED child specialist instead
3. After exploration, reset `exploration_pressure` to 0

This implements Gemini's insight: "the TRM interprets 0 as an instruction to explore alternative recursive refinements." The 0-signal doesn't damage confidence — it opens doors.

---

## Implementation Steps (for Codex)

### Step A: Fix the Five Chokepoints

All changes are backward-compatible. Existing callers that pass `success: bool` continue to work (True → +1, False → -1).

**A.1: `specialist_base.py` — `update_routing_bias()`**

```python
def update_routing_bias(self, child_name: str, success: bool | None = None, *,
                        ternary_outcome: int | None = None, alpha: float = 0.1) -> None:
    outcome = _resolve_ternary(success, ternary_outcome)
    current = float(self.routing_bias.get(child_name, 0.5))
    if outcome == 0:
        return  # hold position — don't punish or reward uncertainty
    target = 1.0 if outcome > 0 else 0.0
    updated = (float(alpha) * target) + ((1.0 - float(alpha)) * current)
    self.routing_bias[child_name] = max(0.0, min(updated, 1.0))
```

**A.2: `specialist_base.py` — `mark_query()`**

```python
def mark_query(self, success: bool | None = None, *, ternary_outcome: int | None = None) -> None:
    outcome = _resolve_ternary(success, ternary_outcome)
    self.query_count += 1
    if outcome > 0:
        self.success_count += 1
    elif outcome < 0:
        self.failure_count += 1
    else:
        self.uncertain_count += 1  # NEW field, default 0
        self.exploration_pressure += 1  # NEW field, default 0
```

Add `uncertain_count: int = 0` and `exploration_pressure: int = 0` to `SpecialistBase.__init__` and serialization.

**A.3: `navigator_specialist.py` — `learn_routing_topology()`**

```python
bucket = specialist_stats.setdefault(specialist, {"success": 0, "failure": 0, "uncertain": 0})
if outcome > 0:
    bucket["success"] += 1
    self.router.adjust_specialist_bias(specialist, +0.02)
elif outcome < 0:
    bucket["failure"] += 1
    self.router.adjust_specialist_bias(specialist, -0.01)
else:
    bucket["uncertain"] += 1
    # No bias adjustment — hold position
```

**A.4: `trm_navigator.py` — `consolidate_weights_from_events()`**

```python
# Check for DefeasibleVerdictEvent first
verdict_trit = event.get("verdict_trit", None)
if verdict_trit is not None:
    success = True if int(verdict_trit) > 0 else (False if int(verdict_trit) < 0 else None)
    ternary_outcome = int(verdict_trit)
else:
    # Legacy: string-match heuristic
    success = ("success" in event_type) or (
        "fail" not in event_type and confidence >= 0.65
    )
    ternary_outcome = 1 if success else -1

self.learn_from_feedback(
    query=query,
    specialist=specialist,
    success=success,
    ternary_outcome=ternary_outcome,
    confidence=confidence,
    domain_hint=...,
    defeat_source=str(event.get("was_defeated_by", "")) or None,
)
```

**A.5: `execution_grammar_detector.py` — `observe_event()`**

```python
outcome = int(event.get("outcome", 0) or 0)
if outcome == 0:
    # NEW: Don't skip — accumulate as "exploratory" polarity
    polarity = "exploratory"
    # ... rest of pattern accumulation with new polarity
else:
    polarity = "positive" if outcome > 0 else "negative"
```

Add `"exploratory"` polarity support in `_pattern_key`, `_rpn_program`, `_build_rule_entry`. Exploratory patterns get `contrastive_recommendation = "explore_alternatives"`.

**Helper function (shared):**

```python
def _resolve_ternary(success: bool | None, ternary_outcome: int | None) -> int:
    if ternary_outcome is not None:
        return max(-1, min(1, int(ternary_outcome)))
    if success is None:
        return 0
    return 1 if success else -1
```

### Step B: DefeasibleVerdictEvent

**B.1:** In `execution_events.py`, add `DefeasibleVerdictEvent` dataclass:

```python
@dataclass
class DefeasibleVerdictEvent:
    stage: str              # "early_gate", "intra_path", "final"
    candidate_id: str
    program_id: str
    verdict_trit: int       # +1, 0, -1
    proof_tag: int          # packed (D, d) trit pair
    rule_strength: int      # strict/defeasible/defeater
    was_defeated_by: str | None
    confidence: float
    timestamp_us: int
    domain_hint: str | None = None
```

**B.2:** In `knowledgeverse.py`, after each defeasible resolution stage, emit events to shadow copy buffer. Only emit for non-neutral verdicts (don't flood shadow copy with thousands of 0-verdicts on routine queries — only emit 0s when there was actual conflict detected).

### Step C: Auto-Generated Defeater Rules from Defeats

**C.1:** In `execution_grammar_detector.py`, add method:

```python
def observe_defeasible_verdict(self, event: DefeasibleVerdictEvent) -> dict[str, Any]:
    """Process defeasible verdict into Grammar patterns."""
    if event.verdict_trit == 0 and event.was_defeated_by is None:
        return {}  # no conflict detected, skip
    outcome = event.verdict_trit
    polarity = "positive" if outcome > 0 else ("negative" if outcome < 0 else "exploratory")
    # ... accumulate as defeasible-specific pattern
    # When promoted, defeated patterns become defeater rules (rule_strength=-1)
    # Exploratory patterns become conflict-zone markers
```

**C.2:** During sleep-time `_stage_b_logic`, process DefeasibleVerdictEvents through this method.

### Step D: Exploration Pressure in Routing

**D.1:** In `specialist_base.py`, when `exploration_pressure >= 3`:
- Flag the node for exploration mode
- `_dispatch_swarm_weights()` in knowledgeverse.py checks this flag
- If in exploration mode, diversify swarm weights instead of concentrating on favorites

**D.2:** Reset `exploration_pressure` to 0 after one exploration cycle.

### Step E: `trm_navigator.learn_from_feedback()` — Full Ternary

```python
def learn_from_feedback(
    self,
    *,
    query: str,
    specialist: str,
    success: bool | None = None,
    ternary_outcome: int | None = None,
    confidence: float | None = None,
    domain_hint: str | None = None,
    defeat_source: str | None = None,
) -> None:
    outcome = _resolve_ternary(success, ternary_outcome)
    # Thread ternary_outcome through to all downstream calls
    self.navigator_specialist.learn_routing_topology(
        query=query, specialist=specialist,
        success=(outcome > 0) if outcome != 0 else None,
        ternary_outcome=outcome,
    )
    node = self._resolve_specialist_node(...)
    node.mark_query(ternary_outcome=outcome)
    if node.parent is not None:
        node.parent.update_routing_bias(node.name, ternary_outcome=outcome)
    # ... spawner.observe() also gets ternary
```

### Step F: Tests

1. **Chokepoint regression tests:**
   - `update_routing_bias(ternary_outcome=0)` → bias unchanged
   - `mark_query(ternary_outcome=0)` → uncertain_count increments, exploration_pressure increments
   - `learn_routing_topology(ternary_outcome=0)` → no bias adjustment, uncertain counter increments
   - `consolidate_weights_from_events` with `verdict_trit=0` → no weight change
   - `execution_grammar_detector.observe_event(outcome=0)` → exploratory pattern accumulated (not skipped)

2. **Contrastive learning tests:**
   - `-1` verdict with `was_defeated_by` → auto-generated defeater rule in Grammar Galaxy
   - `+1` verdict → trust_weight of contributing rule increases
   - Recurrent `-1` pattern → promoted to Grammar Galaxy with `rule_strength=-1`
   - Recurrent `0` pattern → promoted as conflict-zone marker with `explore_alternatives`

3. **Backward compatibility:**
   - `success=True` still works (maps to ternary_outcome=+1)
   - `success=False` still works (maps to ternary_outcome=-1)
   - All existing execution event flow unchanged
   - No benchmark regression

4. **Full benchmark:** ARC 10/10, Math 20/20, GSM8K ≥ 2/10, MMLU ≥ 17/50

---

## Why This Closes the Loop

**The key insight from Gemini, enhanced with our contrastive infrastructure:**

The contrastive learning system (`execution_quality_tracker`, `execution_grammar_detector`) was built to handle ternary outcomes. It already counts `uncertain_count`, computes `ternary_trend`, generates anti-patterns with `contrastive_recommendation`. But the TRM learning path (`specialist_base`, `navigator_specialist`, `trm_navigator`) collapses everything to binary before consuming it.

**What the five fixes do:**

1. **Stop punishing exploration** — 0-signals no longer push routing_bias toward 0.0
2. **Stop miscounting uncertainty as failure** — uncertain_count separated from failure_count
3. **Stop damaging routing topology on ambiguity** — no bias adjustment on 0
4. **Start using defeasible verdicts directly** — verdict_trit consumed without string-match guessing
5. **Start accumulating exploratory patterns** — 0-outcomes generate Grammar metadata instead of being dropped

**The result:** Every defeasible verdict (+1, 0, -1) flows through the FULL contrastive learning pipeline: quality tracking → grammar detection → shadow copy → sleep-time consolidation → TRM weight update → routing change → better future reasoning.

This is the neurosymbolic integration loop that Gemini described: "every neural hypothesis is symbolically checked, and every symbolic validation actively tunes the neural weights."

---

## Sovereignty Compliance

| Component | Path | Sovereign? |
|-----------|------|-----------|
| DefeasibleVerdictEvent recording | Post-hot-path (Python OK) | N/A |
| Ternary learn_from_feedback | Weight persistence (Python OK) | N/A |
| Auto-generated defeater rules | Ingestion (Python OK) | N/A |
| Exploration pressure routing | Weight dispatch (Python OK) | N/A |
| Exploratory grammar patterns | Grammar Galaxy (Python OK) | N/A |
| Hot-path defeasible resolution | PTX kernel (unchanged) | YES |
| Hot-path scoring | RPN on GPU (unchanged) | YES |

**Zero changes to hot path.** All fixes are in the learning/feedback path (post-query and sleep-time). The sovereign PTX kernels are untouched.
