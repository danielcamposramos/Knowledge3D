# NSI Closed Loop — Defeasible Feedback into TRM Learning

**Date:** 2026-03-16
**Author:** Claude (Architecture Partner), insight from Gemini (NotebookLM research)
**Status:** Architecture Spec — closes the neurosymbolic integration loop

---

## The Gap Gemini Identified

We have defeasible logic flowing FORWARD (rules → verdicts → scoring). We have TRM learning flowing from execution events (shadow copy → sleep-time → weight updates). But these two paths aren't connected. The defeasible resolver produces rich ternary verdicts (+1, 0, -1) that the TRM never learns from.

**Current flow (broken loop):**
```
Grammar Rules ──→ Defeasible Resolver ──→ Scoring ──→ Answer
                                                        │
Execution Events ──→ Shadow Copy ──→ SleepTime ──→ TRM Weight Update
                                          (disconnected)
```

**Closed loop (what we need):**
```
Grammar Rules ──→ Defeasible Resolver ──→ Scoring ──→ Answer
       ↑                   │                            │
       │                   ▼                            ▼
       │         Defeasible Verdict Events    Execution Events
       │                   │                            │
       │                   ▼                            ▼
       │              Shadow Copy ◄─────────────────────┘
       │                   │
       │                   ▼
       │              SleepTime Consolidation
       │                   │
       │                   ▼
       └──── TRM Weight Update (ternary routing)
                   │
                   ▼
           New Grammar Rules (auto-detected anti-patterns from defeats)
```

---

## Four Concrete Gaps to Close

### Gap 1: Defeasible Verdicts → Shadow Copy Events

**What's missing:** After `_apply_defeasible_specialist_resolution()` (all three stages), the verdicts are used for scoring but never recorded as events.

**Fix:** After each defeasible resolution stage, emit a `DefeasibleVerdictEvent` to the shadow copy:

```python
@dataclass
class DefeasibleVerdictEvent:
    stage: str              # "early_gate", "intra_path", "final"
    candidate_id: str
    program_id: str         # Grammar rule that was evaluated
    verdict_trit: int       # +1, 0, -1
    proof_tag: int          # packed (D, d) pair
    rule_strength: int      # strict/defeasible/defeater
    was_defeated_by: str | None  # rule_id of the superior rule, if defeated
    confidence: float       # mean confidence of supporting workers
    timestamp_us: int
```

**Key insight:** The `was_defeated_by` field is gold for learning. When the TRM sees "rule X was defeated by rule Y in context Z," it learns to prefer Y over X in similar future contexts WITHOUT needing to re-run the defeasible resolver.

### Gap 2: learn_from_feedback — Binary → Ternary

**What's missing:** `trm_navigator.learn_from_feedback()` takes `success: bool`. The 0-signal (undetermined) is lost — it's forced into True or False.

**Fix:** Extend to accept ternary outcome:

```python
def learn_from_feedback(
    self,
    *,
    query: str,
    specialist: str,
    success: bool | None = None,  # None = undetermined (0-signal)
    ternary_outcome: int = 0,     # explicit trit: +1, 0, -1
    confidence: float | None = None,
    domain_hint: str | None = None,
    defeat_source: str | None = None,  # which rule defeated this one
) -> None:
```

**Routing weight update logic:**

| Outcome | Signal | TRM Update |
|---------|--------|------------|
| +1 (proven) | Strengthen this routing path | `weight += learning_rate * confidence` |
| -1 (defeated) | Weaken AND record anti-pattern | `weight -= learning_rate * confidence` |
| 0 (undetermined) | DON'T update weight, but flag for exploration | `exploration_pressure += 1` |

**The 0-signal is the key innovation:** It doesn't push the weight in either direction — instead it increases "exploration pressure" on that specialist node. When exploration pressure exceeds a threshold, the TRM tries alternative routing paths. This is the "don't prematurely prune" behavior Gemini described.

### Gap 3: Defeasible Defeats → Auto-Generated Anti-Pattern Rules

**What's missing:** `execution_grammar_detector.py` creates anti-patterns from tool chain failures. But defeasible defeats are a STRONGER signal — they come from explicit rule conflicts, not just statistical failure observation.

**Fix:** When a defeasible verdict is -1 (defeated), and `was_defeated_by` is known, generate a Grammar Galaxy anti-pattern rule:

```python
GrammarRule(
    rule_id=f"antipattern_{defeated_rule_id}_defeated_by_{superior_rule_id}",
    rule_strength=-1,  # defeater
    superior_to=[defeated_rule_id],  # blocks the defeated rule
    trust_weight=0.5,  # starts at 0.5, grows with recurrence
    rpn_program="...",  # the inverted/blocking program
    semantics={
        "source": "defeasible_auto_detected",
        "pattern_type": "defeasible_defeat_antipattern",
        "defeated_rule": defeated_rule_id,
        "defeating_rule": superior_rule_id,
        "occurrence_count": 1,
        "ternary_confidence": -1,
        "contrastive_recommendation": "block_in_context",
    },
)
```

**This is the self-improving loop:** The system learns its own defeaters from runtime behavior. Over time, the Grammar Galaxy accumulates a web of superiority relations that were DISCOVERED, not authored. Sleep-time consolidation prunes weak ones and strengthens strong ones.

### Gap 4: Exploration Routing on 0-Signal

**What's missing:** When a defeasible verdict is 0 (undetermined — conflicting rules with no clear winner), the system treats it as "no opinion." But 0 should mean "explore alternatives."

**Fix:** In the swarm dispatch, when a candidate has `specialist_defeasible_verdict = 0.0` AND `specialist_proof_tag` indicates undetermined:

1. Don't penalize the candidate in scoring (current behavior, correct)
2. BUT flag the path for "exploration mode" in the next swarm iteration
3. In exploration mode, the swarm tries alternative Grammar rules that weren't tried before
4. This maps to the TRM's `exploration_pressure` counter from Gap 2

**Implementation:** Add `exploration_candidates` list to the swarm context. After final defeasible resolution, any candidate with proof_tag = (0, 0) gets added. On the next query tick (daemon mode), these undetermined candidates get priority exploration.

---

## Implementation Steps (for Codex)

### Step A: DefeasibleVerdictEvent (Gap 1)

1. In `execution_events.py`, add `DefeasibleVerdictEvent` dataclass
2. In `knowledgeverse.py`, after each defeasible resolution stage, emit events to shadow copy
3. Ensure events include `stage`, `verdict_trit`, `was_defeated_by`, `program_id`

### Step B: Ternary learn_from_feedback (Gap 2)

1. In `trm_navigator.py`, extend `learn_from_feedback()` to accept `ternary_outcome: int` and `defeat_source: str | None`
2. Update routing weight logic: +1 strengthens, -1 weakens, 0 increases exploration_pressure
3. Add `exploration_pressure` counter to specialist tree nodes
4. In `consolidate_weights_from_events()`, process `DefeasibleVerdictEvent`s alongside existing execution events

### Step C: Auto-Generated Defeater Rules (Gap 3)

1. In `execution_grammar_detector.py`, add method `_defeasible_defeat_to_grammar_rule(verdict_event)` that creates a defeater GrammarRule from a -1 verdict
2. During sleep-time consolidation (`_stage_b_logic`), collect defeasible defeat events and generate Grammar anti-pattern rules
3. New rules get `rule_strength = -1` (defeater) and `superior_to = [defeated_rule_id]`
4. Trust_weight starts at 0.5, increases with recurrence

### Step D: Exploration Routing (Gap 4)

1. In `_apply_defeasible_specialist_resolution()`, collect candidates with proof_tag = (0, 0) into an `exploration_candidates` field on the knowledgeverse
2. In daemon tick mode (Phase C), these candidates get priority in the next reasoning cycle
3. For now (non-daemon): just thread `specialist_exploration_pressure` into the candidate dict for logging/analysis

### Step E: Tests

1. Test that defeasible verdicts generate shadow copy events
2. Test that -1 verdicts create auto-detected defeater rules via sleep-time
3. Test that 0 verdicts increase exploration_pressure without changing routing weights
4. Test that +1 verdicts strengthen routing weights
5. Test backward compat: existing execution event flow unchanged

### Step F: Full Benchmark

Must hold: ARC 10/10, Math 20/20, GSM8K ≥ 2/10, MMLU ≥ 17/50.

The closed loop won't improve benchmark scores immediately (it needs multiple query cycles to learn). But it MUST NOT regress, and the events/anti-patterns should be observable in logs.

---

## Why This Is the Missing Piece

Daniel said it: "this was the missing piece." Here's why:

**Before:** The TRM navigates Galaxy, picks rules, runs them. If a rule fails, the execution event says "fail" and weights update slowly via binary success/failure. The system has no idea WHY it failed.

**After:** The TRM navigates Galaxy, picks rules, runs them through defeasible resolution. If a rule is DEFEATED (not just "failed" — specifically contradicted by a superior rule), the system knows:
- WHICH rule lost (`was_defeated_by`)
- WHY it lost (superiority relation — the more specific rule won)
- WHAT to do about it (generate defeater anti-pattern, weaken routing weight)
- WHEN to explore instead of commit (0-signal → exploration pressure)

This is the difference between "I tried something and it didn't work" (binary) and "I tried X, but Y is specifically better because of rule Z, and in future contexts where both apply, I should prefer Y" (defeasible ternary).

**The game engine analogy completes:** Not only does the NPC (TRM) check for collisions (defeasible conflicts), it LEARNS from them. A game NPC that walks into a wall once and never tries that path again. That's what the closed loop gives us.

---

## Sovereignty Compliance

| Component | Path | Sovereign? |
|-----------|------|-----------|
| DefeasibleVerdictEvent recording | Post-hot-path (Python OK) | N/A |
| learn_from_feedback update | Weight persistence (Python OK) | N/A |
| Auto-generated defeater rules | Ingestion (Python OK) | N/A |
| Exploration pressure routing | Weight dispatch (Python OK) | N/A |
| Hot-path defeasible resolution | PTX kernel (unchanged) | YES |

**Zero changes to hot path.** All new logic is in the feedback/learning path, which is Python-side and runs post-query or during sleep-time. The sovereign PTX kernel is untouched.
