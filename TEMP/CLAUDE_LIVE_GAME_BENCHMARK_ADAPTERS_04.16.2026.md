# Live-Game Benchmark Adapters + RPN-Encoded Shadow Weights

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-16
**For:** Codex
**Builds on:**
- `TEMP/CLAUDE_MEANING_CENTRIC_SPECIALIST_ROUTER_04.16.2026.md`
- `TEMP/CLAUDE_MEANING_CENTRIC_SPECIALIST_ROUTER_ADDENDUM_04.16.2026.md`
- `TEMP/CODEX_MEANING_CENTRIC_ROUTER_REPORT_2026-04-16.md`

**Doctrine source:**
- `docs/vocabulary/SLEEPTIME_PROTOCOL_SPECIFICATION.md`
- `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` §2.3 (symlink execution chain)
- `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` ("programs before opcodes")
- `docs/vocabulary/TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md`
- [MEMORY.md](../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md) → `project_benchmarks_as_natural_activity`, `feedback_runs_are_training`, `feedback_no_fallbacks_ever_including_sleeptime`

---

## 0. Frame (read this first)

**K3D is a living always-on AI, not a benchmark runner.**

- Benchmarks are **natural queries arriving at the always-on tablet**, indistinguishable from a user asking the same question. The AI must not know it is being "benchmarked."
- A benchmark sweep is **a day in the life of the AI** — it lives, answers, and dreams (sleep-time consolidation). Every run = training data.
- Training is **shadow-weight sleep-time compute**, not epoch-batch backprop. Shadow weights absorb; validation gate promotes.
- Weights themselves are **procedural RPN programs** over the base, not float32 tensors. Same fast-precise-machine-friendly math engine as everything else.

If you catch yourself typing `if task_type == "LHE"`, `run_benchmark(...)`, `epoch`, `batch_size`, or `nn.Module` — stop. That is the old way.

---

## 1. Drift to fix first (small, but flag now)

**Drift found in Codex's commit:** [navigator_specialist.py:1702-1726](../knowledge3d/knowledgeverse/navigator_specialist.py#L1702-L1726) — `_meaning_class_hint` derives meaning class from Python keyword matches (`"because" in lowered`, `"solve" in lowered`, etc.) to generate bootstrap labels for sleep-time training samples.

This is Python reasoning logic. Daniel's rule (feedback_no_fallbacks_ever_including_sleeptime): **no Python regex / string ops for reasoning anywhere — not hot path, not sleep-time, nowhere.**

**Replacement (per MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.3):**

Derive the bootstrap meaning-class label from the **retrieved stars' symlinks**, not from the query text. The stars already carry the routing signal:

```
for star in trace.retrieved_stars:
    add star.math_refs     → vote for NUMERIC_COMPUTE
    add star.visual_refs   → vote for SPATIAL_TRANSFORM
    add star.grammar_refs  → vote for (depends on linked program kind)
    add star.reality_refs  → vote for FACTUAL_RECALL / MULTI_HOP_INFERENCE
    add star.meta_refs[forward_entity_extraction] → vote for MULTI_HOP_INFERENCE
    add star.meta_refs[comparative_evidence]      → vote for COMPARATIVE_CHOICE
```

Normalize → that is the bootstrap `meaning_class_target` for the contrastive pair. No keyword matching.

**Task:** delete `_meaning_class_hint` entirely. Replace its single caller in `_trace_to_training_sample` with a symlink-vote helper that reads the trace's recorded retrieved stars. If the trace has no retrieved stars (should be rare), the sample is dropped, not defaulted.

---

## 2. Benchmark adapters → natural-query senders

### 2.1 What changes in the adapters

Current adapters (e.g. [benchmarks/last_humanity_exam.py](../benchmarks/last_humanity_exam.py), [scripts/benchmark_math_comparison.py](../scripts/benchmark_math_comparison.py), `arc_agi_*`) still package queries with benchmark metadata (`task_type="LHE_TASK"`, `competition="MMLU"`, etc.). They must stop.

**New contract for every adapter:**

```
# wrong (current, old way):
knowledgeverse.answer(
    query=question,
    task_type="LHE_TASK",
    competition="LHE",
    options=options,
)

# right (new, living AI):
tablet.submit(
    envelope=TabletEnvelope(
        payload_text=question_as_a_user_would_type_it,
        options=options_if_multiple_choice,
        source="live_query",   # not "benchmark"
        session_id=session_uuid,
    )
)
```

- The envelope carries **only what a human user would supply**: question text, options (if MC), optional attachments. No benchmark label, no `task_type`, no `competition`, no `dataset`.
- `tablet.submit` dispatches to the TRM game loop same as any other query. The AI answers. The answer comes back on the tablet.
- The adapter logs `{question, expected, received, trace_id, timestamp}` to a sidecar JSONL for post-hoc scoring. **Scoring reads the log after the fact** — it does not branch the AI's behavior.

### 2.2 Files to touch

- `benchmarks/last_humanity_exam.py` + `benchmarks/lhe_sender.py` — strip benchmark metadata from envelope; write answers to `runs/<session>/lhe_natural_queries.jsonl`.
- `benchmarks/arc_agi.py` (+ `benchmark_arc_agi_comparison.py`) — same: submit the grid as a natural `input_grid` attachment, not with `task_type="ARC_TASK"`.
- `scripts/benchmark_math_comparison.py`, `scripts/benchmark_lhe_comparison.py`, `scripts/iterative_learning_marathon.py` — all must use `tablet.submit` with natural envelopes.
- `scripts/run_headless_tablet_benchmarks.py` — the existing harness stays but the **submit site** becomes natural-query only. The runner can still know which suite it's running (for the logs' `suite` field), but that suite label must not enter the envelope.

### 2.3 Scoring as observation, not branching

Create `scripts/score_session_log.py`:
- Reads a session JSONL, joins with ground truth from the benchmark's reference file, emits per-suite accuracy + per-meaning-class accuracy.
- Zero interaction with the live AI. Pure post-hoc.
- This is the only place benchmark names appear. Treat it like a log analyzer, not a harness.

### 2.4 Success check

```
grep -nE '(task_type|competition|benchmark|dataset)\s*=' \
    benchmarks/ scripts/benchmark_*.py scripts/iterative_learning_marathon.py \
    scripts/run_headless_tablet_benchmarks.py
```
→ zero hits inside the envelope-construction sites. Only acceptable hit: the logger's `suite=<name>` field for post-hoc scoring.

---

## 3. Game-loop flow (not execution flow)

Every natural query follows the same game tick. Benchmarks do not get a shortcut path.

```
Tablet receives envelope
  → TRM game tick begins (trm_step_fused.ptx)
    → Perceive:  frustum cull over Galaxy neighborhood by query embedding
    → Navigate:  LED-A* + Morton Octree to retrieved stars
    → Route:     Navigator lane emits (meaning_class_dist, halting_weights)
                 — from swarm.forward(..., specialist="navigator")
    → Dispatch:  Jarvis (Worker 8) reads star symlinks → assigns workers 0-7
    → Reason:    Nine-chain swarm executes RPN chains in parallel
    → Halt:      Halting gate consumes halting_weights + worker outputs
    → Answer:    Winning RPN result materializes → tablet response
  → Trace written to VRAM task buffer (same slot as query)
  → Python extracts answer from tablet → returns to adapter
  → (later) sleep-time drains trace buffer → dreams
```

**Implementation task:** confirm the headless tablet path ([knowledge3d/bridge/headless_tablet.py](../knowledge3d/bridge/headless_tablet.py)) invokes the game tick through the sovereign hot path for every envelope — no special branching for "benchmark" sources. Add a test `tests/bridge/test_tablet_single_path.py` that submits three envelopes (one MMLU-shaped, one GSM8K-shaped, one ARC-shaped) **with identical envelope schema** and asserts all three produce the same `program_id` (sovereign dispatch) and populate `trace_star_ids`.

---

## 4. Sleep-time shadow-weight training (not traditional training)

### 4.1 Vocabulary changes (these matter — they shape the code)

| Old word | New word | Why |
|---|---|---|
| `epoch` | `dream_cycle` | Sleep-time compute, not gradient descent on a dataset |
| `batch` | `consolidation_wave` | Groups of traces consolidated together |
| `train_specialist_epoch` (keep, but rename internal path) | `consolidate_specialist_dream_cycle` | Honest name |
| `loss` | `contrast_signal` | It's ternary contrastive, not scalar loss |
| `learning_rate` | `absorption_rate` | Shadow weights absorb at a rate, then gate |
| `validation` | `gate_check` | Shadow→live promotion gate |

Add these as aliases in [adaptive_swarm.py](../knowledge3d/cranium/adaptive_swarm.py). Old names may stay as deprecated shims for one release, but new code uses the new names. Sleep-time code ([sleeptime.py](../knowledge3d/knowledgeverse/sleeptime.py)) switches fully.

### 4.2 Shadow weights promotion gate

Per `TERNARY_CONTRASTIVE_LEARNING_SPECIFICATION.md`:

```
1. Navigator shadow adapter absorbs contrastive pairs from recent traces.
2. After each consolidation_wave, gate_check:
   - Replay N validation traces through both shadow and live.
   - If shadow matches ≥ live on ≥ K cases AND degrades < M cases → promote.
   - Else → discard shadow, keep live. Log the failed promotion.
3. Promotion is atomic: VRAM weight pointer swap, no half-states.
```

Already partially present in [trm_adapters.py](../knowledge3d/cranium/trm_adapters.py) (`SelfUpdatingAdapter` has shadow logic). **Reuse**, do not reinvent. Add a test `tests/knowledgeverse/test_navigator_shadow_promotion.py` that:
- Submits 20 queries → 20 traces.
- Runs one dream cycle.
- Asserts shadow was promoted OR rejected with a reason logged (never silent).
- Asserts live weights are either identical or strictly different bytes after the gate.

### 4.3 No training outside sleep-time

**Grep-enforced invariant:**
```
grep -nE 'train_specialist|consolidate_specialist' knowledge3d/ \
  | grep -v sleeptime.py | grep -v adaptive_swarm.py | grep -v tests/
```
→ zero hits. Training only happens inside sleep-time. Period.

---

## 5. RPN-encoded procedural weights (the biggest leap)

### 5.1 Why

Current specialist adapter weights are stored as dense float32 matrices (via `SelfUpdatingAdapter`). That violates the K3D principle that **everything is procedural RPN over shared primitives**. The fix: encode the specialist delta as an **RPN program over the base weights**, not as a separate tensor.

This is the same principle that makes glyphs, rooms, and math problems all the same substrate. The weights should be too.

### 5.2 Schema — ProceduralAdapterWeights

New module `knowledge3d/cranium/procedural_adapter_weights.py` (the one new file this spec authorizes). Reuses `RPNMathCore`:

```
class ProceduralAdapterWeights:
    """
    Weights as RPN programs over base weights.

    Storage: a single RPN program blob (KB, not MB).
    Materialization: execute program on VRAM base weights → adapter delta.
    Same math engine as Galaxy stars, same opcodes, same math core.
    """
    base_weights_ref: StarRef      # symlink to the base weight star in the Galaxy
    delta_rpn: RPN_Program         # the delta as RPN ops (LOAD_BASE, SCALE, LOW_RANK_ADD, ...)
    ternary_mask: [Trit]           # +1/0/-1 per output dim — hybrid binary + ternary

    def materialize_to_vram(self, ctx: RPNContext) -> VramHandle:
        """Execute delta_rpn against base_weights_ref → adapter weights in VRAM."""

    def absorb_contrast(self, trace_pair: ContrastPair) -> None:
        """Update delta_rpn ops via contrastive signal. Shadow copy only."""
```

### 5.3 RPN ops the adapter needs

Extend `RPN_DOMAIN_OPCODE_REGISTRY.md` with a small LoRA-specific op set (under the "Extended" tier, `0xA0-0xFF`):

- `LORA_LOAD_BASE` — push base weights ref onto stack.
- `LORA_LOW_RANK_ADD rank=r` — pop two factor matrices, add their product as a low-rank delta.
- `LORA_SCALE alpha` — scale top of stack by alpha.
- `LORA_TERNARY_MASK` — pop ternary mask, apply {−1, 0, +1} gating per output dim.
- `LORA_SHADOW_ABSORB` — absorb contrastive signal, update shadow program in-place.

Opcode numbers TBD — coordinate with the opcode registry owner. Commit the registry delta in the same PR.

### 5.4 Migration path (one step, not a long ramp)

- At cold boot, if a specialist has a legacy dense-tensor adapter, convert it to `ProceduralAdapterWeights` on the fly (base weights ref = base star, delta_rpn = one `LORA_LOW_RANK_ADD` op whose factors are the legacy matrices). Persist the procedural form; discard the dense form on next save.
- From that point on, **only procedural adapters exist**.
- Add migration test `tests/cranium/test_procedural_adapter_migration.py`.

### 5.5 Storage win (motivating number)

The navigator adapter (64 → 8+9, rank 8) as dense float32 = ~20 KB.
As RPN program + low-rank factors = ~2-4 KB (program overhead is fixed; factors dominate).
Over 30+ specialists the storage win compounds, but more importantly: **weights live in the Galaxy like anything else**. They can be introspected, symlinked, and evolved by the same pipeline.

---

## 6. Order of operations for Codex

Strict ordering. Do not parallelize:

1. **Fix the `_meaning_class_hint` drift** (§1). Smallest diff; unblocks honest sleep-time labels.
2. **Flip adapters to natural queries** (§2). No routing changes yet — just strip benchmark metadata from envelope construction.
3. **Confirm single game-loop path** (§3). Add the tablet-single-path test.
4. **Rename training vocabulary + shadow gate test** (§4). Behavior mostly unchanged; names align.
5. **ProceduralAdapterWeights** (§5). Biggest leap — save for last, after 1-4 prove the rest.

---

## 7. Tooling reminders

- **`mcp__k3d-knowledge__qdrant-find`** before coding each step — query for "sleep-time consolidation", "shadow weights", "RPN opcode registry extended tier", "tablet envelope schema".
- **`mcp__ollama-specialists__kimi_swarm`** for multi-angle review. **Pass `timeout_ms=240000`** (240 s). The 120 s default kills it mid-synthesis. Use `think=True` for §5's RPN adapter design — it is a genuine architectural step.
- **`mcp__ollama-specialists__ask_coder`** to draft the RPN opcode implementations. Review the draft with kimi_swarm before committing.
- **`mcp__ollama-specialists__plan_task`** before §5 — do not just start coding the procedural adapter.

---

## 8. Sovereignty checks (all must pass before PR)

1. `grep -nE '_meaning_class_hint' knowledge3d/` → **zero hits** (step 1 delete complete).
2. Natural-query invariant grep (§2.4) → **zero hits** in envelope sites.
3. Training-outside-sleeptime grep (§4.3) → **zero hits**.
4. `grep -nE 'import numpy|import torch|import cupy' knowledge3d/cranium/procedural_adapter_weights.py` → **zero hits** (no bulk libs in the new module).
5. Janet regression (`"Janet had 16 ducks…"`) still returns `18` on sovereign dispatch.
6. New tests (§3, §4.2, §5.4) all green.
7. Existing 9 router tests from Codex's report stay green.

---

## 9. Handoff line for your next report

> "Benchmark adapters now submit natural queries indistinguishable from user input. Sleep-time consolidates via shadow-weight promotion gate, with meaning-class labels derived from retrieved-star symlinks (no Python keyword matching). Specialist weights are ProceduralAdapterWeights — RPN programs over base weights, materialized on demand. Every query, including what used to be called a benchmark, now flows through the one game loop. The AI does not know it is being benchmarked."

If that line is not truthfully say-able, one of steps 1-5 was skipped.
