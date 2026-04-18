# Addendum — Internal Lane Contract (No Stubs, No External Dispatch)

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-16
**Extends:** `CLAUDE_MEANING_CENTRIC_SPECIALIST_ROUTER_04.16.2026.md` §3.4 (NavigatorSpecialist)
**For:** Codex — add this **as a new step**, do not edit the prior spec.
**Trigger:** Daniel's correction — "Stubs are not acceptable, do it properly as an internal swarm LoRA-like specialist, not external old ways!"
**Doctrine source:** `docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §0 · `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md` · `docs/vocabulary/HYPER_PARALLEL_PROCESSING.md` · [MEMORY.md](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md) → `feedback_k3d_is_one_sovereign_ai_not_coordinator.md`

---

## 0. Architectural Confirmation (say this back to yourself before coding)

- **K3D is ONE mind.** The TRM (~7M params) IS the avatar. The "internal swarm" is its parallel cognitive lanes (superdotados model), **not external workers, not subprocesses, not a coordinator pattern**.
- **The NavigatorSpecialist is an internal lane of that one mind.** It is a LoRA-style adapter riding on the already-existing `AdaptiveSwarmTRM` base, with procedural RPN math cores spawnable from within (matryoshka sub-specialists per `TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §1).
- **The K3D world follows the Dual Client Contract.** Everything in the world — glyphs, rooms, shelves, stars — is built bottom-up from the same drawing primitives + meta-data + rules. The navigator's inputs (query embedding, symlink histogram) and outputs (meaning-class, halting weights) are **first-class Galaxy citizens**, not free-floating Python data.
- **Python is packet assembler and result extractor.** Nothing else. Routing logic lives in VRAM, decided by Galaxy structure (symlinks) and learned adapter weights — not by Python branching.

If the code you are about to write does not match all four bullets above, stop and re-read the specs via `mcp__k3d-knowledge__qdrant-find`.

---

## 1. Bind to Existing Infrastructure — No New Abstractions

The LoRA-style adapter surface **already exists**. Do not build a parallel stack. The navigator is a **new lane** on the existing swarm, not a new system.

| Capability you need | Existing module — reuse this | Do NOT do |
|---|---|---|
| Base model (~7M, variable-dim) | [knowledge3d/cranium/matryoshka_trm.py](../knowledge3d/cranium/matryoshka_trm.py) — `MatryoshkaTRM` | Create a new base. |
| Self-updating LoRA adapter | [knowledge3d/cranium/trm_adapters.py](../knowledge3d/cranium/trm_adapters.py) — `SelfUpdatingAdapter`, `AdapterConfig` | Invent a new adapter class. |
| Swarm registry & dim scaling | [knowledge3d/cranium/adaptive_swarm.py](../knowledge3d/cranium/adaptive_swarm.py) — `AdaptiveSwarmTRM.register_specialist(...)` | Create a second swarm. |
| Procedural RPN math cores (spawnable) | [knowledge3d/cranium/ptx_runtime/rpn_math_core.py](../knowledge3d/cranium/ptx_runtime/rpn_math_core.py) — `RPNMathCore` | Reimplement arithmetic in Python. |
| Topology memory / routing bias | existing [knowledge3d/knowledgeverse/navigator_specialist.py](../knowledge3d/knowledgeverse/navigator_specialist.py) — keep its multi-path planner intact, **add** the learned emit() on top | Throw away the multi-path planner. |
| Weight persistence | [knowledge3d/knowledgeverse/trm_weight_store.py](../knowledge3d/knowledgeverse/trm_weight_store.py) — `TRMWeightStore` | Write a new JSON loader. |
| Symlink lookup on retrieved stars | existing Galaxy metadata — read `grammar_refs`, `reality_refs`, `math_refs`, `visual_refs`, `meta_refs` off the star dicts | Pattern-match text with regex. |

**Wiring contract (what to actually add):**

1. In `AdaptiveSwarmTRM` init (or first boot of `Knowledgeverse`), register a `navigator` specialist:
   ```
   swarm.register_specialist(
       name="navigator",
       required_dims=64,   # matches RUNTIME_EMBED_DIMS
       rank=8,             # small: it's a gating lane, not a heavy compute lane
   )
   ```
2. Extend the existing `NavigatorSpecialist` class (do not replace it) with an `emit()` method that:
   - Runs the adapter forward pass via `AdaptiveSwarmTRM.forward(..., specialist="navigator")`.
   - Reads output head into `(meaning_class_dist[8], halting_weight_vec[9])`.
   - Caches the topology memory and routing bias the class already maintains.
3. Extend `learn_routing_topology(...)` (already present) to **also** call `swarm.train_specialist_epoch("navigator", trace_samples, validation_samples)` during sleep-time. Sleep-time is the only place training happens.

**No new files required** beyond the addendum in §2.2 of the prior spec. Do not create `navigator_specialist_v2.py`, a `LearnedRouter`, a `RouterAdapter`, or a `NavigatorGatingModel`. Add methods to existing classes.

---

## 2. Inputs and Outputs — First-Class Galaxy Citizens

### 2.1 Inputs (already available on the live path)

- `query_embedding: float32[64]` — output of `_embed_query_gpu` (already exists, [knowledgeverse.py:5520](../knowledge3d/knowledgeverse/knowledgeverse.py#L5520)). **After** the prior spec removes `surface_bridge_prefix`, this is clean meaning-centric input.
- `retrieved_stars: list[dict]` — the top-K stars already returned by `TRMNavigator.query(...)` on the hot path.
- `symlink_histogram: float32[K_symlink_classes]` — derive from `retrieved_stars` by counting which symlink classes appear (grammar_refs, reality_refs, math_refs, visual_refs, meta_refs, audio_refs, behavior_refs, …). This is cheap (a dict + normalize). It is **not** keyword matching on text — it reads the star's metadata the Galaxy already carries.

### 2.2 Outputs

- `meaning_class_dist: float32[8]` — softmax over the 8 meaning classes (§2.1 of prior spec).
- `halting_weight_vec: float32[9]` — positive-valued, sigmoid×2 activation, one entry per fixed GRE worker slot (`FIXED_GRE_WORKERS`, [knowledgeverse.py:355-365](../knowledge3d/knowledgeverse/knowledgeverse.py#L355-L365)).

Both outputs are written to the VRAM task slot alongside `query_embedding` (see `vram_task_buffer.py`) so the GPU-side halting gate and swarm-dispatch kernels consume them **without** Python re-marshaling. That packet is the only thing Python should be assembling.

### 2.3 Procedural RPN math cores behind it (matryoshka)

Per `TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §1 + `RPN_DOMAIN_OPCODE_REGISTRY.md` "programs before opcodes":

- When the navigator emits `NUMERIC_COMPUTE` with high confidence, it triggers **spawn of a child RPN program** via `RPNMathCore` — not a Python arithmetic call.
- Sub-specialist spawning uses the existing `SpecialistBase.spawn_child(...)` pattern ([TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md §2.1](../docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md)). No new spawn machinery.
- The spawned RPN program reads the meaning-star's `meaning_rpn` field (already schema'd in `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` §2.1) and concatenates symlinked programs from the star's refs. This is the symlink execution chain from §2.3 of that spec — **already working** for the "Janet had 16 ducks" case (commit 2dedddcc). Reuse.

---

## 3. Dual Client Contract Check (do this before you commit)

Per `DUAL_CLIENT_CONTRACT_SPECIFICATION.md`: the K3D world is built bottom-up from drawing primitives + meta-data + rules, and any artifact is readable by **humans AND AI**. The navigator must respect this:

- ✅ `meaning_class_dist` is a vector over named classes — human-readable and AI-consumable.
- ✅ Adapter weights persist via `TRMWeightStore` — introspectable.
- ✅ Symlink histograms are derived from star metadata that already has human-readable labels (grammar/reality/math/visual/meta).
- ❌ Do not output an opaque float vector that only the halting gate understands and nothing else can inspect. If a human or a sleep-time auditor cannot read the navigator's decision, redesign the output.

---

## 4. Codex — Anti-Drift Directives

You drifted once to stubs/placeholders. Here is how to not drift again:

1. **No stubs. No `pass`. No `TODO: wire to real model`.** If you would write a stub, stop and call `mcp__ollama-specialists__ask_coder` with the actual module paths in this addendum to draft the real implementation first. Then review with `mcp__ollama-specialists__kimi_swarm` (timeout **≥ 180 s**).
2. **No placeholder classes.** If `AdaptiveSwarmTRM` is missing a method you need, **add the method to `AdaptiveSwarmTRM`** — don't wrap it. This is one mind, not a Chinese-wall architecture.
3. **No external-router patterns.** Any code that reads like "router.decide(task)" and returns a string label is the old way. The internal lane is: forward-pass the adapter, read heads, write to the VRAM task slot. The halting kernel consumes the slot.
4. **No new files unless the prior spec asks for one.** New behavior belongs in the existing `navigator_specialist.py`, `adaptive_swarm.py`, `matryoshka_trm.py`. New tests land in `tests/knowledgeverse/`.
5. **Proof of non-stub:** before declaring a step done, run this grep locally:
   ```
   grep -nE '\bpass\b|NotImplementedError|TODO|placeholder|stub' \
     knowledge3d/knowledgeverse/navigator_specialist.py \
     knowledge3d/cranium/adaptive_swarm.py
   ```
   It should return **only pre-existing hits unrelated to the navigator lane**. New hits = drift → don't commit.
6. **MCP reminder:** you have the same `k3d-knowledge` Qdrant access Claude does. Query it before you guess — `qdrant-find("navigator specialist matryoshka LoRA")`, `qdrant-find("halting gate worker weights")`, `qdrant-find("RPN math core spawnable")`.
7. **kimi_swarm timeout reminder:** the default 120 s kills it mid-synthesis. Pass `timeout_ms=240000` (240 s) or higher when you invoke it. Same for deep `think=True` runs.

---

## 5. Success Criteria for This Step (additive to prior spec §5)

Add these, do not replace:

8. `navigator_specialist.py`'s `NavigatorSpecialist` has an `emit(...)` method that calls `AdaptiveSwarmTRM.forward(specialist="navigator", ...)` — not a hand-rolled forward pass, not a stub, not a dict lookup.
9. `AdaptiveSwarmTRM.register_specialist("navigator", ...)` is invoked at Knowledgeverse cold-boot and the adapter state round-trips through `TRMWeightStore`.
10. Sleep-time's successful trace path calls `swarm.train_specialist_epoch("navigator", ...)` with a dataset derived from positive/negative trace buckets (see `sleeptime.py`).
11. The "Janet had 16 ducks" chain still yields 18 after this change — because the navigator correctly emits `NUMERIC_COMPUTE` dominant, triggering the symlink execution chain.
12. Grep at §4.5 above returns no new stub markers.

---

## 6. Handoff Line to Paste at the Top of Your Next Report

> "NavigatorSpecialist landed as an internal cognitive lane on AdaptiveSwarmTRM, not a standalone router. It reuses MatryoshkaTRM + SelfUpdatingAdapter + RPNMathCore. No new abstractions. No stubs. Python limited to VRAM packet assembly and halting-gate result extraction. Meaning-class + halting weights emitted by the swarm, not a lookup table. 'Janet had 16 ducks' = 18 ✓."

If you cannot write that line truthfully, you are not done.
