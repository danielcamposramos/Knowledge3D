# Meaning-Centric Specialist Router — Remove Benchmark-Named Routing

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-16
**For:** Codex
**Supersedes:** `CLAUDE_RESIDENT_WORLD_LIFECYCLE_04.15.2026.md` Fix B (LHE as distinct surface kind) — **architecturally wrong**, revoke it.
**Doctrine source:** `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` §2.3 · `docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §0–§1b · `docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md`

---

## 0. Why This Spec Exists (Daniel's correction, 2026-04-16)

> "We do not need to name LHE or any other benchmark. We are constructing the next generation of embodied procedural AI, not a benchmark machine. LHE is multi-hop questions — and so are many other real user queries this AI will see along its life. Our knowledgeverse is meaning-centric, so must be the reasoning. Meaning-centric but RPN-actionable (using the stars' metadata). The router was supposed to be a specialist, not deterministic."

Current code treats `LHE`, `MMLU`, `ARC`, `GSM8K` as **first-class routing categories**. That is a benchmark harness, not a living AI. Benchmarks are natural queries ([MEMORY.md](../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/MEMORY.md) · `project_benchmarks_as_natural_activity`); the AI must route by **meaning**, not by benchmark label.

Canonical doctrine (`MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` §2.3):

> "Python NEVER decides 'this is a math problem, route to math.' The meaning stars' symlinks SAY it involves math. Jarvis follows symlinks. The Galaxy's structure IS the routing logic."

This spec removes the benchmark-named scaffolding and replaces it with a **meaning-centric specialist router** driven by star metadata (RPN-actionable symlinks) and a learned routing bias, not lookup tables keyed on benchmark names.

---

## 1. Anti-Pattern Audit (what must be removed / reshaped)

### 1.1 Benchmark-named tables in `knowledgeverse.py`

| Location | Symbol | Problem |
|---|---|---|
| [knowledge3d/knowledgeverse/knowledgeverse.py:384-390](../knowledge3d/knowledgeverse/knowledgeverse.py#L384-L390) | `HALTING_WEIGHT_TABLE` | Halting weights keyed on `GAME_2D`/`MATH`/`QUESTION`/`LHE`. A halting gate should weight by **which reasoning kernels converged**, not by benchmark label. |
| [knowledge3d/knowledgeverse/knowledgeverse.py:740-766](../knowledge3d/knowledgeverse/knowledgeverse.py#L740-L766) | `_normalize_semantic_task_type` | Maps `MMLU_TASK → QUESTION`, `LHE_TASK → LHE`, etc. A benchmark label → surface-kind lookup. Must become **meaning-class inference from query content**, not from the benchmark's name. |
| [knowledge3d/knowledgeverse/knowledgeverse.py:4329-4420](../knowledge3d/knowledgeverse/knowledgeverse.py#L4329-L4420) | `_infer_query_mode`, `_task_specialist_name` | Branch on benchmark declared task type. Hot-path `if task_type == "LHE"` branching violates MEANING_CENTRIC §2.3. |
| [knowledge3d/knowledgeverse/knowledgeverse.py:5502-5510](../knowledge3d/knowledgeverse/knowledgeverse.py#L5502-L5510) | `surface_bridge_prefix` | **Especially egregious:** prepends hardcoded English prompt-engineering strings (e.g. `"multi-hop chained reasoning evidence chain graph traversal inference"`) to the query text *before embedding*. This is surface-form poisoning of a meaning-centric embedding. |
| [knowledge3d/knowledgeverse/knowledgeverse.py:12858-12859](../knowledge3d/knowledgeverse/knowledgeverse.py#L12858-L12859) | `_resolve_halting_weights` call site | Reads `HALTING_WEIGHT_TABLE` by benchmark-named surface kind. |

### 1.2 Benchmark-named families in `sovereign_hot_path.py`

| Location | Symbol | Problem |
|---|---|---|
| [knowledge3d/knowledgeverse/sovereign_hot_path.py:79-87](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L79-L87) | `MEANING_FAMILY_ROUTE_MINIMA` | Required-roles-per-family keyed on `GAME_2D`/`MATH`/`QUESTION`/`LHE`. |
| [knowledge3d/knowledgeverse/sovereign_hot_path.py:88-95](../knowledge3d/knowledgeverse/sovereign_hot_path.py#L88-L95) | `MEANING_ROUTE_CLOSURE_MINIMA` | Closure-per-family keyed on benchmark names. |
| Loader stars | `route_family` field in sovereign-exempt loader | Every foundational star is stamped with a benchmark-shaped `route_family`. |

### 1.3 Fix B of the resident-world spec (2026-04-15)

The previous spec told Codex to add `LHE` as a **distinct surface kind** alongside `QUESTION`. Daniel's correction invalidates that direction. **Roll it back** — there should be no distinct `LHE` family; there should be no benchmark-named families at all.

---

## 2. Target Architecture — Meaning-Centric Specialist Router

### 2.1 Meaning classes (replace benchmark families)

Routing categories are **meaning-classes of reasoning**, derivable from the query's star footprint (not from its dataset label). Use this canonical set (aligned with `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md` layers + `TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §1):

| Meaning-class id | What it captures | Example natural queries |
|---|---|---|
| `FACTUAL_RECALL` | Single hop, single fact | "capital of France" · MMLU definition Qs |
| `MULTI_HOP_INFERENCE` | Chain 2+ facts to answer | LHE multi-hop · "who taught the teacher of X" |
| `NUMERIC_COMPUTE` | Quantities, arithmetic, formulae | "2+3" · GSM8K word problems · IMO algebra |
| `SPATIAL_TRANSFORM` | Grid / shape / mesh reasoning | ARC-AGI · ARC-3 game frames · 3D manipulation |
| `DEFINITION_LOOKUP` | What-is / meaning-of | Glossary · dictionary · grammar terms |
| `COMPARATIVE_CHOICE` | Pick option by evidence | MMLU multi-choice · elimination logic |
| `GENERATIVE_COMPOSITION` | Create new artifact (text / drawing / RPN) | Creative writing · draw-a-cat · synthesize |
| `GROUNDED_DIALOG` | Conversational intent | Chat · clarification · meta-questions |

**These classes are expressed as meta-stars in the Galaxy** (one star per class) so they are introspectable, symlink-able, and evolve with learning. They are **not** an enum on a Python dict. The set above is the seed — the avatar may spawn sub-classes (matryoshka) via sleep-time (`sleeptime_protocol_specification.md` §5).

### 2.2 Routing source of truth = star symlinks

Per `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` §2.3:

> "The key insight: Python NEVER decides 'this is a math problem'. The meaning stars' symlinks SAY it involves math."

Therefore the router reads the **query's retrieved meaning stars**, walks their symlinks (`grammar_refs`, `reality_refs`, `math_refs`, `visual_refs`, `meta_refs`), and dispatches specialists to workers based on which symlinks are present. No benchmark label ever enters the hot path.

### 2.3 Router as learned specialist (not lookup table)

Per `TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` §1b:

- **NavigatorSpecialist** is itself a specialist (a small LoRA-style adapter inside TRM) that reads `(query_embedding, retrieved_star_symlink_histogram)` and emits a **meaning-class distribution** + per-kernel halting weight vector.
- Its weights are trained by sleep-time (`ternary_contrastive_learning_specification.md`): successful traces reinforce the emitted distribution; anti-pattern hits weaken it.
- At cold start it falls back to a **uniform prior** + pure symlink-histogram — not to a benchmark-name lookup. The bootstrap behavior must not encode benchmark names.

### 2.4 Halting weights become kernel-addressed, not family-addressed

`HALTING_WEIGHT_TABLE` currently indexes 9-vector kernel weights by benchmark surface kind. Replace with:

- A fixed 9-entry vector of **kernel identities** (already encoded via `FIXED_GRE_WORKERS` and `HALTING_WORKER_CODE_ORDER`).
- A `halting_weight_vector` **emitted by the learned NavigatorSpecialist** per query, conditioned on meaning-class distribution and symlink histogram — not on benchmark name.
- Default (prior) is uniform (`1.0` × 9). Sleep-time shapes the prior over time.

---

## 3. Implementation Steps for Codex

**Priority ordering: 1 → 4 is strict. Do not reorder.**

> **Reminders for tooling** (you have access to the same MCPs Claude does):
> - Use `mcp__k3d-knowledge__qdrant-find` on the spec before you write code. Saves re-reading full specs.
> - Use `mcp__ollama-specialists__kimi_swarm` for multi-angle review of each step (think=True). **Timeout must be ≥ 180 seconds** — the default 120s will kill kimi_swarm mid-synthesis. If your harness uses 120s, bump it.
> - Use `mcp__ollama-specialists__ask_coder` to draft the Python edits, then review with `kimi_swarm`.
> - Use `mcp__ollama-specialists__plan_task` before touching the hot path — validate your plan against the specs first.

### Step 1 — Remove benchmark-named surface kinds from hot-path embedding

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

1. **Delete** `surface_bridge_prefix` (lines ~5502-5510) entirely. No benchmark-conditioned prompt prefixing. The embedder receives the raw query text.
2. **Delete** the `_normalize_semantic_task_type` aliases that reference `MMLU_TASK`, `LHE_TASK`, `GSM8K_TASK`, `IMO_TASK`, `ARC`, `ARC_TASK`. Keep only generic kinds (`GENERAL`, `GRAMMAR`, `CHAT`, `INTERACTION`) — and document them as *legacy* for one release, scheduled for removal once §3.3 lands.
3. **Replace** `_infer_query_mode` and `_task_specialist_name` with a single thin shim that returns `"GENERAL"` unless the router specialist (§3.3) has emitted a meaning-class. The shim MUST NOT consult `competition`, `benchmark`, `dataset`, `subject`, or `domain_hint` payload keys. Grep the codebase for those keys in the hot path and delete their branches.

**Sovereignty check:** `grep -nE '(MMLU|LHE|GSM8K|ARC_TASK|benchmark|competition)' knowledge3d/knowledgeverse/knowledgeverse.py` should return **zero hits inside functions on the query hot path**. Ingestion/logging may still reference them.

### Step 2 — Remove benchmark-named family tables

**File:** `knowledge3d/knowledgeverse/sovereign_hot_path.py`

1. **Delete** `MEANING_FAMILY_ROUTE_MINIMA` and `MEANING_ROUTE_CLOSURE_MINIMA` (lines 79-95).
2. Replace with `MEANING_CLASS_ROUTE_MINIMA` keyed on the eight meaning-classes from §2.1:

   ```python
   MEANING_CLASS_ROUTE_MINIMA = {
       "FACTUAL_RECALL":          {"routers": 1, "executors": 2, "validators": 2, "anti_patterns": 2},
       "MULTI_HOP_INFERENCE":     {"routers": 1, "executors": 4, "validators": 3, "anti_patterns": 2},
       "NUMERIC_COMPUTE":         {"routers": 1, "executors": 5, "validators": 3, "anti_patterns": 2},
       "SPATIAL_TRANSFORM":       {"routers": 1, "executors": 4, "validators": 2, "anti_patterns": 3},
       "DEFINITION_LOOKUP":       {"routers": 1, "executors": 2, "validators": 2, "anti_patterns": 2},
       "COMPARATIVE_CHOICE":      {"routers": 1, "executors": 3, "validators": 3, "anti_patterns": 3},
       "GENERATIVE_COMPOSITION":  {"routers": 1, "executors": 3, "validators": 2, "anti_patterns": 2},
       "GROUNDED_DIALOG":         {"routers": 1, "executors": 2, "validators": 2, "anti_patterns": 2},
   }
   ```

3. Add a single closure `MEANING_CLASS_ROUTE_CLOSURE_MINIMA` keyed on the same eight meaning-classes (same shape as existing `MEANING_ROUTE_CLOSURE_MINIMA`).
4. Every `route_family` read in this file (grep hits at lines 1948, 1966, 2149, plus loader) must be renamed to `meaning_class` and use the new table. Loader stars keep their `route_family` field **temporarily** (for artifact backwards-compat) but emit a derived `meaning_class` that is what the router consumes.

### Step 3 — Repoint halting weights to kernel-addressed emission

**File:** `knowledge3d/knowledgeverse/knowledgeverse.py`

1. Replace `HALTING_WEIGHT_TABLE` with a single named default constant `HALTING_WEIGHT_PRIOR_UNIFORM = (1.0,) * 9`.
2. Rewrite `_resolve_halting_weights(task_type)` to:
   - Call `self._navigator_specialist.emit(query_embedding, symlink_histogram)` (stub OK at §3.3).
   - If specialist not ready, return `HALTING_WEIGHT_PRIOR_UNIFORM`.
3. **No branch** on benchmark surface kind. Period.

### Step 4 — Introduce NavigatorSpecialist (stub with learned path)

**New file:** `knowledge3d/cranium/specialists/navigator_specialist.py`

Thin Python shim over a GPU-resident LoRA-style adapter (shape: `[64 → 32 → (8 meaning classes + 9 halting weights)]`). Contract:

```python
class NavigatorSpecialist:
    """
    Learned router. NOT a deterministic lookup.

    Input:
      query_embedding: float32[64]  (RUNTIME_EMBED_DIMS)
      symlink_histogram: float32[SYMLINK_CLASS_COUNT]  (derived from top-K retrieved stars)

    Output:
      meaning_class_dist: float32[8]   (softmax over 8 classes in §2.1)
      halting_weight_vec: float32[9]   (positive weights, no normalization; sigmoid * 2.0)
    """

    def emit(
        self,
        query_embedding: list[float],
        symlink_histogram: list[float],
    ) -> tuple[list[float], list[float]]: ...

    def update_from_trace(
        self,
        trace: dict[str, Any],  # sleep-time emits; includes positive/negative/anti_pattern signals
    ) -> None: ...
```

- Back the forward pass with an existing GRE specialist kernel (`gre_specialist_mlp.cu` or equivalent — match what bridges/sovereign_bridges.py exposes).
- Backward pass: delegate to `ternary_contrastive_learning_specification.md` + existing `rlwhf_policy.py` update hook. Do not duplicate trainer logic.
- At boot, if weights file absent: initialize to `prior_uniform` (meaning_class_dist uniform 1/8, halting uniform 1.0). No benchmark-named priors.

**Hook it in** at [knowledgeverse.py:11962-12040](../knowledge3d/knowledgeverse/knowledgeverse.py#L11962-L12040) (`_apply_specialist_swarm_features`) and at `_resolve_halting_weights`. Route minima lookups use the top-1 emitted meaning class.

### Step 5 — Ingestion-path loader decouples `route_family` from meaning

**File:** the foundational loader (search `route_family = str(star.get("route_family")`).

- Keep `route_family` in the ingestion-side star payload for backwards-compat **but stop using benchmark names**. Instead, infer `route_family` from the star's content — specifically its symlinks and RPN program class (math_refs → NUMERIC_COMPUTE; visual_refs → SPATIAL_TRANSFORM; etc.).
- If a star has no symlinks yet, mark `route_family = "FACTUAL_RECALL"` (safe default, not a benchmark name).
- Stars that currently have `route_family = "LHE"` / `"MMLU"` etc. must be re-stamped during the next artifact rebuild with their inferred meaning-class.

### Step 6 — Tests (sovereignty + meaning-class)

Add to `tests/knowledgeverse/`:

1. `test_router_no_benchmark_names.py` — greps the hot path (`knowledgeverse.py` query-path functions, `sovereign_hot_path.py`) for `"LHE"`, `"MMLU"`, `"GSM8K"`, `"ARC_TASK"` string literals. Asserts zero hits.
2. `test_navigator_specialist_prior.py` — NavigatorSpecialist with no training produces uniform distributions (no benchmark-leaning bias).
3. `test_meaning_class_routing.py` — five natural queries (one pure-factual, one multi-hop, one numeric, one spatial, one definition). Assert each routes to the correct meaning-class **without** any benchmark label in the input.
4. Non-regression: existing benchmark sweeps (`benchmarks/last_humanity_exam.py`, `benchmarks/arc_agi.py`) still run — but they now submit questions as *natural queries*, not `LHE_TASK` / `ARC_TASK` metadata. The harness is a submitter, not a router.

---

## 4. What NOT to Do

- ❌ Do not add `LHE` as a distinct surface kind (revoke the previous spec's Fix B).
- ❌ Do not conditionally prepend English strings to the embedder input (`surface_bridge_prefix`) even "just for fallback."
- ❌ Do not keep benchmark-named keys "temporarily for compatibility" anywhere in the query hot path. If they must persist for ingestion/logging, isolate them in a submodule named `_legacy_benchmark_labels.py` and do not import from it in the hot path.
- ❌ Do not rewrite the NavigatorSpecialist as a giant if/else over benchmark hints. It is a learned adapter.
- ❌ Do not drop benchmark sweeps — they are health checks, not modes. Just stop treating them as special.

---

## 5. Success Criteria

1. `grep -nE '(MMLU|LHE|GSM8K|ARC_TASK|competition|benchmark)' knowledge3d/knowledgeverse/knowledgeverse.py` — **zero hits in any function executed during query resolution** (ingestion/logging OK).
2. Same grep on `knowledge3d/knowledgeverse/sovereign_hot_path.py` — **zero hits**.
3. `tests/knowledgeverse/test_router_no_benchmark_names.py` passes.
4. `tests/knowledgeverse/test_navigator_specialist_prior.py` passes (uniform prior at cold start).
5. One-question smokes — MMLU, LHE, GSM8K, ARC — all produce non-empty `trace_star_ids` and `fired_kernels`, with `meaning_class` emitted by the NavigatorSpecialist matching the expected class for the query's content (not for its benchmark label).
6. 50-question sweep: no `arc2` empty-answer or `mmlu` ctypes wedge caused by benchmark-family dispatch mismatch. (The ctypes wedge itself is a separate bug — see `CLAUDE_RESIDENT_WORLD_LIFECYCLE_04.15.2026.md` §follow-up.)
7. Sleep-time: after N sweeps, the NavigatorSpecialist's emitted `halting_weight_vec` has shifted from uniform prior — proving the router is learning, not looking up.

---

## 6. Docs Deliverables (also for Codex)

1. Append a **Router Contract** section to `docs/vocabulary/TRM_SPECIALIST_MATRYOSHKA_ARCHITECTURE.md` pinning the 8 meaning-classes (§2.1) + NavigatorSpecialist I/O signature.
2. Update `docs/ROADMAP.md` with the phase marker: "Meaning-Centric Router Landed".
3. Add a `TEMP/CODEX_MEANING_CENTRIC_ROUTER_REPORT_<date>.md` summarizing: files changed, greps run, tests passing, before/after kernel counts, before/after hot-path Python line count (continuing the ~200-line target).

---

## 7. Handoff Note to Future Claude / Codex Instances

- **Claude = architecture only.** Do not run code in next review — inspect diffs, run grep through the Grep tool, point Codex back to the specs.
- **Codex = implementation.** Use the Ollama MCPs. Respect the 180-second kimi_swarm timeout. Parallelize kimi_swarm + ask_coder + plan_task when you can.
- **Daniel's rule (stated 14 times):** **no numpy, cupy, scipy, sympy in hot path.** The NavigatorSpecialist's forward pass is PTX + bridges, not numpy. Training may use bulk libraries (sleep-time) but inference cannot.
- **Sovereignty principle:** the router is a specialist *inside* the one AI. No `if task_type ==` branching in the hot path. Ever.
