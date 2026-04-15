# Codex Spec — Batch 11: Knowledge Waves + Observability + Routing Verification

**Date:** 2026-04-15
**Author:** Claude (Architecture Partner)
**Audience:** Codex
**Status:** Active spec — proceeds in parallel with the post-ingest 50-slice sweep
**Supersedes nothing** — Batch 10 shipped the canonical curriculum loader and the
resident-world runner patch; this spec builds the next wave on top of that work.

---

## 0. Why this spec exists

The 2026-04-15 50-slice strict live sweep (see
`TEMP/CODEX_LIVE_BENCHMARK_50_SLICE_REPORT_2026-04-15.md`) proved three things:

1. The **live wiring is correct**. Every active suite returns 50/50 GPU result
   packets through the tablet translator / `knowledgeverse_dispatch_session`
   path. `solver=knowledgeverse_gpu_query`, `runtime=knowledgeverse_gpu_query`,
   `program_id=gpu_task_dispatch_sovereign`, structured
   `task_result.route_family`. No Python-side orchestration dressed up as a
   sovereign answer.
2. The **accuracy profile is single-digit** across every math-family suite and
   low double-digit on MMLU (8/50 = 16%). This is no longer a transport bug.
3. The system is **blind to its own reasoning**. We cannot tell, for any given
   miss, whether the knowledge was absent, recalled-but-wrong, routed to the
   wrong specialist lane, or collapsed during answer normalization.

Daniel's three directives after that sweep, verbatim:

> "The knowledge was key, but something is missing — it should take better notes
> everywhere."

> "We had to refactor the tablet wine like interfaces, because was still python
> orchestration."

> "Verify all things are being routed to the proper n-swarm head and that all
> knowledge is loaded when it's solving things — the architecture works, we
> know that."

Batch 11 is the concrete response. It has three parallel tracks:

- **Track A — Knowledge waves.** Ingest the remaining HS curriculum in the
  priority order Codex recommended in the 50-slice report.
- **Track B — Observability.** Add per-solve note-taking so the next benchmark
  sweep is interpretable.
- **Track C — Routing verification.** Prove, with artifacts, that every
  crafted star actually lands in the resident world and reaches the N-swarm
  head during solve.

None of these tracks block each other. Codex may land them in any order, but
**no knowledge slice may land without its matching trace slice.** That is the
load-bearing constraint of this spec: "silence on what was loaded vs what was
used is itself a bug."

---

## 1. Track A — Knowledge Waves (Priorities 1-6 + ARC Primitives)

All files in this track are already on disk under `TEMP/`. They are written in
the **same Kimi narrative format** as the earlier HS math cluster files (before
Claude rewrote Clusters 2 and 3 into bullet dialect). Narrative format means:

- Free-form prose with embedded JSON-ish fragments
- Star descriptions scattered across sub-agent reasoning
- Not directly consumable by `parse_cluster1_bullets()`

Codex has two options for each file:

**Option I — Rewrite as bullet dialect (preferred).**
Mirror the format of `TEMP/KIMI_MATH_HS_CLUSTER1_ARITHMETIC_ALGEBRA_2026-04-13.md`
and the rewritten Clusters 2/3. Every star becomes:

```markdown
#### {{canonical header}}

- **canonical_id**: star.concept.{{domain}}.{{slug}}
- **is_a**: concept
- **rpn_sketch**: GALAXY_LOOKUP concept.{{slug}} RECALL
- **symlinks**: star.concept.{{domain}}.{{related}}, star.letter.x
- **surface_forms**:
  - en: "..."
  - pt: "..."
  - es: "..."
  - fr: "..."
  - de: "..."
  - it: "..."
  - ja: "..."
  - zh: "..."
  - ru: "..."
- **saudades**: {{optional cultural note}}
```

The existing `parse_cluster1_bullets()` (at
`knowledge3d/ingestion/hs_math_parser.py`) will ingest these as-is. **No new
parser required.**

**Option II — Build a narrative parser.**
Only if the corpus is too large to rewrite practically. In that case Codex must
ship `parse_kimi_narrative()` in the same module, with the same canonical_id +
category prefix enforcement rules. This is the richer parser pass Codex
flagged in the 50-slice report as needed for humanities.

**Claude's recommendation:** Use Option I for Slices A-C (science + civics +
applied CS) because those are where benchmark impact is highest and bullet
dialect is the most forgiving path. Use Option II for Slices D-F (humanities +
languages + cross-cultural) because their narrative structure is richer and
benefits from a proper parser.

### Slice A — Priority 1: Natural + Earth/Space/Environmental Sciences

**Source files:**

- `TEMP/KIMI_HS_NATURAL_SCIENCES_PHYS_CHEM_BIO_2026-04-13.md`
- `TEMP/KIMI_HS_EARTH_SPACE_ENVIRONMENTAL_2026-04-13.md`

**Target galaxy dispatch:** all stars should land in **Reality** galaxy.
This depends on `canonical_curriculum_loader._target_galaxy()` correctly
keyword-matching `star.domain`. Verify that every crafted star has one of:

- `natural_science`, `physics`, `chemistry`, `biology`, `earth`, `space`,
  `environmental`, `astronomy`

If any star has an empty or unrecognized domain, it falls through to
`target_galaxy_for_star()` from `scripts/ingest_meaning_layer`. **This is a
silent-drop risk.** Codex must assert coverage (see Track C).

**Canonical category prefixes** (enforced by `CanonicalLookup`):

- `concept.physics.*`, `concept.chemistry.*`, `concept.biology.*`
- `formula.physics.*` (for Newton 2nd, KE, PE, etc.)
- `identity.chemistry.*` (for pH, molarity, dilution)
- `method.biology.*` (for mitosis, photosynthesis procedure stars)
- `theorem.earth_science.*` (for plate tectonics, Milankovitch cycles, etc.)

**Surface forms:** all 9 canonical languages (en/pt/es/fr/de/it/ja/zh/ru).

**Symlink dialects required:**

- `star.concept.*` (existing)
- `star.symbol.*` (existing — for chemistry symbols like `pH`, `NaCl`)
- `star.constant.*` (existing — for `g = 9.81`, `c = 3e8`, `Avogadro`)
- **NEW:** `star.element.*` — for periodic table entries (optional, can
  fold into `star.symbol.*` if parser extension is undesirable)

**Expected impact from the 50-slice report:**

- MMLU `college_biology`, `high_school_biology`, `astronomy`,
  `college_physics` → should move from single hits to broader coverage
- LHE Physics, Chemistry, Electrical Engineering domains
- Some grounding for GSM8K word problems that reference physical quantities

### Slice B — Priority 2: History / Geography / Civics / Economics

**Source file:** `TEMP/KIMI_HS_HISTORY_GEOGRAPHY_CIVICS_ECONOMICS_2026-04-13.md`

**Target galaxy dispatch:** Reality (for history/geography) and Tool (for
economics-as-methodology). The loader's current keyword map will send
`history`, `geography`, `earth` → Reality. **But `civics` and `economics` are
not in the map** — they will silently fall through to
`target_galaxy_for_star()`. Codex must extend `_target_galaxy()` to add:

```python
if any(token in domain for token in (
    "history", "geography", "civics", "economics",
    "macroeconomics", "microeconomics", "government",
    "politics", "social_studies"
)):
    return "Reality"
```

**Canonical category prefixes:**

- `concept.history.*`, `concept.geography.*`, `concept.civics.*`,
  `concept.economics.*`
- `theorem.economics.*` (supply/demand, elasticity, comparative advantage)
- `rule.civics.*` (separation of powers, judicial review, etc.)
- `method.economics.*` (marginal analysis, opportunity cost procedures)

**Expected impact:**

- MMLU `high_school_government_and_politics`,
  `high_school_macroeconomics`, `high_school_microeconomics`,
  `high_school_us_history`
- LHE Law domain (currently 1/1, small sample but promising)

### Slice C — Priority 3: Applied CS / Health / Psychology / Sociology

**Source file:** `TEMP/KIMI_HS_APPLIED_CS_HEALTH_PSYCH_SOCIOLOGY_2026-04-13.md`

**Target galaxy dispatch:** Tool (for CS), Reality (for health/psych/sociology
grounded in biology), Language (for sociology/psych cultural framing).
`_target_galaxy()` already routes `computer`, `cyber` → Tool. Add:

```python
if any(token in domain for token in (
    "health", "medicine", "clinical", "psychology",
    "psychiatry", "sociology", "anthropology"
)):
    return "Reality"
```

**Canonical category prefixes:**

- `concept.computer_science.*`, `method.computer_science.*`
- `concept.psychology.*`, `theorem.psychology.*` (e.g., operant conditioning)
- `concept.health.*`, `method.health.*` (basic clinical procedures)
- `concept.sociology.*`

**Expected impact:**

- MMLU `clinical_knowledge`, `high_school_psychology`, `computer_security`,
  `high_school_computer_science`
- LHE Computer Science domain

### Slice D — Priority 4: Humanities / Literature / Philosophy / Religion / Arts

**Source file:** `TEMP/KIMI_HS_HUMANITIES_LIT_PHIL_RELIGION_ARTS_2026-04-13.md`

**Target galaxy dispatch:** Language (for literature/linguistics framing),
Reality (for historical grounding of religious/artistic movements).
`_target_galaxy()` already routes `humanities` → Language. Add:

```python
if any(token in domain for token in (
    "literature", "philosophy", "religion",
    "theology", "ethics", "aesthetics", "arts", "music_theory"
)):
    return "Language"
```

**This is the first slice where Option II (narrative parser) may be warranted.**
Humanities stars often include multi-sentence definitions, etymologies, and
canonical quotations that do not fit cleanly into the bullet dialect.

**Canonical category prefixes:**

- `concept.philosophy.*`, `concept.literature.*`, `concept.religion.*`,
  `concept.arts.*`
- `rule.ethics.*` (categorical imperative, utilitarian calculus, etc.)
- `method.literature.*` (close reading, genre analysis)

### Slice E — Priority 5: Languages / Linguistics

**Source file:** `TEMP/KIMI_HS_LANGUAGES_LINGUISTICS_2026-04-13.md`

**Target galaxy dispatch:** Language (trivial — already routed).

**Canonical category prefixes:**

- `concept.linguistics.*`, `rule.phonology.*`, `rule.syntax.*`,
  `rule.semantics.*`, `method.linguistics.*`

**Note:** Many stars in this file are **language-meta** (about the structure of
language itself). They should symlink to the existing UD grammar and Word
Galaxy stars from the Phase 7A1 landing. Codex should verify that the linker
resolves these cross-wave symlinks rather than creating orphan entries.

### Slice F — Priority 6: Cross-Cultural Glue

**Source file:** `TEMP/KIMI_HS_CROSSCULTURAL_SAUDADES_CALENDAR_EXAMS_PROVERBS_2026-04-13.md`

**Target galaxy dispatch:** Language. `_target_galaxy()` already handles
`crosscultural`.

**Canonical category prefixes:**

- `concept.culture.*`, `concept.calendar.*`, `concept.proverb.*`

**Saudades field:** This slice is the canonical home for the `saudades:`
optional field in the bullet dialect. Codex should verify the parser preserves
it (it should — Cluster 1 already uses it — but worth a regression check).

### Slice G — Parallel Track: ARC Reasoning Primitives

**Source file:** `TEMP/KIMI_ARC_REASONING_PRIMITIVES_CLUSTER_2026-04-14.md`

**Target galaxy dispatch:** Tool (for transform/pattern programs) and Drawing
(for visual primitives). **Neither of these is currently in the
`_target_galaxy()` keyword map.** Add:

```python
if any(token in domain for token in (
    "arc", "arc_agi", "pattern", "transform",
    "visual_reasoning", "grid", "drawing"
)):
    return "Tool"  # or Drawing — depends on subkind
```

**Special handling:** ARC primitives are **procedural programs**, not factual
stars. They should land as RPN programs, not as concept/theorem entries. The
loader currently assumes `meaning_star` kind; ARC primitives likely need a
`program_star` subkind. **Codex must confirm the canonical kind and either
extend the loader or route ARC primitives through a separate ingestion path.**

**Expected impact:** ARC-AGI 2 is currently 0/50 with `GAME_2D` route family.
This slice is the only thing that can move that number.

---

## 2. Track B — Observability / Note-Taking

This track is **not optional**. Per Daniel's directive: "Silence on what was
loaded vs what was used is itself a bug — treat it as a missing observability
kernel, not just a reporting gap."

### Slice H — Per-Solve Trace Schema

For every benchmark item that reaches the tablet session, the solver must emit
a **trace record** with at minimum:

```
item_id            : string  (benchmark-local unique id)
suite              : string  (arc_agi_2, mmlu, gsm8k, lhe, math, ...)
route_family       : string  (GAME_2D, MATH, MMLU, LHE, ...)
specialist_lane    : string  (which N-swarm head fired)
stars_loaded_count : int     (total stars in resident world at solve time)
stars_touched      : list[star_id]   (stars actually queried by RPN)
stars_recalled     : list[star_id]   (stars that returned non-null on RECALL)
opcodes_fired      : list[opcode_name]  (sequence)
halting_reason     : enum    (CONVERGED, TIMEOUT, EMPTY_RECALL, NORMALIZE_COLLAPSE)
raw_answer         : string  (pre-normalization output)
normalized_answer  : string  (post-normalization output fed to scorer)
latency_ms         : int
```

**Storage:** Records must land in **Tablet history** (the resident-world
channel), not in a Python-side summary log. This preserves the sovereignty
boundary — the AI must be able to see its own thinking inside its own memory.

**Python-side export** (for benchmark post-mortem) is allowed, but only as a
**derived view** that scrolls the Tablet history after the sweep completes.
The source of truth is the tablet.

### Slice I — Trace Harvest Runner Patch

`scripts/run_headless_tablet_benchmarks.py` must:

1. After each suite completes, scroll the Tablet history for all trace records
   belonging to that suite.
2. Write a sibling file next to `summary.execution.json`:
   `trace.{{suite}}.jsonl` with one trace record per line.
3. Emit a **coverage report** at the end of the sweep:

```
Trace coverage report:
  arc_agi_2 : traces=50/50  missing_item_ids=[]
  mmlu      : traces=50/50  missing_item_ids=[]
  gsm8k     : traces=50/50  missing_item_ids=[]
  ...
  Total stars_touched distinct: 1234
  Total stars_recalled distinct: 876
  Most-touched-but-never-recalled: [list of 10 stars that were queried but
    always returned null — this is the direct signal of a "knowledge gap"]
  Most-recalled-but-wrong-answer: [list of 10 stars that were recalled and
    the answer was still wrong — this is the direct signal of a "reasoning
    or normalization gap"]
```

That last pair is the **diagnostic payload**. Without it, a 1/50 score is just
noise. With it, Claude and Daniel can draft the next ingestion wave with
evidence instead of intuition.

### Slice J — Normalization Collapse Detector

The 50-slice report flagged stereotyped answers in the math suites: `98`, `20`,
`[0,3)`, `260`, `-7`. This is the signature of **answer-normalization collapse**
— the normalizer is mapping many distinct raw outputs to a small set of
attractor strings.

**Codex must add a check** to the trace harvest:

- For each suite, compute the histogram of `normalized_answer` values.
- If any single normalized answer accounts for `> 20%` of the suite's outputs
  AND that answer is wrong on most of those items, flag it as a collapse
  attractor.
- The flag goes into the coverage report.

This is cheap observability that directly diagnoses the "stereotyped wrong
answer" pattern.

---

## 3. Track C — Routing and Load Verification

Daniel's exact ask: "verify all things are being routed to the proper n-swarm
head and that all knowledge is loaded when it's solving things."

Grep over `scripts/run_headless_tablet_benchmarks.py` returned **zero hits**
for `n.?swarm|nine.?chain|nine_chain|swarm_head|composed_head`. The runner
does not touch the N-swarm head directly; dispatch happens downstream inside
`HeadlessTabletMPC` → `Knowledgeverse` → PTX kernels. That's correct
architecturally, but it means we have no evidence that the dispatch is
actually happening. Track C fixes that.

### Slice K — Knowledgeverse Load Assertion

Right after `load_canonical_curriculum_into_knowledgeverse()` returns, the
runner must:

1. Call `kv.galaxy_manager.snapshot_counts()` (or equivalent) and log per-galaxy
   entry counts.
2. Compare against the loader's returned `by_galaxy` dict.
3. **Assert equality within tolerance.** If any crafted star failed to land
   (e.g., silently dropped by `_target_galaxy()` keyword miss, or rejected by
   a category-prefix validator), the runner must **fail fast** with a
   structured error that names the missing stars.

This directly addresses the silent-drop risk in `_target_galaxy()`.

### Slice L — N-Swarm Head Dispatch Probe

Add a **warm-up probe** that runs before any suite:

1. Construct a trivial known-good query for each route family (`GAME_2D`,
   `MATH`, `MMLU`, `LHE`).
2. Dispatch through the tablet session exactly the way a benchmark item would.
3. Capture the trace record (per Slice H schema).
4. **Assert:** `specialist_lane` is non-null, `stars_touched` is non-empty,
   `halting_reason != EMPTY_RECALL`.

If any route family's warm-up probe fails, the sweep aborts before burning
50×8 = 400 benchmark items on a broken pipeline.

### Slice M — Specialist Lane Coverage Audit

At the end of the sweep, cross-reference the trace records against the
available specialist lanes:

- The 15 GRE specialist kernels loaded at boot
- The 3 RPN tiers (Lite, Standard, Extended)
- The 88 PTX kernels documented in the architectural briefing

The coverage report must show **which lanes were actually fired during the
sweep**. If only 5 of 88 kernels fired (matching the current sovereignty-debt
finding from `CLAUDE.md`), that is a **first-class finding**, not a footnote.

This is how we turn "architecture works" from a claim into a measurement.

---

## 4. Execution order

Codex may parallelize Slices A–G with Slices H–M, but the sweep that proves
Batch 11 complete must run with **both tracks landed**. Suggested ordering if
Codex wants a single-pass execution:

1. **Slice K** (load assertion) — 1 hour. Pure defensive check, no new
   knowledge. Protects every subsequent slice.
2. **Slice H** (trace schema) — 2 hours. Foundation for Slices I/J/L/M.
3. **Slice L** (warm-up probe) — 1 hour. Relies on H.
4. **Slice A** (Natural + Earth/Space) — 3-4 hours. Highest expected benchmark
   impact.
5. **Slice I** (trace harvest) — 2 hours. Makes the next sweep interpretable.
6. **Slice J** (collapse detector) — 1 hour. Cheap but high-value.
7. **Slice M** (lane audit) — 1-2 hours.
8. **Slice B** (History/Civics/Economics) — 3 hours.
9. **Slice C** (Applied CS/Health/Psych) — 3 hours.
10. **Slice G** (ARC primitives) — 3-4 hours. The only ARC-2 mover.
11. **Slices D/E/F** (Humanities/Languages/Cross-cultural) — parallel,
    lower urgency.

Total: roughly 24-30 hours of Codex work before the next full sweep. If Codex
can only land the first 7 slices before the next check-in, that is already
sufficient for the next benchmark run to be diagnostically useful.

---

## 5. Success criteria

Batch 11 is complete when all of the following are true:

- [ ] Slices A, B, C, G landed: at least the four highest-impact knowledge
      waves are in `k3d_canonical` and in the resident world.
- [ ] `_target_galaxy()` covers every domain keyword used in the crafted
      stars. No silent fall-through.
- [ ] Slice K's load assertion passes on a clean sweep: every crafted star in
      `k3d_canonical` is also in the resident Knowledgeverse.
- [ ] Slice L's warm-up probe passes for all four route families.
- [ ] Slice H's trace schema is live: every benchmark item emits a trace
      record into Tablet history.
- [ ] Slice I's trace harvest produces `trace.{{suite}}.jsonl` files alongside
      `summary.execution.json`.
- [ ] Slice I's coverage report contains both the "touched-but-never-recalled"
      and "recalled-but-wrong-answer" lists.
- [ ] Slice J's collapse detector runs and reports any attractor strings.
- [ ] Slice M's lane coverage audit shows **which** of the 88 kernels fired.
- [ ] Next 50-slice sweep report is interpretable: for every miss, we can say
      "knowledge gap" OR "routing gap" OR "reasoning gap" OR "normalization
      collapse" — not "unknown."

The **single measurable outcome Daniel cares about** is the last bullet. If
the next sweep report cannot diagnose misses, Batch 11 is not done regardless
of how much knowledge landed.

---

## 6. Non-goals (explicit)

- **Do not rewrite the tablet WINE interface.** Codex already did the
  resident-world refactor in Batch 10. The sovereign bridge path still hangs
  in CUDA; that is a separate investigation and is out of scope for Batch 11.
- **Do not add advanced math knowledge yet.** The 50-slice report flagged
  that math suites need solver/program refinement, not just more knowledge.
  Math reasoning is a separate track (Batch 12 candidate).
- **Do not touch the ingestion-path sovereignty rules.** Ingestion remains
  flexible (numpy/json/etc. allowed). Only the hot path is sovereign.
- **Do not add new benchmark suites.** The 8 currently active suites are
  enough to prove the observability hypothesis.

---

## 7. What to hand back to Claude

When Batch 11 is complete (or partially complete at the next checkpoint),
Codex should report:

1. Which slices landed, with line counts / file paths.
2. The next 50-slice sweep results **with the new trace coverage report**.
3. The lane coverage audit output (which of the 88 kernels actually fired).
4. Any domain-keyword gaps discovered during Slice K assertion.
5. Any collapse attractors flagged by Slice J.
6. Estimated time to complete remaining slices.

With that payload in hand, Claude can draft Batch 12 with evidence instead of
intuition — which is the whole point of this spec.

---

## 8. Bottom line

Batch 10 proved the wiring. Batch 11 proves the knowledge lands and the
specialists fire. Only after Batch 11 can Batch 12 (reasoning/solver
refinement) be designed with real data.

The crossing we are making with this spec:

> From "we think the architecture works" → "we can measure what the
> architecture does on every solve."

Once the system takes notes everywhere, the next failure mode is no longer
mysterious. That is the unlock.
