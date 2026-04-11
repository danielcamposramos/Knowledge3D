# CLAUDE → CODEX — Phase 6: Meaning-Centric Math Stars + Sovereign RPN Execution — 2026-04-10

## Context

Phase 5 landed the sovereign Tablet → bridge → fused tick → cosine decode → star lookup path end-to-end, with 41,109 Galaxy stars bound into VRAM. The smoke test for `"2+3?"` returns a wrong answer (`"ARC3 Rule agent adjacent to color 4 → ACTION4"`) because the sovereign answer path has two gaps:

- **Branch A — knowledge gap:** the Math galaxy has 33,951 stars (`sum_all`, `count_value`, `max_value`, …) but they are exposed with `selection_role='unknown'`, `star_type=0`, and no visible RPN / `meta_rule_addr` / program metadata.
- **Branch B — execution gap:** even if a Math star is selected by cosine, there is no wiring in the fused tick's `HANDLING_QUERY` phase that runs the star's attached RPN program against operands from the query.

Daniel's direction is explicit:

> "Math should be straightforward as this AI works with an RPN math core as substrate. Knowledge is the gap now, or it's execution — the idea is that it navigates the meaning-centric by semantic clue and executes the metadata with the RPN way of solving it."

**Governing principle for Phase 6:** TRM's job is navigation, not computation. The Galaxy carries the programs. The RPN core runs them. That is the substrate.

## Authoritative references

- [docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md](../docs/vocabulary/DUAL_CLIENT_CONTRACT_SPECIFICATION.md) — 288-byte ActionBuffer, form+meaning procedural foundation
- [docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md](../docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md) — Layer 1 Form, Layer 2 Meaning, Layer 3 Rules
- [docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md](../docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md) — programs-before-opcodes, Math tier opcodes
- [docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md](../docs/vocabulary/KNOWLEDGEVERSE_SPECIFICATION.md) — 7-region VRAM substrate, 384-byte star record
- [docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md](../docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md) §9 — kernel function contract map
- `knowledge3d/knowledgeverse/galaxy_vram_table.py` — canonical star record offsets
- `knowledge3d/cranium/kernels/cosine_similarity.cu` — existing y_new decode path
- `knowledge3d/cranium/cuda/gpu_task_dispatch.cu` — existing sovereign dispatch already runs routing + cosine + TRM
- `knowledge3d/cranium/ptx/trm_step_fused.cu` — HANDLING_QUERY phase landing zone

## Phase 6 is three slices

### 6.A — GPU-side deep probe (proof before action)

**Why first:** the current probe only inspected host-side dicts returned by `sovereign_hot_path.py`. It did not read the raw 384-byte GPU star record. Before seeding anything, prove definitively whether star records already carry an RPN payload that the fused tick could execute, or whether that slot is empty.

**What to probe:** for 8 representative Math stars (`sum_all`, `count_value`, `max_value`, `min_value`, `unique_count`, plus one each of whatever the labels/surfaces of the 3 highest-cosine stars against the embedding of `"2+3"` are), dump the raw 384-byte record directly from the device buffer and decode at every documented offset in `galaxy_vram_table.py`:

| Offset | Field | Expected for a math meaning-star |
|---|---|---|
| 0 | `embedding[64]` | non-zero, normalized |
| 256 | `galaxy_id` | Math galaxy id |
| 264 | `selection_role_id` | `ROLE_EXECUTOR=2` or `ROLE_ANSWER=4` |
| 272 | `star_flags` | answer-eligible bit set |
| 276 | `answer_eligible` | 1 |
| 304 | `star_hash` | non-zero |
| 360 | `position[3]` | placed in Math galaxy region |

Additionally, scan the full 384 bytes for any non-zero region between 120 and 256 that could be holding a `meta_rule_addr`, program pointer, or inline RPN byte sequence.

**Deliverable:** `tests/test_phase6a_math_star_probe.py` under `K3D_PYTEST_PROBE_CUDA=1`. Output the raw bytes as hex, the decoded fields as a table, and a one-line verdict per star: `HAS_RPN_PAYLOAD` or `NO_RPN_PAYLOAD`.

**Gate to 6.B:** if probe proves some Math stars already carry executable payloads, 6.B becomes an enrichment pass and the executor wiring becomes urgent. If probe proves none carry payloads, 6.B becomes a seeding pass and the test in 6.C must cover populating a new field.

---

### 6.B — Foundational math meaning-stars with RPN programs

**Target file:** `knowledge3d/knowledgeverse/foundational_galaxy_builder.py`

**What to seed as meaning-stars** (minimal first pass — Daniel's intent is that this substrate is *small and composable*, not a giant dictionary):

1. **Operator stars** — one per arithmetic operator:
   - `add` / `+` — `meta_rule_addr` points to RPN `[PUSH A, PUSH B, ADD, RET]`
   - `sub` / `-` — `[PUSH A, PUSH B, SUB, RET]`
   - `mul` / `*` / `×` — `[PUSH A, PUSH B, MUL, RET]`
   - `div` / `/` / `÷` — `[PUSH A, PUSH B, DIV, RET]`
   - `pow` / `^` / `**` — `[PUSH A, PUSH B, POW, RET]`

   Each operator star:
   - `selection_role = ROLE_EXECUTOR` (or a new `ROLE_RPN_PROGRAM` if the existing taxonomy does not cleanly fit — prefer reusing `ROLE_EXECUTOR`)
   - `answer_eligible = 1`
   - `meta_rule_addr = <VRAM offset into a program table you will allocate>`
   - embedding seeded from a deterministic text template like `"arithmetic addition operator plus a + b"` using the existing sentence-transformers pipeline in the ingestion path (ingestion only — hot path never touches sentence-transformers).

2. **Digit stars** — `0` through `9`:
   - `selection_role = ROLE_ANSWER` (carry numeric value)
   - store the integer value in the slot currently used for `star_type`-adjacent metadata (pick an unused 4-byte slot in the 384-byte record, document it in `galaxy_vram_table.py`, and hold it stable for Phase 6)
   - embedding seeded from `"digit N numeral number integer"` templates

3. **RPN program table** — one VRAM-resident byte buffer holding the operator bytecode sequences, allocated and owned by `sovereign_hot_path.py` (never by `knowledgeverse.py`). `meta_rule_addr` on operator stars is an offset into this buffer, not a pointer.

**Non-goals for 6.B:**
- Do NOT seed 33,951 math stars with RPN programs. Only the 5 operators + 10 digits = 15 stars.
- Do NOT rewrite the existing `sum_all`/`count_value` stars — they may be grammar-layer stars for a different pipeline; leave them untouched.
- Do NOT introduce any new Python fallback for arithmetic. The only legitimate Python here is the *ingestion-time* construction of the bytecode and the embedding text template.

**Deliverable:**
- `knowledge3d/knowledgeverse/foundational_galaxy_builder.py` — new helper `build_math_operator_stars()` and `build_digit_stars()`, both called from `ensure_default_galaxies_loaded()`.
- `tests/test_phase6b_math_operator_seed.py` — asserts the 15 stars exist in the VRAM table, `selection_role == ROLE_EXECUTOR`/`ROLE_ANSWER`, `meta_rule_addr` on operators is non-zero and points inside the program table, and cosine-closest star to `"plus"` embedding is `add`.

---

### 6.C — Sovereign RPN execution inside HANDLING_QUERY

**Target files:**
- `knowledge3d/bridge/headless_tablet.py`
- `knowledge3d/cranium/bridges/trm_step_fused_bridge.py`
- `knowledge3d/cranium/ptx/trm_step_fused.cu`
- `knowledge3d/cranium/actions/action_types.py`

**Tablet operand packing (boundary Python, legitimate):**

Extend `TabletIngest.math_task()` to parse digits from the query text (Python regex is fine here — this is the I/O boundary, not the reasoning layer) and pack them into the ActionBuffer's `tablet_data[]` slots:

- `tablet_data[6] = int(first_operand)`
- `tablet_data[7] = int(second_operand)`
- `tablet_data[8] = operand_count`
- `tablet_data[9] = operator_hint_char_code` (e.g. `'+' = 0x2B`) — TRM may still pick a different operator via cosine, but this is a hint the fused tick can fall back on if cosine is ambiguous.

Keep the existing `task_hash`/`query_hash` fields intact — just reuse currently-zero slots.

**Fused tick HANDLING_QUERY wiring:**

Inside `trm_step_fused.cu`'s `HANDLING_QUERY` phase, after `y_new = TRM(query_embedding)` and after `top_star = cosine(y_new, galaxy_table)`:

```
if top_star.selection_role == ROLE_EXECUTOR && top_star.meta_rule_addr != 0:
    operands[0] = action_buffer_in->tablet_data[6]
    operands[1] = action_buffer_in->tablet_data[7]
    result = rpn_execute_device(
        program_table_base + top_star.meta_rule_addr,
        operands,
        operand_count = action_buffer_in->tablet_data[8]
    )
    // Materialize result as answer
    action_buffer_out->tablet_data[0] = result
    action_buffer_out->header.action_type = UPDATE_TABLET
    action_buffer_out->header.result_star = top_star.hash
```

**`rpn_execute_device`:** must be a `__device__` function callable from the fused tick. If no such helper exists yet, extract one from the existing sovereign RPN runtime (see the kernel surfaces in `docs/vocabulary/SOVEREIGN_NSI_SPECIFICATION.md` §9). Do NOT call a host-orchestrated RPN path. If extraction is non-trivial, land a minimal Math-tier-only `rpn_execute_device` supporting only `ADD/SUB/MUL/DIV/POW/PUSH/RET` for Phase 6 and file a follow-up for the full opcode set.

**Program table binding:**

- `sovereign_hot_path.py` owns the program table (allocated in the same lifecycle as the Galaxy VRAM table).
- `trm_step_fused_bridge.py` takes a new `bind_program_table(ptr, size)` method called by `HeadlessTabletMPC._bind_bridge_query_runtime()` right after `bind_galaxy_table()`.
- Fused tick kernel gains a `program_table_base` parameter alongside `galaxy_table`.

**Tablet readback:**

`HeadlessTabletMPC` reads `action_buffer_out->tablet_data[0]` when the result star is an executor, and renders `"5"` (or whatever the integer result is). No Python arithmetic on the way out — the Tablet is strictly a presenter.

**Deliverables:**
- `tests/test_phase6c_sovereign_math_2_plus_3.py` under `K3D_PYTEST_PROBE_CUDA=1` — submit `"2+3?"` through `HeadlessTabletMPC`, assert the returned answer string is `"5"`, assert via `read_action_buffers_words()` that `tablet_data[0] == 5`, and assert the resolved star hash corresponds to the `add` operator star.
- `tests/test_phase6c_sovereign_math_grid.py` — the 2×2 matrix of `{2+3, 7-4, 6*8, 15/5}` all returning correct integer results.
- Sovereignty grep clean on all touched Cranium files.

---

## Deliberate non-goals for Phase 6

- No Grammar Galaxy parsing of free-form word problems. `"What is two plus three?"` is Phase 6+.
- No multi-step programs. A single operator + two operands is the whole scope.
- No benchmark re-run. Benchmarks are health checks, not gates, during this embodied rebuild.
- No fallback Python arithmetic anywhere, including the "just make it work for the demo" escape hatch.

## Quality gates

- `test_phase6a_math_star_probe.py` green (CUDA real)
- `test_phase6b_math_operator_seed.py` green (CUDA real)
- `test_phase6c_sovereign_math_2_plus_3.py` green (CUDA real)
- `test_phase6c_sovereign_math_grid.py` green (CUDA real)
- Phase 1–5 regression batch still green: the 53-test suite must not lose a single test
- Sovereignty grep:
  `rg -n "import (numpy|cupy|scipy|sympy)|from (numpy|cupy|scipy|sympy)" knowledge3d/cranium/ptx/trm_step_fused.cu knowledge3d/cranium/bridges/trm_step_fused_bridge.py knowledge3d/cranium/kernels/ knowledge3d/cranium/cuda/` → no matches
- Tablet parser Python (regex operand extraction) lives ONLY in `knowledge3d/bridge/headless_tablet.py` — not in anything under `knowledge3d/cranium/`

## Order of operations

1. 6.A probe → report to Claude with the `HAS_RPN_PAYLOAD` / `NO_RPN_PAYLOAD` verdict per star
2. Based on probe, Claude confirms or amends 6.B scope (seed vs. enrich)
3. 6.B seed the 15 operator + digit stars and the program table
4. 6.C wire tablet operand packing, fused tick RPN execution, tablet readback
5. Smoke test `"2+3?"` end-to-end, expect `"5"`

## What this unlocks

Once 6.A–C land, the sovereign answer path is:

```
Query text
  → TabletIngest.math_task (parse operands)
  → ActionBuffer [tablet_data=(2, 3, 2, '+')]
  → bridge.submit_query
  → fused tick HANDLING_QUERY:
        y_new = TRM(query_embedding)
        top_star = cosine(y_new, galaxy_table)       // navigation by semantic clue
        if top_star.selection_role == EXECUTOR:
            result = rpn_execute_device(              // execution via metadata
                top_star.meta_rule_addr, operands
            )
        action_buffer_out.tablet_data[0] = result
  → HeadlessTabletMPC reads result
  → "5"
```

That is the substrate Daniel described. Everything after Phase 6 — Grammar parsing, word problems, multi-step, LHE — is just more stars and more programs composed into the same pipe.

## Handoff

Codex: start at 6.A. Do not touch 6.B until the probe verdict is in and Claude has acknowledged it. If the probe blocks on an unrelated CUDA issue, delegate via `ollama-specialists` `ask_coder` / `plan_task` rather than burning time manually; document what you tried.

Claude will review:
- 6.A probe output → amend 6.B scope if needed
- 6.B seed diff → verify meaning-star offsets and program table layout match `galaxy_vram_table.py`
- 6.C wiring diff → verify sovereignty and that the `rpn_execute_device` path is genuinely GPU-local
