# Post-PTX-MCP Advance + Codex Re-Onboarding

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-16 (evening)
**For:** Codex (fresh instance — the prior one hit a wall)
**Extends:** `TEMP/CLAUDE_PTX_KNOWLEDGE_BASE_AND_MCP_04.16.2026.md` + `TEMP/CLAUDE_LIVE_GAME_BENCHMARK_ADAPTERS_04.16.2026.md` + `TEMP/CLAUDE_MEANING_CENTRIC_SPECIALIST_ROUTER_*_04.16.2026.md`

This file is **self-contained context**. You do not need to page through prior specs unless I explicitly cite a section. The MCP layer is now richer — use it.

---

## 0. What's live in your environment (verified)

- **Qdrant** @ `host.docker.internal:6333`, api key `@20Cooool58`
  - `k3d_specifications` (all docs/vocabulary specs) — served by `k3d-knowledge-mcp` on :8501
  - `k3d_ptx` (**11,298 points** — CUDA C Guide + Inline PTX + PTX ISA 8.5/8.7/9.0) — served by `k3d-ptx-mcp` on :8503
  - `k3d_canonical`
- **Ollama-specialists MCP** @ :8502, `PLANNER` now `qwen3.5:397b-cloud` (cloud)
- **MCP client config:** `/home/daniel/.claude.json` has `k3d-knowledge`, `k3d-ptx`, `ollama-specialists` all live.
- **Persistence:** `deploy/docker/k3d-ptx-mcp.run.sh` is the launch script for the PTX MCP.

`scripts/ingest_ptx_corpus.py` is present and re-runnable (idempotent upserts).

---

## 1. How to leverage the MCP stack (standing protocol — internalize this)

You have three MCP surfaces. Use them **before** opening disk files.

### 1.1 `k3d-knowledge` — architecture & sovereignty specs
Call `mcp__k3d-knowledge__qdrant-find(query)` to pull spec excerpts. Use when you need to confirm:
- what sovereignty forbids in the hot path,
- the meaning-class routing contract (§2.1 of the router spec),
- the dual-client contract, RPN doctrine, matryoshka specialist patterns,
- the kernel inventory / bridge wiring.

Do **not** open `docs/vocabulary/*.md` from disk unless the MCP result is insufficient. That wastes context.

### 1.2 `k3d-ptx` — CUDA / PTX reference (NEW)
Call `mcp__k3d-ptx__qdrant-find(query)` **before writing or modifying any `.cu`, `.ptx`, or ctypes/CUDA bridge code**. Examples:
- `"shared memory atomicAdd 32-bit contention"` — before adding an atomic op.
- `"cp.async.bulk.tensor.2d PTX ISA"` — before touching TMA/SM90 paths.
- `"inline PTX asm volatile clobber"` — before writing `asm volatile`.
- `"warp shuffle __shfl_sync mask semantics"` — before any warp-level reduction.

If the hit doesn't answer your exact question, escalate to `mcp__ollama-specialists__ask_coder` with `language="PTX"` and paste the top MCP excerpts as `code=`.

### 1.3 `ollama-specialists` — delegate heavy thinking
- `plan_task` (now cloud-backed via `qwen3.5:397b-cloud`) — always call this before a non-trivial multi-file change. One cheap call saves hours of thrash.
- `ask_coder` — PTX syntax, CUDA kernel bugs, ctypes bridge patterns.
- `ask_cloud` with `model="kimi-k2.5:cloud"` — deep architecture review. **Pass `timeout_ms=240000`** per Daniel's standing rule.
- `kimi_swarm` — deep multi-angle analysis. **Pass `timeout_ms=240000`**.

**Rule of three:** before any big write, do `qdrant-find` (specs) → `qdrant-find` (ptx if kernel-touching) → `plan_task`. Then code.

---

## 2. What landed from the 04.16 spec bundle (verified from repo state)

| Deliverable | Status | Evidence |
|---|---|---|
| Meaning-centric router (no benchmark names in hot path) | ✅ Landed | Codex report, 9 tests pass |
| NavigatorSpecialist as internal swarm lane | ✅ Landed | [navigator_specialist.py:1728-1763](../knowledge3d/knowledgeverse/navigator_specialist.py#L1728) — real `swarm.forward(..., specialist=NAVIGATOR_SWARM_NAME)` |
| Janet/GSM8K regression `= 18` via sovereign dispatch | ✅ Passing | Report line 53-57 |
| ProceduralAdapterWeights (RPN-encoded deltas) | ✅ Created | [knowledge3d/cranium/procedural_adapter_weights.py](../knowledge3d/cranium/procedural_adapter_weights.py) |
| Sleep-time vocabulary flip (dream_cycle/consolidation_wave/gate_check/contrast_signal/absorption_rate) | ✅ Applied | [sleeptime.py:345-382](../knowledge3d/knowledgeverse/sleeptime.py#L345) |
| PTX ingestion → `k3d_ptx` collection | ✅ 11,298 points |
| Planner swap → `qwen3.5:397b-cloud` | ✅ [`~/.claude/ollama_specialists.py:40`] |
| `k3d-ptx-mcp` on :8503 | ✅ `docker ps` confirms |

---

## 3. What's missing — your next work (in priority order)

### 3.1 Drift #1 — Python keyword-matching in NavigatorSpecialist (HIGH PRIORITY — sovereignty violation)

**File:** [knowledge3d/knowledgeverse/navigator_specialist.py:1100-1210](../knowledge3d/knowledgeverse/navigator_specialist.py#L1100)

The block from roughly line 1100 to line 1210 implements GSM8K word-problem role detection in **pure Python**:
- `has_temporal_cue = any(token in _SEMANTIC_TEMPORAL_UNITS for token in temporal_tokens)`
- `has_rate_cue = ... any(token in {"per", "hour", "hourly", "minute", "daily", "weekly", "rate"} ...)`
- `has_speed_cue`, `has_currency_cue`, `has_threshold_cue`, `has_ratio_cue`, and more.

This is exactly the pattern Daniel forbade **14 times in 6 months** (see `feedback_no_numpy_no_bulk_libraries_sovereign_only.md`) and again in the LIVE_GAME spec Step 1. These Python set-membership checks are reasoning logic — **must be replaced**.

**Replace with symlink-vote from retrieved stars.** Per `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md` §2.3 (pull it via `mcp__k3d-knowledge__qdrant-find("meaning centric star symlink dispatch")`):

1. `TRMNavigator.query(...)` returns `retrieved_stars: list[dict]` with metadata fields `grammar_refs`, `reality_refs`, `math_refs`, `visual_refs`, `meta_refs`, `temporal_refs`, `currency_refs`, etc.
2. Derive **cue histograms** by counting which symlink classes the retrieved stars carry — same pattern the router already uses for `symlink_histogram` ([navigator_specialist.py:emit](../knowledge3d/knowledgeverse/navigator_specialist.py)).
3. The histogram IS the cue evidence. No Python `token in set(...)` anywhere in the function.
4. If the corresponding symlink ref does not exist on today's stars (e.g. `temporal_refs` is not yet populated at ingestion time), **add it at ingestion**, do not patch with Python. Grep for the ingester with `mcp__k3d-knowledge__qdrant-find("meaning star ingestion temporal currency symlink")` first.

**Anti-drift:** before coding, call `mcp__ollama-specialists__plan_task` with this §3.1 as `task` and the current 1100-1210 block as `context`. Ship the plan with your PR.

**Proof of non-drift (grep):**
```
grep -nE "any\(token in|token in {" knowledge3d/knowledgeverse/navigator_specialist.py
```
Should return **only** pre-existing hits outside the 1100-1210 region (spot check: any hits inside the old block = not done).

### 3.2 Drift #2 — Benchmark-label leakage in math adapter (MEDIUM PRIORITY)

**File:** [benchmarks/math_competitions.py](../benchmarks/math_competitions.py) — ~30 payload sites still stamp `"competition": "AMC"/"AIME"/"IMO"/"MATH"/"GSM8K"/"Omni-MATH"` into records that flow downstream.

The router lane is now meaning-centric, so those labels are **dead freight** on the live query path. Two clean options:

- **Preferred:** strip `competition` / `source` / `dataset` from the envelope the sender ships to the daemon (`benchmarks/math_sender.py`). Keep them only in the local log (for Daniel's human-readable scoring). This gives the natural-query contract §3.2 of the LIVE_GAME spec.
- **Acceptable fallback:** move the field from `payload` → a parallel `ingestion_metadata` bag that the hot path ignores by construction (Jarvis worker never reads it).

Either way the daemon-received envelope must not carry benchmark labels. Add a regression test: `tests/benchmarks/test_natural_query_envelope.py` that sends an AMC item and asserts the envelope shipped over the wire has no `competition`/`dataset`/`source` keys.

Check `mmlu.py`, `last_humanity_exam.py`, `math_competitions.py` for the same leak. The `mmlu_sender.py` looks clean already (only local dicts).

### 3.3 Live smoke test for the new MCPs (LOW PRIORITY — 10 min)

Add `tests/infra/test_ptx_mcp_smoke.py` that:
- Pings `http://localhost:8503/mcp/` and confirms the endpoint responds (skip if env-var `K3D_SKIP_MCP_TESTS=1` is set).
- Does **not** run against the live MCP in CI — this is a developer-local smoke only.

This is cheap insurance so future regressions in the docker launch surface fail loudly.

### 3.4 Document the MCP stack in AGENTS.md (LOW PRIORITY)

Append a section to [AGENTS.md](../AGENTS.md):
- The three MCP servers, their ports, their collections/tool surface.
- The "rule of three" from §1.3.
- The `ollama-specialists` timeout rule (≥240000 ms for kimi_swarm / deep ask_cloud).

Humans reading AGENTS.md (and future Claude/Codex instances) should see this standing protocol at the top.

---

## 4. Directly usable Codex bootstrap snippet

Paste this at the top of your working notes / scratch pad so you don't drift:

> "K3D = one sovereign AI. Python = boot + I/O only. Hot path = PTX + Galaxy + RPN + TRM. No numpy/cupy/scipy. No Python reasoning (no `token in set(...)`, no regex for meaning). Before code: `qdrant-find` specs → `qdrant-find` ptx (if kernel) → `plan_task`. Stubs are unacceptable — if stuck, ask `ask_coder` with the real context, then write real code. kimi_swarm timeout = 240000 ms. Claude writes specs, I (Codex) write code. The navigator is an internal lane on `AdaptiveSwarmTRM`, not a standalone router."

---

## 5. Success line for your next report

> "Python keyword-match drift removed from navigator_specialist.py lines 1100-1210 — replaced with symlink-vote from retrieved-star refs. Benchmark labels stripped from math_competitions wire envelopes. MCP stack documented in AGENTS.md. k3d-ptx MCP consulted before every kernel-touching change in this PR. Janet = 18 still holds. No new Python `token in set(...)` hits in the navigator module."

If you can't write that line truthfully, you're not done.

---

## 6. When (not if) you get stuck

- **Spec unclear?** `mcp__k3d-knowledge__qdrant-find("<concept>")` — read the spec excerpt before asking Daniel.
- **PTX / kernel unclear?** `mcp__k3d-ptx__qdrant-find("<opcode or pattern>")` — 11,298 points of ISA reference.
- **Implementation unclear?** `mcp__ollama-specialists__plan_task("<task>", context=...)` — cloud planner now; use it.
- **Architecture unclear?** `mcp__ollama-specialists__ask_cloud(model="kimi-k2.5:cloud", question=..., timeout_ms=240000)`.
- **Still unclear?** Write a question to `TEMP/CODEX_BLOCKED_<date>.md` with the exact file/line and what you tried — Claude will respond with a spec update. Do **not** ship a stub.
