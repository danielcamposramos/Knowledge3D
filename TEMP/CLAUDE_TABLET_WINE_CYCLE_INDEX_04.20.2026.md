# Tablet WINE Spec Cycle — 2026-04-20 Index

**Author:** Claude (Architecture Partner)
**Date:** 2026-04-20
**Status:** Cycle complete — four specs ready for Codex
**Umbrella:** `TEMP/CLAUDE_TABLET_AS_PROCEDURAL_INTERFACE_04.20.2026.md`

---

## Why this index exists

This session landed four tightly-coupled specs that make the Tablet a first-class procedural interface for every "old paradigm" the living AI needs to serve. They were written in parallel and cross-reference each other; this index is the one-stop entry point for Codex (or any future Claude instance) to pick up the full cycle.

**Paradigm framing (Daniel, 2026-04-20):**
> "K3D is a live game with a living AI inside using the Tablet as a virtual interface (procedural) to old paradigms: LLMs, benchmarks answering, old chat interfaces … ingestion of documents … using sleeptime compute to transmute temporary stars to actual house knowledge. For now it only needs to work, not be pretty."

---

## The four specs

| # | Spec | Status | Owns |
|---|---|---|---|
| 1 | [Umbrella — Tablet as Procedural Interface](CLAUDE_TABLET_AS_PROCEDURAL_INTERFACE_04.20.2026.md) | Ready | Contract shape, boundary rules, paradigm inventory, sovereignty gate |
| 2 | [Proceduralize WINE v2](CLAUDE_SMART_PROCEDURALIZER_SPEC_V2_04.20.2026.md) | Ready | External LLM-assisted proceduralization via Ollama cloud; MeaningCentricStar output |
| 3 | [Chat WINE](CLAUDE_CHAT_WINE_SPEC_04.20.2026.md) | Ready | Multi-turn free-text CHAT surface, in-process CLI, stateless server |
| 4 | [Ingest WINE + Sleeptime Transmutation](CLAUDE_INGEST_WINE_SPEC_04.20.2026.md) | Ready | Document ingestion queue, temporary-star region, sleeptime promote/merge/discard |

**All four share** the canonical WINE triad (`<SURFACE>_ROUTE_GALAXIES`, `build_<surface>_route`, `build_<surface>_task`, `<surface>_envelope`) and hand a single `TabletEnvelope` to the sovereign core.

---

## Key design invariants (consistent across all four)

1. **WINE is pure envelope construction.** No reasoning. No external calls outside `tablet/wine/`. No in-line calling between WINEs — they emit envelopes, the daemon dispatches.
2. **`TabletIngest.<surface>_task` is the factory.** Adds one classmethod per paradigm; construction logic lives there.
3. **CHAT is synchronous** (user waits). **INGEST is asynchronous** (returns a receipt; sleeptime finishes the work). **PROCEDURALIZE is asynchronous** (external LLM round-trip). **Benchmarks** run at harness cadence.
4. **Sleeptime IS hot path** (corrected 2026-04-20). It is the model's learning phase — consolidates *knowledge* (stars) AND *logic* (TRM + specialist weights) in VRAM, fast. Only the final save-to-disk step uses outside libraries. See `feedback_sleeptime_is_hot_path.md` and Ingest spec §10.
5. **Temporary-star region** (Ingest §7) is the staging ground. Sleeptime (Ingest §10) is the promotion mechanism. Nothing lands in House without earning it.
6. **Sovereignty grep is the drift detector.** `grep -rn "ollama\|requests\|httpx" knowledge3d/ --include="*.py" | grep -v "tablet/wine/"` must stay empty.
7. **Registry is append-only.** New `SURFACE_KIND_INGEST` is additive; no existing codes renumbered (per `feedback_expand_not_replace_opcodes.md`).
8. **Paper scope holds.** Specs target GSM8K / Math Competitions lift (paper deadline 2026-11-08), NOT full Grammar 103k enrichment or ARC scope expansion.

---

## Codex landing order (dependency chain)

1. **Umbrella read-through** — understand the contract boundaries.
2. **Chat WINE** — lands first because (a) it reuses existing `chat_specialist`, (b) it drives the `handle_command_inprocess` refactor every other CLI depends on, (c) it has the tightest regression gate ("what is 2+3?" → "5"). Wins confidence.
3. **Ingest WINE** — adds `SURFACE_KIND_INGEST`, `ingest_task`, temporary-star region. No sleeptime promotion yet — just the receipt path.
4. **Sleeptime transmutation** — bring the sleeptime runner online; close the ingest → promote → next-chat-turn loop.
5. **Proceduralize WINE v2** — move `knowledge3d/ingestion/proceduralizer_wine.py` → `knowledge3d/tablet/wine/proceduralize_wine.py`. Fix drift; pass sovereignty grep; confirm benchmark non-regression.

**Parallelizable:** Chat + Ingest + Proceduralize v2 can land in any order after Umbrella is read. Sleeptime depends on Ingest.

---

## Global test gate

After the cycle lands, the following all pass:

```bash
# 1. No drift outside WINE
grep -rn "^import \(ollama\|requests\|httpx\)" knowledge3d/ --include="*.py" \
  | grep -v "tablet/wine/"     # must be empty

# 2. All three new surfaces answer
python -m knowledge3d.tablet.chat --text "what is 2+3?"
# → {"status":"ok","response":"5","gpu_execution":true,...}

python -m knowledge3d.tablet.ingest --source file:///tmp/hello.md --mime text/markdown
# → {"status":"ok","result_kind":"ingest_receipt","ingest_id":"..."}

python -m knowledge3d.tablet.sleeptime --run-once
# → tick telemetry JSON; promoted/merged/discarded counts

# 3. Benchmark non-regression
pytest benchmarks/math_*.py -k "pinned"
# → ARC 10/10, Math 20/20 green
```

---

## Outstanding non-blocking questions

Consolidated from the four specs' §"Open Questions" tails:

- **Chat:** history persistence? bytes cap tuning? streaming design?
- **Ingest:** s3:// default? stream-ingest > 4k chunks? temporary region retention after daemon restart?
- **Sleeptime:** 60s idle literal vs House-constant driven? discard retention policy?
- **Proceduralize v2:** model profile pinning vs floating? retry envelope for plan_limit_consumed?

None block the cycle. All are tuning calls; MVP defaults are documented inline.

---

## Files the cycle creates or edits

| File | New / Edit | Spec |
|---|---|---|
| `knowledge3d/tablet/wine/chat_wine.py` | New | Chat |
| `knowledge3d/tablet/wine/ingest_wine.py` | New | Ingest |
| `knowledge3d/tablet/wine/proceduralize_wine.py` | New (moves from `ingestion/`) | Proceduralize v2 |
| `knowledge3d/tablet/chat.py` | New | Chat |
| `knowledge3d/tablet/ingest.py` | New | Ingest |
| `knowledge3d/tablet/sleeptime.py` | New | Ingest §10 |
| `knowledge3d/knowledgeverse/sleeptime_ingest.py` (or extend `sleep_time_micro.py`) | New or extend | Ingest §10 |
| `knowledge3d/bridge/headless_tablet.py` | Edit | Chat + Ingest (adds `chat_task`, `ingest_task`, `SURFACE_KIND_INGEST`) |
| `knowledge3d/daemon/main.py` | Edit | Chat + Ingest (extracts `handle_command_inprocess`; adds INGEST handler; CHAT handler goes through WINE) |
| `knowledge3d/knowledgeverse/knowledgeverse.py` | Edit | Ingest (adds `enqueue_ingest`) |
| `knowledge3d/ingestion/proceduralizer_wine.py` | Delete (after move) | Proceduralize v2 |
| `tests/tablet/test_chat_wine.py`, `test_chat_cli.py`, `test_ingest_wine.py`, `test_ingest_cli.py`, `test_sleeptime_ingest.py`, proceduralize v2 tests | New | All |

No deletions of existing `knowledge3d/ingestion/*.py` parser modules — the proceduralizer WINE drives them.

---

## Success definition

The cycle is "shipped" when:

1. A user types in chat → gets an answer (existing path, now routed through WINE).
2. A user ingests a document → gets a receipt immediately.
3. Sleeptime tick promotes the document's knowledge into House.
4. The next chat turn naturally answers better because semantic gravity pulls in the new stars.
5. No benchmark regressed. No drift outside `tablet/wine/`.

That's a working living AI with a procedural tablet. Pretty can come later.
