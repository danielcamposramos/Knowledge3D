---
date: 2026-04-20
author: Claude (architecture partner, Codex limit-locked)
status: spec v2 — supersedes v1 (v1 misframed as internal pipeline)
scope: external WINE-wired proceduralizer; GSM8K + Math Competitions only
paper: Paper A, deadline 2026-11-08, accuracy rubric 3→4.5 target
related:
  - TEMP/CLAUDE_SMART_PROCEDURALIZER_SPEC_04.20.2026.md (v1 — superseded)
  - TEMP/kimi_swarm_smart_proceduralizer_04.20.2026.md (design notes, still valid)
  - docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.1 (authoritative schema)
  - docs/vocabulary/FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §1.4 (defeasibility)
  - knowledge3d/tools/knowledge_proceduralizer.py (1640 LOC, live)
  - knowledge3d/ingestion/proceduralizer_wine.py (359 LOC, live)
  - knowledge3d/bridge/headless_tablet.py (TabletEnvelope + TabletIngest)
  - TEMP/CLAUDE_PAPER_A_SKELETON_DRAFT_04.19.2026.md
---

# Smart Proceduralizer v2 — External Tool via Tablet WINE

## 0. Corrections since v1

v1 misframed the proceduralizer as an internal pipeline. Daniel corrected:

> "We will not need internal access to the proceduralizer, this is an
> outside tool. It must be able to be called as needed (by user query for
> example — as a tool) but not internal. We'll wire external LLMs using
> a tablet WINE interface (translator)."

Plus three groundings confirmed by sub-agent exploration:

1. **Target schema exists.** `MeaningCentricStar` in
   `docs/vocabulary/MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.1`
   is the authoritative shape. Field is `meaning_rpn` (single program
   string), **not** `arg_keys` + `eval_program`. Identity is `star_id`
   (content hash), **not** `meaning_hash`. v1 invented field names; v2
   uses the spec verbatim.
2. **Old_Attempts already has proceduralizer.**
   `knowledge3d/tools/knowledge_proceduralizer.py` is the live module
   (1640 LOC, 4 cloud models via `PROCEDURALIZER_MODEL_PROFILES`,
   emits 4-layer `knowledge_packets`). `knowledge3d/ingestion/proceduralizer_wine.py`
   (359 LOC) is the transport bridge. Adapt these; do not re-invent.
3. **Paper-scoped.** Focus = GSM8K + Math Competitions only (current
   10% / 0% → target ≥50% / ≥30%). Skip Grammar's 103k-entry
   enrichment and ARC scope. Paper deadline 2026-11-08.

## 1. The external-tool framing

```
┌──────────────────────────────────────────────────────────────────┐
│                      OUTSIDE K3D (no sovereignty)                  │
│                                                                    │
│   user / benchmark harness / paper eval script                    │
│         │                                                          │
│         │  1. invoke proceduralizer tool                           │
│         ▼                                                          │
│   ExternalProceduralizerTool (new, thin)                          │
│         │                                                          │
│         │  2. TabletEnvelope (surface_kind=PROCEDURALIZE)          │
│         ▼                                                          │
├──────────────────────────────────────────────────────────────────┤
│                   TABLET WINE BOUNDARY (translator)                │
│                                                                    │
│   tablet/wine/proceduralize_wine.py (NEW — moves from              │
│     ingestion/proceduralizer_wine.py into tablet/wine/ per the     │
│     WINE location contract)                                        │
│         │                                                          │
│         │  3. ProceduralizerRequest (existing dataclass)           │
│         ▼                                                          │
│   LLM transport: Ollama @ :11434 / :8502 MCP                       │
│   Model profiles: qwen3.5:397b-cloud, kimi-k2-thinking:cloud,     │
│     glm-5:cloud, deepseek-v3.2:cloud                               │
│         │                                                          │
│         │  4. ProceduralizerReceipt → parse_bundle()               │
│         ▼                                                          │
├──────────────────────────────────────────────────────────────────┤
│              SOVEREIGN INGRESS (Region 7 Stargate)                 │
│                                                                    │
│   MeaningCentricStar build (schema §2.1) → star_id hash →         │
│     Matryoshka embeddings → write via existing stargate_feeder     │
│                                                                    │
│   Output: JSONL append on /K3D/Knowledge3D.local/galaxies/         │
│     proceduralized_<split>.jsonl (conforms to §2.1 exactly)        │
└──────────────────────────────────────────────────────────────────┘
```

Key constraints:
- The proceduralizer **never** runs inside knowledgeverse's hot path.
- The sovereign AI, when it needs enrichment, sends a
  `TabletEnvelope(surface_kind=PROCEDURALIZE)` through WINE exactly the
  way it already sends MATH / GAME_2D / QUESTION envelopes. The WINE
  module is the only place LLMs get called.
- Results come back as fully-formed `MeaningCentricStar` entries;
  Region 7 writes them; hot path reads them.

## 2. Target schema — `MeaningCentricStar` (authoritative)

From `MEANING_CENTRIC_STAR_SCHEMA_SPECIFICATION.md §2.1` verbatim. The
proceduralizer **must** emit exactly this shape — no invented fields.

```
MeaningCentricStar {
    star_id:        ContentHash(meaning_rpn || meaning_class || domain)
    meaning_class:  concept | relation | action | property | meta
    meaning_rpn:    RPN_Program        // Layer 2 — THE program
    domain:         DomainPath         // e.g., "math/arithmetic"
    taxonomy_refs:  [StarRef]
    surface_forms:  { "<iso>": { word_ref, char_refs[] } }
    visual_rpn:     RPN_Program        // optional but expected
    visual_refs:    [StarRef]
    audio_rpn:      RPN_Program        // optional
    audio_refs:     [StarRef]
    pronunciations: { "<iso>": AudioRef }
    behavior_rpn:   RPN_Program        // Layer 3 interaction
    reality_refs:   [StarRef]
    grammar_refs:   [StarRef]          // Layer 3 rule refs
    meta_refs:      [StarRef]          // Layer 4
    house_position: Vec3
    house_room:     RoomRef
    confidence:     Trit               // +1/0/−1
    polarity:       Trit               // +1/0/−1
    embeddings:     { tier_64, tier_128, tier_512, tier_2048 }
    component_refs: [StarRef]
    composite_of:   [StarRef]
}
```

Defeasibility (from `FOUNDATIONAL_KNOWLEDGE_SPECIFICATION.md §1.4`) lives
on **Layer-3 GrammarRule stars** specifically, via `rule_strength`,
`superior_to`, `trust_weight` on the Grammar star (not on every star).
For math-path enrichment (GSM8K/Math), the ordinary `MeaningCentricStar`
with `meaning_rpn` + `confidence` + `polarity` is sufficient; GrammarRule
stars are emitted alongside when the LLM proposes a reusable
rewrite/transformation.

### Binding contract for math materialization

Because v1 invented `arg_keys` / `eval_program`, here is the correct
binder path in v2: the proceduralizer emits `meaning_rpn` as a program
string using registered opcodes with **positional numeric literals
already bound in**, not placeholders. For queries that require late-
binding query numbers, the star carries a sibling GrammarRule star with
`rule_rpn` that names slot tokens (e.g. `#N`) registered in the RPN
opcode registry. The existing math path at `knowledgeverse.py:5931` is
a transitional helper that v2 **intentionally does not couple to** —
the long-term path is meaning_rpn execution, not ARG_X substitution.

Concretely for the paper MVP slice: each enriched math entry carries

- a **canonical example** star with fully-bound `meaning_rpn` and
  a `supervision_answer` in metadata (these drive the benchmark);
- **one or more** Layer-3 GrammarRule stars in `grammar_refs` whose
  `rule_rpn` is the abstracted template (with named slots registered
  under a reserved opcode block, see §6).

This matches what the live `PROCEDURALIZER_BUNDLE_JSON_SCHEMA` in
`knowledge_proceduralizer.py` already emits (knowledge_packets with
`layer_kind ∈ {form, meaning, rule, meta_rule}`) — v2 is wiring, not
schema change.

## 3. Salvage plan — adapt, don't reinvent

| File | Current role | v2 disposition |
|---|---|---|
| `knowledge3d/tools/knowledge_proceduralizer.py` | 1640 LOC batch pipeline; cloud model routing; `PROCEDURALIZER_BUNDLE_JSON_SCHEMA`; request/receipt/bundle chain | **Keep core.** Extract orchestration into WINE + Stargate sides. Python chunking, domain inference, retry envelope stay host-side (legit Region 7). |
| `knowledge3d/ingestion/proceduralizer_wine.py` | 359 LOC WINE bridge; `ProceduralizerRequest`/`Receipt`; Ollama transport; retry + failure detection | **Move to `knowledge3d/tablet/wine/proceduralize_wine.py`.** Current location is the drift Daniel flagged: a file named `*_wine.py` lives outside `tablet/wine/`. Move + rename + conform to `build_*_task` / `*_envelope` factory pattern that other WINE modules use. |
| `knowledge3d/tablet/wine/math_wine.py` | Template pattern | Copy the `build_math_task` / `math_envelope` factory shape when writing the proceduralize wine module. |
| `knowledge3d/bridge/headless_tablet.py` | `TabletIngest` static factory + `TabletEnvelope` dataclass | Add `TabletIngest.proceduralize_task(...)` factory. No schema change — new `surface_kind = "PROCEDURALIZE"` constant. |
| `knowledge3d/ingestion/stargate_feeder.py` (or equivalent) | Region 7 ingress | Extend to consume `ProceduralizerReceipt.parsed_bundle` → `MeaningCentricStar` writer. |
| `Old_Attempts/repo_archive/knowledge3d/ingestion/pdf_augmenter.py` | 248 LOC simpler template | Reference only, no import. |

Net: **no new tool code**. Move one file, add one factory method, extend
one feeder. The new work is the spec for (a) the `TabletEnvelope`
contract for proceduralize, (b) bundle-to-`MeaningCentricStar` mapping,
(c) gate validation, (d) registry opcode reservation.

## 4. New: `tablet/wine/proceduralize_wine.py`

Factory shape (mirrors `math_wine.py` and `game2d_wine.py`):

```python
PROCEDURALIZE_ROUTE_GALAXIES = (
    "Math", "Reality", "Word", "Grammar", "Number", "Character",
)

SURFACE_KIND_PROCEDURALIZE = "PROCEDURALIZE"

def build_proceduralize_route(
    *, specialist: str = "auto",
    domain_hint: str | None = None,
    galaxies: Sequence[str] | None = None,
    route_policy: str = ROUTE_POLICY_ALL_LIVE_GALAXIES,
) -> dict[str, Any]: ...

def build_proceduralize_task(
    *, task_id: str,
    source_kind: str,            # "benchmark_row" | "pdf_chunk" | "raw_text"
    content: str,
    domain_hint: str | None = None,
    context_chunks: Sequence[str] = (),
    supervision_answer: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]: ...

def proceduralize_envelope(
    *, task_id: str,
    source_kind: str,
    content: str,
    domain_hint: str | None = None,
    context_chunks: Sequence[str] = (),
    supervision_answer: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    return TabletIngest.proceduralize_task(
        task_id=task_id,
        source_kind=source_kind,
        content=content,
        domain_hint=domain_hint,
        context_chunks=tuple(context_chunks),
        supervision_answer=supervision_answer,
        metadata=dict(metadata or {}),
    )
```

Consumption side (the bridge, which moved from `ingestion/`):

```python
class ProceduralizerWineBridge:
    """Translator: TabletEnvelope(PROCEDURALIZE) -> ProceduralizerReceipt."""

    def translate(self, envelope: TabletEnvelope) -> ProceduralizerRequest:
        assert envelope.surface_kind == SURFACE_KIND_PROCEDURALIZE
        task = envelope.task or {}
        return ProceduralizerRequest(
            source_id=envelope.task_id,
            source_kind=task["source_kind"],
            content=task["content"],
            context_chunks=tuple(task.get("context_chunks") or ()),
            domain_hint=task.get("domain_hint"),
            supervision_answer=task.get("supervision_answer"),
            mode="standard",
            quality_profile="quality",
        )

    def invoke(self, request: ProceduralizerRequest) -> ProceduralizerReceipt:
        """Existing path — keep as-is in proceduralizer_wine (transport)."""
        ...
```

The existing `ProceduralizerRequest` / `ProceduralizerReceipt` shape
stays unchanged. v2 is wiring.

## 5. `TabletIngest.proceduralize_task`

Addition to `headless_tablet.py`. Symmetric with `game2d_task`,
`math_task`, `question_task`:

```python
@staticmethod
def proceduralize_task(
    *, task_id: str,
    source_kind: str,
    content: str,
    domain_hint: str | None = None,
    context_chunks: Sequence[str] = (),
    supervision_answer: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TabletEnvelope:
    return TabletEnvelope(
        surface_kind=SURFACE_KIND_PROCEDURALIZE,
        task_id=str(task_id),
        query=f"Proceduralize ({source_kind}): {content[:120]}",
        specialist="auto",
        galaxies=list(PROCEDURALIZE_ROUTE_GALAXIES),
        route_policy=ROUTE_POLICY_ALL_LIVE_GALAXIES,
        task={
            "source_kind": source_kind,
            "content": content,
            "context_chunks": list(context_chunks),
            "domain_hint": domain_hint,
            "supervision_answer": supervision_answer,
        },
        metadata=dict(metadata or {}),
    )
```

## 6. Opcode registry — what to reserve

For the Layer-3 GrammarRule stars emitted by the proceduralizer, the
rule_rpn uses **slot tokens**. Registry is append-only per
[feedback_opcode_range_reservation_protocol](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_opcode_range_reservation_protocol.md).

Reserve in `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md §11`:

| Range | Owner spec | Purpose |
|---|---|---|
| `0x300–0x30F` | This spec | Math-template slot tokens: `SLOT_N0`, `SLOT_N1`, … up to 16 slots |
| `0x310–0x31F` | This spec | Slot binding ops: `SLOT_BIND`, `SLOT_COERCE_INT`, `SLOT_COERCE_FRAC`, `SLOT_RANGE` |

These replace v1's `ARG_X` placeholder idea with proper registered
opcodes. No collision — registry is currently unassigned in this band
(§11 blocks `0x2DB` and prior are active; `0x300+` is clear).

## 7. Gates (adapted from kimi_swarm design notes)

| Gate | Check |
|---|---|
| G1 | Bundle schema conforms to `PROCEDURALIZER_BUNDLE_JSON_SCHEMA` (already in live code; keep) |
| G2 | Every emitted star's `star_id` equals `ContentHash(meaning_rpn ‖ meaning_class ‖ domain)` |
| G3 | Every token in `meaning_rpn` / `rule_rpn` is a registered opcode, a slot token (§6), or a literal |
| G4 | RPN stack simulation: final depth 1, no underflow |
| G5 | If `supervision_answer` present: symbolic slot-bound eval matches |
| G6 | `rpn_opcodes.lower_to_ptx(meaning_rpn)` dry-run succeeds |
| G7 | Matryoshka `embeddings` have all four tiers (64/128/512/2048) |

Gates run host-side during the Region 7 write. Failure returns a
`ProceduralizerReceipt` with `failure_code` and is subject to the
existing retry envelope (3 transient + 2 plan-limit waves — already in
`proceduralizer_wine.py`).

## 8. Paper-aligned scope (non-negotiable)

From the paper-scope sub-agent (which read
`TEMP/CLAUDE_PAPER_A_SKELETON_DRAFT_04.19.2026.md`):

- Paper deadline **2026-11-08** (202 days).
- Rubric target **≥4.3/5.0**; accuracy axis 3→4.5 is where the
  proceduralizer contributes.
- Scope v2 ships against **only** GSM8K + Math Competitions. Grammar
  103k enrichment and ARC lift are out of scope for this spec.
- Expected lift: GSM8K 10% → ≥50% (paper needs ≥5/10), Math 0% →
  ≥30% (paper needs ≥3/10). Validate on same 10-q harnesses already
  in place.

Out-of-scope (explicit, so Codex doesn't drift):
- MMLU enrichment (indirect benefit, defer)
- LHE Grammar rule_strength backfill (separate spec)
- New MCP server for opcode registry (nice-to-have, defer — LLM can be
  given the opcode list inline in the prompt)
- "Query knowledgeverse from outside" tool — Daniel said "at some
  point", not now

## 9. Success criteria

Ship green when all four hold:

1. `tablet/wine/proceduralize_wine.py` exists with the three factory
   functions; `ingestion/proceduralizer_wine.py` is gone (moved).
   `TabletIngest.proceduralize_task` exists.
2. External caller (benchmark harness) can build a PROCEDURALIZE
   envelope, pass it to `ProceduralizerWineBridge.translate().invoke()`,
   and receive a `ProceduralizerReceipt` whose `parsed_bundle` emits
   `MeaningCentricStar` entries that pass G1–G7.
3. Running the enrichment pass over
   `/K3D/Knowledge3D.local/galaxies/proceduralized_gsm8k_train_10.jsonl`
   produces `proceduralized_gsm8k_train_10.meaning_stars.jsonl`
   conforming to `§2.1`. GSM8K-10 accuracy rises to **≥5/10** on the
   same harness (`benchmarks/gsm8k.py --max-tasks 10`).
4. Math-10 accuracy rises to **≥3/10** on
   `benchmarks/math_competitions.py --max-tasks 10`.

Non-success: if (3) or (4) misses, the receipt/bundle schema is the
most likely culprit — debug the star_id hash and opcode-token
coverage first before re-prompting.

## 10. Handoff

**Claude stays architecture.** This spec + the kimi_swarm design notes
are the handoff. When Codex returns from limit, pickup sequence:

1. Read this spec + §2.1 of the star schema spec + live
   `proceduralizer_wine.py` + `knowledge_proceduralizer.py`.
2. Move `proceduralizer_wine.py` into `tablet/wine/`, rename to
   `proceduralize_wine.py`, add factory shape.
3. Add `TabletIngest.proceduralize_task` + `SURFACE_KIND_PROCEDURALIZE`.
4. Reserve `0x300–0x31F` in registry §11 (same commit as step 3).
5. Add G1–G7 gates in Region 7 writer.
6. Run the two 10-q harnesses and commit deltas.

Pilot guidance (if Codex stays locked and Daniel authorizes per
[project_claude_runs_during_codex_limit](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/project_claude_runs_during_codex_limit.md)):
steps 2-4 are pure plumbing and WINE conformance — safe pilot scope.
Steps 5-6 are code + measurement, real coding territory.

---

**What v2 is NOT:**
- Not a new internal pipeline (v1 was wrong on this)
- Not a new entry schema (MeaningCentricStar already exists)
- Not a new MCP server (defer opcode-MCP)
- Not a Grammar-103k enrichment (separate work)
- Not an ARC-path lift (proceduralizer doesn't lift ARC)

**What v2 IS:**
A one-file move + one factory add + one registry reservation, wiring
an already-existing LLM bridge through the WINE pattern every other
surface uses, targeted at the two benchmarks that move the paper's
accuracy rubric.
