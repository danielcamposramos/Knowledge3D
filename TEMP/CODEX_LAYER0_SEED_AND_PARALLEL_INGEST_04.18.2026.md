# Codex Runbook — Layer 0 Seed + Parallel Ingestion Build-Out

**Date:** 2026-04-18
**Role:** Codex — IMPLEMENT + RUN. Source edits authorized for this task.
**Authoritative spec:**
`TEMP/CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md` (read it first).
**Agent prompts:** `prompts/ingestion/*.md` (do not modify; reference only).

---

## Context

Daniel's directive (2026-04-18): the ingestion path is not producing the
1.5 GB VRAM load we expect at boot. Root cause is three-fold: Layer 0
drawing primitives are stranded in `galaxy_pending/`, their IDs are
non-canonical (`PRIM_LINE` vs `drawing_primitive_line`), and the
canonical Qdrant registry is under-seeded. Upper layers therefore have
no anchor to symlink to, and stars ship without actionable RPN.

This runbook fixes that in **eight** ordered gates (0 through 7) plus a
smoke run. Gate 0 is new (2026-04-18 turn 2): opcode reservation + Python-
drift pre-flight. Gates 6–7 are new (virtual-page PoC + NAS corpus sweep).

---

## Daniel's 2026-04-18 locked decisions (authoritative)

Per spec §12:

1. `data/ingest_proposals/**` is **git-tracked** (deterministic PoC).
2. Corpus root = `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/`.
   **Process locally, NEVER copy binaries into the repo.** Manifest records
   NAS paths + SHA-256 + ingestion state.
3. `data/ocr_cache/<pdf_sha>/<page>.json` is **git-tracked**. No re-paying
   the cloud-vision bill on re-ingestion.

---

## Gate-Ordered Tasks

Each gate must be green before the next. Do not parallelize across
gates. You MAY parallelize within a gate's subtasks where clearly
independent.

---

### Gate 0 — Opcode Reservation + Python-Drift Pre-Flight (BLOCKS EVERYTHING)

**0A. Reserve VIRTUAL_PAGE opcode range in the registry**

Amend `docs/vocabulary/RPN_DOMAIN_OPCODE_REGISTRY.md` §11 with one new
row:

```
| 0x1D0–0x1FF | RESERVED | VIRTUAL_PAGE_* | 2026-04-18 | Ingestion-path
  graph-grammar RPN for virtual pages (see spec
  TEMP/CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md §13). Sub-families:
  PAGE_* / FRAME_* / TABLE_* / PARAGRAPH_* / LINE_* / RUN_* / *_EMIT* /
  LAYOUT_* / STYLE_* / BIDI_* / SCRIPT_* / HYPHEN_*. Specific numbers
  NOT assigned in this reservation — only the range. |
```

Per `memory/feedback_opcode_range_reservation_protocol.md`: the range must
appear in the registry BEFORE any implementation lane references it.
Commit this amendment alone, push the commit, verify the row is present.

**0B. Python-drift grep (must return zero)**

```bash
# Hot-path modules: knowledgeverse + cranium only.
grep -RnE "if .*\.surface_kind ==|if .*\.kind ==|route_by_kind|dispatch_by_kind|galaxy_switch|mode_switch" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
    | tee /tmp/drift_grep.log
test ! -s /tmp/drift_grep.log && echo GATE_0B_OK || echo GATE_0B_FAIL
```

Any hit = drift; hand back to Claude with the grep output. Do not attempt
to "clean up" — triage decides whether the hit is legacy-scheduled-for-
removal or genuinely load-bearing.

**0C. Hot-path library bans (grep)**

```bash
# Modules imported by knowledgeverse.boot() MUST NOT import these:
grep -RnE "^\s*import (numpy|cupy|scipy|sympy|torch)|^\s*from (numpy|cupy|scipy|sympy|torch) import" \
    knowledge3d/cranium/ knowledge3d/knowledgeverse/ \
    | tee /tmp/banned_libs.log
test ! -s /tmp/banned_libs.log && echo GATE_0C_OK || echo GATE_0C_FAIL
```

**Gate-0 check:** all three subsections green. If any fail, fix the
violation in its own commit before proceeding. Do not mix Gate-0 fixes
with Gate-1 work.

---

### Gate 1 — Canonical Layer-0 Seed (BLOCKS EVERYTHING BELOW)

**1A. Fix `knowledge3d/ingestion/atomic/drawing_grammar_builder.py`**

Current defect (confirmed 2026-04-18): emits IDs like `PRIM_LINE`,
`PRIM_CIRCLE`, etc. Required: use `canonical_drawing_primitive_id(name)`
from `knowledge3d/ingestion/canonical_lookup.py` to mint
`drawing_primitive_line`, `drawing_primitive_circle`, … for all 7
primitives.

Changes:
1. Import `canonical_drawing_primitive_id` and `CanonicalLookup` at top.
2. For every primitive dict, replace `"id": "PRIM_LINE"` with
   `"id": canonical_drawing_primitive_id("line")`. Same for
   arc/quad/cubic/circle/rect/tri.
3. At end of build, instantiate `CanonicalLookup()` and call
   `lookup.register(kind="drawing_primitive", key=name, star_id=sid,
   text=<short name>, doc=<visual_rpn>, metadata=<attrs>)` for each.
4. Write primitives JSONL to BOTH the `--output` path (legacy) AND the
   live-House primitives path. Coordinate exact live path:
   - Query live loader: `grep -rn "drawing_primitive" knowledge3d/knowledgeverse/ | head`
     to find the loader's expected location.
   - Default assumption: `data/house/drawing_primitives.jsonl` — verify
     against the loader before shipping.

**1B. Extend `scripts/ingest_canonical_to_qdrant.py`**

Currently seeds 3 drawing primitives. Required: all 7, plus extended
meaning_class and all 12 symlink_kind field paths.

Changes:
1. In the drawing-primitive seed loop, iterate all 7:
   `["line", "arc", "quad", "cubic", "circle", "rect", "tri"]`.
2. Extend `meaning_class` seed to the full 11-item vocabulary:
   `["drawing", "glyph", "word", "number", "grammar", "reality",
   "math", "audio", "object_3d", "tool", "game_2d"]`.
3. Extend `symlink_kind` seed to all 12 field paths from
   `docs/vocabulary/CANONICAL_REGISTRY_SPECIFICATION.md` §7:
   `["taxonomy_refs", "meta_refs", "grammar_refs", "reality_refs",
   "visual_refs", "audio_refs", "char_refs", "component_refs",
   "composite_of", "surface_forms_word_ref", "glyph_refs", "rpn_refs"]`.

**1C. Run Gate 1**

```bash
bash scripts/k3d_env.sh -e k3d-cranium python scripts/ingest_canonical_to_qdrant.py
bash scripts/k3d_env.sh -e k3d-cranium python -m knowledge3d.ingestion.atomic.drawing_grammar_builder \
    --output data/house/drawing_primitives.jsonl
```

**Gate-1 check (MUST pass before Gate 2):**
```bash
bash scripts/k3d_env.sh -e k3d-cranium python -c "
from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
lookup = CanonicalLookup()
for k in ['line','arc','quad','cubic','circle','rect','tri']:
    sid = lookup.find_star_id('drawing_primitive', k)
    assert sid == f'drawing_primitive_{k}', (k, sid)
print('GATE_1_OK')
"
```

Expected output: `GATE_1_OK`. Any assertion failure: stop, triage, hand
back to Claude with stderr tail.

---

### Gate 2 — Canonical Registry Kinds Complete

**2A. Add seed rows for Phase 7.A.1 kinds**

Extend `scripts/ingest_canonical_to_qdrant.py` with vocabulary seeds
(not per-language instances) for:
- `letter_star` — seed at least one exemplar per script family
  (Latin, Cyrillic, Greek, CJK, Arabic, Devanagari) — ~6 exemplars.
- `font_glyph` — exemplars for at least 3 core fonts
  (e.g., `dejavu-sans`, `liberation-serif`, `noto-sans-cjk`).
- `word_lemma` — at least one exemplar per target language
  (`por`, `eng`, `esp`, `fra`, `deu`, `jpn`, `zho`, `ara`, `rus`,
  `hin`, `kor`).
- `math_symbol` — core set: `plus, minus, times, div, eq, lt, gt,
  frac, sum, prod, int, sqrt, binom, infty, pi, alpha, beta, gamma`.
- `grammar_rule` — exemplars: `por_article_agreement`,
  `eng_subject_verb_agreement`, `jpn_particle_wa`.
- `rpn_template` — exemplars: `reality_kinematics_v_linear`,
  `math_quadratic_formula`, `grammar_substitute_simple`.

**2B. Document `key` format for each kind**

Add docstrings to `knowledge3d/ingestion/canonical_lookup.py` describing
the exact `key` format per kind. Reference
`TEMP/CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md` §6.1 table.

**Gate-2 check:**
```bash
bash scripts/k3d_env.sh -e k3d-cranium python -c "
from knowledge3d.ingestion.canonical_lookup import CanonicalLookup
lookup = CanonicalLookup()
kinds = ['drawing_primitive','meaning_class','symlink_kind',
         'letter_star','font_glyph','word_lemma','math_symbol',
         'grammar_rule','rpn_template']
for k in kinds:
    # expect at least one seed row per kind
    pass  # implement as registry scroll-count > 0 per kind
print('GATE_2_OK')
"
```

Implementation hint: `CanonicalLookup` needs a `count_by_kind(kind)`
helper or use Qdrant `scroll` with filter on `kind`. Pick the cheaper
option.

---

### Gate 3 — Parallel Ingestion Infrastructure

**3A. Extend `knowledge3d/ingestion/corpus_manifest.py` with locking**

Per spec §4.2. Add fields to `CorpusEntry`:
```python
locked_by: str | None = None
locked_at: str | None = None
lock_kind: str | None = None  # "extract" | "stitch" | "ocr"
```

Add methods to `CorpusManifest`:
- `claim_next_available(agent_id: str, lock_kind: str) -> CorpusEntry | None`
  - Atomic via `fcntl.flock(LOCK_EX)` on the manifest JSON file.
  - Picks topologically-first entry with `ingested=False` AND
    `locked_by is None` AND all dependencies have `ingested=True`.
  - Sets `locked_by`, `locked_at=datetime.utcnow().isoformat()`,
    `lock_kind`.
  - Persists manifest atomically (write to tmp, rename).
- `release(agent_id: str, entry_id: str, success: bool) -> None`
  - Clears lock fields. If `success`, sets `ingested=True`.
  - Persists atomically.
- `expire_stale_locks(max_age_seconds: int = 1800) -> list[str]`
  - Scans for locks older than threshold, clears them, returns list of
    reclaimed entry_ids for logging.

Write tests at `tests/ingestion/test_corpus_manifest_locks.py`:
- Single-process: claim, release, re-claim works.
- Two-process: spawn two subprocesses, each tries to claim; verify
  exactly one succeeds per entry, no double-lock.
- Stale-lock expiration: set `locked_at` to 2h ago, call expire,
  verify reclaim.

**3B. Build `scripts/ingest_parallel_agents.py` orchestrator**

Responsibilities:
1. Load manifest.
2. Spawn Agent A workers (default 2) and Agent B workers (default 2)
   via `multiprocessing.Process` (NOT thread pool — PIL/requests would
   contend on GIL).
3. Agent A loop: claim(extract) → read file → call OCR sidecar if
   needed → call Ollama cloud (kimi-k2.5:cloud) with
   `prompts/ingestion/SYSTEM_PROMPT_INGEST_AGENT.md` + task template
   filled with `agent_role=extractor_A` → receive proposal JSON →
   validate against schema → write to `data/ingest_proposals/` →
   release(success=True).
4. Agent B loop: pull proposal paths from a multiprocessing.Queue the
   Agent A workers populate → call Ollama cloud with
   `SYSTEM_PROMPT_INGEST_AGENT.md` + task template filled with
   `agent_role=stitcher_B` → execute returned plan via CanonicalLookup
   + symlink_helpers → write to live House → release(success=True).
5. On failure in any worker: release(success=False), log, continue with
   next entry (do NOT crash the orchestrator — one bad file should not
   halt the pipeline).
6. Periodic `expire_stale_locks()` in a watchdog thread.

CLI:
```bash
scripts/ingest_parallel_agents.py \
    --manifest data/corpus_manifest.json \
    --extractors 2 --stitchers 2 \
    --max-entries 10 \
    --live-house-root data/house/
```

**3C. Build OCR sidecar `scripts/ocr_sidecar_service.py`**

Minimal FastAPI or plain-socket service:
- Endpoint: `POST /ocr` with `{page_id, image_base64, dpi}` body.
- Calls `mcp__ollama-specialists__ask_cloud` with
  `SYSTEM_PROMPT_OCR_AGENT.md` as system and the image as user content
  (base64 inline per Ollama vision protocol).
- Cache: `data/ocr_cache/<pdf_sha>/<page>.json`. Check cache before
  calling cloud.
- Returns OCR JSON per the contract in `SYSTEM_PROMPT_OCR_AGENT.md`.

Client helper `knowledge3d/ingestion/ocr_client.py`:
```python
def process(image_bytes: bytes, page_id: str, dpi: int = 200) -> dict:
    ...
```

**Gate-3 check:**
Run the locking tests + a 2-file smoke:
```bash
bash scripts/k3d_env.sh -e k3d-cranium pytest tests/ingestion/test_corpus_manifest_locks.py -v
bash scripts/k3d_env.sh -e k3d-cranium python scripts/ingest_parallel_agents.py \
    --manifest data/corpus_manifest.json --extractors 2 --stitchers 2 --max-entries 2 \
    --live-house-root data/house/
```
Must show 2 files processed, no lock violations.

---

### Gate 4 — Symlink Backfill for Existing Stars

**4A. Audit coverage**

Build `scripts/audit_symlink_coverage.py` that walks every
`data/house/**/*.jsonl`, loads each star, and reports:
- Total stars by layer.
- % with at least one resolved upward symlink terminating at a Layer-0
  canonical primitive.
- List of star_ids missing symlinks (first 100 per layer for triage).

Hand the JSON audit report to Claude for review before Gate 4B.

**4B. Stitch missing symlinks**

Run the parallel agent pipeline in "stitch-only" mode over the
`needs_symlinks` queue produced by 4A. Agent A drafts proposals
containing ONLY new symlinks (no new stars). Agent B commits.

CLI:
```bash
bash scripts/k3d_env.sh -e k3d-cranium python scripts/ingest_parallel_agents.py \
    --mode stitch-only \
    --input-queue data/audit/needs_symlinks.jsonl \
    --extractors 2 --stitchers 2 \
    --live-house-root data/house/
```

**Gate-4 check:** re-run `audit_symlink_coverage.py` — coverage must be
≥ 95% across all layers.

---

### Gate 5 — VRAM-Load Regression Check

Daniel's symptom was "modest load" vs prior 1.5 GB peak. Measure it:

```bash
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json, subprocess, time
from knowledge3d.knowledgeverse.knowledgeverse import Knowledgeverse
kv = Knowledgeverse(boot_mode='full')
kv.boot()
# Read nvidia-smi memory.used for our PID
out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                '--format=csv,noheader,nounits']).decode().strip()
print(f'VRAM_MB={out}')
"
```

**Gate-5 check:** `VRAM_MB >= 1024`. If less, Gate 3 or Gate 4 did not
achieve the symlink resolution density the loader needs to materialize
the full four-layer cascade — hand back to Claude with a gap analysis.

---

### Gate 6 — Virtual Page PoC (Daniel's Layer-3 Extension)

Per spec §13. Implement the minimum viable virtual-page compiler +
proof-of-concept artifact.

**6A. VIRTUAL_PAGE opcode dispatcher stubs**

For every sub-family reserved in Gate 0A, add a dispatcher stub in the
PTX runtime. Stubs are "log and halt" placeholders — they validate the
opcode decoder reaches the slot; they do not execute layout yet. This
keeps the hot path honest: an opcode that has no dispatcher HARD FAILS
at runtime rather than silently skipping.

Location: `knowledge3d/cranium/ptx_runtime/rpn_opcodes.py` —
EXTEND ONLY, do not renumber existing entries. Append VIRTUAL_PAGE_*
entries pointing to the stubs.

**6B. Virtual-page compiler in Agent B**

Extend `scripts/ingest_parallel_agents.py` Agent B stitcher with a
`virtual_page_compile(ocr_json, resolved_terminals) -> rpn_bytes`
function following the pipeline in spec §13.5:
1. Classify OCR blocks → graph nodes.
2. Build contains + reading_order edges.
3. Cluster runs → StyleScope stars.
4. Resolve terminals via CanonicalLookup (fail hard on miss).
5. Walk graph in reading order → emit opcode stream.
6. Emit `virtual_page_<doc_sha>_<page_n>` star.

**6C. PoC smoke: 1-page document**

Generate a synthetic-but-realistic 1-page PDF matching the PoC target
in spec §13.6 (heading, 2 paragraphs, 3×3 table, figure+caption).
Commit to `TEMP/virtual_page_poc/source/poc.pdf`.

Run:
```bash
bash scripts/k3d_env.sh -e k3d-cranium python scripts/ingest_parallel_agents.py \
    --manifest-single TEMP/virtual_page_poc/source/poc.pdf \
    --extractors 1 --stitchers 1 \
    --live-house-root data/house/ \
    --dump-poc TEMP/virtual_page_poc/
```

**Gate-6 check:**
```bash
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json, pathlib
poc = pathlib.Path('TEMP/virtual_page_poc')
star = json.loads((poc / 'virtual_page.json').read_text())
assert 'meaning_rpn' in star
assert len(star['page_graph_refs']) >= 30, len(star['page_graph_refs'])
assert len(star['visual_refs']) >= 8
assert len(star.get('matryoshka_embeddings', {})) == 4
# Determinism: re-run must produce byte-identical RPN
import hashlib
rpn1 = hashlib.sha256(star['meaning_rpn'].encode()).hexdigest()
print('GATE_6_OK', 'rpn_sha256=' + rpn1)
"
```

Commit PoC artifacts (star JSON, graph-node JSONs, RPN disassembly,
OCR cache) to `TEMP/virtual_page_poc/` — these are tracked per Daniel's
decision 1.

---

### Gate 7 — NAS Corpus Manifest + Initial Sweep

Per Daniel's decision 2. Build the corpus manifest from the NAS root,
run a 10-entry hand-picked sweep, verify the pipeline holds at scale.

**7A. Build NAS manifest**

`scripts/build_nas_manifest.py`:
- Walk `/mnt/arquivos/0 ChatGPTs/DataBase/EchoSystems Default Libraries/`
- Filter: include `.pdf`, `.md`, `.txt`, `.json`, `.epub`. Exclude:
  system files, empty files, `.DS_Store`, `Thumbs.db`, `*.tmp`,
  any path segment starting with `.` (hidden).
- Compute SHA-256 per file (stream, don't load full).
- Emit `data/corpus_manifest_nas.json` with `CorpusEntry` rows,
  `path` = absolute NAS path, `ingested=False`, all lock fields None.
- Topological dependency inference: foundational (`Tier 1`) = anything
  under a folder named `foundational|primitives|alphabet|numbers`;
  `Tier 2` = anything under a subject folder; `Tier 3` = remainder.
- **Do NOT copy file content into the repo.** Manifest stores paths
  + hashes only.

**7B. Hand-picked 10-entry smoke sweep**

Claude picks 10 entries for the first sweep (mix of foundational
math, language, and one scanned-PDF for OCR path exercise). List
written to `TEMP/initial_sweep_10.txt` as one path per line.

Run:
```bash
bash scripts/k3d_env.sh -e k3d-cranium python scripts/ingest_parallel_agents.py \
    --manifest data/corpus_manifest_nas.json \
    --filter-file TEMP/initial_sweep_10.txt \
    --extractors 2 --stitchers 2 \
    --live-house-root data/house/
```

**Gate-7 check:**
```bash
# All 10 entries ingested, no lock violations, no canonical_lookup_miss.
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
m = json.load(open('data/corpus_manifest_nas.json'))
filtered = [e for e in m['entries'] if e['path'] in open('TEMP/initial_sweep_10.txt').read().split()]
assert all(e['ingested'] for e in filtered), [e['path'] for e in filtered if not e['ingested']]
assert all(e['locked_by'] is None for e in filtered), 'residual locks'
print('GATE_7_OK', len(filtered))
"
```

---

### Gate 8 — Benchmark Non-Regression

ARC 10/10, Math 20/20, LHE 10/10 must stay pinned:

```bash
bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.arc_agi_2 import ARCAGI2Benchmark
b = ARCAGI2Benchmark(dataset_version='arc_agi_2', max_tasks=10)
s = b.run_benchmark(use_enriched=True)
assert s['correct'] == 10, s
print('ARC_OK', s['correct'])
"

bash scripts/k3d_env.sh -e k3d-cranium python -c "
import json
from benchmarks.last_humanity_exam import LastHumanityExamBenchmark
b = LastHumanityExamBenchmark(max_questions=10)
s = b.run_benchmark(use_enriched=True)
print('LHE', s['correct'], '/', s['total_questions'])
"

bash scripts/k3d_env.sh -e k3d-cranium python -c "
from benchmarks.math_competitions import MathCompetitionsBenchmark
b = MathCompetitionsBenchmark(dataset_path='/K3D/K3D_llama_cpp/datasets/math',
                              max_questions=20)
s = b.run_benchmark(use_enriched=True)
print('MATH', s.get('correct'), '/', s.get('total_questions'))
"
```

If any regresses: Gate 4 over-stitched or Gate 7 introduced a bad
symlink. Hand back to Claude with failing task IDs + diff vs prior.
**Math benchmark is the load-bearing one** — with sovereign math PTX
cores + actionable RPN per math star, math should not only hold but
improve.

---

## Commit Policy

Commit after each green gate. One commit per gate:

```bash
git add <files touched this gate>
git commit -m "ingest(gate{N}): <one-line summary>"
```

Example gate-1 commit:
```
ingest(gate1): canonical Layer-0 seed — fix drawing_grammar_builder IDs, seed all 7 primitives
```

No push. No amend. No `--no-verify`.

---

## Forbidden

- Silent canonical-ID synthesis. If `CanonicalLookup.find_star_id()`
  misses and the referenced entity is NOT in the current proposal's
  `needs_new_star` list: raise. Do not invent.
- Writing to live House from Agent A (extractor). Only Agent B writes.
- Using pytesseract / tesseract / easyocr anywhere in the new path.
  Vision model only.
- Touching `knowledge3d/cranium/**` or any PTX code.
- Adding 384-dim or 768-dim embedding tiers to stars. Matryoshka
  {64,128,512,2048} only.
- Re-numbering existing opcodes or renaming canonical `key` values.
- `--no-verify`, `--amend`, `push --force`.

---

## Reporting

### On PASS (each gate) — one line to stdout:
```
GATE_N_OK: <1-line summary>
```

### On FAIL — exact format, no prose:
```
gate: N
command: <exact command>
exit_code: <number>
stderr_tail_80:
<last 80 lines>
commit_head: <git rev-parse HEAD>
```

Hand back to Claude for triage. Do not attempt to work around.

---

## Post-Compaction Reload Protocol

1. Re-read `TEMP/CLAUDE_INGESTION_SYMLINK_REWIRE_04.18.2026.md`.
2. Re-read this runbook.
3. Re-read `prompts/ingestion/SYSTEM_PROMPT_INGEST_AGENT.md` and
   `prompts/ingestion/SYSTEM_PROMPT_OCR_AGENT.md`.
4. `git log --oneline -10` — identify which gate is next by the last
   `ingest(gateN)` commit.
5. `git status --short` — expect clean working tree between gates.

---

## One-Sentence Summary

Reserve VIRTUAL_PAGE opcodes + pre-flight the drift grep, seed canonical
Layer-0 IDs, wire the two-agent parallel ingest + OCR sidecar, backfill
symlinks, ship a virtual-page PoC, run a 10-entry NAS sweep, and keep
ARC/Math/LHE pinned — all without touching the sovereign hot path.
