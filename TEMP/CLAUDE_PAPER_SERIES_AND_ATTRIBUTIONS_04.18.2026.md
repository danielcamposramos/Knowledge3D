# Paper Series & Attributions Re-Evaluation — 2026-04-18

**Supersedes:** `TEMP/CLAUDE_PAPER_MVP_PLAN_04.18.2026.md` (v2)
**Trigger:** Daniel — *"deep investigation on our attributions file, consider the ternary logic integration and the form to meaning logic, as well as the layered (inspired by OSI) architecture, the hyper-modular architecture idea and 'gravity cohered by meaning' concept of the knowledgeverse — this by itself is a paper subject — re evaluate the plan"*

**One-line verdict:** The ARC Prize 2026 Paper Track is one paper. K3D carries at least **six** paper subjects. Plan the series now, protect Paper A's novelty score by citing (not squeezing in) the other five, and fix `ATTRIBUTIONS.md` before any preprint leaves the repo.

---

## Part A — Attribution Audit

### A.0 Restoration event (2026-04-18 evening)

`ATTRIBUTIONS.md` was discovered to have been destroyed by commit `c742e2dd` ("Land staged ARC3, CAS, physics, and benchmark updates"): 2,291 lines removed, leaving only an 87-line Transfer Yard stub. The pre-destruction 2,310-line file has been restored from `c742e2dd~1` and the Transfer Yard section merged in as §1.4, yielding the current **2,400-line** authoritative file. All analysis below is against this restored file.

### A.1 Current state (post-restoration)

`ATTRIBUTIONS.md` (2,400 lines) documents most of K3D's foundational prior art. The table below reconciles the A.3 list from v1 of this spec against the restored file.

### A.2 Novelty-rubric risk

The ARC Prize 2026 paper rubric scores **Novelty (0-5)** and (implicitly) **Completeness**. A reviewer's first check is: *what did they build, what did they borrow?* Without a clean attribution boundary, every novelty claim reads as possibly lifted. Target: **ATTRIBUTIONS.md is complete and cross-linked from Paper A §3 Prior Work BEFORE any submission is drafted.**

### A.3 External prior art — reconciliation after restoration

Status key: ✅ present · ⚠ present but citation needs tightening · ❌ missing, must be added.

| # | Entity | Status in restored file | Location / action |
|---|---|---|---|
| 1 | Tiny Recursive Model (TRM) — Jolicoeur-Martineau et al. | ✅ | §1.2.1 |
| 2 | Boris Knyazev — graph reasoning / optimization lineage | ✅ | §1.2.2 |
| 3 | ARC-AGI (Chollet) | ✅ | §1.3 |
| 4 | Transfer Yard (Abu El Haijaa et al. 2024) | ✅ | §1.4 (merged this session) |
| 5 | Morton codes (Morton 1966 IBM) | ✅ | §2.3 (line 760) |
| 6 | Setun / Balanced Ternary (Brusentsov) | ✅ | §3.1 + §3.5 (duplicate — merge in cleanup) |
| 7 | Matryoshka Representation Learning | ⚠ | §3.4 credits Qwen but NOT the Kusupati et al. NeurIPS 2022 paper by name — **must add primary citation** |
| 8 | Defeasible Logic (Nute) / SPINdle (Lam & Governatori) / Christoph Dorn contribution | ✅ | §4.4 |
| 9 | Milton Ponson — mathematical grounding | ✅ | §4.1 |
| 10 | PM-KR Community Group + early ingressors | ✅ | §4.3 (6 sub-entries) |
| 11 | Apollo 11 guidance computer — modular engineering | ✅ | §4.2 |
| 12 | CUDA/PTX, Ollama, PyMuPDF, Tesseract, Wine | ✅ | §5.1–§5.5 |
| 13 | DeepSeek-OCR | ✅ | §1.1 |
| 14 | Tesla 3-6-9 vortex math | ✅ | §3.2 |
| 15 | .kkrieger / procedural generation | ✅ | §2.4 |
| 16 | **BitNet b1.58** (Microsoft 2024, arXiv 2402.17764) | ❌ | **Add §3.6** — 5 trits/byte packing for weight matrices; cites our 20× compression / 82% energy claims |
| 17 | **Method of Loci** (Yates 1966, *Art of Memory*; Simonides of Ceos) | ❌ | **Add §3.7** — House = Memory Palace paradigm origin |
| 18 | **A\*** (Hart, Nilsson & Raphael 1968) | ❌ | **Add §2.5** — `led_astar_*.ptx` |
| 19 | **LoRA** (Hu et al. 2021, arXiv 2106.09685) | ❌ | **Add §3.8** — specialist adapters + shadow copy enhancement |
| 20 | **Transformer / SwiGLU** (Vaswani 2017; Shazeer 2020) | ❌ | **Add §3.9** — TRM internal MLP |
| 21 | **OSI Reference Model** (ISO/IEC 7498-1:1984) | ❌ | **Add §4.5** — layered-stack *inspiration*, explicitly NOT implementation |
| 22 | **RETE** (Forgy 1982) | ❌ | **Add §3.10** — `RETE_*` opcodes in 0xA0-0xF1 reasoning block |
| 23 | **Knuth TAOCP Vol. 2 §4.1** — optimal-radix argument for ternary | ❌ | **Add as subsection of §3.1** — "hardware imperative" theoretical anchor |
| 24 | **Kusupati et al. 2022** (Matryoshka primary) | ❌ | Tighten §3.4 — add arXiv 2205.13147 + authors |

### A.4 Structural fixes in the restored file (discovered during restoration)

These are pre-existing section-numbering collisions inherited from the 2,310-line original. Codex should clean these up in a follow-up PR (NOT in the restoration commit — keep restore diff minimal):

- Two `## 5.` sections: `## 5. Software & Tools` (L1387) AND `## 5. Datasets & Corpora` (L1492). Rename second to `## 5A. Datasets & Corpora` or promote to `## 5a.`.
- Two `## 6.` sections: `## 6. K3D's Novel Contributions` (L1695) AND `## 6. Universal Procedural Display Stack` (L1871). Renumber second to `## 6A.` or merge into `## 2a. Game Industry Techniques` sibling.
- Two `## 7.` sections: `## 7. Paper Preparation` (L1793) AND `## 7. Carbon Impact & Future-Proofing Philosophy` (L2046). Renumber second.
- Two `### 3.4` sections: `### 3.4 Thinking Tags / Chain-of-Thought` (L848) AND `### 3.4 Qwen-embedding: Matryoshka Representations` (L878). Renumber Qwen section to `### 3.6 Qwen/Matryoshka` (and slot BitNet/Method of Loci/LoRA/Transformer/RETE per A.3 above).
- Two `### 3.5` sections: `### 3.5 Multi-Modal Fusion` (L863) AND `### 3.5 Setun Computer: Balanced Ternary Logic` (L960). Setun §3.5 duplicates §3.1 — merge into §3.1, elevate unique content, delete the stale §3.5.

### A.4 Required attribution entries (K3D-internal coinages — credit OUR people)

| # | Term / concept | Credited to | Date / source |
|---|---|---|---|
| C1 | **Knowledge3D / K3D** | Daniel Campos Ramos | Project founding |
| C2 | **Knowledgeverse** | Daniel Campos Ramos | 2026-03-05 (term definition spec) |
| C3 | **Spatial General Intelligence (SGI)** | Daniel Campos Ramos (PM-KR CG coinage) | 2026-03-04; see `SGI_TERM_ORIGIN_PROOF.md` |
| C4 | **Superhuman General Intelligence (SHGI)** | Daniel Campos Ramos | 2026-03-05 |
| C5 | **Hyper-Modular Architecture** | Daniel Campos Ramos | 2026-02-20 |
| C6 | **Hyper-Parallel Processing** | Daniel Campos Ramos | 2026-03-16 |
| C7 | **TRM-as-Avatar paradigm** | Daniel Campos Ramos | — *(game-loop framing, House-Palace embodiment)* |
| C8 | **Meaning-Centric Star Schema** | Daniel Campos Ramos (with Christoph Dorn) | 2026-03 |
| C9 | **Semantic gravity cohered by meaning** | **Christoph Dorn** (ternary force formula F = T(s₁,s₂)·M(s₁)·M(s₂)/d²) | 2026-03; with Daniel and Claude |
| C10 | **Transfer Yard default + tier variants** | K3D team adaptation | 2026 |
| C11 | **Absolute Sovereignty Purge** | Daniel Campos Ramos (ruling) | 2026-04-18 |
| C12 | **PM-KR Community Group** | Chair Daniel Campos Ramos; Co-Chair Milton Ponson; contributors inc. Christoph Dorn | Institutional credit |

### A.5 Expansion plan (execution) — revised post-restoration

**Two-commit strategy** to keep the restoration diff legible:

**Commit 1 — Restoration (this session, already landed in working tree):**
- `c742e2dd~1:ATTRIBUTIONS.md` restored verbatim
- §1.4 Transfer Yard Algorithm merged in (was the only legitimate Codex addition)
- Net: 105 → 2,400 lines

**Commit 2 — Gap-fill (next session, Claude drafts section texts, Codex merges):**

1. Add the **8 missing external attributions** from A.3 rows 16-23 as new subsections, each ~20-40 lines matching the existing §1.3/§1.4 template (source, what-it-is, what-we-adapted, academic citation, credit line).
2. Tighten §3.4 Matryoshka to cite Kusupati et al. 2022 as primary (keep Qwen credit as secondary adaptation).
3. Add Knuth TAOCP Vol. 2 §4.1 optimal-radix citation inside §3.1 Setun to anchor the "hardware imperative" theoretical claim.
4. Add a new `## K3D-Internal Coinages` top-level section covering A.4 rows C1-C12 (Knowledge3D, Knowledgeverse, SGI, SHGI, Hyper-Modular, Hyper-Parallel, TRM-as-Avatar, Meaning-Centric Star, Semantic Gravity, Transfer Yard K3D adaptation, Absolute Sovereignty Purge, PM-KR CG).
5. Fix the section-numbering collisions listed in A.4 (five renames) — separate PR to keep the attribution content diff reviewable.

**Codex is executor-only here**; Claude writes all prose. This prevents a re-destruction incident.

---

## Part B — Paper Series Roadmap

### B.1 Why a series (not one paper)

Daniel's five concepts each pass the **standalone-paper test**:

1. **Absolute sovereignty + PTX-native reasoning** — Paper A (ARC Prize 2026)
2. **Form → Meaning 4-layer architecture + Dual Client Contract** — Paper D
3. **Layered cognitive stack (OSI-inspired, NSI closed loop)** — Paper F
4. **Hyper-Modular Architecture + Hyper-Parallel Processing** — Paper C
5. **Semantic gravity cohered by meaning** — Paper B *(Daniel-led, Christoph co-author — CORRECTED 2026-04-19)*
6. **Ternary logic as hardware imperative** — Paper E

Squeezing six subjects into one six-page Paper-Track submission **destroys** each subject's Theory and Completeness scores. Citing the other five as companion preprints **protects** Paper A's Novelty score and seeds the rest of the series.

### B.2 Series map

| Paper | Subject | Venue target | Lead | Status |
|---|---|---|---|---|
| **A** | Sovereign Substrate + TRM-as-Avatar on ARC-AGI-3 | ARC Prize 2026 Paper Track (Kaggle) — Nov 8 deadline | Daniel + Claude (drafting) | **Priority 1** |
| **B** | Semantic Gravity Cohered by Meaning — ternary force in working memory | arXiv cs.AI + PM-KR CG note | **Daniel Campos Ramos (first)** + Christoph Dorn (second, term coiner) — CORRECTED 2026-04-19 | After A submits; Christoph notified |
| **C** | Hyper-Modular + Hyper-Parallel: one unified paradigm for procedural substrates | arXiv cs.SE / cs.AI | Daniel | After A submits |
| **D** | Form → Meaning: a 4-layer representation bridging humans and AI | arXiv cs.AI + W3C CG note | Daniel + CG collaborators | After A submits |
| **E** | Ternary as the hardware imperative — Setun resurrected on NVIDIA | arXiv cs.AR / cs.AI | Daniel | After A submits; pairs well with BitNet conversation |
| **F** | Layered Sovereign Cognitive Stack — OSI inspiration, NSI closed loop | arXiv cs.AI | Daniel + Milton | After C and D |

### B.3 Authorship policy (proposed — needs Daniel sign-off)

- **Paper A:** Daniel Campos Ramos (first); Christoph Dorn (second, contributor); Milton Ponson (institutional, W3C CG).
- **Paper B (CORRECTED 2026-04-19):** **Daniel Campos Ramos as first author** — the *idea* and the *formula* `F = T(s₁,s₂) · M(s₁) · M(s₂) / d²` are his. **Christoph Dorn as second author** — he coined the *phrase* "semantic gravity cohered by meaning" after Daniel explained the idea and formula to him. See [`feedback_semantic_gravity_provenance_corrected.md`](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_semantic_gravity_provenance_corrected.md) and [`ATTRIBUTIONS.md §4.4.1`](../ATTRIBUTIONS.md). This reverses the earlier draft which had Christoph first.
- **Papers C-F:** Daniel first; co-authors slot in as subjects demand.

---

## Part C — Paper A (ARC Prize 2026) Adjusted Scope

### C.1 What stays from v2

- §1 Competition facts (2026 rubric, Nov-8 deadline, arXiv preprint allowed)
- §2 Linked code track anchored to ARC-AGI-3 (`ls20-9607627b` Level 1 live solve)
- §4 Three-contribution set:
  - **C1** Absolute Sovereignty on the reasoning hot path
  - **C2** TRM-as-Avatar game-loop architecture
  - **C3** ActionBuffer 288-byte binary contract
- §5 Six required sections mapped into 6 pages
- §6 Python-exit arc as the spine
- §9 Risk register

### C.2 What changes from v2

**Replace §5 Approach §§3-5 with explicit *deferred-to-companion* framing.** The new Paper A Approach opens with:

> "K3D rests on six design choices. This paper formalizes three of them; the other three are the subjects of companion preprints cited in §3 Prior Work and §6 Conclusion."

Then lists all six with one-line summaries and links to the five companion papers (even if B-F are not yet written — list them as "in preparation" with target arXiv categories). This:

- Saves ~1.5 pages in the 6-page budget
- Signals to reviewers that K3D is a **research program**, not a single paper
- Routes "but what about X?" reviewer objections to the companion-paper pointer

### C.3 Novelty axis: what specifically is new in Paper A (for the rubric)

Anchors reference restored `ATTRIBUTIONS.md` sections (post-commit-1 state).

| Claim | Evidence | Prior-art anchor | Genuinely novel? |
|---|---|---|---|
| Zero-fallback sovereign reasoning substrate on consumer GPU | `scripts/sovereignty_preflight.sh` clean full-tree; Phase 7.6 live_server purge spec | ATTR §1.2.1 (TRM base), §5.1 (CUDA/PTX) | ✅ — sovereignty-as-absolute-rule is K3D-novel; add forward cite to Paper F |
| TRM-as-Avatar game-loop (not Python orchestration) | `trm_step_fused.ptx`; 4000→200 LoC Python target; Phase D spec 03-23 → Phase 7.6 04-18 arc | ATTR §1.2.1 (TRM architecture by Jolicoeur-Martineau) | ✅ — *embodiment framing* is K3D-novel (TRM paper does not specify game-loop execution model); coinage C7 |
| ActionBuffer 288-byte host/device contract with no deserialization | `knowledge3d/cranium/sovereign/loader.py` | — (no direct prior art) | ✅ — K3D-novel |
| Python-exit arc as an *architectural lesson* about hybrid-path drift | §6 six dated artifacts: Phase D 03-23 → Phase 6.C 04-11 → Kill Python Dispatch 04-17 → Absolute Sovereignty Purge 04-18 → Phase 7.6 04-18 | ATTR §4.4 (Christoph Dorn — defeasible logic context for rule-strength argument) | ✅ — case-study contribution; coinage C11 (Absolute Sovereignty Purge ruling) |
| Live ARC-AGI-3 solve on `ls20-9607627b` Level 1 via sovereign path | Run trace + House JSONL; ARC-3 SDK WINE proceduralization | ATTR §1.3 (ARC-AGI by Chollet) | ⚠ — empirical result, NOT theoretical novelty; use as §5 Results evidence, not §4 Contribution |
| Transfer Yard Algorithm default-on tiers 1/2/3 with measured 18-28× vs NumPy | `infix_to_rpn.py`, `lightweight_rpn.py`, tier-dispatch benchmarks | ATTR §1.4 (Transfer Yard by Abu El Haijaa et al. 2024) | ⚠ — we adapt, paper cites their 15-51% claim; position as "validation + GPU adaptation" |
| Ternary-first execution block (TERNARY_* 0x100-0x10F + TQUANT) measured 850-1000× vs Python for logic | Opcode dispatch benchmarks | ATTR §3.1 (Setun/Brusentsov), §3.5 (duplicate — cleanup) — **needs BitNet b1.58 entry (A.3 row 16)** | ⚠ — defer deep ternary contribution to Paper E; Paper A only mentions ternary block as a sovereignty-enabler |

**Non-goals restated:** no modeling claims (C2 describes the substrate the model runs on, not architectural novelty of the MLP itself); no SHGI claims; no ARC-AGI-2 code submission; no venue hedging.

---

## Part D — Timeline

| Window | Paper A (ARC Prize 2026) | Attributions | Series B-F |
|---|---|---|---|
| **Apr 18 – Apr 25** | Freeze scope (this spec); confirm authorship policy with Daniel | Claude drafts ATTRIBUTIONS §2-5 text; Codex lands file; **§4.4.1 semantic-gravity provenance split + §6.4 Form→Meaning external validation + Material Enablers section all landed 2026-04-18/19** | Christoph notified re: Paper B (Daniel first, him second) |
| **Apr 26 – May 10** | Sonnet drafts §§1-2 Intro/Prior-Work using new ATTRIBUTIONS as source of truth; Haiku builds tables/notation glossary | ATTRIBUTIONS merged to `main` | Paper B outline co-authored with Christoph |
| **May 11 – May 31** | Full v1 draft complete; ask_cloud lit sweep; kimi_swarm A/B critique | — | Paper C and D outlines |
| **Jun 1 – Jun 30** | arXiv preprint v1 → cs.AI; Kaggle competition milestone | — | Paper B draft v1 |
| **Jul – Oct** | Ablations + reviewer pre-empt + Nov-8 polish | — | Papers C, D outlines → drafts |
| **Nov 8, 2026** | **Paper A submitted to ARC Prize 2026 Paper Track** | — | — |
| **Dec 2026+** | Results Dec-4; post-mortem | — | Paper B arXiv; Papers C-F staggered through 2027 |

---

## Part E — Actions (who does what)

### E.1 Claude (architecture)

1. ✅ This spec.
2. **Next:** Author `ATTRIBUTIONS.md` §2-5 section texts (draft prose for each row in A.3 and A.4) — hand off to Codex as a clean PR diff.
3. **After Daniel sign-off on authorship policy:** Draft Paper A skeleton following v2 §5 with the C.2 defer-framing baked in.
4. **Defer:** Papers B-F outlines start only after Paper A skeleton locks.

### E.2 Daniel (decisions only architecture-partner-scope cannot make)

1. **Series framing:** endorse or amend the A-through-F map.
2. **Paper B authorship (resolved 2026-04-19):** Daniel Campos Ramos first author (idea + formula); Christoph Dorn second author (term coinage). Decision recorded in [`feedback_semantic_gravity_provenance_corrected.md`](../../home/daniel/.claude/projects/-K3D-GitHub-Knowledge3D/memory/feedback_semantic_gravity_provenance_corrected.md) and ATTRIBUTIONS §4.4.1. Christoph should be notified before preprint upload.
3. **ATTRIBUTIONS expansion:** endorse the A.3 + A.4 entry list, flag any missing attributions.
4. **Non-goals:** confirm the non-goals in §C.2 (no SHGI claims, no ARC-2 submission, no venue hedging).

### E.3 Codex (implementation — after this spec approved)

1. ATTRIBUTIONS.md structural expansion (§2-5 scaffolding) once Claude's section-text drafts land.
2. Cross-link ATTRIBUTIONS.md from `docs/vocabulary/README.md` and `CLAUDE.md`.
3. Nothing else from Codex until Paper A skeleton exists.

### E.4 Ollama specialists (standing delegation)

- `plan_task` → Paper A §§1-2 section outlines (before Sonnet drafting begins)
- `ask_cloud` → literature sweep for each A.3 attribution (fill in missing dates/DOIs)
- `kimi_swarm` → A/B critique of Paper A C1/C2/C3 claims for reviewer pre-empt
- `extract_facts` → pull numbers and run-traces for Paper A §5 Results
- `summarize` → compress each `docs/vocabulary/*.md` spec into one paragraph for Paper A §3 Prior Work bullets

---

## Diff vs. v1 of this spec (what changed after the restoration)

1. **A.0 added** — documents the Codex-destruction incident and restoration of 2,310 lines from `c742e2dd~1`.
2. **A.3 re-scoped** — reduced from "14 missing" to "8 genuinely missing" (BitNet b1.58, Method of Loci, A\*, LoRA, Transformer/SwiGLU, OSI, RETE, Knuth optimal-radix) after discovering TRM/ARC-AGI/Setun/Matryoshka/Morton/Defeasible-Logic/Christoph-Dorn/Milton-Ponson were ALREADY present in the restored file.
3. **A.4 added** — structural fixes (section-numbering collisions pre-existing in the 2,310-line original).
4. **A.5 changed** — from "one big expansion PR" to "two-commit strategy" (restoration done; gap-fill + cleanup later).
5. **C.3 rewritten** — novelty-claim table now anchors to actually-present ATTRIBUTIONS sections with explicit "genuinely novel?" column; adds two new rows (Transfer Yard tier-dispatch + ternary execution block); demotes Live ARC-AGI-3 solve from contribution to Results evidence.
6. **Part D timeline unchanged** — restoration commit slots into Apr 18-25 window; gap-fill commit into Apr 26-May 10.

## Closing

The ARC Prize 2026 deadline is Nov 8. We have 29 weeks. One paper is achievable and defensible; six papers in six pages is neither. Post-restoration, the attribution foundation is materially stronger than I first believed — most of K3D's prior art is already documented. Paper A locks on three contributions (C1 sovereignty, C2 TRM-as-Avatar, C3 ActionBuffer); the other five concepts become companion preprints (Papers B-F); the 8 remaining missing attributions land before Paper A §3 Prior Work is drafted.

**Gating questions for Daniel:**

1. Endorse the series framing (Papers A-F) + two-commit ATTRIBUTIONS strategy?
2. Christoph Dorn as first author on Paper B (semantic gravity) — ping him?
3. Greenlight Claude to draft the 8 missing attribution section texts (BitNet b1.58, Method of Loci, A\*, LoRA, Transformer/SwiGLU, OSI, RETE, Knuth optimal-radix) so Codex can merge them as Commit 2?
